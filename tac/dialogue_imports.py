"""Conversas locais, validadas e separadas por grupo. Nenhum conteúdo é executado."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from .conversation_lessons import canonical, prompt_text
from .db import now
from .diagnostics import check_resources
from .learning import Cancelled

SPLITS = ('train', 'validation', 'test')
FIELDS = {'messages', 'answer', 'skill', 'group', 'split'}
MAX_RECORDS = 20000
MAX_EXAMPLES = 50000
MAX_LINE_BYTES = 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def prompt_key(row):
    # Considera o contexto que realmente chega ao chat, além da forma completa.
    return digest(canonical(prompt_text(row['messages'][-7:])))


def automatic_split(group):
    bucket = int(hashlib.sha256(('TAC-groups-v1:' + group).encode()).hexdigest()[:8], 16) % 1000
    return 'train' if bucket < 800 else 'validation' if bucket < 900 else 'test'


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Campo JSON repetido: ' + key)
        result[key] = value
    return result


def read_records(raw, suffix):
    """Retorna registros com linha de origem; erros não ocultam outros registros."""
    text = raw.decode('utf-8-sig', errors='strict')
    if suffix == '.jsonl':
        for line_number, line in enumerate(text.split('\n'), 1):
            if not line.strip():
                continue
            try:
                if len(line.encode('utf-8')) > MAX_LINE_BYTES:
                    raise ValueError('Linha maior que 1 MiB. Divida as conversas.')
                value = json.loads(line, object_pairs_hook=_json_object)
                yield line_number, value, None
            except (ValueError, RecursionError) as exc:
                yield line_number, None, 'JSON inválido: ' + str(exc)
        return
    if suffix != '.txt':
        raise ValueError('Importe conversas em .jsonl ou .txt.')
    block, start, started = [], 1, False
    for number, line in enumerate(text.splitlines(), 1):
        if line == '=== conversa ===':
            if block:
                yield _txt_record(block, start) if started else (start,None,'Falta o cabeçalho === conversa === antes da conversa.')
            block, start, started = [], number, True
        elif line.strip() or block:
            block.append((number, line))
    if block:
        yield _txt_record(block, start) if started else (start,None,'Falta o cabeçalho === conversa === antes da conversa.')


def _txt_record(block, start):
    header, messages, current = {}, [], None
    try:
        for number, line in block:
            if not line.strip():
                if current is not None:
                    current[1] += '\n'
                continue
            if len(line.encode('utf-8')) > MAX_LINE_BYTES:
                raise ValueError('Linha maior que 1 MiB.')
            if line.startswith('  '):
                if current is None:
                    raise ValueError('Continuação de fala sem uma fala anterior.')
                current[1] += '\n' + line[2:]
                continue
            match = re.fullmatch(r'(group|skill|split):\s*(\S+)\s*', line)
            if match and not messages:
                key, value = match.groups()
                if key in header:
                    raise ValueError('Campo repetido no cabeçalho: ' + key)
                header[key] = value
                continue
            match = re.match(r'^(Usuário|TAC):\s?(.*)$', line)
            if not match:
                raise ValueError(f'Papel inválido ou linha {number} sem marcador Usuário:/TAC:. Use dois espaços para continuar uma fala.')
            current = ['user' if match[1] == 'Usuário' else 'assistant', match[2]]
            messages.append(current)
        if not messages or messages[-1][0] != 'assistant':
            raise ValueError('Conversa TXT incompleta: termine com uma resposta TAC:.')
        return start, {**header, 'messages':[{'role':r,'content':t.strip()} for r,t in messages[:-1]],
                       'answer':messages[-1][1].strip()}, None
    except ValueError as exc:
        return start, {'group':header.get('group')}, str(exc)


def normalize_record(record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError('Cada registro precisa conter somente messages, answer, skill, group e split.')
    for key, limit in (('group',80), ('skill',64)):
        if not isinstance(record[key], str) or not re.fullmatch(r'[a-z0-9_.-]{1,' + str(limit) + '}', record[key]):
            raise ValueError(f'{key} inválido: use até {limit} letras minúsculas sem acentos, números, ponto, hífen ou sublinhado.')
    if record['split'] not in (*SPLITS,'auto'):
        raise ValueError('split inválido: use train, validation, test ou auto.')
    messages = record['messages']
    if not isinstance(messages, list) or not 1 <= len(messages) <= 101:
        raise ValueError('messages precisa ter de 1 a 101 falas.')
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message,dict) or set(message) != {'role','content'}:
            raise ValueError('Cada fala precisa conter role e content.')
        role, content = message['role'], message['content']
        if role not in ('user','assistant'):
            raise ValueError(f'Papel inválido na fala {index+1}: use user ou assistant.')
        if not isinstance(content,str) or not content.strip() or len(content) > 200000 or '\x00' in content:
            raise ValueError(f'Texto vazio, inválido ou muito grande na fala {index+1}.')
        normalized.append([role,content.strip()])
    if normalized[-1][0] != 'user':
        raise ValueError('A última mensagem de messages precisa ser do usuário (user).')
    if any(role != ('user' if i % 2 == 0 else 'assistant') for i,(role,_) in enumerate(normalized)):
        raise ValueError('As falas precisam alternar entre usuário e TAC, começando pelo usuário.')
    answer = record['answer']
    if not isinstance(answer,str) or not answer.strip() or len(answer) > 200000 or '\x00' in answer:
        raise ValueError('answer precisa ser uma resposta não vazia, com tamanho permitido.')
    return {**record, 'messages':normalized, 'answer':answer.strip()}


def derive_examples(record, tokenizer, context, line):
    sequence = [*record['messages'], ['assistant',record['answer']]]
    episode = digest([record['group'], sequence])
    rows, warnings = [], []
    for index in range(1, len(sequence), 2):
        answer = sequence[index][1]
        size = len(tokenizer.encode(' ' + answer, special=False)) + 1
        if size >= context - 4:
            raise ValueError(f'Resposta da fala {index+1} com {size} tokens. Limite: {context-5}, contando espaço inicial e encerramento. A resposta não foi cortada.')
        row = {'messages':sequence[:index], 'answer':answer,'skill':record['skill'],'group':record['group'],
               'episode':episode, 'turn':index//2, 'source_line':line, 'imported':True, 'mixture':'imported'}
        if size > 128:
            warnings.append(f'Linha {line}, fala {index+1}: a resposta cabe no treino, mas o chat gera no máximo 128 novos tokens.')
        prefix = len(tokenizer.encode(prompt_text(row['messages'][-7:]), special=False))
        if len(row['messages']) > 7 or prefix > context - size:
            warnings.append(f'Linha {line}, fala {index+1}: o histórico será recortado para caber no contexto do chat.')
        row['prompt_hash'] = prompt_key(row)
        row['example_hash'] = digest([canonical(prompt_text(row['messages'][-7:])), answer])
        rows.append(row)
    return rows, warnings


class ConversationImports:
    def __init__(self, config, db, gate):
        self.config, self.db, self.gate = config, db, gate

    def revision(self):
        return digest(self.db.query('SELECT id,content_hash,enabled FROM dialogue_imports ORDER BY id'))

    def list(self):
        return self.db.query('SELECT i.*,count(e.id) AS examples,count(DISTINCT e.group_id) AS groups FROM dialogue_imports i LEFT JOIN dialogue_examples e ON e.import_id=i.id GROUP BY i.id ORDER BY i.created DESC,i.id')

    def set_enabled(self, import_id, enabled):
        with self.gate.writing():
            self.db.execute('UPDATE dialogue_imports SET enabled=? WHERE id=?',(int(enabled),import_id))

    def preview(self, path, tokenizer, context=512, reserved=None, program_train=None, progress=lambda *_:None, cancel=None):
        path = Path(path).resolve()
        if path.suffix.lower() not in ('.jsonl','.txt') or not path.is_file():
            raise ValueError('Escolha um arquivo de conversas .jsonl ou .txt.')
        if path.stat().st_size > self.config.get('max_file_mb') * 2**20:
            raise ValueError('Arquivo maior que o limite de importação configurado.')
        with path.open('rb') as stream:
            raw = stream.read(int(self.config.get('max_file_mb') * 2**20) + 1)
        if len(raw) > self.config.get('max_file_mb') * 2**20:
            raise ValueError('Arquivo maior que o limite configurado.')
        report = {'records':0,'valid_records':0,'rejected':[],'duplicates':0,'warnings':[], 'unknown_example_count':0}
        records, requests, candidate_count = [], defaultdict(set), 0
        for line, value, error in read_records(raw, path.suffix.lower()):
            if cancel is not None and cancel.is_set():
                raise Cancelled('Conferência de conversas cancelada.')
            report['records'] += 1
            if report['records'] > MAX_RECORDS:
                raise ValueError(f'Limite de {MAX_RECORDS} registros por arquivo. Divida o material.')
            if report['records'] % 50 == 0:
                check_resources(self.config)
                progress(report['records'], f"Conferindo registro {report['records']}…")
            group = value.get('group') if isinstance(value,dict) else None
            known = None
            try:
                if error:
                    raise ValueError(error)
                record = normalize_record(value)
                known = (len(record['messages'])+1)//2
                # Mesmo uma conversa com resposta longa não pode ocultar conflito de split.
                requests[record['group']].add(record['split'])
                rows, warnings = derive_examples(record,tokenizer,context,line)
                if candidate_count + len(rows) > MAX_EXAMPLES:
                    raise ValueError(f'Limite de {MAX_EXAMPLES} exemplos. Divida o arquivo.')
                records.append((line,record,rows))
                candidate_count += len(rows)
                report['warnings'].extend(warnings)
                report['valid_records'] += 1
            except ValueError as exc:
                report['rejected'].append({'line':line,'group':group,'reason':str(exc),'examples':known})
                if known is None:
                    report['unknown_example_count'] += 1
        if not report['records']:
            raise ValueError('Arquivo vazio: nenhuma conversa foi encontrada.')
        saved_groups = {r['id']:r['split'] for r in self.db.query('SELECT id,split FROM dialogue_groups')}
        assignments, conflicts = {}, set()
        for group, values in requests.items():
            explicit = values - {'auto'}
            if len(explicit) > 1 or (group in saved_groups and explicit and saved_groups[group] not in explicit):
                conflicts.add(group)
            else:
                assignments[group] = saved_groups.get(group) or next(iter(explicit), automatic_split(group))
        existing = self.db.query('SELECT e.prompt_hash,e.example_hash,g.split FROM dialogue_examples e JOIN dialogue_groups g ON g.id=e.group_id')
        prompt_splits = defaultdict(set)
        examples = set()
        for row in existing:
            prompt_splits[row['prompt_hash']].add(row['split'])
            examples.add(row['example_hash'])
        rows = []
        for line, record, candidates in records:
            if record['group'] in conflicts:
                report['rejected'].append({'line':line,'group':record['group'],'reason':'O mesmo grupo tem destinos diferentes. O grupo foi recusado nesta importação.','examples':len(candidates)})
            else:
                rows.extend({**r,'split':assignments[r['group']]} for r in candidates)
        # Entre grupos novos, reserva primeiro os exemplos de avaliação.
        priority = {'test':0,'validation':1,'train':2}
        rows.sort(key=lambda r:(priority[r['split']],r['source_line'],r['turn']))
        accepted = []
        for row in rows:
            reason = None
            if row['prompt_hash'] in (reserved or set()):
                reason = 'Exemplo coincide com uma avaliação interna reservada.'
            elif row['split'] != 'train' and row['prompt_hash'] in (program_train or set()):
                reason = 'Esta entrada já aparece no treino do programa; não serve como avaliação importada nova.'
            elif prompt_splits[row['prompt_hash']] - {row['split']}:
                reason = 'Entrada repetida em outro destino de treino/avaliação.'
            elif row['example_hash'] in examples:
                report['duplicates'] += 1
                continue
            if reason:
                report['rejected'].append({'line':row['source_line'],'group':row['group'],'turn':row['turn'],'reason':reason,'examples':1})
            else:
                accepted.append(row)
                examples.add(row['example_hash'])
                prompt_splits[row['prompt_hash']].add(row['split'])
        report.update(accepted_examples=len(accepted), groups=len({r['group'] for r in accepted}),
                      by_split={s:{'groups':len({r['group'] for r in accepted if r['split']==s}),
                                   'examples':sum(r['split']==s for r in accepted)} for s in SPLITS},
                      rejected_examples=sum(r['examples'] or 0 for r in report['rejected']),
                      rejected_records=len({r['line'] for r in report['rejected']}))
        return {'path':str(path),'filename':path.name,'content_hash':hashlib.sha256(raw).hexdigest(),
                'tokenizer_hash':digest(tokenizer.state()) if hasattr(tokenizer,'state') else digest(tokenizer.version),
                'context':context,'rows':accepted,'report':report,'revision':self.revision()}

    def commit(self, preview):
        if not preview['rows']:
            raise ValueError('Nenhum exemplo válido para importar.')
        path = Path(preview['path'])
        with path.open('rb') as stream:
            raw = stream.read(int(self.config.get('max_file_mb') * 2**20) + 1)
        if len(raw) > self.config.get('max_file_mb') * 2**20:
            raise ValueError('Arquivo maior que o limite configurado.')
        if hashlib.sha256(raw).hexdigest() != preview['content_hash']:
            raise ValueError('O arquivo mudou depois da conferência. Confira novamente.')
        import_id = uuid.uuid4().hex
        folder = self.config.data/'dialogue_imports'
        target = folder/(import_id + path.suffix.lower())
        try:
            with self.gate.writing(), self.db.lock, self.db.connect() as c:
                self.gate.check()
                c.execute('BEGIN IMMEDIATE')
                if self.revision() != preview['revision']:
                    raise ValueError('A lista de conversas mudou. Confira o arquivo novamente.')
                folder.mkdir(parents=True,exist_ok=True)
                temp = target.with_suffix('.tmp')
                with temp.open('wb') as f:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp,target)
                c.execute('INSERT INTO dialogue_imports VALUES(?,?,?,?,?,?,?,?)',
                          (import_id,target.name,str(path),preview['content_hash'],preview['tokenizer_hash'],json.dumps(preview['report'],ensure_ascii=False),1,now()))
                for row in preview['rows']:
                    self.gate.check()
                    c.execute('INSERT OR IGNORE INTO dialogue_groups VALUES(?,?,?)',(row['group'],row['split'],now()))
                    actual = c.execute('SELECT split FROM dialogue_groups WHERE id=?',(row['group'],)).fetchone()[0]
                    if actual != row['split']:
                        raise ValueError('O destino de um grupo mudou. Confira novamente.')
                    c.execute('INSERT INTO dialogue_examples(import_id,group_id,skill,messages,answer,prompt_hash,example_hash,episode,turn,source_line,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        (import_id,row['group'],row['skill'],json.dumps(row['messages'],ensure_ascii=False),row['answer'],row['prompt_hash'],row['example_hash'],row['episode'],row['turn'],row['source_line'],now()))
                c.execute('INSERT INTO events(kind,detail,created) VALUES(?,?,?)',('importacao_conversas',
                          json.dumps({'id':import_id,'arquivo':path.name,'conversas':preview['report']['groups'],
                                      'respostas':preview['report']['accepted_examples']},ensure_ascii=False),now()))
            return {'id':import_id,**preview['report']}
        except BaseException:
            target.unlink(missing_ok=True)
            target.with_suffix('.tmp').unlink(missing_ok=True)
            raise

    def snapshot(self, tokenizer_hash):
        rows = self.db.query('SELECT e.*,g.split,i.tokenizer_hash FROM dialogue_examples e JOIN dialogue_imports i ON i.id=e.import_id JOIN dialogue_groups g ON g.id=e.group_id WHERE i.enabled=1 ORDER BY e.id')
        if any(r['tokenizer_hash'] != tokenizer_hash for r in rows):
            raise ValueError('Há conversas conferidas com outro vocabulário. Confira novamente antes de treinar.')
        return [{ 'messages':json.loads(r['messages']),'answer':r['answer'],'skill':r['skill'],'group':r['group_id'],
                  'episode':r['episode'],'turn':r['turn'],'source_line':r['source_line'],'import_id':r['import_id'],
                  'imported':True,'mixture':'imported','split':r['split']} for r in rows]


def merge_imports(data, rows):
    """Acrescenta um pool de treino; benchmarks internos continuam idênticos."""
    from .neural.conversation import ConversationData
    reserved = {prompt_key(r) for key,values in data.sets.items() if key != 'train' for r in values}
    internal_train = {prompt_key(r) for r in data.sets['train']}
    kept, skipped = [], []
    for row in rows:
        if prompt_key(row) in reserved or (row['split'] != 'train' and prompt_key(row) in internal_train):
            skipped.append({'group':row['group'],'line':row['source_line'],'reason':'Sobreposição com os exercícios internos atuais.'})
        else:
            kept.append(row)
    if not kept:
        return data, skipped
    # Uma avaliação importada não pode aparecer no pool antigo de gradientes.
    heldout = {prompt_key(r) for r in kept if r['split'] != 'train'}
    own_train = [r for r in kept if r['split']=='train' and prompt_key(r) not in heldout]
    sets = {**data.sets,'train':[*data.sets['train'],*own_train],
            'validation_imported':[r for r in kept if r['split']=='validation'],
            'test_imported':[r for r in kept if r['split']=='test']}
    return ConversationData(sets,data.tokenizer), skipped


def training_counts(data):
    user = [r for r in data.sets['train'] if r.get('imported')]
    return {'user_groups':len({r['group'] for r in user}),'user_examples':len(user),
            'program_examples':len(data.sets['train'])-len(user),
            'reserved':{s: {'groups':len({r['group'] for r in data.sets.get(s+'_imported',[])}),
                            'examples':len(data.sets.get(s+'_imported',[]))} for s in ('validation','test')}}


def imported_kind(step, micro, accumulation):
    """11 importados, 11 revisados, 12 linguagem, 6 antigos a cada 40 lotes."""
    position = (step * accumulation + micro) % 40
    if position < 22:
        return 'imported' if position % 2 == 0 else 'reviewed'
    return 'language' if position < 34 else 'legacy'
