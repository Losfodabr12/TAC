"""Estatísticas sem copiar textos de datasets para o arquivo de diagnóstico."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter,defaultdict

from .conversation_lessons import canonical,prompt_text


def side(split):
    return 'train' if split=='train' else 'validation' if split.startswith('validation') else 'test' if split.startswith('test') else split


def fingerprint(text):
    return hashlib.sha256(canonical(text).encode('utf-8',errors='replace')).digest()


def length_stats(values):
    return {'n':len(values),'min':min(values) if values else None,'max':max(values) if values else None,
            'media':sum(values)/len(values) if values else None}


def overlap(keys,groups):
    pairs={}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        pairs[a+'__'+b]={'entradas_normalizadas_repetidas':len(keys[a]&keys[b]),
                         'grupos_repetidos':len(groups[a]&groups[b])}
    return {'comparacoes':pairs,'metodo':'Igualdade normalizada de entradas; group/episode quando registrados. Não detecta toda paráfrase nem todos os recortes de contexto.',
            'independencia_semantica':'não verificada; não se pode afirmar ausência total de vazamento.'}


def analyze_dataset(value, dataset_id, privacy, contexts=()):
    result={'dataset_id':privacy.identifier(dataset_id),'estado':'analisado','textos_exportados':False,
            'contextos_referenciados':sorted(set(contexts)),'splits':{}}
    expected=value.get('hash')
    if 'sets' in value:
        kind='dialogo'; sets=value['sets']
        content={'sets':sets,'tokenizer':value['tokenizer_state']} if value.get('tokenizer_state') else sets
    elif 'rows' in value:
        kind='classificador'; sets=value['rows']; content=sets; expected=dataset_id
    elif isinstance(value.get('train'),list) and isinstance(value.get('validation'),list):
        kind='leitura'; sets={s:value[s] for s in ('train','validation')}
        content={key:value.get(key,[]) for key in ('train','validation','provenance')}
    else:
        return {**result,'estado':'formato_nao_reconhecido'}
    result['tipo']=kind
    observed=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
    result['integridade']={'hash_esperado':privacy.identifier(expected),'hash_confere':observed==expected if expected else None,
                         'nome_confere_hash':dataset_id==observed}
    keys=defaultdict(set); groups=defaultdict(set); precise_keys=defaultdict(set)
    tokenizer=None
    if kind=='dialogo':
        from .neural.bpe import LocalBPETokenizer
        from .neural.tokenizer import ByteTokenizer
        tokenizer=LocalBPETokenizer.from_state(value['tokenizer_state']) if value.get('tokenizer_state') else ByteTokenizer()
        result['tokenizador']={'tipo':tokenizer.version,'vocab_size':tokenizer.vocab_size}
    for split,rows in sets.items():
        if not isinstance(rows,list):continue
        stats={'exemplos':len(rows),'categorias':{},'mistura':{},'anomalias':Counter()}
        categories=Counter(); mixtures=Counter(); lengths=[]; answers=[]; prefix_lengths=[]; duplicates=Counter(); targets=defaultdict(set)
        for row in rows:
            try:
                if kind=='leitura':
                    if not isinstance(row,str):raise ValueError()
                    text=row; answer=None; skill='texto'; mixture='texto'
                else:
                    skill=row.get('skill',row.get('intent','indisponivel'))
                    mixture=row.get('mixture','legacy' if kind=='dialogo' else 'classificador')
                    answer=row.get('answer')
                    messages=row.get('messages',[])
                    text=prompt_text(messages[-7:] if row.get('imported') else messages) if kind=='dialogo' else row['text']
                    group=row.get('group',row.get('episode'))
                    if group is not None:groups[side(split)].add(fingerprint(str(group)))
                categories[privacy.key(skill)]+=1; mixtures[privacy.key(mixture)]+=1
                key=fingerprint(text); keys[side(split)].add(key); duplicates[key]+=1
                precise_keys[side(split)].add(hashlib.sha256(' '.join(unicodedata.normalize('NFKC',text).casefold().split()).encode('utf-8')).digest())
                lengths.append(len(text))
                if not text.strip():stats['anomalias']['entradas_vazias']+=1
                if answer is not None:
                    targets[key].add(fingerprint(answer))
                    size=len(tokenizer.encode(' '+answer,special=False))+1
                    prefix=len(tokenizer.encode(text,special=False))+1
                    answers.append(size); prefix_lengths.append(prefix)
                    if size>128:stats['anomalias']['respostas_maiores_que_geracao_128']+=1
                if kind=='leitura' and '\ufffd' in text:stats['anomalias']['textos_com_caractere_substituto']+=1
            except (ValueError,TypeError,KeyError,UnicodeError):
                stats['anomalias']['registros_invalidos']+=1
        stats['categorias']=dict(categories); stats['mistura']=dict(mixtures)
        stats['tamanho_entrada_caracteres']=length_stats(lengths)
        stats['duplicatas_entrada_no_split']=sum(n-1 for n in duplicates.values())
        stats['entradas_com_respostas_diferentes']=sum(len(t)>1 for t in targets.values())
        if tokenizer:
            stats['tamanho_resposta_tokens_com_espaco_e_eos']=length_stats(answers)
            stats['por_contexto']={str(ctx):{'respostas_nao_cabem':sum(n>=ctx-4 for n in answers),
                'prefixos_recortados_no_treino':sum(n+p>ctx+1 for n,p in zip(answers,prefix_lengths))} for ctx in contexts}
        stats['anomalias']=dict(stats['anomalias'])
        result['splits'][privacy.key(split)]=stats
    result['vazamento']=overlap(keys,groups)
    result['vazamento']['aviso_normalizacao']='A comparação antiga remove operadores e acentos. Duplicatas normalizadas não são prova de perguntas equivalentes ou rótulos errados.'
    result['vazamento']['comparacao_conservadora_preserva_operadores']={a+'__'+b:len(precise_keys[a]&precise_keys[b])
        for a,b in (('train','validation'),('train','test'),('validation','test'))}
    if kind=='leitura':
        result['origens_contagem']=dict(Counter('documento' if 'source_id' in p else 'correcao' if 'correction_id' in p else 'autoral_ou_outra' for p in value.get('provenance',[]) if isinstance(p,dict)))
        result['limite_separacao']='Leitura normalmente separa partes do mesmo trecho; não reserva documentos inteiros. Igualdade de trechos não mede similaridade entre livros.'
    return result
