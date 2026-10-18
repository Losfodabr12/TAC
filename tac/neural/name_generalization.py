"""Dados sintéticos e medidas limitadas de nomes. Não busca nem responde pelo modelo."""
from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter

import torch
from torch.nn import functional as F

from ..conversation_lessons import canonical, prompt_text
from ..dialogue_imports import digest
from .conversation import ConversationData
from .controlled_eval import trace_reply

TRAIN_FAMILIES = tuple('t' + str(i) for i in range(8))
DEV_FAMILIES = tuple('v' + str(i) for i in range(4))
TEST_FAMILIES = tuple('s' + str(i) for i in range(4))

COMMON = (
    ['Ana', 'Bruno', 'Carla', 'Diego', 'Elisa', 'Felipe', 'Gabriela', 'Hugo',
     'Isabela', 'João', 'Karen', 'Lucas', 'Márcia', 'Natan', 'Olívia', 'Pedro'],
    ['Renata', 'Sérgio', 'Talita', 'Vítor', 'Wesley', 'Yasmin', 'Cecília', 'Davi'],
    ['André', 'Beatriz', 'Cláudio', 'Débora', 'Eduardo', 'Flávia', 'Gustavo', 'Helena'])
RARE = (
    ['Zuleica', 'Eudóxia', 'Apolônio', 'Odete', 'Iolanda', 'Jaciara', 'Ubiratã', 'Érico',
     'Íris', 'Guaraci', 'Crisóstomo', 'Maíra', 'Átila', 'Noêmia', 'Gonçalo', 'Zoraide'],
    ['Açucena', 'Dâmaris', 'Ilídio', 'Orestes', 'Tarsila', 'Nereu', 'Zênia', 'Eloá'],
    ['Agnelo', 'Berenice', 'Calixto', 'Doroteia', 'Etelvina', 'Firmino', 'Guiomar', 'Hermenegildo'])


def name_pools():
    pools, used, prior_parts = [], set(), set()
    for split, count in enumerate((160, 48, 48)):
        pool = [{'name': n, 'kind': k} for k, ns in [('comum', COMMON[split]), ('raro', RARE[split])] for n in ns]
        used.update(canonical(r['name']) for r in pool)
        rng = random.Random(7301 + split * 9109)
        syllables = ['za', 'ri', 'lu', 'vê', 'né', 'ki', 'tá', 'xo', 'mi', 'çá', 'fy', 'wá', 'jo', 'pü', 'dô', 'lé']
        while len(pool) < count:
            n = ''.join(rng.choice(syllables) for _ in range(rng.randint(2, 7))).capitalize()
            if len(pool) % 7 == 0:
                n += '-' + ''.join(rng.choice(syllables) for _ in range(2)).capitalize()
            if canonical(n) in used or any(canonical(p) in prior_parts for p in re.split(r"[-'\s]+", n)):
                continue
            used.add(canonical(n))
            pool.append({'name': n, 'kind': 'inventado'})
        pools.append(pool)
        prior_parts.update(canonical(p) for e in pool for p in re.split(r"[-'\s]+", e['name']))
    return pools


def single_answer(name, role, style):
    if role == 'previous':
        return [name + '.', 'O nome anterior era ' + name + '.', 'Antes você usava ' + name + '.'][style % 3]
    return [name + '.', 'Seu nome atual é ' + name + '.', 'Você se chama ' + name + '.',
            'O nome informado é ' + name + '.'][style % 4]


def episode(family, a, b, c, style):
    """Cada família altera também o fluxo da conversa, não só nomes/saudações."""
    turns = []
    def add(user, slots, skill):
        if len(slots) == 2:
            previous, current = slots['previous'], slots['current']
            answer = [f'Antes: {previous}; agora: {current}.',
                      f'O nome anterior era {previous}; o atual é {current}.',
                      f'Agora: {current}; antes: {previous}.'][style % 3]
        else:
            role, name = next(iter(slots.items()))
            answer = single_answer(name, role, style + len(turns))
        turns.append((user, answer, slots, skill))
    if family == 't0':
        add(f'Oi, meu nome é {a}.', {'current': a}, 'identificacao')
        add('Como eu me chamo?', {'current': a}, 'recuperacao')
        add(f'Corrigindo: meu nome é {b}.', {'current': b}, 'correcao')
        add('Qual nome eu tinha informado antes da correção?', {'previous': a}, 'nome_anterior')
    elif family == 't1':
        add(f'Pode me chamar de {a}.', {'current': a}, 'identificacao')
        add(f'Prefiro trocar meu nome nesta conversa para {b}.', {'current': b}, 'correcao')
        add('Compare meu nome de antes com o atual.', {'previous': a, 'current': b}, 'contraste')
    elif family == 't2':
        add(f'Sou {a}; {c} é meu colega.', {'current': a}, 'identificacao')
        add(f'{c} vai conversar depois. Qual é o MEU nome?', {'current': a}, 'recuperacao')
        add(f'Houve um erro: para mim, use {b}.', {'current': b}, 'correcao')
        add('Como você deve me chamar agora?', {'current': b}, 'apos_correcao')
    elif family == 't3':
        add(f'Anote meu nome exatamente como escrevi: {a}.', {'current': a}, 'copia')
        add('Repita meu nome sem mudar as letras.', {'current': a}, 'recuperacao')
        add(f'Atualize a anotação: {b}.', {'current': b}, 'correcao')
        add('Quais eram e são os nomes anotados?', {'previous': a, 'current': b}, 'contraste')
    elif family == 't4':
        add(f'Quem está falando é {a}.', {'current': a}, 'identificacao')
        add(f'Assinei errado. Minha assinatura correta é {b}.', {'current': b}, 'correcao')
        add('Recupere a assinatura que eu substituí.', {'previous': a}, 'nome_anterior')
    elif family == 't5':
        add(f'Meu nome nesta conversa será {a}.', {'current': a}, 'identificacao')
        add(f'Troque para {c}.', {'current': c}, 'correcao')
        add(f'Mude outra vez: agora é {b}.', {'current': b}, 'correcao')
        add('Diga o nome imediatamente anterior e o que vale agora.', {'previous': c, 'current': b}, 'contraste')
    elif family == 't6':
        add(f'Entre nós: {c} é minha amiga, eu sou {a}.', {'current': a}, 'identificacao')
        add('Quem está conversando com você?', {'current': a}, 'recuperacao')
        add(f'De agora em diante, me chame de {b}.', {'current': b}, 'correcao')
    elif family == 't7':
        add(f'No cadastro está {a}. Esse é meu nome.', {'current': a}, 'identificacao')
        add('Confirme o nome do cadastro.', {'current': a}, 'recuperacao')
        add(f'Acabei de corrigir o cadastro para {b}.', {'current': b}, 'correcao')
        add('Que nome está atualizado?', {'current': b}, 'apos_correcao')
    elif family == 'v0':
        add(f'Para evitar confusão, {c} é outra pessoa. Eu me apresento como {a}.', {'current': a}, 'identificacao')
        add(f'Escrevi {a} por engano; substitua por {b} quando se referir a mim.', {'current': b}, 'correcao')
        add('Se fosse me apresentar a alguém neste instante, que nome usaria?', {'current': b}, 'apos_correcao')
        add('Recupere as duas versões do meu nome na ordem anterior, atual.', {'previous': a, 'current': b}, 'contraste')
    elif family == 'v1':
        add(f'Estou preenchendo uma ficha. No campo de nome escrevi {a}.', {'current': a}, 'identificacao')
        add(f'A ficha voltou para revisão: o campo passou a {b}.', {'current': b}, 'correcao')
        add('Qual era o valor desse campo na primeira versão?', {'previous': a}, 'nome_anterior')
    elif family == 'v2':
        add(f'Chame-me de {a} enquanto falamos.', {'current': a}, 'identificacao')
        add(f'Vou manter esse tratamento por enquanto. {c} é o nome do meu irmão.', {'current': a}, 'recuperacao')
        add(f'Não é para usar meu irmão como referência. Atualizei meu próprio nome para {b}.', {'current': b}, 'correcao')
        add('Ao dirigir a palavra a mim, qual tratamento você deve escolher?', {'current': b}, 'apos_correcao')
    elif family == 'v3':
        add(f'Você recebeu uma mensagem assinada por {a}. Sou eu.', {'current': a}, 'identificacao')
        add(f'Na próxima mensagem a assinatura será {c}; considere a troca.', {'current': c}, 'correcao')
        add(f'A assinatura mudou novamente para {b}.', {'current': b}, 'correcao')
        add('Sem voltar à primeira assinatura, mostre a penúltima e a última.', {'previous': c, 'current': b}, 'contraste')
    elif family == 's0':
        add(f'Pode registrar quem está aqui: {a}.', {'current': a}, 'identificacao')
        add(f'Esse registro não vale mais para mim. Coloque {b} no lugar.', {'current': b}, 'correcao')
        add('Se você tivesse que responder por mim agora, que nome colocaria?', {'current': b}, 'apos_correcao')
    elif family == 's1':
        add(f'O crachá com {a} é o meu; o com {c} é de outra pessoa.', {'current': a}, 'identificacao')
        add('Qual deles me identifica?', {'current': a}, 'recuperacao')
        add(f'Refizeram meu crachá com {b}.', {'current': b}, 'correcao')
        add('Informe o que havia no crachá e o que há no novo.', {'previous': a, 'current': b}, 'contraste')
    elif family == 's2':
        add(f'Ao ler minha mensagem, veja o remetente: {a}.', {'current': a}, 'identificacao')
        add(f'Passei a usar outro remetente, {b}.', {'current': b}, 'correcao')
        add('Qual remetente deixei de usar?', {'previous': a}, 'nome_anterior')
    elif family == 's3':
        add(f'Apresentei-me como {a}.', {'current': a}, 'identificacao')
        add(f'Depois da apresentação passei a assinar {c}.', {'current': c}, 'correcao')
        add(f'Finalmente decidi ficar com {b}.', {'current': b}, 'correcao')
        add('Qual nome ficou para trás imediatamente antes da decisão final?', {'previous': c}, 'nome_anterior')
    else:
        raise ValueError('Família desconhecida.')
    return turns


def make_panel(label, pool, families, per_family):
    rows = []
    for fi, family in enumerate(families):
        for j in range(per_family):
            i = fi * per_family + j
            indices = [(i * 7) % len(pool), (i * 13 + 31) % len(pool), (i * 11 + 17) % len(pool)]
            for k in range(1, 3):
                while indices[k] in indices[:k]:
                    indices[k] = (indices[k] + 1) % len(pool)
            entries = [pool[k] for k in indices]
            a, b, c = [e['name'] for e in entries]
            messages = []
            group = f'{label}_{family}_{j:03d}'
            for turn, (question, answer, slots, skill) in enumerate(episode(family, a, b, c, i)):
                messages.append(['user', question])
                rows.append({'messages': [list(m) for m in messages], 'answer': answer, 'skill': skill,
                             'group': group, 'episode': group, 'turn': turn, 'family': family, 'imported': True,
                             'mixture': 'imported', 'target_names': slots, 'source_names': [a, b, c],
                             'name_kinds': {e['name']: e['kind'] for e in entries}})
                messages.append(['assistant', answer])
    return rows


def create_sets():
    train, dev, final = name_pools()
    panels = {'train': make_panel('train', train, TRAIN_FAMILIES, 32),
              'train_probe': make_panel('train', train, TRAIN_FAMILIES, 32),
              'dev_names': make_panel('dev_names', dev, TRAIN_FAMILIES, 2),
              'dev_structures': make_panel('dev_structures', train, DEV_FAMILIES, 4),
              'validation': make_panel('validation', dev, DEV_FAMILIES, 4)}
    # Probe contém episódios inteiros que REALMENTE estão no treino.
    selected = {f'train_{f}_{j:03d}' for f in TRAIN_FAMILIES for j in (0, 17)}
    panels['train_probe'] = [r for r in panels['train'] if r['group'] in selected]
    test = make_panel('final', final, TEST_FAMILIES, 4)
    return panels, test, {'train': train, 'development': dev, 'test': final}


def audit_sets(panels, test, inventories, tokenizer):
    names = {s: {canonical(e['name']) for e in pool} for s, pool in inventories.items()}
    assert not names['train'] & names['development'] and not names['train'] & names['test'] and not names['development'] & names['test']
    assert not set(TRAIN_FAMILIES) & set(DEV_FAMILIES) and not set(TRAIN_FAMILIES) & set(TEST_FAMILIES) and not set(DEV_FAMILIES) & set(TEST_FAMILIES)
    all_sets = {**panels, 'test': test}
    audit = {}
    for label, rows in all_sets.items():
        data = ConversationData({'train': rows}, tokenizer)
        lengths, crop, bad_copy = [], [], []
        for row in rows:
            x, y = data.encode(row, 512)
            lengths.append(len(x))
            prompt = prompt_text(row['messages'][-7:])
            if len(row['messages']) > 7 or len(tokenizer.encode(prompt, special=False)) + int((y != 0).sum()) > 512:
                crop.append(row['group'])
            if any(n not in prompt for n in row['target_names'].values()):
                bad_copy.append(row['group'])
        audit[label] = {'examples': len(rows), 'conversations': len({r['group'] for r in rows}),
                        'by_skill': dict(Counter(r['skill'] for r in rows)), 'families': sorted({r['family'] for r in rows}),
                        'max_sequence': max(lengths), 'cropped': crop, 'expected_name_not_in_prompt': bad_copy,
                        'digest': digest(rows)}
        assert not crop and not bad_copy
    # Não basta confiar nos IDs das famílias: verifica nomes nos textos reais do treino.
    train_text = '\n'.join(prompt_text(r['messages']) + '\n' + r['answer'] for r in panels['train'])
    forbidden = inventories['development'] + inventories['test']
    contamination = [e['name'] for e in forbidden if re.search(r'(?<!\w)' + re.escape(e['name']) + r'(?!\w)', train_text, re.I)]
    assert not contamination, contamination
    tokens = {}
    train_chars = set(''.join(e['name'] for e in inventories['train']))
    for label, pool in inventories.items():
        entries = []
        for e in pool:
            ids = tokenizer.encode(e['name'], special=False)
            decoded = tokenizer.decode(ids)
            assert decoded == e['name']
            entries.append({**e, 'characters': len(e['name']), 'tokens': len(ids),
                            'byte_base_tokens': sum(i < 259 for i in ids),
                            'characters_not_in_train_inventory': sorted(set(e['name']) - train_chars),
                            'roundtrip_exact': True})
        tokens[label] = entries
    return {'panels': audit, 'name_pools': {s: len(v) for s, v in inventories.items()},
            'strict_main_split_disjoint': True, 'reserved_names_in_training': contamination,
            'diagnostic_panels': 'dev_names reutiliza estruturas de treino; dev_structures reutiliza nomes de treino. validation e test não reutilizam nenhum dos dois.',
            'tokenizer_audit': tokens, 'architecture_copy_mechanism': 'atenção causal; não existe pointer/copy head nem decodificação restrita a nomes'}


def name_weights(data, row, context, factor):
    """Repondera somente os tokens de nomes no alvo; não altera sequência/contexto."""
    _, y = data.encode(row, context)
    weight = (y != 0).float()
    text = ' ' + row['answer']
    spans = []
    for name in row['target_names'].values():
        start = text.index(name)
        spans.append((len(text[:start].encode()), len(text[:start + len(name)].encode())))
    answer_ids = data.tokenizer.encode(text, special=False) + [data.tokenizer.eos]
    target_indices = (y != 0).nonzero().flatten().tolist()
    assert len(target_indices) == len(answer_ids)
    offset = 0
    for position, token in zip(target_indices, answer_ids):
        size = len(data.tokenizer.tokens.get(token, b''))
        if any(offset < end and offset + size > start for start, end in spans):
            weight[position] = factor
        offset += size
    return weight


def weighted_loss(logits, y, weights):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0, reduction='none').view_as(y)
    return (loss * weights).sum() / weights.sum()


def parse_answer(text, requested):
    text = text.strip().rstrip('.!?').strip()
    pair_patterns = [(r'Antes:\s*(.+?);\s*agora:\s*(.+)', ('previous', 'current')),
                     (r'O nome anterior era\s+(.+?);\s*o atual é\s+(.+)', ('previous', 'current')),
                     (r'Agora:\s*(.+?);\s*antes:\s*(.+)', ('current', 'previous'))]
    for pattern, roles in pair_patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            return dict(zip(roles, match.groups()))
    for role, prefix in [('previous', r'(?:O nome (?:anterior|antigo) era|Antes você usava)'),
                         ('current', r'(?:Seu nome(?: atual)? é|Você se chama|O nome informado é|Agora você se chama|Você é|Prazer,)')]:
        match = re.fullmatch(prefix + r'\s+(.+)', text, re.I)
        if match:
            return {role: match[1]}
    # Uma resposta curta sem marcador só é interpretável para uma pergunta de um nome.
    return {next(iter(requested)): text} if len(requested) == 1 else {}


def score_name_answer(actual, expected, trace, source_names):
    parsed = parse_answer(actual, expected)
    wanted = sorted(canonical(n) for n in expected.values())
    got = sorted(canonical(n) for n in parsed.values())
    identity = wanted == got
    characters = sorted(expected.values()) == sorted(parsed.values())
    contextual = set(parsed) == set(expected) and all(canonical(parsed[k]) == canonical(v) for k, v in expected.items())
    coherent = contextual and trace['eos'] and not trace['user_role_emitted'] and not trace['repeated_trigram_3_times']
    if contextual:
        error = 'correct' if characters else 'changed_characters'
    elif identity:
        error = 'swapped_roles'
    elif any(canonical(v) in {canonical(n) for n in source_names} for v in parsed.values()):
        error = 'wrong_name_from_context'
    elif parsed and any(canonical(v).startswith(canonical(n)[:3]) for v in parsed.values() for n in expected.values()):
        error = 'partial_or_mutated_copy'
    else:
        error = 'unverified_or_out_of_context'
    return {'name_identity': identity, 'characters_exact': characters, 'context_correct': contextual,
            'coherence_restricted_proxy': coherent, 'parsed': parsed, 'error_type': error,
            'semantic_human_review': None}


def summarize(outputs):
    keys = ('literal_exact', 'name_identity', 'characters_exact', 'context_correct', 'coherence_restricted_proxy')
    result = {'total': len(outputs), **{key: sum(r[key] for r in outputs) for key in keys},
              'eos': sum(r['trace']['eos'] for r in outputs),
              'error_types': dict(Counter(r['error_type'] for r in outputs))}
    result['rates'] = {k: result[k] / len(outputs) if outputs else None for k in keys}
    return result


@torch.inference_mode()
def evaluate_names(model, rows, device, own_history=False, gate=None):
    model.eval()
    outputs, histories = [], {}
    for row in rows:
        if gate:
            gate.check()
        if own_history:
            history = histories.setdefault(row['episode'], [])
            history.append(row['messages'][-1])
            messages = history[-7:]
        else:
            messages = row['messages'][-7:]
        # Não recebe expected, listas de nomes nem rótulo da habilidade.
        actual, trace = trace_reply(model, messages, device, max_tokens=80)
        scored = score_name_answer(actual, row['target_names'], trace, row['source_names'])
        outputs.append({'group': row['group'], 'turn': row['turn'], 'family': row['family'], 'skill': row['skill'],
                        'messages': [list(m) for m in messages], 'expected': row['answer'],
                        'target_names': row['target_names'], 'name_kinds': row['name_kinds'],
                        'actual': actual, 'trace': trace, 'literal_exact': actual == row['answer'], **scored})
        if own_history:
            history.append(['assistant', actual])
    return {**summarize(outputs), 'by_skill': {s: summarize([r for r in outputs if r['skill'] == s]) for s in sorted({r['skill'] for r in outputs})},
            'by_family': {s: summarize([r for r in outputs if r['family'] == s]) for s in sorted({r['family'] for r in outputs})},
            'by_name_kind': {s: summarize([r for r in outputs if s in {r['name_kinds'][n] for n in r['target_names'].values()}])
                             for s in ('comum', 'raro', 'inventado')},
            'outputs': outputs, 'own_history': own_history,
            'semantic_metric': 'Não há avaliador semântico geral. A coerência é proxy restrita, com gramática explícita, vínculo factual, EOS e ausência de repetição.'}
