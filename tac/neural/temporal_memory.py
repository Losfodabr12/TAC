"""Variantes experimentais de exposição temporal. Não modifica os dados originais."""
from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import torch

from ..conversation_lessons import prompt_text
from ..dialogue_imports import digest
from .file_learning import EpochSampler

PRIORITY = ('nome_anterior', 'contraste', 'apos_correcao')
ARMS = ('original', 'intercalado', 'redistribuido', 'temporal_explicito')


def original_plan(rows, seed, count=1920):
    sampler = EpochSampler(rows, seed)
    indexes = {id(r): i for i, r in enumerate(rows)}
    return [indexes[id(r)] for r in sampler.take(count)]


def interleaved_plan(plan, rows):
    """Intercala proporcionalmente ao longo da rodada, sem alterar multiplicidades."""
    queues = defaultdict(deque)
    for index in plan:
        queues[rows[index]['skill']].append(index)
    total = {skill: len(q) for skill, q in queues.items()}
    used = Counter()
    result = []
    while len(result) < len(plan):
        skill = min((s for s in queues if queues[s]), key=lambda s: ((used[s] + .5) / total[s], s))
        result.append(queues[skill].popleft())
        used[skill] += 1
    assert Counter(result) == Counter(plan)
    return result


def redistributed_plan(plan, rows, token_counts, seed):
    """Maior exposição temporal, cobertura total, máximo 3 usos e orçamento igual.

    Troca somente exposições repetidas de correção/recuperação por um exemplo
    temporal de mesmo tamanho de alvo. Mantém pelo menos uma visão de cada
    exemplo; não cria linhas duplicadas no dataset nem excede o teto original.
    """
    rng = random.Random(seed + 670031)
    counts = Counter(plan)
    candidates = defaultdict(list)
    for i, row in enumerate(rows):
        if row['skill'] in PRIORITY:
            candidates[token_counts[i]].append(i)
    for pool in candidates.values():
        rng.shuffle(pool)
    result = list(plan)
    positions = list(range(len(plan)))
    rng.shuffle(positions)
    for position in positions:
        original = result[position]
        if rows[original]['skill'] not in ('correcao', 'recuperacao') or counts[original] <= 1:
            continue
        pool = [i for i in candidates[token_counts[original]] if counts[i] < 3]
        if not pool:
            continue
        recipient = min(pool, key=lambda i: (counts[i], PRIORITY.index(rows[i]['skill']), i))
        result[position] = recipient
        counts[original] -= 1
        counts[recipient] += 1
    assert set(result) == set(plan) and max(counts.values()) <= 3
    assert [token_counts[i] for i in result] == [token_counts[i] for i in plan]
    return result


def temporal_render(rows):
    """Mesmos 928 grupos/alvos; pista temporal e terceira mudança intermediária.

    Esta é uma VARIANTE de apresentação; não é declarada idêntica ao dataset.
    Só os testes de amostragem usam texto integralmente idêntico ao original.
    """
    by_group = defaultdict(dict)
    inventory = {}
    for row in rows:
        by_group[row['group']][row['turn']] = row
        inventory.update(row['name_kinds'])
    names = sorted(inventory)
    labels = {'identificacao': 'Informação inicial do nome: ', 'copia': 'Informação inicial do nome: ',
              'recuperacao': 'Recupere o nome que ainda vale: ',
              'correcao': 'Correção: substitua a informação anterior. ',
              'apos_correcao': 'Considere a informação mais recente: ',
              'nome_anterior': 'Recupere o nome imediatamente anterior ao atual: ',
              'contraste': 'Diferencie o nome substituído do atual, nessa ordem: '}
    result = []
    for row in rows:
        source_names = list(row['source_names'])
        kinds = dict(row['name_kinds'])
        extra = None
        if row['family'] == 't5':
            available = [n for n in names if n not in source_names]
            extra = available[int(digest(row['group'])[:8], 16) % len(available)]
            source_names.append(extra)
            kinds[extra] = inventory[extra]
        messages = []
        for index, (role, text) in enumerate(row['messages']):
            if role == 'user':
                turn = index // 2
                skill = by_group[row['group']][turn]['skill']
                if extra and turn == 1:
                    text = f'Primeiro mudei o nome para {extra}. Depois, {text[0].lower() + text[1:]}'
                text = labels[skill] + text
            messages.append([role, text])
        result.append({**row, 'messages': messages, 'source_names': source_names, 'name_kinds': kinds,
                       'temporal_changes': 3 if row['family'] == 't5' and row['turn'] >= 2 else None})
    assert len(result) == len(rows)
    assert all(a['answer'] == b['answer'] and a['target_names'] == b['target_names']
               for a, b in zip(rows, result))
    return result


def exposure(plan, rows, token_counts):
    seen = Counter(plan)
    return {'unique_examples': len(seen), 'presentations': len(plan),
            'repetitions': len(plan) - len(seen), 'minimum_presentations': min(seen.values()),
            'maximum_presentations': max(seen.values()),
            'by_skill': dict(Counter(rows[i]['skill'] for i in plan)),
            'unique_by_skill': dict(Counter(rows[i]['skill'] for i in seen)),
            'supervised_tokens': sum(token_counts[i] for i in plan),
            'supervised_tokens_by_skill': {s: sum(token_counts[i] for i in plan if rows[i]['skill'] == s)
                                          for s in sorted({r['skill'] for r in rows})},
            'per_example': dict(seen)}


def copy_probe(panels):
    names = {n: kind for r in panels['validation'] for n, kind in r['name_kinds'].items()}
    return [{'messages': [['user', f'Repita exatamente, sem mudar as letras, este nome: {n}.']],
             'answer': n + '.', 'skill': 'copia_isolada', 'group': f'copy_{i}', 'episode': f'copy_{i}',
             'turn': 0, 'family': 'copy_probe', 'imported': True, 'target_names': {'current': n},
             'source_names': [n], 'name_kinds': {n: kind}}
            for i, (n, kind) in enumerate(sorted(names.items()))]


def temporal_probe(panels):
    """Desenvolvimento, nomes reservados anteriores; nenhuma resposta-ouro no histórico."""
    kinds = {n: kind for r in panels['validation'] for n, kind in r['name_kinds'].items()}
    names = sorted(kinds)
    result = []
    questions = {
        'atual': ['Qual nome vale neste momento?', 'Como devo assinar agora?', 'Recupere o nome mais recente.', 'Que nome você usaria para falar comigo?'],
        'anterior': ['Qual nome foi substituído pela última mudança?', 'Lembre o nome logo antes do atual.', 'Que nome eu usava imediatamente antes?', 'Qual era o penúltimo nome?'],
        'explicar_mudanca': ['Explique a mudança: qual nome deixou de valer e qual passou a valer?',
                            'Descreva a última troca de nome, do anterior para o atual.',
                            'Mostre a transição mais recente entre os dois nomes.',
                            'Diga como o registro mudou na última correção.']}
    for changes in (2, 3):
        for i in range(8):
            seq = [names[(i * 3 + j * 7) % len(names)] for j in range(changes + 1)]
            assert len(set(seq)) == len(seq)
            a = seq[0]
            if changes == 2:
                prefix = [['user', f'Para nosso registro, eu uso {a}.'], ['assistant', 'Entendido.'],
                          ['user', f'Esse registro mudou para {seq[1]}.'], ['assistant', 'Registrei a mudança.'],
                          ['user', f'Uma nova revisão mudou meu nome para {seq[2]}.'], ['assistant', 'Recebi a revisão.']]
            else:
                prefix = [['user', f'No começo, registrei meu nome como {a}.'], ['assistant', 'Entendido.'],
                          ['user', f'A primeira alteração foi para {seq[1]}. Depois fiz uma segunda, para {seq[2]}.'],
                          ['assistant', 'Acompanhei essas alterações.'],
                          ['user', f'Por último, fiz a terceira alteração: meu nome passou a {seq[3]}.'],
                          ['assistant', 'Recebi a última alteração.']]
            for skill, variants in questions.items():
                expected = {'previous': seq[-2], 'current': seq[-1]} if skill == 'explicar_mudanca' else {('current' if skill == 'atual' else 'previous'): seq[-1] if skill == 'atual' else seq[-2]}
                answer = f'Antes: {seq[-2]}; agora: {seq[-1]}.' if skill == 'explicar_mudanca' else next(iter(expected.values())) + '.'
                group = f'dev_temporal_{changes}_{i}_{skill}'
                result.append({'messages': prefix + [['user', variants[i % len(variants)]]], 'answer': answer,
                               'target_names': expected, 'skill': skill, 'family': f'mudancas_{changes}',
                               'group': group, 'episode': group, 'turn': 0, 'imported': True,
                               'source_names': seq, 'name_kinds': {n: kinds[n] for n in seq}, 'changes': changes})
    return result


def audit_rows(rows, data, context=512):
    max_length = 0
    for row in rows:
        x, y = data.encode(row, context)
        text = prompt_text(row['messages'][-7:])
        if len(row['messages']) > 7 or len(data.tokenizer.encode(text, special=False)) + int((y != 0).sum()) > context:
            raise ValueError('Histórico temporal seria truncado.')
        if not all(n in text for n in row['target_names'].values()):
            raise ValueError('Nome-alvo não está no contexto real.')
        max_length = max(max_length, len(x))
    return {'examples': len(rows), 'max_length': max_length, 'cropped': 0, 'missing_target_name': 0,
            'digest': digest(rows)}
