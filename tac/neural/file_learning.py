"""Experimento isolado de aquisição: somente dados explicitamente fornecidos.

Não consulta biblioteca, não escreve no catálogo de modelos e não promove pesos.
Reutiliza validação, máscara e geração da rede real da TAC.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import torch

from ..conversation_lessons import canonical, prompt_text
from ..dialogue_imports import (MAX_EXAMPLES, MAX_RECORDS, derive_examples, digest,
                               normalize_record, read_records)
from .conversation import ConversationData
from .controlled_eval import trace_reply


def adapt_records(raw, tokenizer, context=512):
    """Converte conversas completas sem alterar falas; split explícito é preservado.

    No formato categoria/messages, sem split, todo o arquivo é TREINO por
    autorização do experimento. Não atribui aleatoriamente uma avaliação falsa.
    Registros inválidos são relatados; o runner exige zero rejeições para treinar.
    """
    records, rows, errors, warnings = [], [], [], []
    requests = defaultdict(set)
    adaptations = 0
    count = 0
    for line, value, error in read_records(raw, '.jsonl'):
        count += 1
        if count > MAX_RECORDS:
            raise ValueError('Arquivo excede o limite de registros do importador.')
        try:
            if error:
                raise ValueError(error)
            if isinstance(value, dict) and set(value) == {'categoria', 'messages'}:
                messages = value['messages']
                if not isinstance(messages, list) or not messages:
                    raise ValueError('Conversa completa sem falas.')
                last = messages[-1]
                if not isinstance(last, dict) or set(last) != {'role', 'content'} or last['role'] != 'assistant':
                    raise ValueError('Conversa completa precisa terminar com uma fala assistant.')
                value = {'messages': messages[:-1], 'answer': last['content'],
                         'skill': value['categoria'], 'group': f'arquivo_{digest(value)[:16]}', 'split': 'train'}
                adaptations += 1
            record = normalize_record(value)
            # O experimento não resolve auto silenciosamente: o destino deve ser claro.
            if record['split'] == 'auto':
                raise ValueError('Declare train, validation ou test neste experimento isolado.')
            requests[record['group']].add(record['split'])
            examples, notices = derive_examples(record, tokenizer, context, line)
            if len(rows) + len(examples) > MAX_EXAMPLES:
                raise ValueError('Arquivo excede o limite de exemplos do importador.')
            records.append({**record, 'messages': [{'role': r, 'content': t} for r, t in record['messages']]})
            rows.extend({**r, 'split': record['split']} for r in examples)
            warnings.extend(notices)
        except (ValueError, TypeError) as exc:
            errors.append({'line': line, 'reason': str(exc)})
    if not count:
        raise ValueError('Arquivo sem conversas.')
    conflicts = {group for group, splits in requests.items() if len(splits) > 1}
    for group in sorted(conflicts):
        errors.append({'group': group, 'reason': 'Conversa com destinos diferentes.'})
    if conflicts:
        rows = [r for r in rows if r['group'] not in conflicts]
        records = [r for r in records if r['group'] not in conflicts]
    prompt_splits = defaultdict(set)
    answers = defaultdict(set)
    for row in rows:
        prompt_splits[row['prompt_hash']].add(row['split'])
        answers[row['prompt_hash']].add(row['answer'])
    leaked = [p for p, splits in prompt_splits.items() if len(splits) > 1]
    if leaked:
        errors.append({'reason': 'Entradas idênticas aparecem em destinos diferentes.', 'count': len(leaked)})
    sets = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    data = ConversationData(sets, tokenizer)
    lengths, targets, cropped = [], [], []
    for row in rows:
        x, y = data.encode(row, context)
        prefix_size = len(tokenizer.encode(prompt_text(row['messages']), special=False))
        target_size = int((y != 0).sum())
        lengths.append(len(x))
        targets.append(target_size)
        if len(row['messages']) > 7 or prefix_size > context - target_size:
            cropped.append({'line': row['source_line'], 'turn': row['turn']})
    report = {'input_records': count, 'adapted_records': adaptations, 'accepted_records': len(records),
              'accepted_examples': len(rows), 'by_split': {s: len(rs) for s, rs in sets.items()},
              'conversations': len({r['group'] for r in rows}), 'by_skill': dict(Counter(r['skill'] for r in rows)),
              'rejected': errors, 'warnings': warnings, 'history_cropped': cropped,
              'unique_examples': len({r['example_hash'] for r in rows}),
              'identical_inputs_different_answers': sum(len(a) > 1 for a in answers.values()),
              'maximum_training_sequence_tokens': max(lengths, default=0),
              'maximum_answer_tokens_including_eos': max(targets, default=0),
              'supervised_tokens_per_pass': sum(targets),
              'validation_independent': bool(sets['validation']), 'test_evaluated': False}
    return records, data, report


class EpochSampler:
    """Permutações completas: nenhuma conversa de outra fonte pode entrar."""
    def __init__(self, rows, seed):
        if not rows:
            raise ValueError('Nenhum exemplo de treino.')
        self.rows = rows
        self.rng = torch.Generator().manual_seed(seed)
        self.pending = []
        self.seen = Counter()
        self.tokens = Counter()

    def take(self, size):
        result = []
        for _ in range(size):
            if not self.pending:
                self.pending = torch.randperm(len(self.rows), generator=self.rng).tolist()
            index = self.pending.pop()
            self.seen[index] += 1
            result.append(self.rows[index])
        return result

    def summary(self):
        return {'unique_examples': len(self.seen), 'presentations': sum(self.seen.values()),
                'repetitions': sum(self.seen.values()) - len(self.seen),
                'effective_passes': sum(self.seen.values()) / len(self.rows),
                'per_example_presentations': dict(self.seen),
                'by_skill': dict(Counter({s: sum(n for i, n in self.seen.items() if self.rows[i]['skill'] == s)
                                         for s in {r['skill'] for r in self.rows}})),
                'supervised_tokens_by_skill': dict(self.tokens)}


@torch.inference_mode()
def measured_loss(model, data, rows, device, batch_size=8):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.
    for start in range(0, len(rows), batch_size):
        x, y = data.batch(rows[start:start + batch_size], model.cfg.context, device)
        logits, loss = model(x, y)
        mask = y != 0
        count = int(mask.sum())
        loss_sum += float(loss) * count
        total += count
        correct += int(((logits.argmax(-1) == y) & mask).sum())
    return {'loss': loss_sum / total if total else None, 'target_tokens': total,
            'teacher_forced_token_accuracy': correct / total if total else None}


@torch.inference_mode()
def evaluate_file(model, data, device, own_history=False, max_tokens=128):
    """Somente exemplos ESTUDADOS, geração livre sem olhar a resposta esperada.

    Com own_history, mensagens assistant anteriores vêm das gerações desta sessão.
    Comparação esperada é feita APÓS gerar; não há consulta nem pós-processamento.
    """
    model.eval()
    outputs, histories = [], {}
    for row in sorted(data.sets['train'], key=lambda r: (r['source_line'], r['turn'])):
        if own_history:
            history = histories.setdefault(row['episode'], [])
            history.append(row['messages'][-1])
            messages = history[-7:]
        else:
            messages = row['messages'][-7:]
        actual, trace = trace_reply(model, messages, device, max_tokens=max_tokens)
        outputs.append({'line': row['source_line'], 'group': row['group'], 'turn': row['turn'],
                        'skill': row['skill'], 'messages': [list(m) for m in messages],
                        'expected': row['answer'], 'actual': actual,
                        'exact_literal': actual.strip() == row['answer'].strip(),
                        'exact_normalized': canonical(actual) == canonical(row['answer']), 'trace': trace})
        if own_history:
            history.append(['assistant', actual])
    def summary(part):
        return {'examples': len(part), 'exact_literal': sum(r['exact_literal'] for r in part),
                'exact_normalized': sum(r['exact_normalized'] for r in part),
                'eos': sum(r['trace']['eos'] for r in part),
                'user_role_emitted': sum(r['trace']['user_role_emitted'] for r in part),
                'repeated_trigram': sum(r['trace']['repeated_trigram_3_times'] for r in part)}
    return {**summary(outputs), 'by_skill': {s: summary([r for r in outputs if r['skill'] == s])
                                             for s in sorted({r['skill'] for r in outputs})},
            'outputs': outputs, 'own_history': own_history, 'scope': 'in_sample_training_examples',
            'semantic_score': None, 'semantic_score_reason': 'Exige revisão humana; igualdade não mede equivalência semântica.'}
