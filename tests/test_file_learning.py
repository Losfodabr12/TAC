import json
from types import SimpleNamespace

import pytest
import torch

from tac.learning import LearningGate, LearningDisabled, Cancelled
from tac.neural.file_learning import adapt_records, EpochSampler, evaluate_file
from tac.neural.tokenizer import ByteTokenizer
from scripts.learn_conversation_file import PROTOCOL, protected_step, LiveSettings


def raw_complete():
    return {'categoria': 'nomes', 'messages': [
        {'role': 'user', 'content': 'Sou Ana.'}, {'role': 'assistant', 'content': 'Oi, Ana.'},
        {'role': 'user', 'content': 'Quem sou?'}, {'role': 'assistant', 'content': 'Ana.'}]}


def adapt(value):
    return adapt_records(json.dumps(value, ensure_ascii=False).encode(), ByteTokenizer())


def test_adapter_keeps_every_turn_and_masks_only_answer_plus_eos():
    original = raw_complete()
    records, data, report = adapt(original)
    assert not report['rejected'] and report['accepted_examples'] == 2
    assert records[0]['answer'] == 'Ana.' and records[0]['messages'] == original['messages'][:-1]
    assert report['by_split'] == {'train': 2, 'validation': 0, 'test': 0}
    assert not report['history_cropped']
    row = data.sets['train'][1]
    x, y = data.encode(row, 512)
    assert row['messages'] == [['user', 'Sou Ana.'], ['assistant', 'Oi, Ana.'], ['user', 'Quem sou?']]
    assert data.tokenizer.decode(y.tolist()) == ' Ana.' and int((y == 2).sum()) == 1
    assert int((y != 0).sum()) == len(data.tokenizer.encode(' Ana.', special=False)) + 1
    assert 'Sou Ana.' in data.tokenizer.decode(x.tolist())


@pytest.mark.parametrize('mutation,reason', [
    (lambda v: v['messages'][0].update(role='system'), 'Papel inválido'),
    (lambda v: v['messages'][-1].update(content='a' * 510), 'Limite'),
    (lambda v: v['messages'][-1].update(role='user'), 'terminar'),
])
def test_adapter_reports_importer_errors(mutation, reason):
    value = raw_complete()
    mutation(value)
    _, _, report = adapt(value)
    assert report['rejected'] and reason in report['rejected'][0]['reason']


def test_explicit_validation_stays_out_of_training():
    value = {'messages': [{'role': 'user', 'content': 'A?'}], 'answer': 'B.',
             'skill': 'a', 'group': 'g', 'split': 'validation'}
    _, data, _ = adapt(value)
    assert not data.sets['train'] and len(data.sets['validation']) == 1


def test_group_split_conflict_rejected():
    a = {'messages': [{'role': 'user', 'content': 'A?'}], 'answer': 'B.', 'skill': 'a', 'group': 'g', 'split': 'train'}
    b = {**a, 'split': 'test'}
    _, data, report = adapt_records(('\n'.join(json.dumps(r) for r in (a, b))).encode(), ByteTokenizer())
    assert report['rejected'] and all(not rows for rows in data.sets.values())


def test_epoch_sampler_only_uses_supplied_unique_rows_before_repeating():
    rows = [object() for _ in range(17)]
    a, b = EpochSampler(rows, 42), EpochSampler(rows, 42)
    first = a.take(17)
    assert len({id(r) for r in first}) == 17 and first == b.take(17)
    assert set(a.take(20)).issubset(set(rows)) and sum(a.seen.values()) == 37
    assert set(a.seen.values()) <= {2, 3}


def test_evaluation_uses_own_answers_and_never_expected_as_generated(monkeypatch):
    _, data, _ = adapt(raw_complete())
    seen = []
    def generate(model, messages, device, max_tokens):
        seen.append([list(m) for m in messages])
        return 'Resposta produzida.', {'eos': True, 'user_role_emitted': False, 'repeated_trigram_3_times': False}
    monkeypatch.setattr('tac.neural.file_learning.trace_reply', generate)
    model = SimpleNamespace(eval=lambda: None)
    result = evaluate_file(model, data, 'cpu', own_history=True)
    assert seen[1][1] == ['assistant', 'Resposta produzida.']
    assert result['exact_literal'] == 0 and result['semantic_score'] is None


def test_file_runner_stops_weight_update_when_learning_disabled_or_cancelled(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'learning_enabled': True, 'data_dir': 'data'}))
    gate = LearningGate(LiveSettings(tmp_path), None)
    gate.begin()
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    path.write_text(json.dumps({'learning_enabled': False, 'data_dir': 'data'}))
    with pytest.raises(LearningDisabled):
        protected_step(opt, scaler, gate)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    path.write_text(json.dumps({'learning_enabled': True, 'data_dir': 'data'}))
    gate.cancel.set()
    with pytest.raises(Cancelled):
        protected_step(opt, scaler, gate)
    assert not opt.state
    assert not PROTOCOL['replay'] and not PROTOCOL['kl'] and PROTOCOL['internal_examples'] == 0
    assert not PROTOCOL['promotion'] and not PROTOCOL['test_reserved_read']
