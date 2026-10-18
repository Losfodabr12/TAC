from collections import Counter
import json

import pytest
import torch

from tac.learning import LearningGate, LearningDisabled, Cancelled
from tac.conversation_lessons import prompt_text
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import ConversationData
from tac.neural.name_generalization import create_sets
from tac.neural.temporal_memory import (original_plan, interleaved_plan, redistributed_plan,
    temporal_render, temporal_probe, copy_probe, audit_rows, exposure, PRIORITY)
from scripts.run_temporal_memory import PROTOCOL, protected_step
from scripts.run_temporal_coverage import render_coverage
from scripts.run_controlled_experiments import LiveSettings


@pytest.fixture(scope='module')
def data():
    panels, _, _ = create_sets()
    # Este experimento usa BPE de 1.024 tokens. O BPE vazio seria o modelo
    # de bytes de 259 tokens, com outro orçamento de contexto por caractere.
    # O tokenizer do fixture só vê treino; não lê pesos/dados reais do usuário.
    tok = LocalBPETokenizer.fit([prompt_text(r['messages']) + '\n' + r['answer']
                                 for r in panels['train']], vocab_size=1024)
    return panels, ConversationData({'train': panels['train']}, tok)


def test_interleaving_only_changes_order_not_coverage_or_repetitions(data):
    panels, _ = data
    rows = panels['train']
    plan = original_plan(rows, 42)
    new = interleaved_plan(plan, rows)
    assert new != plan and Counter(new) == Counter(plan)
    assert len(set(new)) == 928 and len(new) == 1920 and max(Counter(new).values()) == 3
    assert new == interleaved_plan(original_plan(rows, 42), rows)


def test_redistribution_keeps_token_budget_all_rows_and_three_use_cap(data):
    panels, dataset = data
    rows = panels['train']
    lengths = [int((dataset.encode(row, 512)[1] != 0).sum()) for row in rows]
    old = original_plan(rows, 417)
    new = redistributed_plan(old, rows, lengths, 417)
    assert [lengths[i] for i in old] == [lengths[i] for i in new]
    assert set(old) == set(new) and max(Counter(new).values()) <= 3
    assert sum(rows[i]['skill'] in PRIORITY for i in new) > sum(rows[i]['skill'] in PRIORITY for i in old)
    assert new == redistributed_plan(old, rows, lengths, 417)
    assert exposure(new, rows, lengths)['unique_examples'] == 928


def test_explicit_variant_preserves_all_labels_and_three_change_order(data):
    panels, dataset = data
    original = panels['train']
    unchanged = json.dumps(original, sort_keys=True)
    new = temporal_render(original)
    assert json.dumps(original, sort_keys=True) == unchanged
    assert len(new) == 928
    for a, b in zip(original, new):
        assert (a['answer'], a['target_names'], a['group']) == (b['answer'], b['target_names'], b['group'])
        assert int((dataset.encode(a, 512)[1] != 0).sum()) == int((dataset.encode(b, 512)[1] != 0).sum())
    row = next(r for r in new if r['family'] == 't5' and r['turn'] == 3)
    initial, current, previous, inserted = row['source_names']
    assert len(set((initial, current, previous, inserted))) == 4
    assert row['messages'][2][1].index(inserted) < row['messages'][2][1].index(previous)
    assert current in row['messages'][4][1]
    assert row['target_names'] == {'previous': previous, 'current': current}
    assert audit_rows(new, dataset)['cropped'] == 0


def test_temporal_probe_has_two_three_changes_and_no_teacher_name_echo(data):
    panels, dataset = data
    rows = temporal_probe(panels)
    assert len(rows) == 48 and {r['changes'] for r in rows} == {2, 3}
    assert {r['skill'] for r in rows} == {'atual', 'anterior', 'explicar_mudanca'}
    train_names = {n for r in panels['train'] for n in r['source_names']}
    assert not train_names & {n for r in rows for n in r['source_names']}
    for row in rows:
        assert len(row['messages']) == 7
        assert all(n not in text for role, text in row['messages'] if role == 'assistant' for n in row['source_names'])
        expected = row['target_names']
        if 'previous' in expected:
            assert expected['previous'] == row['source_names'][-2] != row['source_names'][0]
    assert audit_rows(rows, dataset)['missing_target_name'] == 0
    assert len(copy_probe(panels)) > 0


def test_optimizer_blocked_mid_operation_for_disable_and_cancel(tmp_path):
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps({'learning_enabled': True, 'data_dir': 'data'}))
    gate = LearningGate(LiveSettings(tmp_path), None)
    gate.begin()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.weight.detach().clone()
    model(torch.ones(1, 2)).sum().backward()
    cfg.write_text(json.dumps({'learning_enabled': False, 'data_dir': 'data'}))
    with pytest.raises(LearningDisabled):
        protected_step(optimizer, torch.amp.GradScaler('cuda', enabled=False), gate)
    cfg.write_text(json.dumps({'learning_enabled': True, 'data_dir': 'data'}))
    gate.cancel.set()
    with pytest.raises(Cancelled):
        protected_step(optimizer, torch.amp.GradScaler('cuda', enabled=False), gate)
    assert torch.equal(model.weight, before) and not optimizer.state
    assert not PROTOCOL['automatic_promotion'] and PROTOCOL['internal_training_examples'] == 0


def test_coverage_changes_history_not_targets_and_teaches_intermediate_previous(data):
    panels, dataset = data
    original = panels['train']
    frozen = json.dumps(original, sort_keys=True)
    rows = render_coverage(original)
    assert json.dumps(original, sort_keys=True) == frozen
    assert len(rows) == 928
    for old, new in zip(original, rows):
        assert (old['answer'], old['target_names'], old['group']) == (new['answer'], new['target_names'], new['group'])
    previous = [r for r in rows if r['skill'] == 'nome_anterior']
    assert Counter(r['coverage_changes_after_last_correction'] for r in previous) == {2: 32, 3: 32}
    for row in previous:
        earlier = row['extra_earlier_names']
        target = row['target_names']['previous']
        assert target not in earlier
        assert all(row['messages'][0][1].index(n) < row['messages'][0][1].index(target) for n in earlier)
    assert audit_rows(rows, dataset)['cropped'] == 0
