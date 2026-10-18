import json

import pytest
import torch

from tac.learning import LearningGate, LearningDisabled
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import ConversationData
from tac.neural.name_generalization import (create_sets, audit_sets, name_weights, weighted_loss,
    parse_answer, score_name_answer, evaluate_names, TRAIN_FAMILIES, DEV_FAMILIES, TEST_FAMILIES)
from scripts.run_name_generalization import PROTOCOL, protected_step, LiveSettings


def test_dataset_disjoint_names_families_no_context_loss_or_missing_evidence():
    panels, test, inventories = create_sets()
    audit = audit_sets(panels, test, inventories, LocalBPETokenizer())
    assert len(panels['train']) >= 900 and len(test) >= 50
    assert audit['strict_main_split_disjoint'] and not audit['reserved_names_in_training']
    assert {r['family'] for r in panels['train']} == set(TRAIN_FAMILIES)
    assert {r['family'] for r in panels['validation']} == set(DEV_FAMILIES)
    assert {r['family'] for r in test} == set(TEST_FAMILIES)
    assert all(not r['cropped'] and not r['expected_name_not_in_prompt'] for r in audit['panels'].values())
    assert {r['group'] for r in panels['train_probe']} <= {r['group'] for r in panels['train']}
    assert panels == create_sets()[0]


def test_named_token_weight_masks_prompt_preserves_eos_and_unknown_utf8():
    tok = LocalBPETokenizer()
    row = {'messages': [['user', 'Meu nome é Xü-Çá.']], 'answer': 'Seu nome atual é Xü-Çá.',
           'skill': 'nomes', 'group': 'x', 'imported': True, 'target_names': {'current': 'Xü-Çá'}}
    data = ConversationData({'train': [row]}, tok)
    x, y = data.batch([row], 512, 'cpu')
    w = name_weights(data, row, 512, 3.)
    assert not w[y[0] == 0].any() and w[-1] == 1
    assert tok.decode(y[0][w == 3].tolist()) == 'Xü-Çá'
    assert torch.equal(name_weights(data, row, 512, 1.), (y[0] != 0).float())
    logits = torch.randn(1, y.shape[1], tok.vocab_size, requires_grad=True)
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0)
    assert torch.allclose(ce, weighted_loss(logits, y, (y != 0).float()))
    weighted_loss(logits, y, w[None]).backward()
    assert not logits.grad[y == 0].any()


def trace(eos=True):
    return {'eos': eos, 'user_role_emitted': False, 'repeated_trigram_3_times': False}


def test_metrics_do_not_conflate_identity_characters_or_role_binding():
    expected = {'current': 'Érica'}
    score = score_name_answer('Seu nome atual é erica.', expected, trace(), ['Ana', 'Érica'])
    assert score['name_identity'] and score['context_correct'] and not score['characters_exact']
    score = score_name_answer('O nome anterior era Érica.', expected, trace(), ['Ana', 'Érica'])
    assert score['name_identity'] and score['characters_exact'] and not score['context_correct']
    score = score_name_answer('Ana.', expected, trace(), ['Ana', 'Érica'])
    assert not score['context_correct'] and score['error_type'] == 'wrong_name_from_context'
    score = score_name_answer('Érica.', expected, trace(False), ['Ana', 'Érica'])
    assert score['context_correct'] and not score['coherence_restricted_proxy']
    assert score['semantic_human_review'] is None


def test_two_names_require_ordered_roles_and_do_not_reward_mentioning_both():
    expected = {'previous': 'Lia', 'current': 'Bia'}
    assert parse_answer('Agora: Bia; antes: Lia.', expected) == expected
    bad = score_name_answer('Antes: Bia; agora: Lia.', expected, trace(), ['Lia', 'Bia'])
    assert bad['name_identity'] and not bad['context_correct'] and bad['error_type'] == 'swapped_roles'
    assert not score_name_answer('Lia Bia.', expected, trace(), ['Lia', 'Bia'])['context_correct']


def test_free_generation_receives_no_expected_names_or_labels(monkeypatch):
    panels, _, _ = create_sets()
    rows = panels['validation'][:4]
    seen = []
    def generate(model, messages, device, max_tokens):
        seen.append([list(m) for m in messages])
        return 'Resposta inventada.', trace()
    monkeypatch.setattr('tac.neural.name_generalization.trace_reply', generate)
    class Model:
        def eval(self):
            pass
    result = evaluate_names(Model(), rows, 'cpu', own_history=True)
    assert seen[1][1] == ['assistant', 'Resposta inventada.']
    assert result['context_correct'] == 0 and result['literal_exact'] == 0


def test_experimental_update_blocks_when_learning_switched_off(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'data_dir': 'data', 'learning_enabled': True}))
    gate = LearningGate(LiveSettings(tmp_path), None)
    gate.begin()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    before = model.weight.detach().clone()
    config.write_text(json.dumps({'data_dir': 'data', 'learning_enabled': False}))
    with pytest.raises(LearningDisabled):
        protected_step(optimizer, torch.amp.GradScaler('cuda', enabled=False), gate)
    assert torch.equal(model.weight, before) and not optimizer.state
    assert not PROTOCOL['automatic_promotion'] and not PROTOCOL['automatic_extension']
