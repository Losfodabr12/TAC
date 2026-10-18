import collections
import hashlib
import torch

from tac.fluency_lessons import (batch_kind, lessons, make_fluency_sets,
                                make_language_corpus, score_reply, natural_legacy)
from tac.conversation_lessons import canonical,prompt_text
from tac.neural.conversation import ConversationData,prepare_data,load_checkpoint


def test_reviewed_scenes_are_split_by_whole_episode_and_all_turns_are_supervised():
    source = lessons()
    ids = [{s[0] for s in source[split]} for split in ('train','validation','test')]
    assert len(ids[0]) == 40 and len(ids[1]) == len(ids[2]) == 6
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    data = make_fluency_sets()
    assert len(data['train']) == 40*4*6
    for split in data:
        assert {r['turn'] for r in data[split]} == {0,1,2,3}
    keys = [{canonical(prompt_text(r['messages'])) for r in data[s]} for s in ('train','validation','test')]
    assert not (keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
    language = make_language_corpus()
    assert len(language.train_text) == len(set(language.train_text))
    for split in ('validation','test'):
        assert all(not any(u in text for text in language.train_text) for entry in source[split] for u,a in entry[1])


def test_mixture_is_explicit_and_independent_of_accumulation():
    for accumulation in (1,2,4):
        counter = collections.Counter(batch_kind(step,m,accumulation) for step in range(20//accumulation) for m in range(accumulation))
        assert counter == {'reviewed':11,'language':6,'legacy':3}


def test_pools_are_sampled_separately_and_reproducibly():
    rows = [{'messages':[['user','oi']], 'answer':'Oi.', 'skill':'s', 'mixture':p} for p in ('legacy','reviewed')]
    data = ConversationData({'train':rows})
    for pool in ('legacy','reviewed'):
        sampled = data.sample_pool(4,torch.Generator().manual_seed(7),pool)
        assert all(r['mixture']==pool for r in sampled)


def test_legacy_filter_removes_invented_copy_targets():
    assert not natural_legacy({'skill':'natural_color_correction','answer':'Agora sua cor favorita é qwxydr.','messages':[]})
    assert natural_legacy({'skill':'natural_color_correction','answer':'Agora sua cor favorita é verde.','messages':[]})
    assert not natural_legacy({'skill':'context_location','answer':'Qwxydr está no local sala.','messages':[]})


def test_rules_are_a_relevance_proxy_and_reject_slot_drift():
    assert score_reply('Você pode ensaiar em voz alta.',['ensai|pratic'])['passes_simple_rules']
    assert not score_reply('Agora sua cor favorita é ensaiar.',['ensai|pratic'])['passes_simple_rules']
    assert not score_reply('Certo.',['ensai|pratic'])['passes_simple_rules']
    from tac.fluency_lessons import fluency_ready
    assert fluency_ready({'rule_accuracy':.95,'slot_drift_rate':0},{'rule_accuracy':.95})
    assert not fluency_ready({'rule_accuracy':.3,'slot_drift_rate':0},{'rule_accuracy':.04})


def test_evaluation_uses_actual_answers_for_every_turn(core,monkeypatch):
    rows = [ {'messages':[['user',str(i)]], 'answer':'REFERÊNCIA NÃO ENVIAR','skill':'s','episode':'s',
               'turn':i,'relevance_rules':['pratic']} for i in range(4)]
    data = ConversationData({'train':rows,'validation_fluency':rows})
    calls = []
    def decode(model,messages,device):
        assert not any(t=='REFERÊNCIA NÃO ENVIAR' for r,t in messages)
        if calls:assert messages[-2][1]=='Você pode praticar.'
        calls.append(messages)
        return 'Você pode praticar.'
    monkeypatch.setattr('tac.neural.conversation.decode_reply',decode)
    result = core.conversation.evaluate_fluency(None,data,'cpu')
    assert result['rule_matches']==4 and result['slot_drift_rate']==0


def test_balanced_training_changes_weights_and_resumes_exactly(core,monkeypatch):
    from test_training import setup_training
    from test_generative_conversation import small_sets
    monkeypatch.setattr('tac.neural.conversation.make_sets',small_sets)
    corpus = setup_training(core,5)
    core.perform('Leitor',lambda:core.trainer.train(corpus=corpus),learning=True)
    parent = core.perform('Base',lambda:core.conversation.train(steps=120,learning_rate=.003),learning=True)['checkpoint']
    parent_path = core.trainer.checkpoint_path(parent)
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    original_prepare = prepare_data
    def small_data(cp,large=False,curriculum=None):
        data = original_prepare(cp,large=False)
        if curriculum:
            data = ConversationData({**data.sets,'train':[{**r,'mixture':'legacy'} for r in data.sets['train']]
                +[{**r,'mixture':'reviewed','skill':'reviewed'} for r in data.sets['train']]},data.tokenizer)
        return data
    monkeypatch.setattr('tac.neural.conversation.prepare_data',small_data)
    full = core.perform('Mistura completa',lambda:core.conversation.train(steps=120,initial_checkpoint=parent,curriculum='fluency'),learning=True)
    assert full['initial_hash'] != full['final_hash']
    assert set(full['mixture_counts']) == {'reviewed','language','legacy'}
    core.conversation.activate(parent)
    def stop(step,*_):
        if step==100:core.gate.set_enabled(False)
    partial = core.perform('Interromper',lambda:core.conversation.train(stop,steps=120,initial_checkpoint=parent,curriculum='fluency'),learning=True)
    assert partial['steps']==100 and partial['status']=='interrompido'
    core.gate.set_enabled(True)
    resumed = core.perform('Retomar',lambda:core.conversation.train(resume=partial['checkpoint']),learning=True)
    assert resumed['final_hash']==full['final_hash']
    assert resumed['mixture_counts']==full['mixture_counts']
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==parent_hash
