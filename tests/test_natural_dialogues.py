import json
import hashlib
import torch

from tac.conversation_lessons import canonical, prompt_text
from tac.natural_dialogues import make_natural_sets
from tac.neural.conversation import ConversationData, prepare_data, natural_ready
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.model import ModelConfig, TACTransformer


def test_natural_curriculum_freezes_reserved_forms_and_trains_immediate_corrections():
    sets = make_natural_sets()
    other = make_natural_sets(('que', 'ab', 'co'))
    assert sets['validation'] == other['validation'] and sets['test'] == other['test']
    keys = [{canonical(prompt_text(r['messages'])) for r in sets[s]} for s in ('train','validation','test')]
    assert not (keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
    matches = [r for r in sets['train'] if r['skill']=='natural_color_correction' and r['messages'][-1][1]=='Agora prefiro verde.']
    assert matches and all(r['answer']=='Agora sua cor favorita é verde.' for r in matches)
    inputs = ' '.join(m[1] for r in sets['train'] for m in r['messages'])
    assert 'kkk' in inputs and '😅' in inputs and 'na vdd' in inputs
    from tac.natural_dialogues import SLOTS
    topics = [r for r in sets['train'] if r['skill'] == 'natural_topic_correction']
    assert any(r['answer'] not in {SLOTS['topic']['changed'].format(value=v) for v in SLOTS['topic']['train']} for r in topics)
    # A mesma forma ambígua exige consultar o assunto/nome/cor no histórico.
    for slot in ('color', 'name', 'topic'):
        assert any(r['skill'] == f'natural_{slot}_correction' and r['messages'][-1][1].startswith('Corrigindo:') for r in sets['train'])


def test_combined_curriculum_prevents_reserved_prompts_in_training():
    cp = {'model_config':vars(ModelConfig()), 'tokenizer_state':LocalBPETokenizer().state()}
    data = prepare_data(cp, large=True)
    heldout = {canonical(prompt_text(r['messages'])) for s,rows in data.sets.items() if s!='train' for r in rows}
    assert all(canonical(prompt_text(r['messages'])) not in heldout for r in data.sets['train'])
    assert len(data.sets['validation'])==142 and len(data.sets['test'])==144


def test_closed_sessions_reuse_actual_answers_instead_of_gold(core, monkeypatch):
    row={'messages':[['user','cor inicial azul'],['assistant','RESPOSTA OURO NÃO ENVIAR'],['user','prefiro verde'],
                     ['assistant','Nova cor: verde.'],['user','qual cor?']], 'answer':'Verde.', 'skill':'natural_color_recall'}
    data=ConversationData({'train':[row], 'validation_natural':[row]})
    generated=['Minha resposta gerada.', 'Nova cor: verde.', 'Verde.']
    calls=[]
    def reply(model, messages, device):
        assert all(text!='RESPOSTA OURO NÃO ENVIAR' for _,text in messages)
        if len(calls): assert messages[1][1]=='Minha resposta gerada.'
        calls.append(messages)
        return generated[len(calls)-1]
    monkeypatch.setattr('tac.neural.conversation.decode_reply', reply)
    metrics=core.conversation.evaluate_sessions(None, data, 'cpu')
    assert metrics['examples']==1 and metrics['immediate_accuracy']==metrics['recall_accuracy']==1.


def test_evaluation_accepts_authored_alternative_without_modifying_text(core, monkeypatch):
    row={'messages':[['user','oi']], 'answer':'Oi.', 'acceptable_answers':['Oi.','Olá.'], 'skill':'natural_greeting'}
    data=ConversationData({'train':[row],'validation':[row]})
    model=TACTransformer(ModelConfig(width=32, heads=2, layers=1, context=64))
    monkeypatch.setattr('tac.neural.conversation.decode_reply', lambda *_:'Olá.')
    metrics=core.conversation.evaluate(model,data,'cpu',generation=True)
    assert metrics['exact_accuracy']==1 and metrics['outputs'][0]['actual']=='Olá.'


def test_correction_gate_requires_both_immediate_and_recall_skills():
    groups={f'natural_{slot}_{stage}':{'correct':4,'total':5} for slot in ('color','name','topic') for stage in ('correction','recall')}
    metrics={'by_skill':groups,'exact_accuracy':.8}
    assert natural_ready(metrics, {'exact_accuracy':.3})
    groups['natural_color_correction']['correct']=1
    assert not natural_ready(metrics, {'exact_accuracy':.3})


def test_retention_replays_old_and_new_data_and_resumes_exactly(core, monkeypatch):
    from test_training import setup_training
    from test_generative_conversation import small_sets
    from tac.neural.conversation import load_checkpoint

    monkeypatch.setattr('tac.neural.conversation.make_sets', small_sets)
    corpus = setup_training(core, 5)
    core.perform('Leitor de teste', lambda: core.trainer.train(corpus=corpus), learning=True)
    parent = core.perform('Primeiro dialogo', lambda: core.conversation.train(steps=120, learning_rate=.003), learning=True)['checkpoint']
    path = core.trainer.checkpoint_path(parent)
    previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    previous_data = load_checkpoint(path)['dataset_hash']

    def new_sets():
        sets = small_sets()
        sets['train'].append({'messages':[['user', 'hey']], 'answer':'Oi.', 'skill':'new_style'})
        return sets
    monkeypatch.setattr('tac.neural.conversation.make_sets', new_sets)
    sampled = set()
    real_sample = ConversationData.sample
    def sample(data, *args):
        sampled.add(data.digest)
        return real_sample(data, *args)
    monkeypatch.setattr(ConversationData, 'sample', sample)
    full = core.perform('Retencao completa', lambda: core.conversation.train(steps=120, initial_checkpoint=parent), learning=True)
    assert previous_data in sampled and len(sampled) == 2

    def stop(step, *_):
        if step == 100:
            core.gate.set_enabled(False)
    partial = core.perform('Retencao interrompida', lambda: core.conversation.train(stop, steps=120, initial_checkpoint=parent), learning=True)
    assert partial['status'] == 'interrompido' and partial['steps'] == 100
    cp = load_checkpoint(core.trainer.checkpoint_path(partial['checkpoint']))
    assert cp['training_config']['retention_source'] == parent and cp['training_config']['retention_weight'] == 2.
    core.gate.set_enabled(True)
    resumed = core.perform('Retencao retomada', lambda: core.conversation.train(resume=partial['checkpoint']), learning=True)
    assert full['final_hash'] == resumed['final_hash']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == previous_hash
