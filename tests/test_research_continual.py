"""Controles de pesquisa, sem usar o teste reservado real para desenvolver."""
import copy
import json
import hashlib

import pytest
import torch

from tac.neural.research import precise_prompt,unique_training_view,DeckSampler,audit_data
from tac.neural.conversation import ConversationData
from tac.neural.model import ModelConfig,TACTransformer
from tac.neural.transfer_eval import freeze_suite,evaluate_once,score
from test_imported_training import imported_training


def test_identity_preserves_operations_and_histories_and_never_drops_different_answers():
    assert precise_prompt([['user','2 + 3?']])!=precise_prompt([['user','2 - 3?']])
    assert precise_prompt([['user','Meu nome é Ana.'],['assistant','Certo.'],['user','Meu nome?']])!=precise_prompt(
        [['user','Meu nome é Bia.'],['assistant','Certo.'],['user','Meu nome?']])
    row={'messages':[['user','Oi?']],'answer':'Olá.','skill':'a'}
    data=ConversationData({'train':[row,copy.deepcopy(row),{**row,'answer':'Bom dia.'}]})
    clean=unique_training_view(data)
    assert len(clean.sets['train'])==2 and len(data.sets['train'])==3
    report=audit_data(data,64)['splits']['train']
    assert report['identical_prompt_and_answer_repeats']==1 and report['precise_inputs_with_multiple_answer_texts']==1


def test_audit_reports_real_window_collisions_without_removing_histories():
    tail=' '.join(['lembrar']*80)
    a={'messages':[['user','Nome Ana. '+tail]],'answer':'Ana.','skill':'name','mixture':'imported','imported':True,'group':'a'}
    b={**a,'messages':[['user','Nome Bia. '+tail]],'answer':'Bia.','group':'b'}
    data=ConversationData({'train':[a,b],'validation_imported':[copy.deepcopy(a)]})
    result=audit_data(data,64)
    assert result['splits']['train']['visible_input_collisions_introduced_by_truncation']==1
    assert result['overlap_with_train']['validation_imported']['shared_imported_groups']==1
    assert result['splits']['train']['prefixes_token_cropped']==2 and len(data.sets['train'])==2


def test_decks_cover_groups_without_replacement_and_restore_exactly():
    rows=[{'messages':[['user',str(i)]],'answer':'Oi.','skill':'s','mixture':'imported','group':str(i)} for i in range(3)]
    data=ConversationData({'train':rows,'test_imported':[{**rows[0],'group':'NEVER'}]})
    rng=torch.Generator().manual_seed(88);sampler=DeckSampler(data)
    first=sampler.sample('imported',3,rng)
    assert {r['group'] for r in first}=={'0','1','2'}
    sampler.sample('imported',1,rng)
    saved=json.loads(json.dumps(sampler.state)); rng_state=rng.get_state()
    expected=sampler.sample('imported',21,rng)
    resumed=DeckSampler(data,saved);newrng=torch.Generator();newrng.set_state(rng_state)
    assert resumed.sample('imported',21,newrng)==expected
    assert all(r['group']!='NEVER' for r in expected)


def test_controlled_generation_is_seeded_and_top_one_preserves_greedy():
    torch.manual_seed(98)
    model=TACTransformer(ModelConfig(width=32,heads=2,layers=1,context=32))
    prompt=torch.tensor([[1,30,40,50]])
    greedy=model.generate(prompt,5,greedy=True)
    assert torch.equal(greedy,model.generate(prompt,5,top_k=1,seed=7))
    a=model.generate(prompt,6,seed=9,top_k=30,top_p=.9,repetition_penalty=1.08)
    b=model.generate(prompt,6,seed=9,top_k=30,top_p=.9,repetition_penalty=1.08)
    assert torch.equal(a,b) and model.parameter_count==sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError):model.generate(prompt,top_p=0)


def test_imported_validation_cannot_outweigh_failed_basic_protection():
    from tac.neural.research import protected_selection_key
    good=protected_selection_key({'loss':.1001},{'loss':.1},.4,5.)
    failed=protected_selection_key({'loss':.2},{'loss':.1},.3,.000001)
    assert good < failed
    assert protected_selection_key({'loss':.1001},{'loss':.1},.4,2.) < good


def test_literal_rubric_does_not_claim_general_semantics_or_accept_denied_name():
    episode={'expected_fact':'Zorelia','forbidden_facts':['Tavira'],'exact_reference':'Seu nome é Zorelia.'}
    assert score('Seu nome é Zorelia.',episode)['factual_correct']
    assert not score('Seu nome não é Zorelia.',episode)['factual_correct']
    assert not score('Talvez Zorelia.',episode)['factual_correct']
    assert not score('Tavira e Zorelia.',episode)['context_consistent']
    assert score('Zorelia.',episode)['semantic_equivalence'] is None
    assert score('Uma resposta aberta.',{'expected_fact':None,'exact_reference':None})['factual_correct'] is None


def test_semantic_review_validates_real_answer_and_leaves_unreviewed_unscored():
    from tac.neural.transfer_eval import review_template,validate_review
    result={'suite_sha256':'abc','models':{'model':{'greedy':{'results':[{'id':'one','skill':'name','actual':'Ana.',
        'history':[['user','Quem?']],'expected':'Ana.'}]}}}}
    review=review_template(result)
    assert validate_review(result,review)['by_model_mode_skill'][0]['accuracy'] is None
    review['rows'][0]['semantic_equivalence']=True
    assert validate_review(result,review)['by_model_mode_skill'][0]['accuracy']==1.
    review['rows'][0]['answer']='Outra resposta.'
    with pytest.raises(ValueError):validate_review(result,review)


def test_holdout_is_immutable_once_consumed_and_does_not_train(core,tmp_path,monkeypatch):
    from tac.neural.tokenizer import ByteTokenizer
    data=ConversationData({'train':[{'messages':[['user','Exercício privado de teste.']],'answer':'Sim.','skill':'s'}]},ByteTokenizer())
    path=tmp_path/'sealed.json';suite=freeze_suite(path,data,798,'unit')
    before=path.read_bytes()
    with pytest.raises(FileExistsError):freeze_suite(path,data,798,'unit')
    core.gate.set_enabled(True);core.gate.begin()
    # Stub de inferência verifica o protocolo, não mede inteligência.
    monkeypatch.setattr('tac.neural.transfer_eval.evaluate_suite',lambda *a,**k:{'controlled_test_stub':True})
    result=evaluate_once(path,[('baseline',None),('candidate',None)],'cpu',core.gate)
    with pytest.raises(FileExistsError):evaluate_once(path,[('other',None)],'cpu',core.gate)
    assert path.read_bytes()==before and not core.db.query('SELECT * FROM runs')
    assert len(suite['episodes'])==32 and set(result['models'])=={'baseline','candidate'}


def test_anchor_and_deck_training_resume_real_weights_and_keep_teacher(core,imported_training):
    parent=imported_training;path=core.trainer.checkpoint_path(parent)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    full=core.perform('Controle completo',lambda:core.conversation.train(steps=120,curriculum='fluency',
        stability_policy='anchor_v1',sampler_mode='deck_v1'),learning=True)
    assert full['initial_hash']!=full['final_hash'] and full['mixture_counts']['anchor']==120
    assert full['objective_history'] and full['stability_policy']=='anchor_v1'
    core.conversation.activate(parent)
    def stop(step,message):
        if step==100:core.gate.set_enabled(False)
    partial=core.perform('Parar controle',lambda:core.conversation.train(stop,steps=120,curriculum='fluency',
        stability_policy='anchor_v1',sampler_mode='deck_v1'),learning=True)
    assert partial['status']=='interrompido' and core.conversation.active_id()==parent
    core.gate.set_enabled(True)
    resumed=core.perform('Retomar controle',lambda:core.conversation.train(resume=partial['checkpoint']),learning=True)
    assert resumed['final_hash']==full['final_hash'] and resumed['mixture_counts']==full['mixture_counts']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert resumed['baseline']['loss']==full['baseline']['loss']
    assert resumed['promotion_reference']=='original_execution'


def test_optional_documents_require_selection_and_do_not_train_or_change_library(core,imported_training):
    from tac.neural.document_replay import prepare_document_replay
    parent=core.conversation.active_id()
    core.memory.ingest('Documento escolhido','origem','file','O exemplo selecionado explica funções locais. '*30)
    core.memory.ingest('Documento não escolhido','outra','file','NUNCA_INCLUIR_ESTA_FRASE_PRIVADA. '*30)
    source=next(s for s in core.memory.sources() if s['origin']=='origem')
    sources=core.memory.sources();runs=core.db.query('SELECT id FROM runs')
    result=prepare_document_replay(core,[source['id']])
    raw=(core.config.data/'datasets'/(result['dataset_hash']+'.json')).read_text(encoding='utf-8')
    assert 'NUNCA_INCLUIR_ESTA_FRASE_PRIVADA' not in raw and result['document_sample_fraction']<=.1
    assert core.memory.sources()==sources and core.db.query('SELECT id FROM runs')==runs
    assert core.conversation.active_id()==parent
    core.gate.set_enabled(False)
    with pytest.raises(Exception):prepare_document_replay(core,[source['id']])


def test_ui_saves_separate_dialogue_learning_rate_and_experimental_choices(core,qtapp):
    from tac.ui import Window
    window=Window(core);window.timer.stop()
    reading=core.config.get('learning_rate')
    window.settings_widgets['conversation_learning_rate'].setValue(.00004)
    window.conversation_stability_setting.setCurrentIndex(1)
    window.conversation_generation_setting.setCurrentIndex(1)
    window.save_settings()
    assert core.config.get('conversation_learning_rate')==.00004
    assert core.config.get('conversation_stability')=='anchor_v1' and core.config.get('conversation_generation')=='controlled'
    assert core.config.get('learning_rate')==reading
    window.close()
