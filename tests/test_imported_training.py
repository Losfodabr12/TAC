"""Pesos e retomada reais; casos de falha injetada são apenas testes da política."""
import hashlib

import pytest
import torch

from tac.neural.conversation import ConversationData,load_checkpoint,prepare_data
from tac.neural.training import weight_hash
from test_conversation_imports import record,write_jsonl
from test_generative_conversation import small_sets
from test_training import setup_training


@pytest.fixture
def imported_training(core,monkeypatch):
    monkeypatch.setattr('tac.neural.conversation.make_sets',small_sets)
    corpus=setup_training(core,5)
    core.perform('Leitor real de teste',lambda:core.trainer.train(corpus=corpus),learning=True)
    base=core.perform('Conversa real de teste',lambda:core.conversation.train(steps=120,learning_rate=.003),learning=True)
    assert base['status']=='aprovado'
    parent=core.conversation.active_id()
    def compact_data(cp,large=False,curriculum=None):
        data=prepare_data(cp,large=False)
        if curriculum:
            return ConversationData({**data.sets,'train':[{**r,'mixture':'legacy'} for r in data.sets['train']]
                +[{**r,'mixture':'reviewed','skill':'reviewed'} for r in data.sets['train']]},data.tokenizer)
        return data
    monkeypatch.setattr('tac.neural.conversation.prepare_data',compact_data)
    path=write_jsonl(core.config.root/'conversas.jsonl',[
        record('treino','train',question='Fala aí?',answer='Oi.'),
        record('validacao','validation',question='Boas?',answer='Oi.'),
        record('teste','test',question='E aí?',answer='Oi.')])
    preview=core.conversation.preview_import(path)
    assert preview['report']['accepted_examples']==3
    core.conversation.commit_import(preview)
    return parent


def test_real_imported_steps_retention_and_resume_frozen_data(core,imported_training,monkeypatch):
    parent=imported_training
    path=core.trainer.checkpoint_path(parent)
    original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    prepared=core.conversation.prepare_training(curriculum='fluency')
    assert prepared['counts']['user_groups']==1 and prepared['counts']['program_examples']==8
    full=core.perform('Importadas completo',lambda:core.conversation.train(steps=120,curriculum='fluency',prepared=prepared),learning=True)
    assert full['initial_hash']!=full['final_hash']
    assert full['mixture_counts']=={'imported':33,'reviewed':33,'language':36,'legacy':18}
    assert full['imported_evaluation']['status']=='concluido'
    assert full['imported_evaluation']['informational_only']
    assert full['imported_evaluation']['reference_checkpoint']==parent
    assert full['imported_evaluation']['splits']['test']['candidate']['examples']==1
    core.conversation.activate(parent)
    def stop(step,message):
        if step==100:core.gate.set_enabled(False)
    partial=core.perform('Importadas interrompido',lambda:core.conversation.train(stop,steps=120,curriculum='fluency'),learning=True)
    assert partial['steps']==100 and partial['status']=='interrompido'
    assert core.conversation.active_id()==parent
    core.gate.set_enabled(True); core.gate.begin()
    store=core.conversation.import_store()
    store.set_enabled(store.list()[0]['id'],False)
    resumed_data=core.conversation.prepare_training(curriculum='fluency',resume=partial['checkpoint'])
    assert resumed_data['counts']==prepared['counts']
    resumed=core.perform('Importadas retomado',lambda:core.conversation.train(resume=partial['checkpoint']),learning=True)
    assert resumed['final_hash']==full['final_hash']
    assert resumed['mixture_counts']==full['mixture_counts']
    assert resumed['dataset_hash']==full['dataset_hash']
    assert resumed['imported_evaluation']['reference_checkpoint']==parent
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original_hash
    cp=load_checkpoint(core.trainer.checkpoint_path(resumed['checkpoint']))
    model,_=core.conversation.load(resumed['checkpoint'])
    assert weight_hash(model)==resumed['final_hash']
    assert cp['training_config']['retention_weight']==2.
    assert cp['training_config']['retention_source']==parent


def test_imported_metrics_never_change_gates_or_selection_even_when_they_fail(core,imported_training,monkeypatch):
    parent=imported_training
    results=[]
    for mode in ('real','zero','perfect','error'):
        core.conversation.activate(parent)
        if mode!='real':
            def controlled_info(model,data,device,split,mode=mode):
                if mode=='error':raise RuntimeError('Falha informativa controlada')
                score=int(mode=='perfect')
                return {'examples':1,'groups':1,'by_skill':{'conversa':{'correct':score,'total':1}},'outputs':[]}
            monkeypatch.setattr(core.conversation,'evaluate_imported',controlled_info)
        result=core.perform('Política informativa '+mode,
            lambda:core.conversation.train(steps=40,curriculum='fluency'),learning=True)
        results.append(result)
    expected=results[0]
    assert all(r['status']==expected['status'] for r in results)
    assert all(r['promotion_checks']==expected['promotion_checks'] for r in results)
    assert all(r['final_hash']==expected['final_hash'] and r['selected_step']==expected['selected_step'] for r in results)
    assert results[-1]['imported_evaluation']['status']=='erro'
    assert all('import' not in key for r in results for key in r['promotion_checks'])


def test_rejected_imported_run_keeps_active_files_and_cancellation_blocks_promotion(core,imported_training,monkeypatch):
    parent=imported_training; path=core.trainer.checkpoint_path(parent)
    original=path.read_bytes()
    real_evaluate=core.conversation.evaluate
    def reject(*a,**kwargs):
        result=real_evaluate(*a,**kwargs)
        if kwargs.get('generation'):result['exact_accuracy']=0.
        return result
    monkeypatch.setattr(core.conversation,'evaluate',reject)
    result=core.perform('Rejeição controlada',lambda:core.conversation.train(steps=40,curriculum='fluency'),learning=True)
    assert result['status']=='rejeitado' and result['initial_hash']!=result['final_hash']
    assert result['imported_evaluation']['status']=='concluido'
    assert core.conversation.active_id()==parent and path.read_bytes()==original
    monkeypatch.setattr(core.conversation,'evaluate',real_evaluate)
    real_imported=core.conversation.evaluate_imported
    def cancel(*a,**kwargs):
        core.gate.set_enabled(False)
        return real_imported(*a,**kwargs)
    monkeypatch.setattr(core.conversation,'evaluate_imported',cancel)
    stopped=core.perform('Parar na comparação',lambda:core.conversation.train(steps=40,curriculum='fluency'),learning=True)
    assert stopped['status']=='interrompido'
    assert stopped['imported_evaluation']['status']=='interrompido'
    assert core.conversation.active_id()==parent and path.read_bytes()==original


def test_prepared_confirmation_rejects_registry_changes_before_training(core,imported_training):
    value=core.conversation.prepare_training(curriculum='fluency')
    store=core.conversation.import_store()
    store.set_enabled(store.list()[0]['id'],False)
    with pytest.raises(ValueError,match='material do treino mudou'):
        core.perform('Prévia desatualizada',lambda:core.conversation.train(steps=1,curriculum='fluency',prepared=value),learning=True)
    assert core.conversation.active_id()==imported_training
