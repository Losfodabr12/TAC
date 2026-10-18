"""Auditoria real do ZIP em ambientes isolados; sentinelas são dados de teste."""
import copy
import hashlib
import json
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest

from tac.db import now
from tac.diagnostic_export import NAMES,export_diagnostic,metric_states
from tac.diagnostic_privacy import Privacy
from tac.diagnostic_datasets import analyze_dataset
from tac.neural.conversation import ConversationData
from test_training import setup_training


def unpack(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())==set(NAMES)|{'manifesto.json'}
        files={n:archive.read(n).decode('utf-8') for n in archive.namelist()}
    manifest=json.loads(files['manifesto.json'])
    for name,record in manifest['arquivos'].items():
        assert hashlib.sha256(files[name].encode('utf-8')).hexdigest()==record['sha256']
    for name,text in files.items():
        if name.endswith('.json'):json.loads(text,parse_constant=lambda s:pytest.fail('JSON não finito: '+s))
    return files


def database_state(core):
    with core.db.connect() as c:return list(c.iterdump())


def test_export_works_without_models_or_training_and_outside_project(core,tmp_path,monkeypatch):
    core.gate.set_enabled(False)
    monkeypatch.setattr(core.trainer,'train',lambda *a,**k:pytest.fail('Não pode treinar'))
    monkeypatch.setattr(core.conversation,'train',lambda *a,**k:pytest.fail('Não pode treinar diálogo'))
    before=database_state(core); cfg=core.config.path.read_bytes()
    target=tmp_path/'fora_da_TAC.zip'
    result=export_diagnostic(core,target)
    files=unpack(target)
    assert result['runs']==0 and result['files']==9
    assert not json.loads(files['treinamento.json'])['execucoes']
    assert all(m['estado']=='sem_modelo_ativo_legivel' for m in json.loads(files['arquitetura.json'])['modelos_ativos'])
    assert database_state(core)==before and core.config.path.read_bytes()==cfg
    assert not core.gate.enabled
    assert not any(n.endswith(('.pt','.sqlite3','.jsonl')) for n in files)
    assert 'Nenhuma avaliação nova' in files['avaliacoes.json']


def test_real_checkpoint_metadata_counts_shapes_and_no_weights_or_network_instantiation(core,tmp_path,monkeypatch):
    corpus=setup_training(core,5)
    result=core.perform('Treino real de preparação do teste',lambda:core.trainer.train(corpus=corpus),learning=True)
    path=core.trainer.checkpoint_path(result['checkpoint'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    before=database_state(core)
    def forbidden(*a,**k):pytest.fail('Exportação não instancia redes, avalia, gera ou treina')
    monkeypatch.setattr('tac.neural.model.TACTransformer.__init__',forbidden)
    monkeypatch.setattr(core.trainer,'evaluate',forbidden)
    monkeypatch.setattr(core.trainer,'generate',forbidden)
    target=tmp_path/'real.zip'; export_diagnostic(core,target)
    files=unpack(target)
    model=next(m for m in json.loads(files['arquitetura.json'])['modelos_ativos'] if m['ativo_em']==['active_checkpoint'])
    assert model['parametros_unicos_observados']==result['parameters']
    assert model['contagem_confere'] and model['tensores']['tokens.weight']['shape']==[259,32]
    assert model['arquitetura']['layers']==1 and model['tokenizador']['vocab_size']==259
    training=json.loads(files['treinamento.json'])
    assert training['execucoes'][0]['run_id']==result['run_id']
    assert training['metricas_por_etapa']
    stats=json.loads(files['datasets_estatisticas.json'])['conjuntos'][0]
    assert stats['integridade']['hash_confere'] and stats['splits']['train']['exemplos']==2
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest and database_state(core)==before


def test_privacy_in_all_channels_and_informative_scores_are_preserved(core,tmp_path):
    from tac.conversation_lessons import make_sets
    core.gate.set_enabled(True)
    sensitive='PessoaPrivadaSentinelaZXQ'
    email='sentinela.secreta@example.invalid'
    credential='sk-SEGREDO_NAO_EXPORTAR_1234567890'
    number=1234567890987654321
    private_path=r'C:\Users\PessoaPrivadaSentinelaZXQ\segredo.txt'
    core.memory.ingest(sensitive,private_path,'file',sensitive+' '+email+' '+credential)
    core.db.execute('INSERT INTO interactions(question,answer,mode,sources,created) VALUES(?,?,?,?,?)',
                    (sensitive,credential,'documental','[]',now()))
    row=make_sets()['validation'][0]
    public={**row,'expected':row['answer'],'actual':row['answer'],'exact':True}
    private={**row,'expected':sensitive,'actual':email+' '+credential,'group':sensitive,'skill':sensitive}
    detail={'baseline':{'loss':1.,'examples':1,'token_accuracy':.9},'validation':{'loss':.5,'examples':1,'exact_accuracy':0.,'outputs':[public,private]},
            'promotion_checks':{'respostas_basicas':False},'detail':'Critérios não atingidos: respostas_basicas',
            'imported_evaluation':{'status':'concluido','splits':{'test':{'candidate':{'examples':1,
                    'by_skill':{sensitive:{'correct':0,'total':1}},'outputs':[public]}}}},
            'password':number,'qualquer_dado':number,'error':private_path}
    rid='a'*32
    core.db.execute('INSERT INTO runs(id,kind,status,started,config,dataset_hash,detail) VALUES(?,?,?,?,?,?,?)',
        (rid,'conversation','rejeitado',now(),json.dumps({'steps':6000,'learning_rate':.0003,'api_key':credential,'campo_privado':number}),
         'b'*64,json.dumps(detail,ensure_ascii=False)))
    log=core.config.data/'logs'/'tac.log'
    with log.open('a',encoding='utf-8') as f:
        f.write('2026-10-16 12:00:00,123 ERROR '+sensitive+' '+email+' '+credential+'\n')
        f.write('Traceback (most recent call last):\n  File "'+private_path+'", line 18, in '+sensitive+'\n')
        f.write('RuntimeError: '+credential+'\n')
    target=tmp_path/'privacidade.zip';export_diagnostic(core,target)
    files=unpack(target); combined='\n'.join(files.values())
    for text in (sensitive,email,credential,private_path,str(number)):
        assert text not in combined,text
    assert row['answer'] in files['exemplos_de_respostas.json']
    assert 'respostas_basicas' in files['avaliacoes.json'] and 'RuntimeError' in files['logs.txt']
    entries=json.loads(files['exemplos_de_respostas.json'])['avaliacoes']
    imported=next(e for e in entries if e['conteudo_privado_omitido'])
    assert '[texto:' in json.dumps(imported,ensure_ascii=False)
    evaluations=json.loads(files['avaliacoes.json'])
    states=evaluations['execucoes'][0]['estados_metricas']
    assert any(r['estado']=='nao_registrado' and r['valor'] is None for r in states)
    assert any(r['estado']=='zero_medido' for r in states)


def test_dataset_leakage_and_anomalies_use_counts_without_private_text():
    privacy=Privacy()
    rows={'train':[{'messages':[['user','Pergunta privada ABC']],'answer':'Resposta privada',
                    'skill':'pessoa_secreta','group':'mesmo_grupo','imported':True}],
          'validation':[{'messages':[['user','pergunta privada abc!']],'answer':'Outra',
                    'skill':'pessoa_secreta','group':'mesmo_grupo','imported':True}], 'test':[]}
    data=ConversationData(rows)
    value={'sets':rows,'hash':data.digest}
    result=analyze_dataset(value,data.digest,privacy,[16])
    assert result['vazamento']['comparacoes']['train__validation']=={'entradas_normalizadas_repetidas':1,'grupos_repetidos':1}
    assert result['splits']['train']['por_contexto']['16']['respostas_nao_cabem']==1
    assert result['integridade']['hash_confere']
    assert 'Pergunta privada' not in json.dumps(result) and 'pessoa_secreta' not in json.dumps(result)


def test_tied_embeddings_count_once_even_if_saved_as_separate_storages(tmp_path):
    import torch
    from tac.neural.model import ModelConfig,TACTransformer
    from tac.diagnostic_export import checkpoint_metadata
    model=TACTransformer(ModelConfig(width=32,heads=2,layers=1,context=32,tie_embeddings=True))
    path=tmp_path/'clone.pt'
    # Replica o efeito de copiar cada entrada CUDA separadamente para CPU ao salvar.
    torch.save({'format':'TAC-v1','model_config':vars(model.cfg),
                'model':{k:v.clone() for k,v in model.state_dict().items()}},path)
    meta=checkpoint_metadata(path,Privacy())
    assert meta['parametros_unicos_observados']==model.parameter_count
    assert meta['contagem_confere'] and meta['parametros_unicos_armazenados']>model.parameter_count


def test_zero_is_not_invented_and_empty_set_has_different_state():
    result=metric_states({'baseline':{'loss':.4,'examples':20,'token_accuracy':.9},
                         'validation':{'examples':8,'exact_accuracy':0.},
                         'session_validation':{'examples':0,'recall_accuracy':0.},
                         'unknown':{'accuracy':0.}})
    lookup={r['caminho']:r for r in result}
    assert lookup['/baseline/exact_accuracy']['valor'] is None
    assert lookup['/baseline/exact_accuracy']['estado']=='nao_registrado'
    assert lookup['/validation/exact_accuracy']['estado']=='zero_medido'
    assert lookup['/session_validation/recall_accuracy']['estado']=='zero_sem_amostras'
    assert lookup['/unknown/accuracy']['estado']=='zero_origem_indeterminada'


def test_missing_corrupt_artifacts_and_nonfinite_metrics_are_explicit(core,tmp_path):
    rid='c'*32; cid='d'*32; dsid='e'*64
    core.db.execute('INSERT INTO runs(id,kind,status,started,config,dataset_hash,detail) VALUES(?,?,?,?,?,?,?)',
        (rid,'conversation','erro',now(),'{}',dsid,'{"baseline":{"loss":NaN},"validation":null}'))
    core.db.execute('INSERT INTO checkpoints(id,run_id,filename,step,status,created) VALUES(?,?,?,?,?,?)',
        (cid,rid,cid+'.pt',4,'erro',now()))
    (core.config.data/'datasets'/(dsid+'.json')).write_text('{corrompido',encoding='utf-8')
    target=tmp_path/'incompleto.zip';export_diagnostic(core,target)
    files=unpack(target)
    assert json.loads(files['checkpoints.json'])['checkpoints'][0]['metadados']['estado']=='arquivo_ausente'
    assert json.loads(files['datasets_estatisticas.json'])['conjuntos'][0]['estado']=='indisponivel'
    assert json.loads(files['avaliacoes.json'])['execucoes'][0]['resultado_registrado']['baseline']['loss'] is None


def test_atomic_failure_preserves_existing_destination_and_cleans_only_own_temp(core,tmp_path,monkeypatch):
    target=tmp_path/'anterior.zip';target.write_bytes(b'CONTEUDO_ANTERIOR')
    real=zipfile.ZipFile.writestr; calls=[]
    def fail(self,*args,**kwargs):
        calls.append(1)
        if len(calls)==3:raise OSError('Erro de disco controlado')
        return real(self,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'writestr',fail)
    with pytest.raises(OSError):export_diagnostic(core,target,overwrite=True)
    assert target.read_bytes()==b'CONTEUDO_ANTERIOR'
    assert not list(tmp_path.glob('.tac_diagnostico_*.tmp'))


def test_existing_destination_and_wrong_extension_require_no_overwrite(core,tmp_path):
    target=tmp_path/'old.zip';target.write_bytes(b'old')
    with pytest.raises(FileExistsError):export_diagnostic(core,target)
    with pytest.raises(ValueError):export_diagnostic(core,tmp_path/'weights.pt')
    assert target.read_bytes()==b'old'


def test_button_exports_in_worker_with_learning_disabled(core,qtapp,tmp_path,monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog
    from tac.ui import Window
    from test_backup_ui import pump
    target=tmp_path/'escolhido.zip'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(target),'Diagnóstico ZIP (*.zip)'))
    window=Window(core);window.timer.stop();window.show()
    assert window.export_diagnostic_button.text()=='Exportar diagnóstico completo'
    assert window.export_diagnostic_button.isEnabled() and not core.gate.enabled
    beats=[];timer=QTimer();timer.timeout.connect(lambda:beats.append(1));timer.start(10)
    window.export_diagnostic_button.click()
    pump(qtapp,lambda:window.job is None,timeout=30)
    timer.stop()
    assert target.is_file() and len(beats)>=5
    assert not core.gate.enabled and not core.db.query('SELECT * FROM runs')
    assert 'Nenhum treino foi iniciado' in window.notice.text()
    unpack(target);window.close()
