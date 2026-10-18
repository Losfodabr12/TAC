"""Importação real em bancos isolados; não modifica o acervo do usuário."""
import copy
import hashlib
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path

import pytest
import torch

from tac.dialogue_imports import (ConversationImports, automatic_split, derive_examples,
                                digest, imported_kind, merge_imports, normalize_record,
                                prompt_key, read_records, training_counts)
from tac.learning import Cancelled, LearningDisabled
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import ConversationData


def record(group='casa_001', split='train', question='Minha janela fica aberta?', answer='Sim.'):
    return {'group':group,'skill':'conversa','split':split,
            'messages':[{'role':'user','content':question}],'answer':answer}


def write_jsonl(path, records):
    path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    return path


def store_for(core):
    core.gate.set_enabled(True)
    core.gate.begin()
    return ConversationImports(core.config,core.db,core.gate)


def test_both_complete_examples_have_identical_targets_and_seven_answers(core):
    store = store_for(core)
    root = Path(__file__).resolve().parents[1]/'examples'
    previews = [store.preview(root/('conversas_exemplo'+suffix),LocalBPETokenizer()) for suffix in ('.txt','.jsonl')]
    for value in previews:
        assert value['report']['records']==3 and value['report']['groups']==3
        assert value['report']['accepted_examples']==7 and not value['report']['rejected']
        assert value['report']['by_split']=={'train':{'groups':1,'examples':3},
            'validation':{'groups':1,'examples':2},'test':{'groups':1,'examples':2}}
    assert [(r['example_hash'],r['split'],r['episode'],r['turn']) for r in previews[0]['rows']] == [
        (r['example_hash'],r['split'],r['episode'],r['turn']) for r in previews[1]['rows']]
    assert not core.db.query('SELECT * FROM dialogue_imports')  # Conferir não grava.


def test_invalid_records_report_roles_last_user_long_answer_and_json(core,tmp_path):
    store = store_for(core)
    role = record('papel'); role['messages'][0]['role']='system'
    last = record('ultima'); last['messages'].append({'role':'assistant','content':'Algo.'})
    boundary = record('limite',answer='x'*506)  # 506 bytes + espaço + EOS = 508.
    good = record('valida',answer='x'*505)  # 507 cabe sem alterar o codificador.
    path = write_jsonl(tmp_path/'mixed.jsonl',[role,last,boundary,good])
    with path.open('a',encoding='utf-8') as f:
        f.write('{quebrado}\n')
    preview = store.preview(path,LocalBPETokenizer())
    report = preview['report']
    assert report['accepted_examples']==1 and report['rejected_records']==4
    assert [r['line'] for r in report['rejected']]==[1,2,3,5]
    reasons = ' '.join(r['reason'] for r in report['rejected'])
    assert 'Papel inválido' in reasons and 'última mensagem' in reasons
    assert '508 tokens' in reasons and 'Limite: 507' in reasons and 'JSON inválido' in reasons
    x,y = ConversationData({'train':preview['rows']},LocalBPETokenizer()).encode(preview['rows'][0],512)
    assert len(x)<=512 and int((y!=0).sum())==507


def test_txt_continuation_bom_and_reject_entire_malformed_conversation(core,tmp_path):
    store = store_for(core)
    text = ('=== conversa ===\ngroup: multilinha\nskill: escrita\nsplit: train\n'
            'Usuário: Primeira linha\n  segunda linha\nTAC: Resposta\n  com continuação\n'
            '=== conversa ===\ngroup: ruim\nskill: escrita\nsplit: test\n'
            'Usuário: Isto é inválido\nRobô: Não é um papel aceito\nTAC: Final\n')
    path = tmp_path/'falas.txt'; path.write_text(text,encoding='utf-8-sig')
    value = store.preview(path,LocalBPETokenizer())
    assert value['report']['accepted_examples']==1 and value['report']['rejected_records']==1
    assert value['rows'][0]['messages'][0][1]=='Primeira linha\nsegunda linha'
    assert value['rows'][0]['answer']=='Resposta\ncom continuação'
    assert value['report']['rejected'][0]['group']=='ruim'
    raw = b'group: missing\nskill: s\nsplit: train\n'
    assert 'cabeçalho' in list(read_records(raw,'.txt'))[0][2]


def test_groups_global_stable_across_files_conflicts_and_duplicates(core,tmp_path):
    store = store_for(core); tokenizer = LocalBPETokenizer()
    rows = [record('familia_'+str(i),'auto',question=f'Pergunta independente {i}?') for i in range(30)]
    a = store.preview(write_jsonl(tmp_path/'primeiro.jsonl',rows),tokenizer)
    b = store.preview(write_jsonl(tmp_path/'invertido.jsonl',list(reversed(rows))),tokenizer)
    assignment = lambda p:{r['group']:r['split'] for r in p['rows']}
    assert assignment(a)==assignment(b)=={r['group']:automatic_split(r['group']) for r in rows}
    committed = store.commit(a)
    assert len(store.list())==1 and store.list()[0]['id']==committed['id']
    again = store.preview(tmp_path/'invertido.jsonl',tokenizer)
    assert again['report']['duplicates']==30 and not again['rows']
    group, original = next(iter(assignment(a).items()))
    other = 'test' if original!='test' else 'train'
    conflict = store.preview(write_jsonl(tmp_path/'conflito.jsonl',[
        record(group,other,question='Variação nova do mesmo assunto?'),
        record('novo','train',question='Primeira versão?'),record('novo','validation',question='Segunda versão?')]),tokenizer)
    assert not conflict['rows'] and conflict['report']['rejected_records']==3
    assert store.snapshot(digest(tokenizer.state()))


def test_benchmark_overlap_and_equal_prompt_across_splits_are_rejected(core,tmp_path):
    store = store_for(core); tokenizer=LocalBPETokenizer()
    protected = derive_examples(normalize_record(record(question='Benchmark reservado?')),tokenizer,512,1)[0][0]
    trained = derive_examples(normalize_record(record(question='Já treinada?')),tokenizer,512,1)[0][0]
    path = write_jsonl(tmp_path/'leak.jsonl',[
        record('a',question='Benchmark reservado?'),record('b','test',question='Já treinada?'),
        record('c','train',question='Uma pergunta igual?'),record('d','validation',question='Uma pergunta igual?',answer='Talvez.')])
    result = store.preview(path,tokenizer,reserved={prompt_key(protected)},program_train={prompt_key(trained)})
    assert result['report']['rejected_examples']==3
    assert len(result['rows'])==1 and result['rows'][0]['split']=='validation'


def test_commit_is_gated_atomic_and_rechecks_file_and_registry(core,tmp_path):
    store=store_for(core); tok=LocalBPETokenizer()
    path=write_jsonl(tmp_path/'original.jsonl',[record()]); raw=path.read_bytes()
    preview=store.preview(path,tok)
    core.gate.set_enabled(False)
    with pytest.raises(LearningDisabled):store.commit(preview)
    assert not store.list() and not (core.config.data/'dialogue_imports').exists()
    core.gate.set_enabled(True); core.gate.begin()
    path.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='mudou'):store.commit(preview)
    path.write_bytes(raw)
    second=store.preview(write_jsonl(tmp_path/'other.jsonl',[record('outra',question='Uma outra?')]),tok)
    store.commit(second)
    with pytest.raises(ValueError,match='lista de conversas mudou'):store.commit(preview)
    assert len(store.list())==1 and path.read_bytes()==raw
    cancelled=threading.Event(); cancelled.set()
    with pytest.raises(Cancelled):store.preview(path,tok,cancel=cancelled)


def test_import_persists_and_roundtrips_backups_without_touching_documents(core,tmp_path):
    from tac.core import Core
    from tac.backups import export_backup,import_backup
    store=store_for(core); tok=LocalBPETokenizer()
    core.memory.ingest('Documento','origem','file','Uma memória documental que permanece independente das conversas.')
    source=core.memory.sources()
    path=write_jsonl(tmp_path/'conversas.jsonl',[record()])
    value=store.commit(store.preview(path,tok))
    reopened=Core(core.config.root)
    assert reopened.memory.sources()==source and len(reopened.conversation.import_store().snapshot(digest(tok.state())))==1
    archived=core.config.data/'dialogue_imports'/store.list()[0]['filename']
    assert archived.read_bytes()==path.read_bytes()
    store.set_enabled(value['id'],False)
    assert not store.snapshot(digest(tok.state()))
    store.set_enabled(value['id'],True)
    destination=core.config.root/'import_backup.zip'
    export_backup(core,destination); import_backup(core,destination)
    restored=Core(core.config.root)
    assert not restored.gate.enabled and not restored.config.get('auto_train')
    assert restored.memory.sources()==source
    assert restored.conversation.import_store().snapshot(digest(tok.state()))==store.snapshot(digest(tok.state()))
    assert (restored.config.data/'dialogue_imports'/archived.name).read_bytes()==path.read_bytes()


def test_cancellation_mid_commit_rolls_back_database_and_managed_copy(core,tmp_path,monkeypatch):
    store=store_for(core)
    path=write_jsonl(tmp_path/'interromper.jsonl',[record('primeira'),record('segunda',question='Outra entrada?')])
    preview=store.preview(path,LocalBPETokenizer())
    real_check=core.gate.check
    calls=[]
    def check():
        calls.append(1)
        if len(calls)==3:core.gate.cancel.set()
        real_check()
    monkeypatch.setattr(core.gate,'check',check)
    with pytest.raises(Cancelled):store.commit(preview)
    assert not core.db.query('SELECT * FROM dialogue_imports')
    assert not core.db.query('SELECT * FROM dialogue_groups')
    assert not core.db.query('SELECT * FROM dialogue_examples')
    assert not list((core.config.data/'dialogue_imports').iterdir())
    assert path.is_file()


def test_old_backup_schema_one_still_restores_and_migrates(core):
    from tac.core import Core
    from tac.backups import export_backup,import_backup
    with core.db.connect() as c:
        c.executescript('DROP TABLE dialogue_examples; DROP TABLE dialogue_groups; DROP TABLE dialogue_imports;')
        c.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    target=core.config.root/'backup_antigo.zip'
    export_backup(core,target)
    import_backup(core,target)
    restored=Core(core.config.root)
    assert restored.db.query("SELECT value FROM meta WHERE key='schema_version'")[0]['value']=='2'
    assert not restored.db.query('SELECT * FROM dialogue_examples')
    assert len(list((restored.config.data/'backups').glob('antes_schema2_*.sqlite3')))==1


def test_v1_schema_migration_backs_up_and_preserves_data(tmp_path):
    from tac.db import Database,SCHEMA
    path=tmp_path/'data'/'tac.sqlite3'; path.parent.mkdir()
    with sqlite3.connect(path) as c:
        c.executescript(SCHEMA)
        c.execute("INSERT INTO meta VALUES('active_conversation','preservado')")
    db=Database(path)
    assert db.query("SELECT value FROM meta WHERE key='schema_version'")[0]['value']=='2'
    assert db.query("SELECT value FROM meta WHERE key='active_conversation'")[0]['value']=='preservado'
    backups=list((path.parent/'backups').glob('antes_schema2_*.sqlite3'))
    assert len(backups)==1
    with sqlite3.connect(backups[0]) as c:
        assert c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]=='1'
    Database(path)
    assert len(list((path.parent/'backups').glob('antes_schema2_*.sqlite3')))==1


def test_merge_keeps_old_benchmarks_and_never_samples_imported_holdouts(core,tmp_path):
    from test_generative_conversation import small_sets
    store=store_for(core); tok=LocalBPETokenizer()
    rows=[record('t','train'),record('v','validation',question='Só para validar?'),record('s','test',question='Só para testar?')]
    value=store.preview(write_jsonl(tmp_path/'sets.jsonl',rows),tok); store.commit(value)
    original=ConversationData(small_sets(),tok)
    old=copy.deepcopy(original.sets)
    data,skipped=merge_imports(original,store.snapshot(digest(tok.state())))
    assert not skipped and original.sets==old
    assert data.sets['validation']==old['validation'] and data.sets['test']==old['test']
    for seed in range(3):
        assert all(r['group']=='t' for r in data.sample_pool(100,torch.Generator().manual_seed(seed),'imported'))
    assert training_counts(data)=={'user_groups':1,'user_examples':1,'program_examples':4,
        'reserved':{'validation':{'groups':1,'examples':1},'test':{'groups':1,'examples':1}}}
    for accumulation in (1,2,4):
        assert Counter(imported_kind(step,m,accumulation) for step in range(40//accumulation) for m in range(accumulation))=={
            'imported':11,'reviewed':11,'language':12,'legacy':6}


def test_imported_eval_uses_generated_answers_and_scores_by_skill(core,monkeypatch):
    core.gate.set_enabled(True); core.gate.begin()
    value=record('v','validation',answer='REFERÊNCIA SEGUNDA')
    value['messages']=[{'role':'user','content':'Primeira?'},{'role':'assistant','content':'REFERÊNCIA PRIMEIRA'},
                       {'role':'user','content':'Segunda?'}]
    rows,_=derive_examples(normalize_record(value),LocalBPETokenizer(),512,1)
    data=ConversationData({'train':[],'validation_imported':rows})
    calls=[]
    def decode(model,messages,device):
        assert not any('REFERÊNCIA' in text for role,text in messages)
        if calls:assert messages[-2]==['assistant','RESPOSTA PRÓPRIA']
        calls.append(messages)
        return 'RESPOSTA PRÓPRIA'
    monkeypatch.setattr('tac.neural.conversation.decode_reply',decode)
    class Model:
        def eval(self):return self
    result=core.conversation.evaluate_imported(Model(),data,'cpu','validation_imported')
    assert result['examples']==2 and result['by_skill']=={'conversa':{'correct':0,'total':2}}
    assert len(calls)==2 and result['outputs'][-1]['user']=='Segunda?'


def test_result_shows_both_reserved_sets_by_skill_without_changing_decision(core,qtapp):
    from tac.ui import Window
    from tac.presentation import imported_results_text
    value={'status':'concluido','informational_only':True,'splits':{}}
    for split in ('validation','test'):
        value['splits'][split]={'active':{'examples':4,'by_skill':{'logica':{'correct':1,'total':2},'contexto':{'correct':1,'total':2}}},
                              'candidate':{'examples':4,'by_skill':{'logica':{'correct':2,'total':2},'contexto':{'correct':0,'total':2}}}}
    window=Window(core); window.timer.stop()
    result={'status':'rejeitado','promotion_checks':{'loss_basica':False},'imported_evaluation':value}
    before=copy.deepcopy(result)
    window.conversation_result(result)
    text=window.train_label.text()
    assert 'Rejeitado' in text and 'Validação' in text and 'Teste' in text
    assert 'logica: anterior 1/2; nova 2/2' in text and 'contexto: anterior 1/2; nova 0/2' in text
    assert 'Não muda a aprovação' in text and result==before
    assert 'não concluída' in imported_results_text({'status':'erro'})
    window.close()
