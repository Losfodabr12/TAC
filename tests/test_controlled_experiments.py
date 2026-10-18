import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from tac.learning import LearningGate, LearningDisabled, Cancelled
from tac.neural.controlled import (BalancedImports, Exposure, MatchedRetention,
    ResponseObjective, category, row_key, relation, diagnostic_gradients, vector)
from tac.neural.conversation import ConversationData, ConversationEngine, decode_reply
from tac.neural.controlled_eval import DevelopmentEvaluator, trace_reply, protection_checks
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.tokenizer import ByteTokenizer
from scripts.run_controlled_experiments import (LiveSettings, protected_step, full_retention,
    common_plan, arm_plan)


def row(question, answer='Certo.', skill='general', group='g', history=None, **extra):
    return {'messages':[*list(history or []),['user',question]],'answer':answer,
        'skill':skill,'group':group,**extra}


def replay_fixture():
    tok=ByteTokenizer();turns=[('Oi!','Olá!'),('Como vai?','Tudo bem.')]
    text='\n'.join('Usuário: '+u+'\nTAC: '+a for u,a in turns)
    replay=SimpleNamespace(rows=[torch.tensor(tok.encode(text)),torch.tensor(tok.encode('Uma história curta.'))],
        provenance=[{'authored_scene':'cena'},{'authored_narrative':0}])
    return tok,replay,[['cena',turns]]


def test_response_objective_masks_roles_and_users_keeps_history_and_eos():
    tok,replay,scenes=replay_fixture();obj=ResponseObjective(replay,scenes,tok)
    x0,y0,_=obj.batch([(0,0),(1,0)],variant=False)
    x1,y1,_=obj.batch([(0,0),(1,0)],variant=True)
    assert torch.equal(x0,x1) and torch.equal(y0[1],y1[1])
    assert tok.decode(y1[0].tolist())=='Olá!Tudo bem.'
    assert (y1[0]==tok.eos).sum()==2
    assert (y1[0]!=0).sum() < (y0[0]!=0).sum()
    assert 'Como vai?' in tok.decode(x1[0].tolist())
    assert obj.changed_rows==1 and obj.boundaries==2
    # Cada alvo textual corresponde ao próximo token de x; só EOS pode substituir uma fronteira.
    keep=(y1[0]!=0)&(y1[0]!=tok.eos)
    assert torch.equal(y1[0][keep],y0[0][keep])


def test_response_objective_matches_frozen_bpe():
    from tac.neural.bpe import LocalBPETokenizer
    tok,replay,scenes=replay_fixture()
    texts=[tok.decode(r.tolist()) for r in replay.rows]
    bpe=LocalBPETokenizer.fit(texts,vocab_size=300)
    replay.rows=[torch.tensor(bpe.encode(t)) for t in texts]
    obj=ResponseObjective(replay,scenes,bpe)
    _,y,_=obj.batch([(0,0)],variant=True)
    assert bpe.decode(y[0].tolist()).replace(' ','')=='Olá!Tudobem.'
    assert (y==bpe.eos).sum()==2


def test_balanced_imports_no_repeat_rare_pool_redistributes_and_deterministic():
    questions=['Meu nome é Ana.','Errei, corrija isso.','Não gosto.','Onde moro?', 'Tudo bom?', 'Qual é a capital?']
    rows=[row(q,group=f'{i}_{j}',history=[['user','Moro aqui.'],['assistant','Certo.']] if i==3 else [])
        for i,q in enumerate(questions) for j in range(1 if i==0 else 9)]
    assert len({category(r) for r in rows})==6
    a=BalancedImports(rows,torch.Generator().manual_seed(77))
    b=BalancedImports(rows,torch.Generator().manual_seed(77))
    result=a.sample(len(rows));other=b.sample(len(rows))
    assert [row_key(r) for r in result]==[row_key(r) for r in other]
    assert len({id(r) for r in result})==len(rows)
    assert sum(category(r)=='memoria_nomes' for r in result)==1
    with pytest.raises(ValueError,match='já foram usados'):a.sample(1)


def test_retention_matches_tokens_and_covers_families():
    tok=ByteTokenizer()
    rows=[row('nome '+str(i),'Sim.',skill='memory',history=[['user','Não antes.'],['assistant','Ok.']] if i%2 else []) for i in range(8)]
    matcher=MatchedRetention(rows,tok,{row_key(rows[0])})
    result=matcher.sample([rows[0]]*30,tok,torch.Generator().manual_seed(5),True)
    assert all(len(tok.encode(' '+r['answer']))==len(tok.encode(' '+rows[0]['answer'])) for r in result)
    assert {len(r['messages']) for r in result}=={1,3}
    assert row_key(rows[0]) in {row_key(r) for r in result}
    old=ConversationData({'train':rows},tok)
    data=ConversationData({'train':[], 'validation':[rows[0]]},tok)
    assert rows[0] not in full_retention(data,old)


def test_adam_actual_update_and_live_learning_gate(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'learning_enabled':True,'data_dir':'data'}))
    settings=LiveSettings(tmp_path);gate=LearningGate(settings,None)
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    before=vector(model);model(torch.ones(1,2)).sum().backward()
    protected_step(opt,scaler,gate);delta=vector(model)-before
    assert delta.norm()>0 and relation(delta,vector(model,grad=True))['dot']<0
    (tmp_path/'config.json').write_text(json.dumps({'learning_enabled':False,'data_dir':'data'}))
    before=vector(model)
    with pytest.raises(LearningDisabled):protected_step(opt,scaler,gate)
    assert torch.equal(before,vector(model))


def test_cancel_before_update_prevents_all_weight_changes(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'learning_enabled':True,'data_dir':'data'}))
    gate=LearningGate(LiveSettings(tmp_path),None);gate.begin()
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters());before=vector(model)
    model(torch.ones(1,2)).sum().backward();gate.cancel.set()
    with pytest.raises(Cancelled):protected_step(opt,torch.amp.GradScaler('cuda',enabled=False),gate)
    assert torch.equal(before,vector(model)) and not opt.state


def test_correction_metadata_and_history_are_not_erased():
    r=row('Na verdade é Joana.',skill='correcao',history=[['user','Me chamo Lia.'],['assistant','Oi, Lia.']])
    assert category(r)=='correcoes'
    assert row_key(r)!=row_key({**r,'messages':[['user','Na verdade é Joana.']]})


def test_protection_checks_keep_original_thresholds():
    from tac.neural.conversation import context_ready,natural_ready
    from tac.fluency_lessons import fluency_ready
    skills={s:{'correct':10,'total':10} for s in ('context_color','context_update','context_fact','context_location','logic_condition')}
    natural={s:{'correct':10,'total':10} for s in (f'natural_{slot}_{stage}' for slot in ('color','name','topic') for stage in ('correction','recall'))}
    base={'basic':{'loss':.1,'exact_accuracy':.923,'by_skill':{'unknown':{'correct':9,'total':10}}},
        'context':{'by_skill':skills},'natural':{'by_skill':natural,'exact_accuracy':.9},
        'sessions':{'immediate_accuracy':.9,'recall_accuracy':.9},'fluency':{'rule_accuracy':.8,'slot_drift_rate':0.}}
    checks=protection_checks(base,base)
    assert checks['contexto_anterior']==context_ready(base['context'])
    assert checks['dialogos_e_correcoes']==natural_ready(base['natural'],base['natural'])
    assert checks['conversas_revisadas']==fluency_ready(base['fluency'],base['fluency'])
    candidate={**base,'basic':{**base['basic'],'loss':.10201,'exact_accuracy':.9029}}
    failed=protection_checks(candidate,base)
    assert not failed['loss_basica'] and not failed['respostas_basicas']


def test_diagnostic_gradients_do_not_change_weights_or_parameter_grads():
    torch.manual_seed(5);model=TACTransformer(ModelConfig(context=16,width=32,layers=1,heads=2))
    teacher=TACTransformer(model.cfg);teacher.load_state_dict(model.state_dict());teacher.requires_grad_(False)
    x=torch.tensor([[1,5,6,7]]);y=torch.tensor([[0,6,7,2]])
    before=vector(model)
    vs,loss=diagnostic_gradients(model,teacher,{'legacy':(x,y),'imported':(x,y),'kl':(x,y)},'cpu')
    assert torch.equal(before,vector(model)) and all(p.grad is None for p in model.parameters())
    assert relation(vs['legacy'],vs['imported'])['cosine']==pytest.approx(1.)
    assert loss['kl']==pytest.approx(0.,abs=1e-6)


def test_gradient_cosine_has_mathematical_bounds_and_zero_is_unavailable():
    v=torch.randn(10000,generator=torch.Generator().manual_seed(13))
    assert relation(v,v)['cosine']==pytest.approx(1.)
    assert relation(v,-v)['cosine']==pytest.approx(-1.)
    assert relation(v,torch.zeros_like(v))['cosine'] is None


def test_exposure_counts_supervised_tokens_and_repeats():
    rows=[row('Meu nome é Lia.',skill='memory',group='a')]*2
    exposure=Exposure();exposure.add('imported',rows,torch.tensor([[0,4,2],[0,5,2]]));report=exposure.report()
    assert report['pools']['imported']=={'draws':2,'unique_examples':1,'repeated_draws':1,'max_draws_same_example':2,'unique_groups':1}
    assert report['supervised_tokens_by_category']['imported/memoria_nomes']==4


def test_measured_evaluation_matches_original_exact_without_changing_criteria():
    tok=ByteTokenizer();model=TACTransformer(ModelConfig(context=32,width=32,layers=1,heads=2));model.text_tokenizer=tok
    data=ConversationData({'train':[], 'validation':[row('Oi','Olá')]},tok)
    gate=SimpleNamespace(check=lambda:None)
    original=ConversationEngine(None,None,gate).evaluate(model,data,'cpu',generation=True)
    measured=DevelopmentEvaluator(None,None,gate).measured(model,data,'cpu','validation')
    assert measured['loss']==original['loss']
    assert measured['exact_matches']==original['exact_matches']
    assert measured['outputs'][0]['actual']==original['outputs'][0]['actual']
    assert measured['outputs'][0]['semantic_equivalence'] is None


def test_arms_keep_other_pool_schedules_fixed():
    tok,replay,scenes=replay_fixture();objective=ResponseObjective(replay,scenes,tok)
    rows=[]
    for pool in ('legacy','reviewed','imported'):
        rows.extend(row('Meu nome é Lia '+str(i),'Ok.',skill='name',group=str(i),mixture=pool) for i in range(100))
    data=ConversationData({'train':rows},tok)
    old=rows[:100]
    base,meta=arm_plan(data,objective,old,42,2,'control')
    alt,other=arm_plan(data,objective,old,42,2,'B_balanced')
    assert meta['xy_hashes']['reviewed']==other['xy_hashes']['reviewed']
    assert meta['xy_hashes']['imported']!=other['xy_hashes']['imported']
    same,again=arm_plan(data,objective,old,42,2,'control')
    assert meta==again
