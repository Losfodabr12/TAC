import json
import hashlib
import numpy as np
import pytest
import torch
from tac.learning import Cancelled, LearningDisabled
from tac.neural.model import ModelConfig, count_parameters
from tac.neural.v2_training import model_hash
from tac.neural.v3 import V3Tokenizer, v3_config
from tac.neural.v3_training import V3Trainer


def fixture_v3(core):
    base=core.config.data/'tac_v3';base.mkdir()
    tokenizer=V3Tokenizer.fit(['Uma história de linguagem em português, nomes e números. '*30],vocab_size=300)
    (base/'tokenizer.json').write_text(json.dumps(tokenizer.state()),encoding='utf-8')
    (base/'encoded_ready.json').write_text(json.dumps({'dataset_hash':'isolated-test'}),encoding='utf-8')
    for split in ('train','validation'):
        np.array(tokenizer.encode('Um texto de linguagem. '*100),dtype='<u2').tofile(base/f'language_{split}.u16')
        with (base/f'instruction_{split}.jsonl').open('w',encoding='utf-8') as f:
            for i in range(4):
                prefix=[1]+tokenizer.encode('Pergunta?',special=False)
                seq=prefix+tokenizer.encode(' Resposta.',special=False)+[2]
                f.write(json.dumps({'sequence':seq,'prefix_length':len(prefix)})+'\n')
    files={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
           for p in base.iterdir() if p.is_file() and p.name!='encoded_ready.json'}
    meta={'dataset_hash':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'files':files}
    (base/'training_ready.json').write_text(json.dumps(meta),encoding='utf-8')
    core.gate.set_enabled(True);core.gate.begin()
    cfg=ModelConfig(vocab_size=tokenizer.vocab_size,context=32,width=32,heads=2,layers=1,tie_embeddings=True,rope=True)
    return cfg,tokenizer


def trainer(core,cfg):
    return V3Trainer(core.config,core.db,core.gate,token_budget=128,sft_epochs=1,
                     micro_batch=1,accumulation=2,device='cpu',model_cfg=cfg,checkpoint_every=4)


def test_v3_architecture_and_local_byte_roundtrip():
    assert count_parameters(v3_config())==29_414_400
    tok=V3Tokenizer.fit(['Olá, língua portuguesa! '*20],vocab_size=300)
    text='João e Néreida: café, 37 + 4, 🐺.'
    assert tok.decode(tok.encode(text))==text
    restored=V3Tokenizer.from_state(tok.state())
    assert restored.encode(text)==tok.encode(text)
    assert not set(tok.encode('<bos> <eos> <pad>',special=False)).intersection({0,1,2})


def test_v3_gate_cancellation_rolls_back_sampling(core):
    cfg,_=fixture_v3(core);t=trainer(core,cfg);t.initialize();original=model_hash(t.model)
    hook=next(t.model.parameters()).register_hook(lambda grad:(core.gate.set_enabled(False),grad)[1])
    with pytest.raises((Cancelled,LearningDisabled)):t.update()
    hook.remove()
    assert t.global_step==0 and model_hash(t.model)==original
    assert t.language_deck.position==0 and t.language_deck.order==[]


def test_v3_resume_matches_uninterrupted_optimizer_and_never_activates(core):
    cfg,_=fixture_v3(core)
    core.db.execute("INSERT INTO meta(key,value) VALUES('active_conversation','v2-preserved')")
    a=trainer(core,cfg);a.initialize();a.update();a.save('test_recovery');a.update();expected=model_hash(a.model)
    b=trainer(core,cfg);b.resume();b.update()
    assert model_hash(b.model)==expected and b.global_step==2
    assert core.active_conversation_id()=='v2-preserved'
    assert all(r['filename'].startswith('v3/') for r in core.db.query('SELECT filename FROM checkpoints'))


def test_v3_response_mask_includes_eos_and_ignores_prompt(core):
    cfg,tok=fixture_v3(core);t=trainer(core,cfg)
    x,y=t.data.instruction_batch([0,1])
    n=t.data.rows['train'][0]['prefix_length']
    assert (y[:,:n-1]==0).all()
    assert (y[:,-1]==tok.eos).all() and (y[:,n-1:]!=0).all()


def test_v3_completed_run_never_reinitializes(core,monkeypatch):
    cfg,_=fixture_v3(core);t=trainer(core,cfg)
    (t.base/'training_result.json').write_text(json.dumps({'status':'experimental_nao_promovido'}),encoding='utf-8')
    monkeypatch.setattr(t,'initialize',lambda:pytest.fail('completed run restarted'))
    assert t.run()['status']=='experimental_nao_promovido'


def test_interface_recovery_preserves_live_external_training(core):
    from PySide6.QtCore import QLockFile
    core.db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash) VALUES('v3-live','conversation','treinando','now','{}','test')")
    lock=QLockFile(str(core.config.data/'training.lock'))
    assert lock.tryLock(0)
    try:
        core.recover_interrupted()
        assert core.db.query("SELECT status FROM runs WHERE id='v3-live'")[0]['status']=='treinando'
    finally:lock.unlock()
    core.recover_interrupted()
    assert core.db.query("SELECT status FROM runs WHERE id='v3-live'")[0]['status']=='interrompido'


def test_v3_session_boundary_resumes_same_budget_without_repeated_steps(core):
    cfg,_=fixture_v3(core);a=trainer(core,cfg);a.session_updates=1
    result=a.run(lambda _:None)
    assert result['status']=='sessao_salva' and a.global_step==1
    b=trainer(core,cfg);finished=b.run(lambda _:None)
    assert finished['status']=='experimental_nao_promovido' and b.global_step==4
    assert b.run_id==a.run_id and b.language_tokens==128 and b.presentations==4
    records=[]
    for p in b.base.glob('exposure_*.jsonl'):
        records.extend(json.loads(line) for line in p.read_text().splitlines())
    assert sorted(r['step'] for r in records)==[1,2,3,4]


def revised_fixture(base):
    parent=json.loads((base/'training_ready.json').read_text())
    child=json.loads(json.dumps(parent))
    for split in ('train','validation'):
        path=base/f'instruction_reviewed_{split}.jsonl'
        lines=(base/f'instruction_{split}.jsonl').read_text().splitlines()[:3]
        path.write_text('\n'.join(lines)+'\n')
        child['files'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    child.update(parent_dataset_hash=parent['dataset_hash'],transition_permitted_only_before_sft=True,
                 revision_reason='isolated test correction to unused SFT',instruction_filename='instruction_reviewed_{split}.jsonl')
    child['dataset_hash']=hashlib.sha256(json.dumps(child['files'],sort_keys=True).encode()).hexdigest()
    (base/'sft_reviewed_ready.json').write_text(json.dumps(child))


def test_explicit_unused_sft_revision_preserves_pretrained_weights_and_steps(core):
    cfg,_=fixture_v3(core);a=trainer(core,cfg);a.initialize();a.update();a.save('before-unused-sft-review')
    before=model_hash(a.model);revised_fixture(a.base)
    b=trainer(core,cfg);b.resume()
    assert b.global_step==1 and model_hash(b.model)==before
    assert b.sft_deck.n==3 and not b.sft_deck.order and b.total_sft==3
    assert b.dataset_transition['old_sft_presentations']==0


def test_data_revision_is_rejected_after_supervised_updates(core):
    cfg,_=fixture_v3(core);a=trainer(core,cfg);a.initialize()
    a.phase='instruction';a.new_optimizer();a.update();a.save('sft-already-used')
    revised_fixture(a.base);b=trainer(core,cfg)
    with pytest.raises(ValueError,match='revisão explícita'):b.resume()
