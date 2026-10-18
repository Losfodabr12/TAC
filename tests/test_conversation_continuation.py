import json
import numpy as np
import pytest
import torch
from tac.learning import Cancelled,LearningDisabled
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation_training import ConversationContinuation,response_loss,score_output
from tac.neural.model import ModelConfig,TACTransformer
from tac.neural.v2_data import prompt_text
from tac.neural.v2_training import model_hash


def fixture(core):
    base=core.config.data/'continuation';base.mkdir();old=core.config.data/'tac_v2';old.mkdir()
    tok=LocalBPETokenizer();cfg=ModelConfig(context=64,width=32,layers=1,heads=2,tie_embeddings=True,rope=True)
    torch.manual_seed(731);model=TACTransformer(cfg)
    for s in ('train','validation'):
        np.array([1,*tok.encode('abc '*200,special=False),2],dtype='<u2').tofile(old/('language_'+s+'.u16'))
    for s in ('train','validation','retention'):
        with (base/(s+'.jsonl')).open('w',encoding='utf-8') as f:
            for i in range(4):
                messages=[['user',f'{i} + 1?']];prefix=[1,*tok.encode(prompt_text(messages),special=False)]
                answer=tok.encode(' '+str(i+1),special=False)+[2]
                f.write(json.dumps({'messages':messages,'answer':str(i+1),'sequence':prefix+answer,
                    'prefix_length':len(prefix),'skill':'arithmetic','source':'fixture','family':s,'supervised_tokens':len(answer)})+'\n')
    (base/'ready.json').write_text(json.dumps({'dataset_hash':'fixture-only'}),encoding='utf-8')
    cp={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','generation':'TAC-v2','model_config':vars(cfg),
        'tokenizer_state':tok.state(),'model':model.state_dict(),'step':15}
    torch.save(cp,core.config.data/'checkpoints/ref.pt')
    core.db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash) VALUES('fixture_ref','conversation','reference','now','{}','fixture')")
    core.db.execute("INSERT INTO checkpoints(id,run_id,filename,step,status,created) VALUES('ref','fixture_ref','ref.pt',15,'reference','now')")
    core.db.execute("INSERT INTO meta(key,value) VALUES('active_conversation','ref')")
    core.gate.set_enabled(True);core.gate.begin()
    return base,model_hash(model)


def make(core,base):
    return ConversationContinuation(core.config,core.db,core.gate,base=base,out=base/'reports',device='cpu',max_steps=4,batch=2,micro=1,every=4,checkpoint_every=2)


def test_response_balancing_and_prompt_mask():
    logits=torch.randn(2,4,259);targets=torch.tensor([[0,35,2,0],[35,36,37,2]])
    loss=response_loss(logits,targets)
    changed=logits.clone();changed[0,0]=1000
    assert torch.allclose(response_loss(changed,targets),loss)
    a=response_loss(logits[:1],targets[:1]);b=response_loss(logits[1:],targets[1:])
    assert torch.allclose(loss,(a+b)/2)


def test_disable_during_backward_preserves_weights_and_sampling(core):
    base,initial=fixture(core);t=make(core,base);t.initialize()
    assert model_hash(t.model)==initial
    hook=next(t.model.parameters()).register_hook(lambda grad:(core.gate.set_enabled(False),grad)[1])
    with pytest.raises((Cancelled,LearningDisabled)):t.update()
    hook.remove()
    assert t.step==0 and model_hash(t.model)==initial and t.deck.position==0
    assert not t.deck.order


def test_resume_preserves_optimizer_sampling_and_original_reference(core):
    base,initial=fixture(core);a=make(core,base);a.initialize()
    original=(core.config.data/'checkpoints/ref.pt').read_bytes()
    a.update();a.checkpoint();a.update();expected=model_hash(a.model)
    b=make(core,base);b.resume();b.update()
    assert model_hash(b.model)==expected and b.step==2
    assert (core.config.data/'checkpoints/ref.pt').read_bytes()==original
    assert expected!=initial


def test_scoring_does_not_accept_arithmetic_hallucination_ending_in_four():
    row={'skill':'arithmetic','answer':'4','verification':{'a':2,'b':2,'op':'+'}}
    assert score_output(row,'O resultado é 4.')['structured_correct']
    assert score_output(row,'2 + 2 = 4')['structured_correct']
    assert not score_output(row,'O 2 é 1 + 4 + 5 = 4.')['structured_correct']


def test_sessions_have_independent_histories_and_clear_resets_transformer_context(core,qtapp,monkeypatch,tmp_path):
    from tac.core import Core
    from tac.ui import Window
    calls=[]
    monkeypatch.setattr(core.conversation,'reply',lambda messages:(calls.append(messages) or {'text':'Aline.','sources':[],'mode':'diálogo neural experimental'}))
    core.chat('Meu nome é Aline.',conversation=True)
    other=Core(tmp_path/'separate_session')
    assert not other.neural_history and core.neural_history
    w=Window(core);w.timer.stop();w.clear_conversation()
    core.chat('Outra pergunta.',conversation=True)
    assert calls[-1]==[['user','Outra pergunta.']]
    w.close()


def test_checkpoint_change_does_not_mix_histories_from_different_models(core,monkeypatch):
    active=['A'];calls=[]
    monkeypatch.setattr(core,'active_conversation_id',lambda:active[0])
    monkeypatch.setattr(core.conversation,'reply',lambda messages:(calls.append(messages) or {'text':'Resposta neural de teste.','sources':[],'mode':'diálogo neural experimental'}))
    core.chat('Sou Aline.',conversation=True)
    core.chat('Qual é meu nome?',conversation=True)
    assert calls[-1][0]==['user','Sou Aline.']
    active[0]='B'
    answer=core.chat('O que é uma árvore?',conversation=True)
    assert calls[-1]==[['user','O que é uma árvore?']]
    assert answer['session_reset_reason']=='checkpoint_changed'


def test_completed_run_cannot_silently_restart_training(core,monkeypatch):
    base,_=fixture(core)
    (base/'result.json').write_text(json.dumps({'status':'concluido_experimental','step':4}),encoding='utf-8')
    t=make(core,base)
    monkeypatch.setattr(t,'initialize',lambda:pytest.fail('Execução concluída não pode inicializar outro treino'))
    assert t.run()['step']==4


def test_second_training_process_is_rejected_before_update(core):
    from PySide6.QtCore import QLockFile
    base,_=fixture(core);lock=QLockFile(str(base/'training.lock'))
    assert lock.tryLock(0)
    try:
        with pytest.raises(RuntimeError,match='outro processo'):make(core,base).run()
    finally:lock.unlock()


def test_experimental_selector_switches_reference_and_clears_session(core,qtapp):
    from tac.ui import Window
    from test_backup_ui import pump
    base,_=fixture(core)
    cp=torch.load(core.config.data/'checkpoints/ref.pt',weights_only=True)
    torch.save(cp,core.config.data/'checkpoints/candidate.pt')
    core.db.execute("INSERT INTO checkpoints(id,run_id,filename,step,status,created) VALUES('candidate','fixture_ref','candidate.pt',16,'experimental','now')")
    deployment=core.config.data/'tac_v2_conversation_20261018';deployment.mkdir()
    (deployment/'deployment.json').write_text(json.dumps({'checkpoint':'candidate','reference':'ref','passed_retention_gate':False}),encoding='utf-8')
    w=Window(core);w.timer.stop();w.chat_mode.setCurrentIndex(2)
    core.neural_history=[['user','Dados da sessão anterior'],['assistant','Trecho anterior']]
    w.conversation_version.setCurrentIndex(0)
    pump(qtapp,lambda:w.job is None)
    assert core.active_conversation_id()=='candidate' and not core.neural_history
    assert 'perdeu desempenho' in w.conversation_version_notice.text()
    w.conversation_version.setCurrentIndex(1)
    pump(qtapp,lambda:w.job is None)
    assert core.active_conversation_id()=='ref'
    assert 'Referência v2 anterior' in w.conversation_version_notice.text()
    w.close()
