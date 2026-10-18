"""Verificações de ingestão e de recuperação; não são pilotos da TAC v2."""
import json
from collections import Counter
import numpy as np
import torch
from tac.neural.v2_data import problem,oasst_examples,NearDedup
from tac.neural.v2_training import V2Trainer,model_hash
from tac.neural.model import ModelConfig
from tac.neural.bpe import LocalBPETokenizer


def test_pt_br_tree_uses_parents_never_siblings():
    def row(ident,parent,role,text,rank=None):
        return dict(message_id=ident,parent_id=parent,role=role,text=text,rank=rank,lang='pt-BR',
                    message_tree_id='root',review_result=True,deleted=False,synthetic=False,labels={})
    rows=[row('u',None,'prompter','Quero organizar uma viagem.'),
          row('a1','u','assistant','Qual cidade você pretende visitar?',0),
          row('bad','u','assistant','Outra resposta de um ramo diferente.',1),
          row('u2','a1','prompter','Pretendo visitar Recife em agosto.'),
          row('a2','u2','assistant','Podemos começar pela duração da viagem e pelo orçamento.',0),
          row('missing','absent','assistant','Esta resposta não tem seu histórico.',0)]
    c=Counter();examples=list(oasst_examples(rows,c))
    second=next(e for e in examples if e['source_id']=='a2')
    assert second['ancestor_ids']==['u','a1','u2']
    assert 'bad' not in second['ancestor_ids']
    assert c['oasst_missing_ancestor']==1
    assert all(e['group']=='oasst:root' for e in examples)


def test_privacy_and_duplicate_filters():
    assert problem('Entre em contato comigo em pessoa@example.com.')=='email'
    assert problem('Meu CPF: 123.456.789-09.')=='cpf_cnpj'
    assert problem('Texto quebrado \ufffd.')=='broken_unicode'
    assert problem('Explique por que a água evapora mesmo sem ferver.') is None
    d=NearDedup();text='A educação científica permite compreender fenômenos naturais e formular perguntas sobre suas causas. '*4
    assert not d.duplicate(text)
    assert d.duplicate('  '+text.upper()+' ')


def test_neural_conversation_never_uses_automatic_web_or_library(core,monkeypatch):
    core.gate.set_enabled(True);core.config.update(research_auto=True)
    def forbidden(*a,**kw):raise AssertionError('Modo neural chamou uma fonte externa/documental')
    from tac.research import Researcher
    monkeypatch.setattr(Researcher,'lookup',forbidden)
    monkeypatch.setattr(core.memory,'answer',forbidden)
    monkeypatch.setattr(core.conversation,'reply',lambda messages:{'mode':'diálogo neural experimental',
        'text':'Saída controlada do teste de roteamento.','sources':[],'generation':'TAC-v2'})
    answer=core.chat('O que é fotossíntese?',conversation=True)
    assert answer['mode']=='diálogo neural experimental' and answer['generation']=='TAC-v2'


def fixture_data(core):
    b=core.config.data/'tac_v2';b.mkdir()
    tok=LocalBPETokenizer()
    (b/'tokenizer.json').write_text(json.dumps(tok.state()),encoding='utf-8')
    for split in ('train','validation'):
        values=np.array(tok.encode('A ciência investiga perguntas. Uma conversa considera o contexto. '*12),dtype='<u2')
        values.tofile(b/f'language_{split}.u16')
        with (b/f'instruction_{split}.jsonl').open('w',encoding='utf-8') as f:
            for i in range(4):
                f.write(json.dumps({'sequence':[1,30,31,32,33+i,34,2],'prefix_length':4,
                                   'source':'test_fixture','source_id':str(i),'messages':[['user','Pergunta curta']],
                                   'answer':'Resposta curta'})+'\n')
    (b/'encoded_ready.json').write_text(json.dumps({'context':32,'dataset_hash':'fixture-new-v2'}),encoding='utf-8')
    core.gate.set_enabled(True)


def trainer(core):
    return V2Trainer(core.config,core.db,core.gate,pretrain_steps=3,sft_epochs=1,device='cpu',
        model_cfg=ModelConfig(vocab_size=259,context=32,width=32,heads=2,layers=1,ff_width=64,tie_embeddings=True,rope=True),
        micro_batch=2,accumulation=2,language_context=16,checkpoint_every=100,session_seconds=300)


def test_v2_recovery_does_not_start_legacy_automatic_archives(core):
    assert core.backup_due()
    fixture_data(core);value=trainer(core);value.initialize()
    assert not core.backup_due()


def test_disable_during_backward_and_exact_resume(core,tmp_path):
    fixture_data(core);t=trainer(core);t.initialize();initial=model_hash(t.model)
    called=[]
    def disable(grad):
        if not called:called.append(True);core.gate.set_enabled(False)
        return grad
    handle=next(t.model.parameters()).register_hook(disable)
    stopped=t.run(lambda _:None);handle.remove()
    assert stopped['status']=='interrompido' and stopped['steps']==0
    assert model_hash(t.model)==initial
    assert t.language_deck.position==0 and not t.language_deck.order
    core.gate.set_enabled(True);core.gate.begin();resumed=trainer(core);result=resumed.run(lambda _:None)
    assert result['steps']==4 and result['initial_weight_hash']==initial
    assert result['final_weight_hash']!=initial
    from tac.core import Core
    other=Core(tmp_path/'comparison');other.config.update(cpu_threads=2,ram_mb=4096,reserve_ram_mb=256)
    fixture_data(other);uninterrupted=trainer(other);complete=uninterrupted.run(lambda _:None)
    assert complete['final_weight_hash']==result['final_weight_hash']
    cp=torch.load(core.config.data/'checkpoints'/(result['checkpoint']+'.pt'),map_location='cpu',weights_only=True)
    assert cp['generation']=='TAC-v2' and cp['v2_state']['phase']=='complete'
    assert core.active_conversation_id()==result['checkpoint']
