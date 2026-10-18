"""TAC v2: inicialização própria, linguagem→SFT, checkpoints e LearningGate."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ..config import atomic_json
from ..db import now
from ..learning import Cancelled, LearningDisabled
from ..diagnostics import check_resources
from .model import ModelConfig,TACTransformer
from .bpe import LocalBPETokenizer
from .v2_data import load_jsonl,prompt_text

GENERATION='TAC-v2'


def model_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def v2_config():
    return ModelConfig(vocab_size=1024,context=1024,width=384,heads=6,layers=5,ff_width=1728,tie_embeddings=True,rope=True)


class IndexDeck:
    """Passagem sem reposição, incluindo o fim do deck, com estado exato."""
    def __init__(self,n,seed,state=None):
        self.n=n;self.rng=torch.Generator().manual_seed(seed);self.order=[];self.position=0;self.epochs_started=0
        if state:
            self.order=state['order'];self.position=state['position'];self.epochs_started=state['epochs_started'];self.rng.set_state(state['rng'])
    def take(self,size):
        result=[]
        while len(result)<size:
            if self.position>=len(self.order):
                self.order=torch.randperm(self.n,generator=self.rng).tolist();self.position=0;self.epochs_started+=1
            count=min(size-len(result),self.n-self.position)
            result.extend(self.order[self.position:self.position+count]);self.position+=count
        return result
    def state(self):return {'n':self.n,'order':self.order,'position':self.position,'epochs_started':self.epochs_started,'rng':self.rng.get_state()}


class V2Data:
    def __init__(self,base,device='cpu'):
        self.base=Path(base);self.device=device
        self.meta=json.loads((self.base/'encoded_ready.json').read_text(encoding='utf-8'))
        self.tokenizer=LocalBPETokenizer.from_state(json.loads((self.base/'tokenizer.json').read_text(encoding='utf-8')))
        self.language={s:np.memmap(self.base/f'language_{s}.u16',dtype='<u2',mode='r') for s in ('train','validation')}
        self.rows={s:list(load_jsonl(self.base/f'instruction_{s}.jsonl')) for s in ('train','validation')}
    def language_batch(self,indices,split='train',context=512):
        data=self.language[split]
        sequences=np.stack([data[i*context:i*context+context+1] for i in indices]).astype(np.int64)
        full=torch.from_numpy(sequences).to(self.device)
        return full[:,:-1].contiguous(),full[:,1:].contiguous()
    def instruction_batch(self,indices,split='train'):
        rows=[self.rows[split][i] for i in indices];length=max(len(r['sequence'])-1 for r in rows)
        x=torch.zeros(len(rows),length,dtype=torch.long);y=torch.zeros_like(x)
        for i,r in enumerate(rows):
            seq=torch.tensor(r['sequence'],dtype=torch.long);n=len(seq)-1
            x[i,:n]=seq[:-1];y[i,:n]=seq[1:];y[i,:r['prefix_length']-1]=0
        return x.to(self.device),y.to(self.device)


class V2Trainer:
    def __init__(self,config,db,gate,*,pretrain_steps=12000,sft_epochs=3,device=None,model_cfg=None,
                 micro_batch=4,accumulation=2,language_context=512,checkpoint_every=500,session_seconds=300):
        self.config,self.db,self.gate=config,db,gate
        self.base=config.data/'tac_v2';self.base.mkdir(exist_ok=True)
        self.progress_path=self.base/'progress.json';self.folder=config.data/'checkpoints'
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cfg=model_cfg or v2_config();self.pretrain_steps=pretrain_steps;self.sft_epochs=sft_epochs
        self.micro_batch=micro_batch;self.accumulation=accumulation;self.language_context=language_context
        self.checkpoint_every=checkpoint_every;self.session_seconds=session_seconds
        self.phase='language';self.step=0;self.global_step=0;self.run_id=uuid.uuid4().hex
        self.sft_presentations=0;self.optimizer=None;self.scaler=None;self.initial_hash=None
        self.language_tokens_seen=0;self.supervised_tokens_seen=0;self.precision_skips=0
        self.last_checkpoint=None;self.stop_requested=False;self.session_started=0
        torch.set_num_threads(config.get('cpu_threads'))
        if self.device.type=='cuda':
            torch.cuda.set_per_process_memory_fraction(min(.85,config.get('vram_mb')*2**20/torch.cuda.get_device_properties(0).total_memory))
            self.amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:self.amp_dtype=torch.float32
        self.data=V2Data(self.base,self.device)
        assert self.data.meta['context']<=self.cfg.context
        self.language_deck=IndexDeck((len(self.data.language['train'])-1)//self.language_context,2026101703)
        self.sft_deck=IndexDeck(len(self.data.rows['train']),2026101704)
        self.total_sft_presentations=len(self.data.rows['train'])*sft_epochs
        sft_batches=math.ceil(self.total_sft_presentations/(micro_batch*accumulation))
        self.sft_target_steps=sft_batches+max(0,(sft_batches-1)//9)
        self.model=None

    def new_optimizer(self):
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.learning_rate(),betas=(.9,.95),
                                         weight_decay=.1 if self.phase=='language' else .01)
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.device.type=='cuda' and self.amp_dtype==torch.float16)

    def learning_rate(self):return 3e-4 if self.phase=='language' else 1e-4

    def initialize(self):
        if self.progress_path.exists():raise ValueError('A TAC v2 já foi inicializada: use retomar, sem voltar ao zero.')
        torch.manual_seed(20261017)
        if self.device.type=='cuda':torch.cuda.manual_seed_all(20261017)
        self.model=TACTransformer(self.cfg).to(self.device);self.model.text_tokenizer=self.data.tokenizer
        self.initial_hash=model_hash(self.model);self.new_optimizer()
        params=self.parameters()
        with self.gate.writing():
            self.gate.check()
            self.db.execute('INSERT INTO runs(id,kind,status,started,config,dataset_hash,detail) VALUES(?,?,?,?,?,?,?)',
                (self.run_id,'conversation','treinando',now(),json.dumps(params),self.data.meta['dataset_hash'],json.dumps({'generation':GENERATION,'parameters':self.model.parameter_count,'stage':'initialization'})))
        self.save_checkpoint('inicializado',validation={})

    def parameters(self):
        return {'generation':GENERATION,'pretrain_steps':self.pretrain_steps,'sft_epochs':self.sft_epochs,
                'micro_batch':self.micro_batch,'accumulation':self.accumulation,'language_context':self.language_context,
                'model_config':asdict(self.cfg),'phase':self.phase,'precision':str(self.amp_dtype),
                'seed':20261017,'checkpoint_every':self.checkpoint_every,'session_seconds':self.session_seconds,
                'checkpoint_initialization':'random_from_scratch_no_parent','old_exercises_used':False}

    def state(self):
        return {'phase':self.phase,'step':self.step,'global_step':self.global_step,'sft_presentations':self.sft_presentations,
                'language_tokens_seen':self.language_tokens_seen,'supervised_tokens_seen':self.supervised_tokens_seen,
                'precision_skips':self.precision_skips,'language_deck':self.language_deck.state(),
                'sft_deck':self.sft_deck.state(),'initial_hash':self.initial_hash,'parameters':self.parameters()}

    def save_checkpoint(self,status,validation=None):
        ident=uuid.uuid4().hex;path=self.folder/(ident+'.pt');temp=path.with_suffix('.v2-writing')
        # Este caminho grava exclusivamente checkpoints NOVOS e nunca copia um antigo.
        cp={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','generation':GENERATION,
            'tokenizer':self.data.tokenizer.version,'tokenizer_state':self.data.tokenizer.state(),
            'model_config':asdict(self.cfg),'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            'optimizer':self.optimizer.state_dict(),'scaler':self.scaler.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            'dataset_hash':self.data.meta['dataset_hash'],'step':self.global_step,'run_id':self.run_id,
            'training_config':self.parameters(),'v2_state':self.state(),'validation':validation or {}}
        with self.gate.lock:
            # Uma interrupção por desligamento pode salvar os pesos NOVOS já confirmados;
            # não executa optimizer.step e não reativa a permissão.
            torch.save(cp,temp);os.replace(temp,path)
            self.db.execute('INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)',
                (ident,self.run_id,path.name,self.global_step,(validation or {}).get('loss'),status,now()))
            atomic_json(self.progress_path,{'generation':GENERATION,'checkpoint':ident,'run_id':self.run_id,'phase':self.phase,
                'phase_step':self.step,'global_step':self.global_step,'sft_presentations':self.sft_presentations,
                'language_tokens_seen':self.language_tokens_seen,'supervised_tokens_seen':self.supervised_tokens_seen,
                'status':status,'updated':now(),'initial_weight_hash':self.initial_hash,'dataset_hash':self.data.meta['dataset_hash']})
        self.last_checkpoint=ident;return ident

    def resume(self,checkpoint=None):
        progress=json.loads(self.progress_path.read_text(encoding='utf-8'))
        ident=checkpoint or progress['checkpoint'];rows=self.db.query('SELECT filename FROM checkpoints WHERE id=?',(ident,))
        if not rows:raise ValueError('Checkpoint novo não está registrado')
        path=(self.folder/rows[0]['filename']).resolve()
        if not path.is_relative_to(self.folder.resolve()):raise ValueError('Checkpoint fora da pasta')
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp.get('generation')!=GENERATION:raise ValueError('Retomada permite somente checkpoints da TAC v2')
        if cp['dataset_hash']!=self.data.meta['dataset_hash']:raise ValueError('O corpus mudou desde o checkpoint')
        st=cp['v2_state'];params=st['parameters']
        for name in ('pretrain_steps','sft_epochs','micro_batch','accumulation','language_context'):
            if getattr(self,name)!=params[name]:raise ValueError('Retomada exige orçamento/configuração iguais: '+name)
        self.cfg=ModelConfig(**cp['model_config']);self.model=TACTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(cp['model']);self.model.text_tokenizer=self.data.tokenizer
        self.phase=st['phase'];self.step=st['step'];self.global_step=st['global_step'];self.run_id=cp['run_id']
        for name in ('sft_presentations','language_tokens_seen','supervised_tokens_seen','precision_skips','initial_hash'):setattr(self,name,st[name])
        self.language_deck=IndexDeck(self.language_deck.n,0,st['language_deck']);self.sft_deck=IndexDeck(self.sft_deck.n,0,st['sft_deck'])
        self.new_optimizer();self.optimizer.load_state_dict(cp['optimizer']);self.scaler.load_state_dict(cp['scaler'])
        torch.set_rng_state(cp['torch_rng'])
        if self.device.type=='cuda' and cp['cuda_rng']:torch.cuda.set_rng_state_all(cp['cuda_rng'])
        self.last_checkpoint=ident

    def check_permission(self):
        # Permite desligamento pela interface em outro processo, além do gate local.
        disk=json.loads(self.config.path.read_text(encoding='utf-8'))
        if not disk['learning_enabled']:
            self.config.values['learning_enabled']=False
            raise Cancelled('Aprendizado desligado na interface; checkpoint novo recuperável')
        if (self.base/'PARAR_TREINO').exists() or self.stop_requested:raise Cancelled('Interrupção solicitada da TAC v2')
        self.gate.wait_if_paused()

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval();results={}
        for kind in ('language','instruction'):
            total_loss=0.;count=0
            if kind=='language':
                n=(len(self.data.language['validation'])-1)//self.language_context
                rng=torch.Generator().manual_seed(2026101705)
                indices=torch.randint(n,(min(32,n),),generator=rng).tolist()
            else:indices=list(range(min(64,len(self.data.rows['validation']))))
            for start in range(0,len(indices),self.micro_batch):
                ids=indices[start:start+self.micro_batch]
                x,y=self.data.language_batch(ids,'validation',self.language_context) if kind=='language' else self.data.instruction_batch(ids,'validation')
                with torch.autocast(device_type=self.device.type,dtype=self.amp_dtype,enabled=self.device.type=='cuda'):
                    _,loss=self.model(x,y)
                tokens=int((y!=0).sum());total_loss+=float(loss)*tokens;count+=tokens
            results[kind]={'loss':total_loss/max(1,count),'supervised_tokens':count,'examples_or_windows':len(indices)}
        self.model.train();return results

    def one_update(self):
        self.check_permission()
        language=self.phase=='language' or self.step%10==9
        batch_size=self.micro_batch*self.accumulation
        if language:
            indices=self.language_deck.take(batch_size);batches=[self.data.language_batch(indices[i:i+self.micro_batch],context=self.language_context) for i in range(0,batch_size,self.micro_batch)]
        else:
            count=min(batch_size,self.total_sft_presentations-self.sft_presentations)
            indices=self.sft_deck.take(count)
            # Ordena apenas o lote já escolhido; não altera frequência/seleção de exemplos.
            indices.sort(key=lambda i:len(self.data.rows['train'][i]['sequence']))
            batches=[self.data.instruction_batch(indices[i:i+self.micro_batch]) for i in range(0,count,self.micro_batch)]
        token_counts=[int((y!=0).sum()) for _,y in batches];denominator=sum(token_counts)
        target=self.pretrain_steps if self.phase=='language' else self.sft_target_steps
        warmup=200 if self.phase=='language' else 50
        lr=self.learning_rate()*min(1.,(self.step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(1.,(self.step+1)/max(1,target)))))
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True);self.model.train();loss_value=0.
        for (x,y),n in zip(batches,token_counts):
            self.check_permission()
            with torch.autocast(device_type=self.device.type,dtype=self.amp_dtype,enabled=self.device.type=='cuda'):
                _,loss=self.model(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Perda não finita na TAC v2')
            weighted=loss*(n/denominator);loss_value+=float(weighted.detach())
            self.scaler.scale(weighted).backward()
        self.scaler.unscale_(self.optimizer)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=False)
        if not torch.isfinite(norm):
            self.precision_skips+=1;self.scaler.update();self.optimizer.zero_grad(set_to_none=True)
            if not self.scaler.is_enabled():raise FloatingPointError('Gradiente BF16/FP32 não finito: interrompido com segurança')
            return None
        self.check_permission()
        with self.gate.writing():
            self.gate.check();self.scaler.step(self.optimizer);self.scaler.update()
            self.step+=1;self.global_step+=1
            if language:self.language_tokens_seen+=denominator
            else:self.sft_presentations+=len(indices);self.supervised_tokens_seen+=denominator
        with (self.base/'exposure.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':self.global_step,'phase':self.phase,'kind':'language' if language else 'instruction',
                'indices':indices,'supervised_tokens':denominator,'loss':loss_value,'gradient_norm_before_clip':float(norm),'learning_rate':lr})+'\n')
        return loss_value

    def finish_phase(self,validation):
        if self.phase=='language':
            self.save_checkpoint('linguagem_concluida',validation['language'])
            atomic_json(self.base/'language_result.json',{'generation':GENERATION,'checkpoint':self.last_checkpoint,
                'steps':self.step,'global_steps':self.global_step,'tokens':self.language_tokens_seen,'validation':validation,
                'initial_weight_hash':self.initial_hash,'weight_hash':model_hash(self.model)})
            self.phase='instruction';self.step=0;self.new_optimizer();self.save_checkpoint('inicio_instrucoes',validation['instruction'])
            return
        self.phase='complete';ident=self.save_checkpoint('experimental_v2',validation['instruction'])
        result={'generation':GENERATION,'status':'experimental_disponivel','parameters':self.model.parameter_count,
                'checkpoint':ident,'run_id':self.run_id,'steps':self.global_step,'pretraining_steps':self.pretrain_steps,
                'instruction_phase_updates':self.step,'instruction_presentations':self.sft_presentations,
                'instruction_unique_examples_eligible':len(self.data.rows['train']),
                'language_supervised_tokens':self.language_tokens_seen,'instruction_supervised_tokens':self.supervised_tokens_seen,
                'precision_skips':self.precision_skips,'validation_losses':validation,'initial_weight_hash':self.initial_hash,
                'final_weight_hash':model_hash(self.model),'device':str(self.device),'precision':str(self.amp_dtype),
                'old_weights_loaded':False,'external_models_used':False,'reserved_tests_used':False,
                'availability_is_not_general_intelligence_approval':True}
        with self.gate.writing():
            self.gate.check()
            self.db.execute("INSERT INTO meta(key,value) VALUES('active_conversation',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(ident,))
            self.db.execute('UPDATE runs SET status=?,finished=?,final=?,detail=? WHERE id=?',
                ('experimental_v2',now(),validation['instruction']['loss'],json.dumps(result,ensure_ascii=False),self.run_id))
            self.db.event('tac_v2_disponivel',ident)
        atomic_json(self.base/'training_result.json',result)

    def run(self,progress=print):
        if self.model is None:
            if self.progress_path.exists():self.resume()
            else:self.initialize()
        if self.phase=='complete':
            if (self.base/'training_result.json').exists():return json.loads((self.base/'training_result.json').read_text(encoding='utf-8'))
            self.phase='instruction'  # Publicação interrompida: concluir sem repetir atualizações.
        baselinepath=self.base/'initial_validation.json'
        if self.global_step==0 and not baselinepath.exists():atomic_json(baselinepath,self.evaluate())
        self.session_started=time.monotonic();last_log=self.session_started
        try:
            while self.phase!='complete':
                if self.phase=='language' and self.step>=self.pretrain_steps or self.phase=='instruction' and self.sft_presentations>=self.total_sft_presentations:
                    validation=self.evaluate();self.finish_phase(validation)
                    progress(f'ETAPA: {self.phase}; perdas {validation}');continue
                before_step=self.global_step
                before_language=self.language_deck.state();before_sft=self.sft_deck.state()
                try:
                    loss=self.one_update()
                except BaseException:
                    if self.global_step==before_step:
                        self.language_deck=IndexDeck(self.language_deck.n,0,before_language)
                        self.sft_deck=IndexDeck(self.sft_deck.n,0,before_sft)
                    raise
                if loss is None:continue
                if self.global_step%50==0 or self.global_step==1:
                    check_resources(self.config)
                    elapsed=max(.001,time.monotonic()-last_log);last_log=time.monotonic()
                    progress(f'V2 {self.phase}: passo {self.step}, global {self.global_step}; loss {loss:.4f}; tokens linguagem {self.language_tokens_seen}; alvos SFT {self.supervised_tokens_seen}; lote {elapsed:.2f}s/50')
                    atomic_json(self.base/'live.json',{'phase':self.phase,'step':self.step,'global_step':self.global_step,
                        'loss':loss,'language_tokens_seen':self.language_tokens_seen,'supervised_tokens_seen':self.supervised_tokens_seen,
                        'updated':now(),'device':str(self.device),'process_session_seconds':time.monotonic()-self.session_started})
                if self.global_step%self.checkpoint_every==0:
                    val=self.evaluate();kind='language' if self.phase=='language' else 'instruction'
                    self.save_checkpoint('recuperavel_v2',val[kind]);progress(f'CHECKPOINT V2 {self.last_checkpoint}; {val}')
                    self.db.execute('INSERT INTO metrics(run_id,step,epoch,train_loss,val_loss,created) VALUES(?,?,?,?,?,?)',
                        (self.run_id,self.global_step,0,loss,val[kind]['loss'],now()))
                if time.monotonic()-self.session_started>=self.session_seconds:
                    self.save_checkpoint('sessao_v2');progress(f'Sessão recuperável salva em {self.global_step}; continuando a mesma execução.')
                    self.session_started=time.monotonic()
        except (Cancelled,LearningDisabled,KeyboardInterrupt) as exc:
            ident=self.save_checkpoint('interrompido_v2')
            self.db.execute('UPDATE runs SET status=? WHERE id=?',('interrompido',self.run_id))
            return {'status':'interrompido','generation':GENERATION,'checkpoint':ident,'steps':self.global_step,'reason':str(exc)}
        except Exception:
            if self.model is not None:self.save_checkpoint('erro_v2')
            self.db.execute('UPDATE runs SET status=? WHERE id=?',('erro',self.run_id))
            raise
        return json.loads((self.base/'training_result.json').read_text(encoding='utf-8'))
