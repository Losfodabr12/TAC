"""One bounded v3 run, random initialization, language then supervised responses."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict

import numpy as np
import torch
from PySide6.QtCore import QLockFile

from ..config import atomic_json
from ..db import now
from ..diagnostics import check_resources
from ..learning import Cancelled, LearningDisabled
from .model import TACTransformer, ModelConfig
from .v2_data import load_jsonl
from .v2_training import IndexDeck, V2Data, model_hash
from .v3 import V3Tokenizer, v3_config


class V3Data:
    language_batch = V2Data.language_batch
    instruction_batch = V2Data.instruction_batch

    def __init__(self, base, device):
        self.base, self.device = base, device
        manifest = next(p for p in (base/'sft_reviewed_ready.json',base/'training_ready.json',base/'encoded_ready.json') if p.exists())
        self.meta = json.loads(manifest.read_text(encoding='utf-8'))
        for filename, expected in self.meta.get('files', {}).items():
            path = (base/filename).resolve()
            if path.parent != base.resolve():raise ValueError('Arquivo de corpus fora da pasta v3')
            digest=hashlib.sha256()
            with path.open('rb') as f:
                while chunk:=f.read(2**20):digest.update(chunk)
            if path.stat().st_size!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:
                raise ValueError('Corpus alterado: '+filename)
        self.tokenizer = V3Tokenizer.from_state(json.loads((base/'tokenizer.json').read_text(encoding='utf-8')))
        self.language = {s:np.memmap(base/f'language_{s}.u16', dtype='<u2', mode='r') for s in ('train','validation')}
        pattern = self.meta.get('instruction_filename', 'instruction_{split}.jsonl')
        self.rows = {s:list(load_jsonl(base/pattern.format(split=s))) for s in ('train','validation')}


class V3Trainer:
    def __init__(self, config, db, gate, *, token_budget=120_000_000, sft_epochs=2,
                 micro_batch=2, accumulation=4, device='cuda', model_cfg=None,
                 checkpoint_every=2048):
        self.config,self.db,self.gate=config,db,gate
        self.base=config.data/'tac_v3';self.out=config.root/'reports/tac_v3/rodada_20261018_01'
        if model_cfg is None and not (self.base/'training_ready.json').exists():
            raise ValueError('O corpus revisado da v3 não está pronto: execute a preparação e a curadoria antes do treinamento')
        self.folder=config.data/'checkpoints/v3';self.folder.mkdir(parents=True,exist_ok=True)
        self.out.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(device);self.cfg=model_cfg or v3_config()
        self.micro=micro_batch;self.accumulation=accumulation;self.batch=self.micro*self.accumulation
        self.context=self.cfg.context
        self.token_budget=token_budget;self.pretrain_steps=math.ceil(token_budget/(self.context*self.batch))
        self.sft_epochs=sft_epochs;self.checkpoint_every=checkpoint_every
        self.data=V3Data(self.base,self.device)
        self.total_sft=len(self.data.rows['train'])*sft_epochs
        self.language_deck=IndexDeck((len(self.data.language['train'])-1)//self.context,43171)
        self.sft_deck=IndexDeck(len(self.data.rows['train']),43172)
        if self.language_deck.n*self.context<token_budget or not self.sft_deck.n:
            raise ValueError('Corpus insuficiente para o orçamento sem repetir o pré-treinamento')
        self.phase='language';self.step=self.global_step=self.presentations=self.language_tokens=self.sft_tokens=0
        self.run_id=uuid.uuid4().hex;self.invocation=uuid.uuid4().hex
        self.model=self.optimizer=None;self.last_checkpoint=None;self.stop_requested=False
        self.initial_hash=None;self.started=time.monotonic();self.sft_epoch_saved=0
        self.elapsed_previous=0.
        self.dataset_transition=None
        self.session_updates=None
        torch.set_num_threads(config.get('cpu_threads'))
        self.amp=torch.float32
        if self.device.type=='cuda':
            if not torch.cuda.is_bf16_supported():raise RuntimeError('Esta execução exige CUDA BF16 suportado')
            torch.cuda.set_per_process_memory_fraction(min(.65,config.get('vram_mb')*2**20/torch.cuda.get_device_properties(0).total_memory))
            self.amp=torch.bfloat16

    def params(self):
        return {'generation':'TAC-v3','model_config':asdict(self.cfg),'token_budget':self.token_budget,
                'pretrain_steps':self.pretrain_steps,'sft_epochs':self.sft_epochs,'micro_batch':self.micro,
                'accumulation':self.accumulation,'context':self.context,'seed':4317,'precision':str(self.amp),
                'lr_language':3e-4,'lr_instruction':6e-5,'betas':[.9,.95],'gradient_clip':1.,
                'weight_decay_language':.1,'weight_decay_instruction':.01,'language_replay_every_sft_updates':10,
                'objective':'Mean CE over all supervised response tokens including EOS; prompt and padding masked',
                'checkpoint_every':self.checkpoint_every,'automatic_promotion':False,'initialization':'random_no_parent'}

    def check(self):
        disk=json.loads(self.config.path.read_text(encoding='utf-8'))
        if not disk['learning_enabled']:
            self.config.values['learning_enabled']=False
            raise Cancelled('Aprendizado desligado; parar atualizações da v3')
        if self.stop_requested or (self.base/'PARAR_TREINO').exists():raise Cancelled('Parada solicitada da v3')
        self.gate.wait_if_paused()

    def new_optimizer(self):
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=3e-4 if self.phase=='language' else 6e-5,
            betas=(.9,.95),weight_decay=.1 if self.phase=='language' else .01)

    def initialize(self):
        if (self.base/'progress.json').exists():raise ValueError('A v3 já existe: retome o checkpoint')
        self.check();torch.manual_seed(4317)
        self.model=TACTransformer(self.cfg).to(self.device);self.model.text_tokenizer=self.data.tokenizer
        self.initial_hash=model_hash(self.model);self.new_optimizer()
        with self.gate.writing():
            self.check()
            self.db.execute('INSERT INTO runs(id,kind,status,started,config,dataset_hash,detail) VALUES(?,?,?,?,?,?,?)',
                (self.run_id,'conversation','treinando',now(),json.dumps(self.params()),self.data.meta['dataset_hash'],
                 json.dumps({'generation':'TAC-v3','parameters':self.model.parameter_count,'stage':'random_initialization'})))
        atomic_json(self.out/'PLANO.json',self.params())
        self.save('inicializacao_v3',recoverable=True)

    def state(self):
        return {'phase':self.phase,'step':self.step,'global_step':self.global_step,'presentations':self.presentations,
                'language_tokens':self.language_tokens,'sft_tokens':self.sft_tokens,'initial_hash':self.initial_hash,
                'language_deck':self.language_deck.state(),'sft_deck':self.sft_deck.state(),
                'params':self.params(),'sft_epoch_saved':self.sft_epoch_saved,
                'elapsed_seconds':self.elapsed_previous+time.monotonic()-self.started,
                'dataset_transition':self.dataset_transition}

    def save(self, status, evaluation=None, *, recoverable=True):
        if shutil.disk_usage(self.folder).free<800*2**20:
            raise OSError('Menos de 800 MiB livres: não arriscar preenchimento do disco')
        ident=uuid.uuid4().hex;path=self.folder/(ident+'.pt');temp=path.with_suffix('.writing')
        weights={k:v.detach().cpu() for k,v in self.model.state_dict().items()}
        if self.cfg.tie_embeddings:weights['output.weight']=weights['tokens.weight']
        cp={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','generation':'TAC-v3',
            'model_config':asdict(self.cfg),'tokenizer_state':self.data.tokenizer.state(),'model':weights,
            'dataset_hash':self.data.meta['dataset_hash'],'step':self.global_step,'run_id':self.run_id,
            'v3_state':self.state(),'recoverable':recoverable,'validation':evaluation or {}}
        if recoverable:
            cp.update(optimizer=self.optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])
        with self.gate.lock:
            # A disabled gate permits saving only already-committed new weights.
            torch.save(cp,temp);os.replace(temp,path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.db.execute('INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)',
                (ident,self.run_id,'v3/'+path.name,self.global_step,(evaluation or {}).get('language' if self.phase=='language' else 'instruction',{}).get('loss'),status,now()))
            with (self.base/'checkpoints.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'checkpoint':ident,'phase':self.phase,'step':self.step,'global_step':self.global_step,
                    'presentations':self.presentations,'recoverable':recoverable,'status':status,'file_sha256':digest,
                    'file_bytes':path.stat().st_size,'weights_hash':model_hash(self.model)})+'\n')
            if recoverable:
                atomic_json(self.base/'progress.json',{'generation':'TAC-v3','checkpoint':ident,'run_id':self.run_id,
                    'phase':self.phase,'step':self.step,'global_step':self.global_step,'updated':now(),'status':status})
        if recoverable:self.last_checkpoint=ident
        return ident

    def resume(self, checkpoint=None):
        progress=json.loads((self.base/'progress.json').read_text(encoding='utf-8'))
        from .training import Trainer
        ident=checkpoint or progress['checkpoint']
        path=Trainer(self.config,self.db,self.gate).checkpoint_path(ident)
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp.get('generation')!='TAC-v3' or not cp.get('recoverable'):raise ValueError('Retomada exige checkpoint recuperável v3')
        st=cp['v3_state']
        transitioned=False
        if cp['dataset_hash']!=self.data.meta['dataset_hash']:
            # A recorded correction to UNUSED instruction targets is allowed.
            # All learned pretraining bytes and the tokenizer must be identical.
            parent=json.loads((self.base/'training_ready.json').read_text(encoding='utf-8'))
            valid=(self.data.meta.get('transition_permitted_only_before_sft') is True
                   and self.data.meta.get('parent_dataset_hash')==cp['dataset_hash']==parent['dataset_hash']
                   and st['phase']=='language' and st['presentations']==0
                   and not st['sft_deck']['order'] and st['sft_deck']['epochs_started']==0
                   and hashlib.sha256(json.dumps(parent['files'],sort_keys=True).encode()).hexdigest()==parent['dataset_hash'])
            for filename in ('language_train.u16','language_validation.u16','tokenizer.json'):
                valid=valid and self.data.meta['files'][filename]==parent['files'][filename]
            if not valid:raise ValueError('Corpus da v3 mudou fora de uma revisão explícita de instruções ainda não usadas')
            self.dataset_transition={'from':cp['dataset_hash'],'to':self.data.meta['dataset_hash'],
                'at_step':st['global_step'],'old_sft_presentations':0,'language_and_tokenizer_unchanged':True,
                'reason':self.data.meta['revision_reason']}
            transitioned=True
        else:self.dataset_transition=st.get('dataset_transition')
        if st['params']!=self.params():raise ValueError('Orçamento/configuração de retomada diferente')
        self.model=TACTransformer(self.cfg).to(self.device);self.model.load_state_dict(cp['model'])
        self.model.text_tokenizer=self.data.tokenizer
        for name in ('phase','step','global_step','presentations','language_tokens','sft_tokens','initial_hash','sft_epoch_saved'):
            setattr(self,name,st[name])
        self.new_optimizer();self.optimizer.load_state_dict(cp['optimizer'])
        self.language_deck=IndexDeck(self.language_deck.n,0,st['language_deck'])
        self.sft_deck=IndexDeck(self.sft_deck.n,43172) if transitioned else IndexDeck(self.sft_deck.n,0,st['sft_deck'])
        self.run_id=cp['run_id'];self.last_checkpoint=ident
        self.elapsed_previous=st.get('elapsed_seconds',0.)
        torch.set_rng_state(cp['torch_rng'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])
        if transitioned:atomic_json(self.out/'transicao_dataset_sem_reiniciar_pesos.json',self.dataset_transition)
        moments=sorted({int(value['step']) for value in self.optimizer.state.values() if 'step' in value})
        if moments and moments!=[self.step]:raise ValueError('Passos do otimizador inconsistentes com o checkpoint')
        evidence={'checkpoint':ident,'run_id':self.run_id,'phase':self.phase,'global_step':self.global_step,
                  'phase_step':self.step,'weights_hash_loaded':model_hash(self.model),'optimizer_state_steps':moments,
                  'language_deck_position':self.language_deck.position,'sft_presentations':self.presentations,
                  'dataset_hash':self.data.meta['dataset_hash'],'dataset_transition':self.dataset_transition}
        if (self.base/'checkpoints.jsonl').exists():
            recorded=next(r for r in load_jsonl(self.base/'checkpoints.jsonl') if r['checkpoint']==ident)
            if evidence['weights_hash_loaded']!=recorded['weights_hash']:raise ValueError('Pesos restaurados diferem do hash registrado')
            evidence['weights_match_recorded_checkpoint']=True
        atomic_json(self.out/('retomada_real_'+str(self.global_step)+'.json'),evidence)
        self.db.execute("UPDATE runs SET status='treinando',finished=NULL WHERE id=?",(self.run_id,))

    def update(self):
        self.check();language=self.phase=='language' or self.step%10==9
        deck=self.language_deck if language else self.sft_deck;prior=deck.state()
        count=self.batch if language else min(self.batch,self.total_sft-self.presentations)
        indices=deck.take(count)
        if not language:indices.sort(key=lambda i:len(self.data.rows['train'][i]['sequence']))
        batches=[]
        for start in range(0,count,self.micro):
            ids=indices[start:start+self.micro]
            batches.append(self.data.language_batch(ids,context=self.context) if language else self.data.instruction_batch(ids))
        sizes=[int((y!=0).sum()) for _,y in batches];total=sum(sizes)
        target=self.pretrain_steps if self.phase=='language' else math.ceil(self.total_sft/self.batch/.9)
        peak=3e-4 if self.phase=='language' else 6e-5;warmup=300 if self.phase=='language' else 100
        lr=peak*min(1.,(self.step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*min(1.,(self.step+1)/target))))
        for group in self.optimizer.param_groups:group['lr']=lr
        self.model.train();self.optimizer.zero_grad(set_to_none=True);value=0.
        try:
            for (x,y),n in zip(batches,sizes):
                self.check()
                with torch.autocast(self.device.type,dtype=self.amp,enabled=self.device.type=='cuda'):
                    _,loss=self.model(x,y);weighted=loss*(n/total)
                if not torch.isfinite(weighted):raise FloatingPointError('Loss não finita')
                weighted.backward();value+=float(weighted.detach())
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
            with self.gate.writing():
                self.check();self.gate.check();self.optimizer.step();self.step+=1;self.global_step+=1
                if language:self.language_tokens+=total
                else:self.presentations+=count;self.sft_tokens+=total
        except BaseException:
            if language:self.language_deck=IndexDeck(deck.n,0,prior)
            else:self.sft_deck=IndexDeck(deck.n,0,prior)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        record={'step':self.global_step,'phase_step':self.step,'phase':self.phase,'kind':'language' if language else 'instruction',
                'loss':value,'gradient_norm_before_clip':float(norm),'lr':lr,'indices':indices,'supervised_tokens':total}
        with (self.base/('exposure_'+self.invocation+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        return record

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval();result={}
        for kind in ('language','instruction'):
            count=loss_sum=0
            if kind=='language':
                n=(len(self.data.language['validation'])-1)//self.context
                ids=torch.randperm(n,generator=torch.Generator().manual_seed(43173))[:32].tolist()
            else:
                ids=torch.randperm(len(self.data.rows['validation']),generator=torch.Generator().manual_seed(43174))[:96].tolist()
            for start in range(0,len(ids),self.micro):
                self.check();part=ids[start:start+self.micro]
                x,y=self.data.language_batch(part,'validation',self.context) if kind=='language' else self.data.instruction_batch(part,'validation')
                with torch.autocast(self.device.type,dtype=self.amp,enabled=self.device.type=='cuda'):
                    _,loss=self.model(x,y)
                n=int((y!=0).sum());count+=n;loss_sum+=float(loss)*n
            result[kind]={'loss':loss_sum/max(1,count),'target_tokens':count,'examples_or_windows':len(ids)}
        return result

    def run(self, progress=print):
        lock=QLockFile(str(self.config.data/'training.lock'));lock.setStaleLockTime(0)
        if not lock.tryLock(0):raise RuntimeError('Outro treinamento está em andamento')
        # The v2 continuation already uses its own lock; hold it too to keep
        # the existing interface from starting a second GPU job concurrently.
        old_lock=QLockFile(str(self.config.data/'tac_v2_conversation_20261018/training.lock'))
        old_lock.setStaleLockTime(0)
        if not (self.config.data/'tac_v2_conversation_20261018').exists():old_lock=None
        try:
            if old_lock is not None and not old_lock.tryLock(0):raise RuntimeError('Treinamento v2 em andamento')
            return self._run(progress)
        finally:
            if old_lock is not None:old_lock.unlock()
            lock.unlock()

    def _run(self, progress):
        if (self.base/'training_result.json').exists():
            return json.loads((self.base/'training_result.json').read_text(encoding='utf-8'))
        if self.model is None:
            if (self.base/'progress.json').exists():self.resume()
            else:self.initialize()
        if self.global_step==0:atomic_json(self.out/'initial_validation.json',self.evaluate())
        last=time.monotonic();session_start_step=self.global_step;last_log_step=self.global_step
        try:
            while self.phase!='complete':
                if self.phase=='language' and self.step>=self.pretrain_steps:
                    evaluation=self.evaluate();ident=self.save('v3_pretrain_concluido',evaluation)
                    atomic_json(self.base/'language_result.json',{'checkpoint':ident,'tokens':self.language_tokens,'validation':evaluation})
                    self.phase='instruction';self.step=0;self.new_optimizer()
                    progress('V3: pré-treinamento concluído; iniciando instruções');continue
                if self.phase=='instruction' and self.presentations>=self.total_sft:
                    self.phase='complete';break
                record=self.update()
                if self.global_step%50==0 or self.global_step==1:
                    memory=check_resources(self.config)
                    current=time.monotonic();elapsed=current-last;last=current
                    interval_steps=self.global_step-last_log_step;last_log_step=self.global_step
                    live={**record,'language_tokens':self.language_tokens,'sft_tokens':self.sft_tokens,
                          'sft_presentations':self.presentations,'updated':now(),'elapsed_seconds':self.elapsed_previous+current-self.started,
                          'session_elapsed_seconds':current-self.started,
                          'process_ram_mib':memory['process_mb'],'available_ram_mib':memory['ram_available_mb'],
                          'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20 if self.device.type=='cuda' else 0}
                    atomic_json(self.base/'live.json',live)
                    progress(f"V3 {self.phase} {self.step}/{self.pretrain_steps if self.phase=='language' else 'SFT'}; loss {record['loss']:.4f}; linguagem {self.language_tokens}; SFT {self.presentations}/{self.total_sft}; {elapsed:.1f}s/{interval_steps}")
                if self.phase=='language' and self.step%self.checkpoint_every==0:
                    evaluation=self.evaluate();ident=self.save('v3_recuperavel',evaluation)
                    progress(f'CHECKPOINT V3 {ident}: {evaluation}')
                if self.phase=='instruction' and self.presentations<self.total_sft and self.presentations//len(self.data.rows['train'])>self.sft_epoch_saved:
                    self.sft_epoch_saved=self.presentations//len(self.data.rows['train'])
                    evaluation=self.evaluate();ident=self.save('v3_sft_epoch_'+str(self.sft_epoch_saved),evaluation)
                    atomic_json(self.out/('sft_epoch_'+str(self.sft_epoch_saved)+'.json'),{'checkpoint':ident,'validation':evaluation})
                    progress(f'V3 SFT época {self.sft_epoch_saved}: {evaluation}')
                if self.session_updates and self.global_step-session_start_step>=self.session_updates:
                    if self.global_step%self.checkpoint_every:
                        self.save('v3_sessao_recuperavel')
                    self.db.execute("UPDATE runs SET status='interrompido' WHERE id=?",(self.run_id,))
                    return {'status':'sessao_salva','checkpoint':self.last_checkpoint,'global_step':self.global_step,
                            'budget_completed':False,'reason':'Session boundary; resume same run and budget'}
        except (Cancelled,LearningDisabled,KeyboardInterrupt) as exc:
            ident=self.save('v3_interrompido')
            self.db.execute("UPDATE runs SET status='interrompido' WHERE id=?",(self.run_id,))
            return {'status':'interrompido','checkpoint':ident,'reason':str(exc)}
        except Exception:
            if self.model is not None:self.save('v3_erro')
            self.db.execute("UPDATE runs SET status='erro' WHERE id=?",(self.run_id,))
            raise
        evaluation=self.evaluate();ident=self.save('v3_final_experimental',evaluation)
        result={'generation':'TAC-v3','status':'experimental_nao_promovido','checkpoint':ident,'run_id':self.run_id,
                'parameters':self.model.parameter_count,'global_steps':self.global_step,'language_tokens':self.language_tokens,
                'sft_target_tokens':self.sft_tokens,'sft_presentations':self.presentations,'sft_unique':len(self.data.rows['train']),
                'initial_hash':self.initial_hash,'final_hash':model_hash(self.model),'seconds':self.elapsed_previous+time.monotonic()-self.started,
                'validation':evaluation,'old_weights_loaded':False,'v2_modified':False,'automatic_promotion':False,
                'dataset_hash':self.data.meta['dataset_hash'],'dataset_transition':self.dataset_transition,
                'reserved_tests_opened':False,'external_model_used':False,'precision':str(self.amp),
                'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20 if self.device.type=='cuda' else 0}
        self.db.execute('UPDATE runs SET status=?,finished=?,detail=? WHERE id=?',('experimental_v3',now(),json.dumps(result),self.run_id))
        atomic_json(self.base/'training_result.json',result)
        return result
