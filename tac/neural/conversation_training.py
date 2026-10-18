"""Continuação SFT da TAC v2 existente; estado separado da reconstrução original."""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..config import atomic_json
from ..db import now
from ..learning import Cancelled,LearningDisabled
from .bpe import LocalBPETokenizer
from .conversation import load_checkpoint
from .decoding import conversation_options,decode_conversation
from .model import ModelConfig,TACTransformer
from .v2_data import canonical,load_jsonl
from .v2_training import IndexDeck,model_hash


def response_loss(logits,targets):
    """Média por resposta, sem diluir alvos curtos em respostas longas."""
    values=F.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=0,reduction='none')
    valid=(targets!=0).sum(1).clamp_min(1)
    return (values.sum(1)/valid).mean()


def token_f1(text,expected):
    a=Counter(re.findall(r'\w+',canonical(text)));b=Counter(re.findall(r'\w+',canonical(expected)))
    overlap=sum((a&b).values());return 2*overlap/max(1,sum(a.values())+sum(b.values()))


def score_output(row,text):
    """Rubricas explícitas; sobreposição textual não certifica semântica geral."""
    skill=row.get('skill','general');expected=row['answer'];norm=canonical(text)
    match=bool(re.search(r'(?<!\w)'+re.escape(canonical(expected))+r'(?!\w)',norm))
    correct=None
    if skill=='arithmetic':
        numbers=re.findall(r'-?\d+',text)
        correct=numbers==[expected]
        spec=row.get('verification',{})
        if isinstance(spec,dict):
            a,b,op=spec['a'],spec['b'],spec['op']
            compact=re.sub(r'\s+','',text).replace('*','×').rstrip('.')
            correct=correct or compact==f'{a}{op}{b}={expected}'
        correct=correct and not re.search(r'\bnão\b',norm)
    elif row.get('source')=='TAC-extraction-v2':
        correct=match and not re.search(r'\bnão\b',norm)
        if skill=='ordering':
            names=[canonical(x.strip()) for x in expected.split(',')]
            positions=[list(re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',norm)) for name in names]
            correct=(all(len(found)==1 for found in positions) and not re.search(r'\bnão\b',norm)
                     and [found[0].start() for found in positions]==sorted(found[0].start() for found in positions))
        # Resposta sobre um papel não pode devolver também outro valor presente.
        if skill!='ordering':
            prompt=' '.join(t for _,t in row['messages'])
            candidates=set(re.findall(r'\b(?:Aline|Bianca|Célia|Denise|Elisa|Fábio|Giovana|Heitor|Ingrid|Jonas|Laís|Mauro|Nádia|Otávio|Paula|Renata|Sérgio|Tânia|Ulisses|Vitória|Zilda|Iara|Murilo|Cíntia|Dália|Bento|azul|verde|vermelho|amarelo|roxo|laranja|branco|preto)\b',prompt))
            correct=correct and not any(re.search(r'(?<!\w)'+re.escape(canonical(v))+r'(?!\w)',norm) for v in candidates if canonical(v)!=canonical(expected))
    return {'structured_correct':correct,'contains_reference_span':match,'word_f1':token_f1(text,expected),
            'semantic_judgment':'not_automatically_certified' if correct is None else 'restricted_verifiable_rule'}


class ConversationContinuation:
    def __init__(self,config,db,gate,*,base=None,out=None,device='cuda',max_steps=8000,
                 batch=8,micro=4,every=1000,checkpoint_every=500,reference=None):
        self.config,self.db,self.gate=config,db,gate
        self.base=Path(base or config.data/'tac_v2_conversation_20261018')
        self.out=Path(out or config.root/'reports/conversacao_v2/rodada_20261018_01')
        self.out.mkdir(parents=True,exist_ok=True)
        self.meta=json.loads((self.base/'ready.json').read_text(encoding='utf-8'))
        self.rows={s:list(load_jsonl(self.base/('curated_'+s+'.jsonl' if (self.base/('curated_'+s+'.jsonl')).exists() else s+'.jsonl'))) for s in ('train','validation','retention')}
        self.device=torch.device(device);self.max_steps=max_steps;self.batch=batch;self.micro=micro
        self.every=every;self.checkpoint_every=checkpoint_every;self.step=0;self.run_id=uuid.uuid4().hex
        self.reference=reference;self.stop_requested=False;self.model=None;self.optimizer=None
        self.deck=IndexDeck(len(self.rows['train']),202610183);self.language_deck=None
        self.examples=0;self.targets=0;self.language_tokens=0;self.best=None;self.best_score=-1e9
        self.last_checkpoint=None;self.no_improvement=0;self.baseline=None;self.evaluations=[]
        self._explicit_resume=False
        torch.set_num_threads(config.get('cpu_threads'))
        self.amp=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
        if self.device.type=='cuda':
            torch.cuda.set_per_process_memory_fraction(min(.8,config.get('vram_mb')*2**20/torch.cuda.get_device_properties(0).total_memory))
        self.language={s:np.memmap(config.data/'tac_v2'/('language_'+s+'.u16'),dtype='<u2',mode='r') for s in ('train','validation')}
        self.language_deck=IndexDeck((len(self.language['train'])-1)//256,202610184)

    def params(self):
        return {'generation':'TAC-v2','continuation':True,'max_steps':self.max_steps,'batch':self.batch,'micro':self.micro,'every':self.every,
                'checkpoint_every':self.checkpoint_every,'lr':1e-4,'warmup':100,'weight_decay':.01,
                'gradient_clip':1.,'seed':20261018,'precision':str(self.amp),'language_replay_every':20,
                'objective':'mean response-token CE per example; prompt/padding masked; EOS supervised',
                'source_checkpoint':self.reference,'architecture_changed':False,'dataset_hash':self.meta['dataset_hash']}

    def check(self):
        disk=json.loads(self.config.path.read_text(encoding='utf-8'))
        if not disk['learning_enabled']:
            self.config.values['learning_enabled']=False
            raise Cancelled('Aprendizado desligado na interface; atualizações interrompidas')
        if self.stop_requested or (self.base/'PARAR_TREINO').exists():raise Cancelled('Parada solicitada')
        self.gate.wait_if_paused()

    def initialize(self):
        self.check()
        old=self.db.query("SELECT value FROM meta WHERE key='active_conversation'")
        self.reference=self.reference or old[0]['value']
        record=self.db.query('SELECT filename FROM checkpoints WHERE id=?',(self.reference,))[0]
        path=self.config.data/'checkpoints'/record['filename'];cp=load_checkpoint(path)
        if cp.get('generation')!='TAC-v2':raise ValueError('Continuação exige referência TAC v2')
        self.reference_sha=hashlib.sha256(path.read_bytes()).hexdigest();self.reference_step=cp['step']
        self.tokenizer=LocalBPETokenizer.from_state(cp['tokenizer_state'])
        self.cfg=ModelConfig(**cp['model_config'])
        torch.manual_seed(20261018)
        self.model=TACTransformer(self.cfg).to(self.device);self.model.load_state_dict(cp['model'])
        self.model.text_tokenizer=self.tokenizer
        self.initial_hash=model_hash(self.model)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
        self.reference_path=path
        with self.gate.writing():
            self.gate.check()
            self.db.execute('INSERT INTO runs(id,kind,status,started,config,dataset_hash,detail) VALUES(?,?,?,?,?,?,?)',
                (self.run_id,'conversation','treinando',now(),json.dumps(self.params()),self.meta['dataset_hash'],json.dumps({'generation':'TAC-v2','continuation':True})))
            self.db.execute("INSERT INTO meta(key,value) VALUES('reference_conversation_v2',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(self.reference,))
        atomic_json(self.out/'reference.json',{'checkpoint':self.reference,'file':str(path),'file_sha256':self.reference_sha,
            'weights_hash':self.initial_hash,'parameters':self.model.parameter_count,'reference_step':self.reference_step,'preserved_in_place':True})
        atomic_json(self.out/'PLANO.json',self.params())

    def batch_rows(self,indices,split='train'):
        rows=[self.rows[split][i] for i in indices];length=max(len(r['sequence'])-1 for r in rows)
        x=torch.zeros(len(rows),length,dtype=torch.long);y=torch.zeros_like(x)
        for i,r in enumerate(rows):
            seq=torch.tensor(r['sequence']);n=len(seq)-1
            x[i,:n]=seq[:-1];y[i,:n]=seq[1:];y[i,:r['prefix_length']-1]=0
        return x.to(self.device),y.to(self.device)

    def language_batch(self,indices,split='train'):
        data=self.language[split]
        seq=torch.from_numpy(np.stack([data[i*256:i*256+257] for i in indices]).astype(np.int64)).to(self.device)
        return seq[:,:-1].contiguous(),seq[:,1:].contiguous()

    def update(self):
        self.check();language=self.step%20==19
        deck=self.language_deck if language else self.deck
        prior=deck.state();ids=deck.take(self.batch)
        if not language:ids.sort(key=lambda i:len(self.rows['train'][i]['sequence']))
        lr=1e-4*min(1.,(self.step+1)/100)*(.15+.85*.5*(1+math.cos(math.pi*(self.step+1)/self.max_steps)))
        for g in self.optimizer.param_groups:g['lr']=lr
        self.model.train();self.optimizer.zero_grad(set_to_none=True);value=0.;targets=0
        try:
            for start in range(0,len(ids),self.micro):
                self.check();part=ids[start:start+self.micro]
                x,y=self.language_batch(part) if language else self.batch_rows(part)
                with torch.autocast(self.device.type,dtype=self.amp,enabled=self.device.type=='cuda'):
                    logits,_=self.model(x);loss=response_loss(logits,y)*(len(part)/len(ids))
                if not torch.isfinite(loss):raise FloatingPointError('Loss não finita')
                loss.backward();value+=float(loss.detach());targets+=int((y!=0).sum())
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
            with self.gate.writing():
                self.check();self.gate.check();self.optimizer.step();self.step+=1
                if language:self.language_tokens+=targets
                else:self.examples+=len(ids);self.targets+=targets
        except BaseException:
            if language:self.language_deck=IndexDeck(deck.n,0,prior)
            else:self.deck=IndexDeck(deck.n,0,prior)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        return {'step':self.step,'kind':'language' if language else 'instruction','indices':ids,'loss':value,
                'gradient_norm_before_clip':float(norm),'lr':lr,'supervised_tokens':targets}

    def checkpoint(self,status='recuperavel',evaluation=None,selected=False):
        if shutil.disk_usage(self.config.data).free<512*2**20:raise OSError('Espaço insuficiente para checkpoint seguro')
        ident=uuid.uuid4().hex;path=self.config.data/'checkpoints'/(ident+'.pt');temp=path.with_suffix('.sft-writing')
        if selected:self.best=ident
        state={'step':self.step,'reference':self.reference,'reference_sha':self.reference_sha,'reference_step':self.reference_step,
               'initial_hash':self.initial_hash,'dataset_hash':self.meta['dataset_hash'],'params':self.params(),
               'deck':self.deck.state(),'language_deck':self.language_deck.state(),'examples':self.examples,
               'targets':self.targets,'language_tokens':self.language_tokens,'best':self.best,'best_score':self.best_score,
               'baseline':self.baseline,'no_improvement':self.no_improvement,'evaluations':self.evaluations}
        cp={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','generation':'TAC-v2',
            'model_config':vars(self.cfg),'tokenizer_state':self.tokenizer.state(),'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            'optimizer':self.optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            'dataset_hash':self.meta['dataset_hash'],'step':self.reference_step+self.step,'run_id':self.run_id,
            'continuation_state':state,'validation':evaluation or {}}
        with self.gate.lock:
            torch.save(cp,temp);os.replace(temp,path)
            self.db.execute('INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)',
                (ident,self.run_id,path.name,cp['step'],(evaluation or {}).get('instruction_loss'),status,now()))
        self.last_checkpoint=ident
        atomic_json(self.base/'progress.json',{'checkpoint':ident,'run_id':self.run_id,'step':self.step,'max_steps':self.max_steps,
            'status':status,'examples':self.examples,'supervised_tokens':self.targets,'best':self.best,'updated':now()})
        return ident

    def resume(self,checkpoint=None):
        progress=json.loads((self.base/'progress.json').read_text(encoding='utf-8'))
        self._explicit_resume=checkpoint is not None
        ident=checkpoint or progress['checkpoint']
        path=self.config.data/'checkpoints'/(ident+'.pt');cp=load_checkpoint(path)
        state=cp['continuation_state'];self.reference=state['reference']
        if state['dataset_hash']!=self.meta['dataset_hash']:raise ValueError('Corpus da continuação mudou')
        for key in ('max_steps','batch','micro','every','checkpoint_every'):
            if getattr(self,key)!=state['params'][key]:raise ValueError('Retomada exige a mesma configuração: '+key)
        self.reference_path=self.config.data/'checkpoints'/(self.reference+'.pt')
        self.reference_sha=state['reference_sha'];self.reference_step=state['reference_step'];self.initial_hash=state['initial_hash']
        self.run_id=cp['run_id'];self.step=state['step'];self.cfg=ModelConfig(**cp['model_config'])
        self.tokenizer=LocalBPETokenizer.from_state(cp['tokenizer_state']);self.model=TACTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(cp['model']);self.model.text_tokenizer=self.tokenizer
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
        self.optimizer.load_state_dict(cp['optimizer'])
        self.deck=IndexDeck(len(self.rows['train']),0,state['deck']);self.language_deck=IndexDeck(self.language_deck.n,0,state['language_deck'])
        self.examples=state['examples'];self.targets=state['targets'];self.language_tokens=state['language_tokens']
        self.best=state['best'];self.best_score=state['best_score'];self.baseline=state['baseline']
        self.no_improvement=state['no_improvement'];self.evaluations=state['evaluations'];self.last_checkpoint=ident
        torch.set_rng_state(cp['torch_rng'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])

    def evaluation_rows(self):
        groups=defaultdict(list)
        for i,r in enumerate(self.rows['validation']):
            category='arithmetic' if r['skill']=='arithmetic' else 'context' if r['source']=='TAC-extraction-v2' else 'reading' if r['skill']=='reading' else 'natural'
            groups[category].append(i)
        selected=[]
        for category,ids in sorted(groups.items()):
            random.Random(202610185).shuffle(ids)
            # Contexto estratificado pelos papéis, sem escolher apenas o caso mais fácil.
            if category=='context':
                skillgroups=defaultdict(list)
                for i in ids:skillgroups[self.rows['validation'][i]['skill']].append(i)
                ids=[i for key in sorted(skillgroups) for i in skillgroups[key][:2]]
            selected.extend((category,self.rows['validation'][i]) for i in ids[:14 if category=='context' else 12])
        return selected

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval();losses={};indices=[]
        for category,row in self.evaluation_rows():indices.append(self.rows['validation'].index(row))
        for split,ids in [('validation',indices),('retention',list(range(min(24,len(self.rows['retention'])))))]:
            total=count=0
            for i in ids:
                self.check();x,y=self.batch_rows([i],split);logits,_=self.model(x)
                total+=float(response_loss(logits,y));count+=1
            losses[split]=total/max(1,count)
        total=0
        for i in range(16):
            self.check();x,y=self.language_batch([i*7],'validation');_,loss=self.model(x,y);total+=float(loss)
        losses['language']=total/16
        outputs=[];grouped=defaultdict(list)
        for category,row in self.evaluation_rows():
            self.check()
            result=decode_conversation(self.model,row['messages'],self.device,max_tokens=96,
                                      generation_options=conversation_options('controlled',42))
            score=score_output(row,result['text']);score['loop']=result['stop_reason']=='repetition';score['length']=result['stop_reason']=='length'
            outputs.append({'category':category,'source':row['source'],'skill':row['skill'],'family':row.get('family'),
                            'messages':row['messages'],'expected':row['answer'],'output':result,'score':score})
            grouped[category].append(score)
        summary={}
        for key,values in grouped.items():
            measurable=[v['structured_correct'] for v in values if v['structured_correct'] is not None]
            summary[key]={'n':len(values),'structured_accuracy':sum(measurable)/len(measurable) if measurable else None,
                          'word_f1':sum(v['word_f1'] for v in values)/len(values),
                          'loop_rate':sum(v['loop'] for v in values)/len(values),'length_rate':sum(v['length'] for v in values)/len(values)}
        probes=[]
        for q in ['Opa, tudo bom?','O que é um lobo?','Quanto é 2 + 2?','Crie uma história curta.','Meu nome é Carlos. Qual é meu nome?']:
            self.check();probes.append({'question':q,**decode_conversation(self.model,[['user',q]],self.device,
                    generation_options=conversation_options('controlled',42))})
        open_tasks=[]
        for category,q in [('greeting','Bom dia! Podemos conversar?'),('greeting','Olá, como estão as coisas?'),
            ('explanation','Explique, em poucas palavras, o que é a chuva.'),('explanation','Como as plantas obtêm energia da luz?'),
            ('creative','Imagine um conto de três frases com uma coruja e uma ponte.'),
            ('creative','Redija um bilhete curto agradecendo a ajuda de uma amiga.'),
            ('instruction','Sugira duas atividades tranquilas para fazer em casa.'),
            ('context','Pode me chamar de Lívia. Qual nome acabei de informar?')]:
            self.check();open_tasks.append({'category':category,'question':q,
                **decode_conversation(self.model,[['user',q]],self.device,generation_options=conversation_options('controlled',42))})
        # Retenção também em geração livre, sem escolher apenas por teacher forcing.
        retained=[]
        for row in self.rows['retention'][:6]:
            self.check();result=decode_conversation(self.model,row['messages'],self.device,max_tokens=96,
                 generation_options=conversation_options('controlled',42))
            retained.append({'messages':row['messages'],'expected':row['answer'],'output':result,
                             'word_f1':token_f1(result['text'],row['answer'])})
        a=summary.get('arithmetic',{}).get('structured_accuracy') or 0
        c=summary.get('context',{}).get('structured_accuracy') or 0
        r=summary.get('reading',{}).get('word_f1',0);n=summary.get('natural',{}).get('word_f1',0)
        loop=sum(v['score']['loop'] for v in outputs)/max(1,len(outputs))
        length=sum(v['score']['length'] for v in outputs)/max(1,len(outputs))
        proxy=.35*a+.35*c+.15*r+.15*n-.25*loop-.05*length
        result={'step':self.step,'instruction_loss':losses['validation'],'losses':losses,'by_category':summary,
                'selection_proxy':proxy,'selection_not_general_semantic_certification':True,
                'outputs':outputs,'known_probes_not_used_for_selection':probes,'retention_free':retained,
                'open_generalization_tasks_for_human_review':open_tasks,
                'retention_free_word_f1':sum(v['word_f1'] for v in retained)/max(1,len(retained))}
        atomic_json(self.out/f'eval_{self.step:05d}.json',result)
        self.model.train();return result

    def run(self,progress=print):
        from PySide6.QtCore import QLockFile
        lock=QLockFile(str(self.base/'training.lock'))
        if not lock.tryLock(0):raise RuntimeError('Esta continuação já está sendo executada em outro processo.')
        try:
            result=self.base/'result.json'
            if result.exists() and not self._explicit_resume:
                return json.loads(result.read_text(encoding='utf-8'))
            return self._run(progress)
        finally:lock.unlock()

    def _run(self,progress=print):
        if self.model is None:
            if (self.base/'progress.json').exists():self.resume()
            else:self.initialize()
        if self.baseline is None:
            progress('Avaliando referência antes do primeiro passo...');self.baseline=self.evaluate()
            self.best_score=self.baseline['selection_proxy'];self.checkpoint('inicio_continuacao',self.baseline)
        start=time.monotonic();last_log=start
        try:
            while self.step<self.max_steps:
                row=self.update()
                with (self.base/'exposure.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                if self.step%50==0:
                    progress(f"SFT {self.step}/{self.max_steps}: loss {row['loss']:.4f}; exemplos {self.examples}; alvos {self.targets}; {time.monotonic()-last_log:.1f}s/50")
                    last_log=time.monotonic()
                    atomic_json(self.base/'live.json',{**row,'examples':self.examples,'targets':self.targets,'updated':now()})
                evaluation=None
                if self.step%self.every==0 or self.step==self.max_steps:
                    progress(f'Avaliação de geração livre no passo {self.step}...');evaluation=self.evaluate()
                    eligible=(evaluation['losses']['language']<=self.baseline['losses']['language']*1.3
                              and evaluation['retention_free_word_f1']>=self.baseline['retention_free_word_f1']-.05)
                    score=evaluation['selection_proxy'];improved=eligible and score>self.best_score+.005
                    if improved:self.best_score=score;self.no_improvement=0
                    else:self.no_improvement+=1
                    ident=self.checkpoint('experimental_conversacao_v2',evaluation,selected=improved)
                    if improved:
                        self.best=ident
                        atomic_json(self.base/'best.json',{'checkpoint':ident,'step':self.step,'proxy':score,'eligible':eligible})
                    self.evaluations.append({'step':self.step,'checkpoint':ident,'score':score,'eligible':eligible,
                        'improved':improved,'losses':evaluation['losses'],'by_category':evaluation['by_category']})
                    atomic_json(self.out/'evolucao.json',self.evaluations)
                    progress('AVALIAÇÃO '+json.dumps(self.evaluations[-1],ensure_ascii=False))
                    if self.step>=4000 and self.no_improvement>=3:
                        progress('Parada antecipada: três avaliações sem melhora elegível após pelo menos 4.000 passos.');break
                elif self.step%self.checkpoint_every==0:self.checkpoint()
        except (Cancelled,LearningDisabled,KeyboardInterrupt) as exc:
            ident=self.checkpoint('interrompido_conversacao_v2')
            self.db.execute("UPDATE runs SET status='interrompido',finished=? WHERE id=?",(now(),self.run_id))
            return {'status':'interrompido','checkpoint':ident,'step':self.step,'detail':str(exc)}
        except BaseException:
            self.checkpoint('erro_conversacao_v2');self.db.execute("UPDATE runs SET status='erro',finished=? WHERE id=?",(now(),self.run_id));raise
        ident=self.checkpoint('conversacao_v2_concluida')
        self.db.execute("UPDATE runs SET status='experimental_v2',finished=? WHERE id=?",(now(),self.run_id))
        result={'status':'concluido_experimental','step':self.step,'max_steps':self.max_steps,'run_id':self.run_id,
                'last_checkpoint':ident,'best_checkpoint':self.best,'best_proxy':self.best_score,'reference':self.reference,
                'reference_sha_preserved':hashlib.sha256(self.reference_path.read_bytes()).hexdigest()==self.reference_sha,
                'examples':self.examples,'supervised_tokens':self.targets,'language_tokens':self.language_tokens,
                'parameters':self.model.parameter_count,'final_weight_hash':model_hash(self.model),'seconds_this_session':time.monotonic()-start,
                'unique_train_examples':len(self.rows['train']),'dataset_hash':self.meta['dataset_hash'],'evaluations':self.evaluations,
                'reserved_tests_opened':False,'external_models_used':False,'automatically_published':False}
        atomic_json(self.base/'result.json',result);atomic_json(self.out/'training_result.json',result)
        return result
