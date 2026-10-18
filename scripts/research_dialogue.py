"""Pilotos A/B reais, isolados do registro de modelos ativos. Sem promoção."""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tac.core import Core
from tac.config import atomic_json
from tac.neural.conversation import ConversationData,load_checkpoint
from tac.neural.conversation_corpus import TextReplay
from tac.neural.model import ModelConfig,TACTransformer
from tac.neural.training import weight_hash
from tac.neural.research import audit_data,development_view,anchor_loss,weight_delta,gradient_interference,DeckSampler,unique_training_view
from tac.diagnostics import check_resources
from tac.dialogue_imports import imported_kind

FOLDER=ROOT/'reports/research_0120'


def load_context():
    core=Core(ROOT)
    run=core.db.query("SELECT * FROM runs WHERE kind='conversation' ORDER BY started DESC LIMIT 1")[0]
    params=json.loads(run['config'])
    data=ConversationData.load(core.config.data/'datasets'/('conversation_'+run['dataset_hash']+'.json'))
    parent=core.conversation.active_id(); path=core.trainer.checkpoint_path(parent)
    cp=load_checkpoint(path)
    return core,params,data,parent,cp


def model_from(cp,data,device):
    model=TACTransformer(ModelConfig(**cp['model_config'])).to(device)
    model.load_state_dict(cp['model']); model.text_tokenizer=data.tokenizer
    return model


def evaluate(core,model,dev,device):
    return {key:core.conversation.evaluate(model,dev,device,split=key,generation=True,interruptible=True)
            for key in dev.sets if key!='train' and dev.sets[key]}


def pilot(core,params,data,parent,cp,lr,steps,folder,policy='original',sampler_mode='replacement'):
    if folder.exists():raise FileExistsError('Piloto já registrado. Preserve os resultados e escolha outra pasta de experimento.')
    if not 1<=steps<=params['steps']:raise ValueError('Passos devem caber no scheduler fixo do experimento.')
    import shutil
    estimated=sum(value.numel()*value.element_size() for value in cp['model'].values())
    if shutil.disk_usage(core.config.data).free < estimated+512*2**20:
        raise OSError('Sem espaço para pesos do piloto e margem de 512 MiB; nenhum treino iniciado.')
    core.gate.check(); check_resources(core.config)
    torch.manual_seed(params['seed']); torch.set_num_threads(core.config.get('cpu_threads'))
    device=core.trainer.device()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    teacher=model_from(cp,data,device).eval().requires_grad_(False)
    model=model_from(cp,data,device)
    replay=TextReplay(core.config.data/'datasets'/(params['text_corpus']+'.json'),data.tokenizer)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    rng=torch.Generator().manual_seed(params['seed']); counts=Counter(); metrics=[]
    dev=development_view(data)
    initial=weight_hash(model); sequence=hashlib.sha256(); started=time.monotonic()
    before=gradient_interference(model,teacher,data,replay,device)
    sampler=DeckSampler(data) if sampler_mode=='deck_v1' else None
    for step in range(steps):
        core.gate.wait_if_paused()
        if step%10==0:check_resources(core.config)
        optimizer.param_groups[0]['lr']=lr*min(1.,(step+1)/params['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/params['steps'])))
        optimizer.zero_grad(set_to_none=True); model.train(); by_pool=[]
        for micro in range(params['accumulation']):
            kind=imported_kind(step,micro,params['accumulation']); counts[kind]+=1
            if kind=='language':x,y=replay.batch(params['batch_size'],min(256,model.cfg.context),rng,device)
            else:
                rows=sampler.sample(kind,params['batch_size'],rng) if sampler else data.sample_pool(params['batch_size'],rng,kind)
                x,y=data.batch(rows,model.cfg.context,device)
            sequence.update(x.detach().cpu().numpy().tobytes());sequence.update(y.detach().cpu().numpy().tobytes())
            jitter=min(params.get('position_jitter',0),model.cfg.context-x.shape[1])
            offset=int(torch.randint(jitter+1,(),generator=rng)) if jitter else 0
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits,ce=model(x,y,position_offset=offset); kl=torch.zeros((),device=device)
                if kind=='legacy':
                    with torch.no_grad():ref,_=teacher(x,position_offset=offset)
                    mask=y!=0
                    kl=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),ref[mask].float().softmax(-1),reduction='batchmean')
                loss=ce+params['retention_weight']*kl
            if not torch.isfinite(loss):raise FloatingPointError('Perda não finita no piloto.')
            factor=.5 if policy=='anchor_v1' else 1.
            scaler.scale(loss*factor/params['accumulation']).backward()
            by_pool.append({'pool':kind,'ce':float(ce.detach()),'kl':float(kl.detach())})
        if policy=='anchor_v1':
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                total,ce,kl=anchor_loss(model,teacher,data,rng,device,params['batch_size'],params['retention_weight'])
            scaler.scale(total*.5).backward();counts['anchor']+=1
            by_pool.append({'pool':'anchor','ce':float(ce.detach()),'kl':float(kl.detach())})
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
        if not torch.isfinite(norm):raise FloatingPointError('Gradiente não finito no piloto; comparação interrompida.')
        with core.gate.writing():core.gate.check();scaler.step(optimizer);scaler.update()
        if step%20==0 or step+1==steps:
            metrics.append({'step':step+1,'learning_rate':optimizer.param_groups[0]['lr'],'objectives':by_pool,'grad_norm':float(norm)})
            print('pilot',lr,policy,sampler_mode,step+1,flush=True)
    training_seconds=time.monotonic()-started
    after=gradient_interference(model,teacher,data,replay,device)
    measured=evaluate(core,model,dev,device)
    final=weight_hash(model)
    folder.mkdir(parents=True,exist_ok=False)
    # Checkpoint novo de PESOS, não substitui nenhum checkpoint da aplicação.
    value={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','model_config':vars(model.cfg),
        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'tokenizer_state':data.tokenizer.state(),
        'dataset_hash':data.digest,'step':steps,'research_only':True,'parent':parent}
    with core.gate.writing():torch.save(value,folder/'candidate.pt')
    result={'parent':parent,'dataset_hash':data.digest,'seed':params['seed'],'learning_rate':lr,'steps':steps,
        'schedule_target_steps':params['steps'],'warmup':params['warmup'],'batch_size':params['batch_size'],
        'accumulation':params['accumulation'],'policy':policy,'sampler':sampler_mode,'retention_weight':params['retention_weight'],
        'sequence_hash':sequence.hexdigest(),'initial_hash':initial,'final_hash':final,'weight_change':weight_delta(model,teacher),
        'mixture_counts':dict(counts),'metrics':metrics,'gradient_before':before,'gradient_after':after,'validation':measured,
        'training_seconds':training_seconds,'total_seconds':time.monotonic()-started,'device':str(device),
        'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
        'test_evaluated':False,'promoted':False}
    atomic_json(folder/'result.json',result)
    del optimizer,model,teacher,replay
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','baseline','rates','anchor','deck','practice'))
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--lr',type=float,default=.00002)
    parser.add_argument('--output',type=Path,default=FOLDER,
                        help='Pasta dos relatórios dentro de TAC/reports; use outra pasta para repetir sem substituir resultados.')
    args=parser.parse_args()
    folder=args.output.resolve()
    if not folder.is_relative_to(ROOT/'reports') or folder==ROOT/'reports':
        parser.error('Escolha uma subpasta dentro de TAC/reports.')
    if args.mode=='baseline' and (folder/'development_baseline.json').exists():
        raise FileExistsError('Referência já registrada. Use --output com outra pasta para repetir a experiência.')
    if args.mode=='rates' and any((folder/('lr_'+str(lr))).exists() for lr in (.00002,.00004,.00008)):
        raise FileExistsError('Esta comparação já existe. Use --output com outra pasta.')
    if args.mode in ('anchor','deck','practice') and (folder/args.mode).exists():
        raise FileExistsError('Piloto já registrado. Use --output com outra pasta.')
    folder.mkdir(parents=True,exist_ok=True)
    core,params,data,parent,cp=load_context()
    if args.mode=='audit':
        atomic_json(folder/'dataset_audit.json',audit_data(data,cp['model_config']['context']));print('audit_saved');return
    core.gate.begin() # Usuário autorizou experimentos; o botão continua sendo respeitado.
    if args.mode=='baseline':
        device=core.trainer.device();model=model_from(cp,data,device)
        atomic_json(folder/'development_baseline.json',{'parent':parent,'dataset_hash':data.digest,
            'validation':evaluate(core,model,development_view(data),device),'test_evaluated':False});return
    if args.mode=='rates':
        for lr in (.00002,.00004,.00008):pilot(core,params,data,parent,cp,lr,args.steps,folder/('lr_'+str(lr)))
    else:
        policy='anchor_v1' if args.mode in ('anchor','practice') else 'original'
        selected=unique_training_view(data) if args.mode=='deck' else data
        if args.mode=='practice':
            from tac.transfer_practice import augment
            selected=augment(data)
        pilot(core,params,selected,parent,cp,args.lr,args.steps,folder/args.mode,policy,'deck_v1' if args.mode=='deck' else 'replacement')


if __name__=='__main__':main()
