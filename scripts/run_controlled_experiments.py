"""Pilotos locais v0.13: um fator por braço, sem registro/promover modelos."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.config import DEFAULTS, atomic_json
from tac.diagnostics import check_resources, resources as current_resources
from tac.learning import LearningGate
from tac.dialogue_imports import imported_kind
from tac.fluency_lessons import lessons
from tac.neural.conversation import ConversationData, load_checkpoint
from tac.neural.conversation_corpus import TextReplay
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import weight_hash
from tac.neural.research import precise_prompt, weight_delta
from tac.neural.controlled import (BalancedImports, Exposure, MatchedRetention, ResponseObjective,
    category, row_key, tensor_hash, vector, vector_hash, relation, diagnostic_gradients)
from tac.neural.controlled_eval import DevelopmentEvaluator, protection_checks

OUT=ROOT/'reports/experimentos_0130'
DATASET='4ac9e6e1c37da84650dc0c908a2232f13edf5e95e3df9ea912908d9e6596b2ee'
OLD='52dcde855aed347972bc508c4b64463990706a98c10d3b7b804851cc9d7d8a55'
CORPUS='27d2765ca06a705e89dac022f438018e9340727b71e649eb523dc80116532060'
ARMS=('control','A_response','B_balanced','C_coverage','C_representative')
PROTOCOL={'format':'TAC-controlled-0130-v1','steps':120,'seeds':[42,417,903],
    'learning_rate':.00002,'schedule_steps':6000,'warmup':300,'batch_size':8,'accumulation':2,
    'weight_decay':.01,'clip_grad_norm':1.,'retention_weight':2.,'position_jitter':0,
    'precision':'fp16 autocast CUDA; parâmetros, AdamW e sondas fp32; CPU fp32',
    'optimizer_reset':True,'dropout':0.,'gradient_probe_steps':[1,12,20,60,120],
    'gradient_reduction':'FP32, sem cópias integrais em FP64; todas as posições dos parâmetros',
    'diagnostic_storage':'vetores na GPU quando CUDA; SHA em blocos de 1 MiB; não copiar vetores integrais para RAM',
    'training_time_limit_seconds':150,'probe_time_excluded_from_training_limit':True,
    'generation':'greedy, máximo 128 novos tokens; nenhuma supressão de papéis',
    'imported_generation_panel':60,'final_test':'não aberto; somente SHA-256',
    'parent':'e2a27d90e80c4123bd795d8cf5776ee7','dataset':DATASET,'old_dataset':OLD,'corpus':CORPUS,
    'common_protocol_difference_from_0120':'RNG independente por pool e position_jitter=0 em TODOS os braços; não é reprodução bit a bit da v0.12.',
    'promotion':'proibida neste runner; segunda rodada exige análise dos pilotos'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


class LiveSettings:
    """Lê autorização ao vivo; nunca grava nem normaliza o config do usuário."""
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'config.json';self.values={**DEFAULTS,**read(self.path)}
        self.data=self.root/self.values['data_dir']

    def get(self,key):
        if key=='learning_enabled':return read(self.path).get(key,False)
        return self.values[key]


def model_from(cp,tokenizer,device):
    model=TACTransformer(ModelConfig(**cp['model_config'])).to(device)
    model.load_state_dict(cp['model']);model.text_tokenizer=tokenizer
    return model


def rng(seed,offset):return torch.Generator().manual_seed(seed+offset)


def full_retention(data,old):
    # Não lê o teste final selado. Filtros só de fronteiras já registradas nos datasets.
    forbidden={precise_prompt(r['messages']) for source in (data,old)
        for split,rows in source.sets.items() if split!='train' for r in rows}
    return [r for r in old.sets['train'] if precise_prompt(r['messages']) not in forbidden]


def common_plan(data,replay,seed,steps):
    generators={p:rng(seed,i*100003) for i,p in enumerate(('legacy','reviewed','imported','language'),1)}
    plan=[]
    for step in range(steps):
        for micro in range(PROTOCOL['accumulation']):
            pool=imported_kind(step,micro,PROTOCOL['accumulation']);g=generators[pool]
            if pool=='language':
                items=[]
                for _ in range(PROTOCOL['batch_size']):
                    index=int(torch.randint(len(replay.rows),(),generator=g))
                    start=int(torch.randint(max(1,len(replay.rows[index])-256),(),generator=g))
                    items.append((index,start))
            else:items=data.sample_pool(PROTOCOL['batch_size'],g,pool)
            plan.append((pool,items))
    return plan


def arm_plan(data,objective,full,seed,steps,arm):
    plan=common_plan(data,objective.replay,seed,steps)
    imported=[r for r in data.sets['train'] if r.get('mixture')=='imported']
    balanced=BalancedImports(imported,rng(seed,771001)) if arm=='B_balanced' else None
    retained={row_key(r) for r in data.sets['train'] if r.get('mixture','legacy')=='legacy'}
    matcher=MatchedRetention(full,data.tokenizer,retained) if arm.startswith('C_') else None
    match_rng=rng(seed,991001);batches=[];hashes=defaultdict(hashlib.sha256);xhashes=defaultdict(hashlib.sha256)
    tokens=Counter()
    for pool,items in plan:
        if pool=='language':x,y,rows=objective.batch(items,variant=arm=='A_response')
        else:
            rows=balanced.sample(len(items)) if balanced and pool=='imported' else items
            if matcher and pool=='legacy':rows=matcher.sample(items,data.tokenizer,match_rng,arm=='C_representative')
            x,y=data.batch(rows,512,'cpu')
            if matcher and pool=='legacy':
                original=[int((data.encode(r,512)[1]!=0).sum()) for r in items]
                assert original==[int((v!=0).sum()) for v in y], 'Orçamento de tokens de replay mudou.'
        hashes[pool].update(bytes.fromhex(tensor_hash(x,y)));xhashes[pool].update(x.numpy().tobytes())
        tokens[pool]+=int((y!=0).sum());batches.append((pool,x,y,rows))
    return batches,{'xy_hashes':{p:h.hexdigest() for p,h in hashes.items()},
        'x_hashes':{p:h.hexdigest() for p,h in xhashes.items()},'target_tokens':dict(tokens),
        'micro_batches':dict(Counter(p for p,*_ in batches))}


def fixed_probes(data,objective,seed):
    g=rng(seed,871331);probes={}
    for pool in ('legacy','imported','reviewed'):
        rows=data.sample_pool(2,g,pool);probes[pool]=data.batch(rows,512,'cpu')
    descriptors=[]
    for _ in range(2):
        i=int(torch.randint(len(objective.replay.rows),(),generator=g))
        start=int(torch.randint(max(1,len(objective.replay.rows[i])-256),(),generator=g))
        descriptors.append((i,start))
    x,y,_=objective.batch(descriptors);probes['language']=(x,y);probes['kl']=probes['legacy']
    return probes


@torch.no_grad()
def probe_losses(model,teacher,batches,device):
    out={}
    for pool,(x,y) in batches.items():
        x=x.to(device);y=y.to(device);logits,ce=model(x,y)
        if pool=='kl':
            ref,_=teacher(x);mask=y!=0
            ce=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),ref[mask].float().softmax(-1),reduction='batchmean')
        out[pool]=float(ce)
    return out


def moment_norms(optimizer):
    sums=Counter();steps=[]
    for state in optimizer.state.values():
        for key in ('exp_avg','exp_avg_sq'):sums[key]+=float(state[key].float().square().sum())
        steps.append(float(state['step']))
    return {'l2':{k:v**.5 for k,v in sums.items()},'step_min':min(steps,default=0),'step_max':max(steps,default=0)}


def protected_step(optimizer,scaler,gate):
    with gate.writing():
        gate.check();scaler.step(optimizer);scaler.update()


def pilot(config,gate,engine,cp,data,objective,full,baseline,novel,device,arm,seed,steps,folder):
    if folder.exists():
        if not (folder/'result.json').is_file():
            raise ValueError(f'Piloto incompleto preservado em {folder}. Use --output com uma pasta nova para repetir do mesmo ponto de partida.')
        previous=read(folder/'result.json')
        if previous['status']!='completed' or previous['protocol']!={**PROTOCOL,'steps':steps}:
            raise ValueError('Execução anterior diferente/incompleta; preserve e escolha outra pasta.')
        if previous['checkpoint_kept'] and sha(folder/'candidate.pt')!=previous['checkpoint_sha256']:
            raise ValueError('O arquivo de pesos experimental foi alterado; não reutilizar este resultado silenciosamente.')
        print('preservado',arm,seed,flush=True);return previous
    estimated=sum(t.numel()*t.element_size() for t in cp['model'].values())
    if shutil.disk_usage(folder.parent).free < estimated+512*2**20:
        raise OSError('Margem de disco insuficiente: preservar 512 MiB e os pesos temporários.')
    folder.mkdir(parents=True,exist_ok=False);gate.check();check_resources(config)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    batches,plan=arm_plan(data,objective,full,seed,steps,arm)
    probes=fixed_probes(data,objective,seed)
    model=model_from(cp,data.tokenizer,device);teacher=model_from(cp,data.tokenizer,device).eval().requires_grad_(False)
    initial=weight_hash(model);optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');exposure=Exposure()
    history=[];telemetry=[];started=time.monotonic();probe_seconds=0.;resources=[]
    atomic_json(folder/'plan.json',plan)
    completed=0
    try:
        for index in range(steps):
            gate.wait_if_paused()
            if index%10==0:resources.append({'step':index,**check_resources(config)})
            if time.monotonic()-started-probe_seconds>PROTOCOL['training_time_limit_seconds']:
                raise TimeoutError('Limite seguro de tempo do piloto; não completar número de passos artificialmente.')
            probe=index+1 in PROTOCOL['gradient_probe_steps'];directions={};weighted=None
            if probe:
                t=time.monotonic();model.eval()
                directions,loss_before=diagnostic_gradients(model,teacher,probes,device,device)
                weights_before=vector(model,device=device);moments_before=moment_norms(optimizer)
                # Gradientes FP32 dos LOTES REAIS da atualização, separados por objetivo.
                contributions={};actual_losses={}
                for micro in range(2):
                    pool,x,y,_=batches[index*2+micro];tasks={pool:(x,y)}
                    if pool=='legacy':tasks['kl']=(x,y)
                    vs,ls=diagnostic_gradients(model,teacher,tasks,device,device)
                    for key,value in vs.items():
                        name=f'{micro}:{key}';factor=(2. if key=='kl' else 1.)/2
                        contributions[name]=value*factor;actual_losses[name]=ls[key]
                weighted=sum(contributions.values())
                task_stats={name:{'norm':float(v.norm()),'coefficient':2./2 if name.endswith(':kl') else 1./2}
                    for name,v in contributions.items()}
                pair_stats={a+'__'+b:relation(contributions[a],contributions[b]) for a in contributions for b in contributions if a<b}
                del contributions,vs,value
                probe_seconds+=time.monotonic()-t
            lr=PROTOCOL['learning_rate']*min(1.,(index+1)/PROTOCOL['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*(index+1)/PROTOCOL['schedule_steps'])))
            optimizer.param_groups[0]['lr']=lr;optimizer.zero_grad(set_to_none=True);model.train();objectives=[]
            for micro in range(2):
                gate.check();pool,x,y,rows=batches[index*2+micro];x=x.to(device);y=y.to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    logits,ce=model(x,y);kl=torch.zeros((),device=device)
                    if pool=='legacy':
                        with torch.no_grad():reference,_=teacher(x)
                        mask=y!=0
                        kl=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),reference[mask].float().softmax(-1),reduction='batchmean')
                    loss=ce+2.*kl
                if not torch.isfinite(loss):raise FloatingPointError('Perda não finita; piloto interrompido.')
                scaler.scale(loss/2).backward();exposure.add(pool,rows,y)
                objectives.append({'pool':pool,'cross_entropy':float(ce.detach()),'kl':float(kl.detach()),'targets':int((y!=0).sum())})
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
            if not torch.isfinite(norm):raise FloatingPointError('Gradiente não finito; piloto interrompido sem contar atualização.')
            clipped=vector(model,grad=True,device=device) if probe else None
            protected_step(optimizer,scaler,gate);completed=index+1
            if probe:
                t=time.monotonic();delta=vector(model,device=device)-weights_before;model.eval()
                after=probe_losses(model,teacher,probes,device)
                telemetry.append({'step':completed,'probe_losses_before':loss_before,'probe_losses_after':after,
                    'actual_batch_losses_before':actual_losses,'raw_task_contributions':task_stats,'actual_batch_gradient_pairs':pair_stats,
                    'fixed_probe_pairs':{a+'__'+b:relation(directions[a],directions[b]) for a in directions for b in directions if a<b},
                    'actual_adam_update':{'norm':float(delta.norm()),'sha256':vector_hash(delta),
                        'vs_fixed_gradients':{p:relation(delta,g) for p,g in directions.items()},
                        'vs_clipped_gradient':relation(delta,clipped),'vs_fp32_weighted_gradient':relation(delta,weighted)},
                    'clipped_vs_fp32_weighted_gradient':relation(clipped,weighted),
                    'moments_before':moments_before,'moments_after':moment_norms(optimizer),
                    'interpretation':'dot(gradiente, delta AdamW) positivo prevê aumento LOCAL da loss; perdas antes/depois medem o efeito real dessa única atualização. Sondas pequenas, não prova global.'})
                del delta,weights_before,weighted,clipped,directions
                gc.collect()
                probe_seconds+=time.monotonic()-t
            history.append({'step':completed,'learning_rate':lr,'grad_norm_before_clipping':float(norm),'objectives':objectives})
            if completed==1 or completed%20==0:
                print(arm,seed,completed,'/120','treino',round(time.monotonic()-started-probe_seconds,1),'s',flush=True)
        training_seconds=time.monotonic()-started-probe_seconds
        final=weight_hash(model)
        value={'format':'TAC-chat-v2','task':'TAC-conversation-sft-v2','model_config':vars(model.cfg),
            'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'tokenizer_state':data.tokenizer.state(),
            'dataset_hash':data.digest,'step':completed,'research_only':True,'parent':PROTOCOL['parent'],
            'protocol':{**PROTOCOL,'steps':steps},'arm':arm,'seed':seed}
        checkpoint=folder/'candidate.pt'
        with gate.writing():gate.check();torch.save(value,checkpoint)
        checkpoint_sha=sha(checkpoint);del value
        print('avaliando',arm,seed,flush=True)
        evaluation=engine.all(model,data,device,novel)
        checks=protection_checks(evaluation,baseline)
        result={'status':'completed','arm':arm,'seed':seed,'protocol':{**PROTOCOL,'steps':steps},'steps_completed':completed,
            'initial_weight_hash':initial,'final_weight_hash':final,'checkpoint_sha256':checkpoint_sha,
            'checkpoint_kept':seed==42,'checkpoint_resume':'pesos apenas; não contém momentos para retomada exata',
            'weight_change':weight_delta(model,teacher),'plan':plan,'exposure':exposure.report(),'history':history,
            'gradients_and_adam':telemetry,'evaluation':evaluation,'protection_checks':checks,'passed_original_checks':all(checks.values()),
            'promoted':False,'test_final_evaluated':False,'training_seconds':training_seconds,'probe_seconds':probe_seconds,
            'total_seconds':time.monotonic()-started,'device':str(device),
            'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None,
            'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,'resources':resources}
        atomic_json(folder/'result.json',result)
        if seed!=42:
            # Exclusivamente este artefato NOVO, criado nesta execução, abaixo de sua pasta.
            resolved=checkpoint.resolve()
            if not resolved.is_relative_to(folder.resolve()) or resolved.name!='candidate.pt':raise ValueError('Caminho temporário inválido.')
            resolved.unlink()
        print('concluído',arm,seed,'básicas',evaluation['basic']['exact_matches'], 'importadas',evaluation['imported']['exact_matches'],flush=True)
        return result
    except BaseException as exc:
        atomic_json(folder/'incomplete.json',{'status':'incomplete','error':str(exc),'steps_completed':completed,
            'arm':arm,'seed':seed,'protocol':PROTOCOL,'history':history,'resources_at_failure':current_resources(),'promoted':False})
        raise
    finally:
        del optimizer,model,teacher,batches
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--seeds',nargs='+',type=int,default=PROTOCOL['seeds'])
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--output',type=Path,default=OUT,help='Pasta nova dentro de TAC/reports para reproduzir sem sobrescrever a rodada existente.')
    args=parser.parse_args()
    if args.steps!=120:parser.error('O protocolo congelado usa exatamente 120 passos.')
    if len(set(args.seeds))!=len(args.seeds) or any(s not in PROTOCOL['seeds'] for s in args.seeds):
        parser.error('Escolha seeds distintas entre 42, 417 e 903, conforme o protocolo congelado.')
    OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/'reports') or OUT==ROOT/'reports':parser.error('A pasta de saída deve ficar dentro de TAC/reports.')
    OUT.mkdir(parents=True,exist_ok=True);config=LiveSettings(ROOT)
    # Mesmo lock do desktop: não coexistir com treino automático/interface.
    app=QCoreApplication.instance() or QCoreApplication([]);lock=QLockFile(str(config.data/'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):raise RuntimeError('Feche a TAC para executar pilotos isolados, sem treinamento simultâneo.')
    try:
        gate=LearningGate(config,None);gate.begin();check_resources(config)
        torch.set_num_threads(config.get('cpu_threads'));torch.use_deterministic_algorithms(True)
        device=torch.device('cuda' if torch.cuda.is_available() and config.get('device')!='cpu' else 'cpu')
        if device.type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.cuda.set_per_process_memory_fraction(min(.75,config.get('vram_mb')*2**20/torch.cuda.get_device_properties(0).total_memory))
        if not (OUT/'preservacao_antes.json').exists():
            spec=importlib.util.spec_from_file_location('controlled_readonly_audit',ROOT/'reports/investigacao_0130/audit.py')
            audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            atomic_json(OUT/'preservacao_antes.json',audit.state())
        before=read(OUT/'preservacao_antes.json')
        if before['active']['active_conversation']['id']!=PROTOCOL['parent']:
            raise ValueError('O modelo ativo mudou desde o protocolo original; congele um protocolo novo para essa versão.')
        if sha(ROOT/'config.json')!=before['config_sha256']:raise ValueError('Configuração mudou desde o congelamento; não executar comparação antiga.')
        path=config.data/'checkpoints'/(PROTOCOL['parent']+'.pt')
        if sha(path)!=before['active']['active_conversation']['sha256']:raise ValueError('Checkpoint inicial mudou.')
        cp=load_checkpoint(path);data=ConversationData.load(config.data/'datasets'/f'conversation_{DATASET}.json')
        old=ConversationData.load(config.data/'datasets'/f'conversation_{OLD}.json');full=full_retention(data,old)
        replay=TextReplay(config.data/'datasets'/(CORPUS+'.json'),data.tokenizer)
        objective=ResponseObjective(replay,lessons()['train'],data.tokenizer)
        novel=read(ROOT/'reports/research_0120/novel_development.json')
        protocol={**PROTOCOL,'arms':list(ARMS),'device':str(device),'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None,
            'full_retention_eligible':len(full),'full_retention_original':len(old.sets['train']),
            'current_retention':len([r for r in data.sets['train'] if r.get('mixture','legacy')=='legacy']),
            'response_objective_changed_rows':objective.changed_rows,'response_objective_boundaries':objective.boundaries,
            'full_retention_by_skill':dict(Counter(r['skill'] for r in full)),
            'imported_categories_available':dict(Counter(category(r) for r in data.sets['train'] if r.get('mixture')=='imported')),
            'development_suite_sha256':sha(ROOT/'reports/research_0120/novel_development.json'),
            'sealed_sha256':sha(ROOT/'reports/research_0120/sealed_transfer.json')}
        if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=protocol:raise ValueError('Protocolo mudou; não sobrescrever.')
        atomic_json(OUT/'protocol.json',protocol)
        engine=DevelopmentEvaluator(config,None,gate)
        if (OUT/'active.json').exists():baseline=read(OUT/'active.json')['evaluation']
        else:
            print('Avaliando ativo: somente desenvolvimento.',flush=True)
            model=model_from(cp,data.tokenizer,device)
            baseline=engine.all(model,data,device,novel)
            atomic_json(OUT/'active.json',{'parent':PROTOCOL['parent'],'weight_hash':weight_hash(model),
                'evaluation':baseline,'promoted':False,'test_final_evaluated':False})
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        for seed in args.seeds:
            for arm in args.arms:
                pilot(config,gate,engine,cp,data,objective,full,baseline,novel,device,arm,seed,args.steps,OUT/f'{arm}_s{seed}')
    finally:lock.unlock()


if __name__=='__main__':main()
