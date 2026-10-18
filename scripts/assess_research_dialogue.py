"""Seleciona por validação, aplica critérios anteriores e preserva teste final cego."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_dialogue import load_context,model_from,FOLDER
from tac.config import atomic_json
from tac.neural.conversation import load_checkpoint,context_ready,natural_ready
from tac.fluency_lessons import fluency_ready
from tac.neural.transfer_eval import freeze_suite,evaluate_once


def full_validation(core,model,data,device):
    engine=core.conversation
    return {'basic':engine.evaluate(model,data,device,generation=True,interruptible=True),
        'context':engine.evaluate(model,data,device,split='validation_context',generation=True,interruptible=True),
        'natural':engine.evaluate(model,data,device,split='validation_natural',generation=True,interruptible=True),
        'sessions':engine.evaluate_sessions(model,data,device,interruptible=True),
        'fluency':engine.evaluate_fluency(model,data,device,interruptible=True)}


def old_checks(before,after):
    unknown=after['basic']['by_skill'].get('unknown')
    return {'treino_concluido':True,'loss_basica':after['basic']['loss']<before['basic']['loss']*1.02,
        'respostas_basicas':after['basic']['exact_accuracy']>=max(.60,before['basic']['exact_accuracy']-.02),
        'desconhecimento':not unknown or unknown['correct']/max(1,unknown['total'])>=.8,
        'contexto_anterior':context_ready(after['context']),
        'dialogos_e_correcoes':natural_ready(after['natural'],before['natural']),
        'sessoes_com_respostas_proprias':after['sessions']['immediate_accuracy']>=.8 and after['sessions']['recall_accuracy']>=.8,
        'conversas_revisadas':fluency_ready(after['fluency'],before['fluency'])}


def main():
    if (FOLDER/'novel_development.consumed.json').exists():
        raise FileExistsError('Esta rodada já foi avaliada. Preserve os resultados; não reutilize a avaliação para ajustar o modelo.')
    core,params,data,parent,cp=load_context(); core.gate.begin()
    baseline=json.loads((FOLDER/'development_baseline.json').read_text(encoding='utf-8'))
    records=[]
    for path in sorted(FOLDER.glob('*/result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        current=result['validation']['validation']; reference=baseline['validation']['validation']
        protected=current['loss']<reference['loss']*1.02 and current['exact_accuracy']>=reference['exact_accuracy']-.02
        imported=result['validation']['validation_imported']
        macro=sum(v['correct']/max(1,v['total']) for v in imported['by_skill'].values())/len(imported['by_skill'])
        records.append({'name':path.parent.name,'learning_rate':result['learning_rate'],'basic_loss':current['loss'],
            'basic_exact':current['exact_matches'],'basic_examples':current['examples'],'imported_exact':imported['exact_matches'],
            'imported_examples':imported['examples'],'imported_macro_exact':macro,'imported_loss':imported['loss'],
            'screen_protected':protected,'training_seconds':result['training_seconds'],'vram_mb':result['peak_vram_mb']})
    eligible=[r for r in records if r['screen_protected']]
    selected=min(eligible,key=lambda r:(-r['imported_macro_exact'],r['imported_loss'],r['basic_loss'])) if eligible else min(
        records,key=lambda r:(-r['basic_exact'],r['basic_loss'],-r['imported_macro_exact']))
    atomic_json(FOLDER/'selection.json',{'candidates':records,'selected_for_diagnosis':selected['name'],
        'eligible_screen':len(eligible),'test_used':False,'promoted':False,
        'reason':'Validação importada só escolhe entre versões protegidas. Sem elegíveis: candidato menos regressivo apenas para diagnóstico.'})
    print('Selected for diagnosis:',selected['name'],flush=True)
    device=core.trainer.device(); teacher=model_from(cp,data,device)
    candidate_cp=load_checkpoint(FOLDER/selected['name']/'candidate.pt')
    candidate=model_from(candidate_cp,data,device)
    print('Full baseline validation',flush=True); before=full_validation(core,teacher,data,device)
    print('Full candidate validation',flush=True); after=full_validation(core,candidate,data,device)
    checks=old_checks(before,after)
    atomic_json(FOLDER/'full_validation.json',{'parent':parent,'candidate':selected['name'],'before':before,'after':after,
        'checks':checks,'approved':all(checks.values()),'promoted':False,'test_imported_used':False})
    development=FOLDER/'novel_development.json'
    freeze_suite(development,data,927613,'development')
    print('One-shot novel development + decoding comparison; sealed test stays untouched',flush=True)
    evaluate_once(development,[('active',teacher),(selected['name'],candidate)],device,core.gate)
    atomic_json(FOLDER/'summary.json',{'candidates':records,'chosen':selected['name'],'approved':all(checks.values()),
        'checks':checks,'promoted':False,'sealed_test_consumed':False,
        'scope':'Pilotos curtos, um seed, até 120/300 passos do warmup. Não demonstram resultados de 6000 passos ou fluência geral.'})


if __name__=='__main__':main()
