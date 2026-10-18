"""Disponibiliza candidato experimental solicitado, sem aprová-lo por qualidade."""
import hashlib
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.core import Core
from tac.config import atomic_json
from tac.neural.v2_data import load_jsonl,prompt_text,fingerprint
BASE=ROOT/'data/tac_v2_conversation_20261018';OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=read(BASE/'result.json');reference=read(OUT/'reference.json');expanded=read(OUT/'expanded_comparison.json')
    # Os painéis anteriores permanecem intactos. Entre os candidatos comparados
    # na validação ampliada, o de 4.000 supera o de 1.000 nas quatro categorias,
    # mas continua abaixo da referência na medida de retenção.
    candidates=[r for r in expanded if r['step']>0]
    def rank(r):
        s=r['summary']
        return (.35*s['arithmetic']['structured_accuracy']+.35*s['context']['structured_accuracy']
                +.15*s['reading']['reference_word_f1']+.15*s['natural']['reference_word_f1']
                -.25*sum(x['loops'] for x in s.values())/sum(x['n'] for x in s.values()))
    selected=max(candidates,key=rank)
    assert selected['step']==4000
    cp=selected['checkpoint'];ref=reference['checkpoint']
    assert sha(ROOT/'data/checkpoints'/(ref+'.pt'))==reference['file_sha256']
    assert sha(ROOT/'data/checkpoints'/(cp+'.pt'))==selected['file_sha256']
    exposure=list(load_jsonl(BASE/'exposure.jsonl'));counts=Counter(i for e in exposure if e['kind']=='instruction' for i in e['indices'])
    assert [e['step'] for e in exposure]==list(range(1,result['step']+1))
    assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_norm_before_clip']) for e in exposure)
    assert len(counts)==result['unique_train_examples']
    ready=read(BASE/'ready.json');familysets={};promptsets={};checks={}
    for split in ('train','validation','retention'):
        rows=list(load_jsonl(BASE/('curated_'+split+'.jsonl')))
        checks[split]=sha(BASE/('curated_'+split+'.jsonl'))==ready['files'][split]
        familysets[split]={r['family'] for r in rows};promptsets[split]={fingerprint(prompt_text(r['messages'])) for r in rows}
    assert all(checks.values()) and not familysets['train']&familysets['validation'] and not promptsets['train']&promptsets['validation']
    # Corrige apenas contagens de metadados após a curadoria; conteúdo e hashes
    # usados no treino permanecem imutáveis.
    ready['families']={s:len(v) for s,v in familysets.items()}
    ready['statistics_note']='Family counts recomputed after curation; training/validation file hashes unchanged.'
    atomic_json(BASE/'ready.json',ready);atomic_json(OUT/'DADOS.json',ready);atomic_json(BASE/'sources.json',ready['sources'])
    old=read(ROOT/'data/tac_v2/encoded_ready.json')
    original_checks={name:sha(ROOT/'data/tac_v2'/name)==info['sha256'] for name,info in old['files'].items()}
    original_checks['tokenizer.json']=sha(ROOT/'data/tac_v2/tokenizer.json')==old['tokenizer_sha256']
    assert all(original_checks.values())
    deployment={'generation':'TAC-v2','checkpoint':cp,'step':selected['step'],'reference':ref,
        'passed_retention_gate':False,'quality_approved':False,'availability':'experimental_requested_by_user',
        'selection':'post-training expanded development generation; earlier acceptance scores unchanged',
        'selected_weights_hash':selected['weights_hash'],'reference_weights_hash':reference['weights_hash'],
        'warning':'Melhor candidato experimental comparado; continua com erros e regressão de retenção. Não aprovado como melhoria geral.'}
    core=Core(ROOT)
    with core.gate.writing():
        core.gate.check();core.conversation.activate(cp)
        core.db.execute("UPDATE checkpoints SET status='experimental_conversacao_selecionado' WHERE id=?",(cp,))
        atomic_json(BASE/'deployment.json',deployment)
    atomic_json(OUT/'SELECAO.json',deployment)
    validation={'status':'OK_execution_and_preservation_not_semantic_success','training_updates':result['step'],
        'budget_max_updates':result['max_steps'],'early_stopping':True,
        'early_stopping_reason':'No checkpoint passed frozen retention gate; at least three evaluations without eligible improvement after minimum 4000 updates',
        'training_run_id':result['run_id'],'parameters':10004544,'reference_preserved':True,'reference_file_sha256':reference['file_sha256'],
        'active_checkpoint':core.active_conversation_id(),'active_step_total':14247+selected['step'],
        'checkpoints_saved_this_run':len(core.db.query('SELECT id FROM checkpoints WHERE run_id=?',(result['run_id'],))),
        'training_examples_presented':result['examples'],'unique_examples_presented':len(counts),
        'presentations_per_example':{'min':min(counts.values()),'max':max(counts.values()),'distribution':dict(Counter(counts.values()))},
        'response_supervised_tokens':result['supervised_tokens'],'language_replay_tokens':result['language_tokens'],
        'finite_training_loss_and_gradient_norm':True,'non_finite_updates':0,'corpus_frozen_hash_checks':checks,
        'original_corpus_and_tokenizer_preserved':original_checks,'family_overlap_train_validation':0,
        'prompt_overlap_train_validation':0,'reserved_tests_opened':False,'external_AI_calls':0,
        'old_v1_exercises_used':False,'runtime_fixed_answer_lookup':False,'training_goal_coherent_general_conversation_achieved':False,
        'best_accepted_checkpoint':None,'experimental_available_with_explicit_warning':True,
        'session_isolation':'instances and clear-conversation isolated; cross-checkpoint history now resets; same-session neural carryover still observed',
        'expanded_generation_cases_per_checkpoint':262,'expanded_checkpoints_compared':[x['step'] for x in expanded],
        'manual_review_sample_rows':len(read(OUT/'amostra_revisao.json'))}
    atomic_json(OUT/'validacao_final.json',validation)
    print(json.dumps({'deployment':deployment,'validation':validation},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
