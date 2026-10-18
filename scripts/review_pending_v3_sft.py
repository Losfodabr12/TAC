"""Revise only unused SFT targets; retain every original and all language bytes."""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.neural.v2_data import load_jsonl,prompt_text,fingerprint
from tac.neural.v3 import V3Tokenizer
from tac.config import atomic_json
from prepare_tac_v3 import check,sha

BASE=ROOT/'data/tac_v3';OUT=ROOT/'reports/tac_v3/rodada_20261018_01'


def risk(row):
    if row['skill']=='math':return 'unverified_public_word_problem_category'
    q=row['messages'][-1][1].casefold();answer=row['answer'].casefold()
    if re.search(r'\d\s*[+*/×÷=]\s*\d|\b(?:calcule|quantos|quantas|equação|equações|aritmética|porcentagem|percentagem|multiplique|divida|subtraia|some)\b|problema matemático',q):
        return 'math_not_formally_verified'
    if row['source'].endswith('instruct-aira-dataset-v3'):
        content=' '.join(t for _,t in row['messages'])+' '+row['answer']
        if re.search(r'\b(?:hotel|hotéis|restaurante|restaurantes|pubs?|pousada)\b',content,re.I):
            return 'unverified_local_recommendations'
        if re.search(r'como (?:um )?modelo (?:de linguagem )?(?:de )?(?:ia|inteligência artificial)',answer):
            return 'identity_boilerplate'
    return None


def arithmetic():
    rng=random.Random(43179);seen=set()
    held={(7,8,'+'),(18,7,'-'),(6,4,'*'),(20,5,'/'),(5,9,'+'),(8,3,'-'),(2,2,'+')}
    def key(a,b,op):return (min(a,b),max(a,b),op) if op in '+*' else (a,b,op)
    held={key(*x) for x in held}
    # Preserve the established development prompts; exclude their computations.
    for part,count in [('train',1200),('validation',200)]:
        for index in range(count):
            while True:
                op=rng.choice(['+','-','*','/']);a=rng.randrange(0,51);b=rng.randrange(0,31)
                if op=='*':a%=13;b%=13
                if op=='/':b=b%12+1;a=b*rng.randrange(0,13)
                if op=='-' and b>a:a,b=b,a
                k=key(a,b,op)
                if k not in seen and k not in held:seen.add(k);break
            value={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a//b}[op]()
            symbol={'*':'×','/':'÷'}.get(op,op)
            if part=='train':
                questions=[f'Resolva a conta {a} {symbol} {b}.',f'Informe o resultado de {a} {symbol} {b}.',
                           f'Faça o cálculo: {a} {symbol} {b}.',f'Qual valor obtemos calculando {a} {symbol} {b}?']
            else:
                questions=[f'Determine o valor numérico da expressão {a} {symbol} {b}.',
                           f'Pode calcular esta expressão para mim: {a} {symbol} {b}?']
            question=questions[index%len(questions)]
            answers=[str(value),f'O resultado é {value}.',f'{a} {symbol} {b} = {value}.',
                     f'Calculando {a} {symbol} {b}, obtemos {value}.']
            yield {'messages':[['user',question]],'answer':answers[index%4],'skill':'verified_arithmetic',
                   'source':'TAC-v3-deterministic-arithmetic','source_id':part+':'+str(index),
                   'group':'arithmetic-template:'+part+':'+str(index%len(questions)),'split':part,
                   'verification':{'a':a,'b':b,'op':op,'result':value},'prompt_hash':fingerprint(prompt_text([['user',question]]))}


def main():
    check()
    if (BASE/'sft_reviewed_ready.json').exists():raise RuntimeError('Unused-SFT revision already exists')
    old=json.loads((BASE/'training_ready.json').read_text(encoding='utf-8'))
    progress=json.loads((BASE/'progress.json').read_text(encoding='utf-8'))
    if progress['phase']!='language':raise RuntimeError('Only revise pending SFT before its first update')
    # Checkpoint itself will be checked again at the controlled session boundary.
    rows={s:list(load_jsonl(BASE/f'instruction_curated_{s}.jsonl')) for s in ('train','validation')}
    bad_conversations=set();counters=Counter();rejected=[]
    for part,values in rows.items():
        for row in values:
            reason=risk(row)
            if reason and row.get('conversation_id'):bad_conversations.add(row['conversation_id'])
    tok=V3Tokenizer.from_state(json.loads((BASE/'tokenizer.json').read_text(encoding='utf-8')))
    supplemented=list(arithmetic());outfiles={}
    for part,values in rows.items():
        path=BASE/f'instruction_reviewed_{part}.jsonl';outfiles[part]=path
        with path.open('x',encoding='utf-8') as f:
            for row in values+[r for r in supplemented if r['split']==part]:
                reason=risk(row) if row['source']!='TAC-v3-deterministic-arithmetic' else None
                if not reason and row.get('conversation_id') in bad_conversations:reason='same_conversation_as_excluded_target'
                if reason:
                    counters['excluded:'+part+':'+reason]+=1
                    rejected.append({'split':part,'source_id':row['source_id'],'source':row['source'],'reason':reason})
                    continue
                if 'sequence' not in row:
                    prefix=[1]+tok.encode(prompt_text(row['messages']),special=False)
                    answer=tok.encode(' '+row['answer'],special=False)+[2]
                    row.update(sequence=prefix+answer,prefix_length=len(prefix))
                assert len(row['sequence'])<=1025 and row['sequence'][-1]==2
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                counters[part+':examples']+=1;counters[part+':'+row['skill']]+=1
                counters[part+':target_tokens']+=len(row['sequence'])-row['prefix_length']
    new=json.loads(json.dumps(old))
    new['parent_dataset_hash']=old['dataset_hash']
    new['revision_reason']='Verified upstream target defects; replace unused unverified math, filter local recommendations and identity boilerplate'
    new['instruction_filename']='instruction_reviewed_{split}.jsonl'
    for path in outfiles.values():new['files'][path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    new['dataset_hash']=hashlib.sha256(json.dumps(new['files'],sort_keys=True).encode()).hexdigest()
    new['reviewed_sft_counts']=dict(counters)
    new['unchanged_language_and_tokenizer']={k:old['files'][k] for k in ('language_train.u16','language_validation.u16','tokenizer.json')}
    new['transition_permitted_only_before_sft']=True
    # Verify overlap and fixed development protection before permitting use.
    parsed={s:list(load_jsonl(p)) for s,p in outfiles.items()}
    train_hashes={r['prompt_hash'] for r in parsed['train']}
    assert not train_hashes.intersection(r['prompt_hash'] for r in parsed['validation'])
    assert not {r['group'] for r in parsed['train']}.intersection(r['group'] for r in parsed['validation'])
    dev=json.loads((OUT/'avaliacao_dev.json').read_text(encoding='utf-8'))
    assert not [r for r in dev['cases'] if not r['known_probe'] and fingerprint(prompt_text(r['messages'])) in train_hashes]
    atomic_json(OUT/'revisao_sft_descartes.json',rejected)
    atomic_json(OUT/'revisao_sft_antes_do_uso.json',{'parent_dataset_hash':old['dataset_hash'],
                'new_dataset_hash':new['dataset_hash'],'counts':dict(counters),'weights_restarted':False,
                'pretraining_unchanged':True,'old_targets_presentations':0,'development_prompts_changed':False,
                'exclusion_does_not_mean_all_category_rows_were_false':True,
                'new_arithmetic':'1200 train + 200 dev; computed in Python; different templates and computation tuples; each training example twice',
                'limitations':'Remaining public targets are not universally fact-checked; filtering is conservative.'})
    atomic_json(BASE/'sft_reviewed_ready.json',new)
    print(json.dumps(dict(counters),indent=2),flush=True)


if __name__=='__main__':main()
