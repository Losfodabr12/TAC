"""New derivatives only: preserve acquired rows; filter and add multi-turn dialogue."""
import ast
import json
import math
import operator
import random
import re
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prepare_tac_v3 import BASE,OUT,check,source,sha,family,split,write_line
from tac.config import atomic_json
from tac.neural.v2_data import clean,problem,prompt_text,fingerprint,canonical,load_jsonl
from tac.neural.v3 import V3Tokenizer
sys.stdout.reconfigure(encoding='utf-8',errors='replace')


def numeric(node):
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -numeric(node.operand)
    if isinstance(node,ast.BinOp):
        operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
        if type(node.op) in operations:return operations[type(node.op)](numeric(node.left),numeric(node.right))
    raise ValueError('Expression is not a numeric calculation')


def issue(row):
    text=prompt_text(row['messages'])+' '+row['answer']
    if re.search(r'\b(?:opções|opcoes|alternativas)\s*:|selecione (?:uma|a) (?:das|opção)|\b(?:a|A)\)\s*.+\b(?:b|B)\)',text):return 'multiple_choice'
    if re.search(r'\[(?:I|II|III|IV)\]|\b(?:premissa|hipótese)\s*:',text,re.I):return 'classification_or_broken_options'
    if re.search(r'\b(?:arredondaremos|arredondar(?:emos|emos)?|arredondando)\b',row['answer'],re.I) and row['skill']=='math':
        return 'rounding_word_problem_uncertain'
    if re.search(r'(?m)^(?:Usuário|TAC):',row['answer']):return 'embedded_role_delimiter'
    for expression,expected in re.findall(r'(?<![\w.])(-?\d+(?:[.,]\d+)?(?:\s*[+*/x×÷-]\s*-?\d+(?:[.,]\d+)?)+)\s*=\s*(-?\d+(?:[.,]\d+)?)(?![\w.])',row['answer']):
        try:
            value=numeric(ast.parse(expression.replace(',','.').replace('x','*').replace('×','*').replace('÷','/'),mode='eval').body)
            target=float(expected.replace(',','.'))
            if math.isfinite(value) and abs(value-target)>max(.02,abs(value)*.002):return 'inconsistent_numeric_equality'
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    return None


def fetch():
    from datasets import load_dataset
    if (BASE/'curation_raw_ready.json').exists():return
    info=source('nicholasKluge/instruct-aira-dataset-v3','1ca52371453cc9e52b650ba4314f2ca0a69f58ba')
    info.update(license='Apache-2.0 as declared upstream',split='portuguese (language partition, not a final test)',upstream_synthetic=True)
    seen={};conflicts=set();counts=Counter()
    def accept(row):
        reason=issue(row)
        if reason:counts['reject:'+reason]+=1;return False
        key=fingerprint(prompt_text(row['messages']))
        if key in seen:
            if canonical(seen[key]['answer'])!=canonical(row['answer']):conflicts.add(key)
            counts['duplicate_prompt']+=1;return False
        seen[key]=row;return True
    for part in ('train','validation'):
        for row in load_jsonl(BASE/f'sft_{part}.jsonl'):accept(row)
    stream=load_dataset(info['repo'],revision=info['revision'],split='portuguese',streaming=True)
    accepted=0
    for index,session in enumerate(stream.shuffle(seed=43175,buffer_size=128)):
        if index%100==0:check()
        messages=session['conversations'];texts=[clean(m['content']) for m in messages]
        if any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):counts['aira_roles']+=1;continue
        if any(problem(t) not in (None,'too_short') for t in texts):counts['aira_quality_or_private']+=1;continue
        if any(s in ' '.join(texts).casefold() for s in ('chatgpt','openai','como uma ia','sou a aira','sou a assistente aira')):
            counts['aira_identity_or_format']+=1;continue
        group=family(texts[0])
        for end in range(2,min(len(messages),8)+1,2):
            # Complete prefixes, never a chopped history with missing antecedents.
            if sum(len(t) for t in texts[:end])>4000:break
            row={'messages':[[m['role'],t] for m,t in zip(messages[:end-1],texts[:end-1])],
                 'answer':texts[end-1],'source':info['repo'],'source_revision':info['revision'],
                 'source_id':session['conversation_id']+':'+str(end),'conversation_id':session['conversation_id'],
                 'group':group,'split':split(group),'skill':'natural_multiturn' if end>2 else 'natural_instruction'}
            row['prompt_hash']=fingerprint(prompt_text(row['messages']))
            if accept(row):accepted+=1
        if accepted>=8000:break
    print('AIRA examples accepted',accepted,flush=True)
    files={s:(BASE/f'curated_sft_{s}.jsonl').open('x',encoding='utf-8') for s in ('train','validation')}
    distribution=Counter()
    try:
        for key,row in seen.items():
            if key in conflicts:continue
            write_line(files[row['split']],row);distribution[row['split']+':'+row['skill']]+=1
    finally:
        for f in files.values():f.close()
    atomic_json(BASE/'curation_raw_ready.json',{'new_source':info,'filters':dict(counts),'distribution':dict(distribution),
                'conflicting_prompts_removed':len(conflicts),'factual_quality_certified':False,
                'limitations':['Public synthetic targets can contain factual errors even with upstream score 5.',
                    'Explicit numeric contradictions and questionable rounding filtered; no universal fact checker.',
                    'Question-family grouping uses normalized five-word prefixes, not semantic equivalence proof.']})
    print(json.dumps(dict(distribution)),flush=True)


def encode():
    if not (BASE/'encoded_ready.json').exists():raise RuntimeError('Prepare o corpus/tokenizer antes de codificar as instruções revisadas')
    if (BASE/'training_ready.json').exists():return
    meta=json.loads((BASE/'encoded_ready.json').read_text(encoding='utf-8'))
    tok=V3Tokenizer.from_state(json.loads((BASE/'tokenizer.json').read_text(encoding='utf-8')))
    counts=Counter();groups={};prompts={};conversations={}
    for part in ('train','validation'):
        groups[part]=set();prompts[part]=set();conversations[part]=set()
        with (BASE/f'instruction_curated_{part}.jsonl').open('x',encoding='utf-8') as f:
            for row in load_jsonl(BASE/f'curated_sft_{part}.jsonl'):
                prefix=[1]+tok.encode(prompt_text(row['messages']),special=False)
                answer=tok.encode(' '+row['answer'],special=False)+[2]
                if len(prefix)+len(answer)-1>1024:counts['too_long_'+part]+=1;continue
                row.update(sequence=prefix+answer,prefix_length=len(prefix))
                write_line(f,row);groups[part].add(row['group']);prompts[part].add(row['prompt_hash'])
                if row.get('conversation_id'):conversations[part].add(row['conversation_id'])
                counts[part+':'+row['skill']]+=1;counts[part+':examples']+=1;counts[part+':target_tokens']+=len(answer)
    assert not (groups['train']&groups['validation'] or prompts['train']&prompts['validation'] or conversations['train']&conversations['validation'])
    for part in ('train','validation'):
        p=BASE/f'instruction_curated_{part}.jsonl';meta['files'][p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
    meta['dataset_hash']=__import__('hashlib').sha256(json.dumps(meta['files'],sort_keys=True).encode()).hexdigest()
    meta['curated_sft_counts']=dict(counts);meta['instruction_filename']='instruction_curated_{split}.jsonl'
    meta['train_validation_conversation_overlap']=0
    atomic_json(BASE/'training_ready.json',meta)
    sample=random.Random(43177).sample(list(load_jsonl(BASE/'instruction_curated_train.jsonl')),40)
    atomic_json(OUT/'amostra_revisao_dados.json',[{k:v for k,v in r.items() if k not in ('sequence','prefix_length')} for r in sample])
    print('CURATED_READY',json.dumps(dict(counts)),flush=True)


if __name__=='__main__':
    check()
    if '--encode' in sys.argv:encode()
    else:fetch()
