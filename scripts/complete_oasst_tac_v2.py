"""Completa a seleção pt-BR do OASST2 sem repetir a aquisição FineWeb2."""
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/tac_v2'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8',errors='replace')
os.environ['HF_HOME']=str(BASE/'cache');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from tac.neural.v2_data import load_jsonl,oasst_examples,prompt_text,fingerprint,canonical,split_group


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    from datasets import load_dataset
    meta=json.loads((BASE/'corpus_ready.json').read_text(encoding='utf-8'))
    if meta.get('oasst_portuguese_codes_verified'):print('OASST já completo.');return
    rows=[];counts=Counter()
    for r in load_dataset('OpenAssistant/oasst2',revision=meta['sources']['oasst']['revision'],split='train',streaming=True,
                          filters=[('lang','in',['pt','pt-BR','pt-PT'])]):
        rows.append({k:v for k,v in r.items() if k not in ('user_id','emojis','detoxify','created_date')})
    examples=list(oasst_examples(rows,counts));print('OASST pt/pt-BR/pt-PT:',len(rows),'mensagens;',len(examples),'exemplos; filtros',dict(counts),flush=True)
    allrows=[r for split in ('train','validation') for r in load_jsonl(BASE/f'sft_{split}.jsonl') if r['source']!='OpenAssistant/oasst2']+examples
    answers={};conflicts=set();unique=[];seen=set()
    for r in allrows:
        k=fingerprint(prompt_text(r['messages']));a=canonical(r['answer'])
        if k in answers and answers[k]!=a:conflicts.add(k)
        answers[k]=a
    for r in allrows:
        k=fingerprint(prompt_text(r['messages']))
        if k in conflicts:counts['additional_conflicting_prompts']+=1;continue
        if k in seen:counts['additional_duplicate_prompts']+=1;continue
        seen.add(k);unique.append(r)
    parent={r['group']:r['group'] for r in unique}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    bytext={}
    for r in unique:
        for role,text in r['messages']:
            if role=='user' and len(text)>35:
                key=fingerprint(text)
                if key in bytext:union(r['group'],bytext[key])
                else:bytext[key]=r['group']
    for r in unique:r['split_component']=find(r['group']);r['split']=split_group(r['split_component'])
    for split in ('train','validation'):
        with (BASE/f'sft_{split}.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for r in unique:
                if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Decontaminação adicional do novo material OASST, antes de qualquer treino.
    shingles=set()
    for r in examples:
        for text in [t for _,t in r['messages']]+[r['answer']]:
            words=canonical(text).split()
            shingles.update(hashlib.blake2b(' '.join(words[i:i+20]).encode(),digest_size=8).digest() for i in range(0,max(0,len(words)-19),10))
    docs=Counter();removed=0
    for split in ('train','validation'):
        path=BASE/f'pretrain_{split}.jsonl';temp=path.with_suffix('.new')
        with temp.open('x',encoding='utf-8',newline='\n') as f:
            for r in load_jsonl(path):
                words=canonical(r['text']).split()
                if any(hashlib.blake2b(' '.join(words[i:i+20]).encode(),digest_size=8).digest() in shingles for i in range(max(0,len(words)-19))):removed+=1;continue
                f.write(json.dumps(r,ensure_ascii=False)+'\n');docs[split]+=1
        os.replace(temp,path)
    meta['oasst_portuguese_codes_verified']=['pt','pt-BR','pt-PT']
    meta['filter_counters'].update(dict(counts));meta['filter_counters']['oasst_portuguese_messages']=len(rows)
    meta['filter_counters']['fineweb_additional_oasst_overlap_removed']=removed
    meta['pretrain_docs']=dict(docs)
    meta['sft_examples_by_source_and_split']={source+':'+split:n for (source,split),n in Counter((r['source'],r['split']) for r in unique).items()}
    meta['pretrain_jsonl_bytes']=sum((BASE/f'pretrain_{split}.jsonl').stat().st_size for split in ('train','validation'))
    for name in meta['files']:meta['files'][name]={'sha256':sha(BASE/name),'bytes':(BASE/name).stat().st_size}
    meta['preparation_note']='Código de língua efetivo do OASST2 é pt-BR. Corrigido antes do treino; FineWeb não foi baixado novamente.'
    with (BASE/'corpus_ready.json').open('w',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
    assert examples and any(len(r['messages'])>1 for r in examples)
    print('SFT FINAL',meta['sft_examples_by_source_and_split'],'FW sobreposição adicional removida',removed,flush=True)


if __name__=='__main__':main()
