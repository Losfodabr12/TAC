"""Obtém amostra limitada das três fontes públicas, sem modelos externos."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=ROOT/'data/tac_v2'
for key,value in {'HF_HOME':str(BASE/'cache'),'HF_DATASETS_CACHE':str(BASE/'cache/datasets'),
                  'HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_XET':'1','DO_NOT_TRACK':'1',
                  'HF_HUB_DOWNLOAD_TIMEOUT':'90'}.items():os.environ[key]=value
from tac.neural.v2_data import clean,problem,canonical,fingerprint,NearDedup,split_group,oasst_examples,prompt_text,load_jsonl

REPOS={'fineweb':('HuggingFaceFW/fineweb-2','af9c13333eb981300149d5ca60a8e9d659b276b9','ODC-By-1.0'),
       'aya':('CohereLabs/aya_dataset','f9ea04583f02a8f86404ff6c58bf75fe637df8a2','Apache-2.0'),
       'oasst':('OpenAssistant/oasst2','179dd21fc55192153d94adb0e0ce8f69e222bf75','Apache-2.0')}


def save(path,value):
    with path.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'TAC-v2-local-dataset-preparation/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-mib',type=int,default=128)
    args=parser.parse_args();BASE.mkdir(exist_ok=True);(BASE/'licenses').mkdir(exist_ok=True)
    if (BASE/'corpus_ready.json').exists():print('Corpus pronto já existente; nenhuma aquisição repetida.');return
    from datasets import load_dataset
    counters=Counter();start=time.time();sourceinfo={}
    for short,(repo,rev,license) in REPOS.items():
        card=fetch(f'https://huggingface.co/datasets/{repo}/raw/{rev}/README.md')
        (BASE/'licenses'/f'{short}_README.md').write_bytes(card)
        sourceinfo[short]={'repo':repo,'revision':rev,'license':license,'card_sha256':hashlib.sha256(card).hexdigest(),
                           'url':'https://huggingface.co/datasets/'+repo}
    for name,url in [('Apache-2.0.txt','https://www.apache.org/licenses/LICENSE-2.0.txt'),
                     ('ODC-By-1.0.html','https://opendatacommons.org/licenses/by/1-0/'),
                     ('CommonCrawl-terms.html','https://commoncrawl.org/terms-of-use')]:
        (BASE/'licenses'/name).write_bytes(fetch(url))
    save(BASE/'sources.json',sourceinfo)
    sft=[]
    # Somente train. Não baixa demographics nem testes de Aya.
    repo,rev,_=REPOS['aya']
    print('AYA: streaming do treino; filtro por português...',flush=True)
    stream=load_dataset(repo,name='default',split='train',revision=rev,streaming=True)
    for i,row in enumerate(stream):
        counters['aya_rows_scanned']+=1
        if row.get('language_code')!='por':continue
        counters['aya_portuguese_rows']+=1
        p,a=clean(row['inputs']),clean(row['targets']);reason=problem(p) or problem(a)
        if reason:counters['aya_'+reason]+=1;continue
        sft.append({'messages':[['user',p]],'answer':a,'source':repo,'source_id':str(i),
                    'group':'prompt:'+fingerprint(p),'annotation_type':row.get('annotation_type'),'skill':'aya_instruction'})
    print('AYA: aceitos provisórios',len(sft),flush=True)
    repo,rev,_=REPOS['oasst'];pt=[]
    print('OASST2: streaming de train; reconstrução de árvores...',flush=True)
    for row in load_dataset(repo,split='train',revision=rev,streaming=True):
        counters['oasst_rows_scanned']+=1
        if row.get('lang') in ('pt','pt-BR','pt-PT'):
            pt.append({k:v for k,v in row.items() if k not in ('user_id','emojis','detoxify','created_date')})
    counters['oasst_portuguese_messages']=len(pt);sft.extend(oasst_examples(pt,counters));del pt;gc.collect()
    # Remove conflitos exatos de alvo, em vez de escolher o que aumenta a nota.
    answers={};conflicts=set();unique=[];seen=set()
    for r in sft:
        key=fingerprint(prompt_text(r['messages']));target=canonical(r['answer'])
        if key in answers and answers[key]!=target:conflicts.add(key)
        answers[key]=target
    for r in sft:
        key=fingerprint(prompt_text(r['messages']))
        if key in conflicts:counters['sft_conflicting_prompt_removed']+=1;continue
        if key in seen:counters['sft_duplicate_prompt_removed']+=1;continue
        seen.add(key);unique.append(r)
    # Atribui componentes conectados de prompts e árvores a um único split.
    # Repetições exatas de uma fala entre árvores unem as duas, evitando vazamento.
    parent={r['group']:r['group'] for r in unique}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    textgroups={}
    for r in unique:
        for role,text in r['messages']:
            if role=='user' and len(text)>35:
                key=fingerprint(text)
                if key in textgroups:union(r['group'],textgroups[key])
                else:textgroups[key]=r['group']
    for r in unique:r['split']=split_group(find(r['group']));r['split_component']=find(r['group'])
    counts=Counter((r['source'],r['split']) for r in unique)
    for split in ('train','validation'):
        with (BASE/f'sft_{split}.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for r in unique:
                if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert any(r['split']=='validation' for r in unique) and any(r['split']=='train' for r in unique)
    sft_shingles=set()
    for r in unique:
        for text in [t for _,t in r['messages']]+[r['answer']]:
            words=canonical(text).split()
            sft_shingles.update(hashlib.blake2b(' '.join(words[i:i+20]).encode(),digest_size=8).digest() for i in range(0,max(0,len(words)-19),10))
    print('SFT pronto:',dict(counts),flush=True)
    # Nenhum arquivo parquet completo é baixado: FineWeb2 usa stream de por_Latn/train.
    repo,rev,_=REPOS['fineweb'];dedup=NearDedup();domains=Counter();bytes_written=0;accepted=Counter()
    files={split:(BASE/f'pretrain_{split}.jsonl').open('w',encoding='utf-8',newline='\n') for split in ('train','validation')}
    try:
        stream=load_dataset(repo,name='por_Latn',split='train',revision=rev,streaming=True)
        # Ordem de shards aleatória reproduzível com buffer limitado, sem materializar corpus.
        stream=stream.shuffle(seed=20261017,buffer_size=256)
        for row in stream:
            counters['fineweb_rows_scanned']+=1
            if counters['fineweb_rows_scanned']%500==0:
                print('FineWeb2',counters['fineweb_rows_scanned'],'lidos;',sum(accepted.values()),'aceitos;',round(bytes_written/2**20,1),'MiB',flush=True)
                if shutil.disk_usage(ROOT).free<900*2**20:raise OSError('Menos de 900 MiB livres: aquisição limitada para preservar o Windows')
            if float(row.get('language_score',0))<.9:counters['fw_language_score']+=1;continue
            text=clean(row['text']);reason=problem(text,document=True)
            if reason:counters['fw_'+reason]+=1;continue
            url=urlsplit(row.get('url',''));domain=(url.hostname or '').lower()
            if not domain or domains[domain]>=250:counters['fw_domain_cap']+=1;continue
            words=canonical(text).split()
            if any(hashlib.blake2b(' '.join(words[i:i+20]).encode(),digest_size=8).digest() in sft_shingles for i in range(max(0,len(words)-19))):
                counters['fw_overlap_with_sft']+=1;continue
            if dedup.duplicate(text):counters['fw_duplicate_or_near_duplicate']+=1;continue
            split=split_group('domain:'+domain);domains[domain]+=1
            record={'text':text,'source':repo,'revision':rev,'document_id':row['id'],
                    'url':urlunsplit((url.scheme,url.netloc,url.path,'','')),'domain':domain,'dump':row.get('dump'),
                    'language_score':row.get('language_score'),'text_sha256':fingerprint(text),'split':split}
            line=json.dumps(record,ensure_ascii=False)+'\n';size=len(line.encode('utf-8'))
            if bytes_written+size>args.max_mib*2**20:break
            files[split].write(line);accepted[split]+=1;bytes_written+=size
            if counters['fineweb_rows_scanned']>=120000:break
    finally:
        for f in files.values():f.close()
    assert min(accepted.values())>0,'FineWeb precisa de train e validação'
    payload=[BASE/f'{kind}_{split}.jsonl' for kind in ('pretrain','sft') for split in ('train','validation')]
    manifest={'generation':'TAC-v2','status':'CORPUS_PREPARADO','sources':sourceinfo,'acquisition_seconds':time.time()-start,
              'max_sample_mib':args.max_mib,'pretrain_docs':dict(accepted),'pretrain_jsonl_bytes':bytes_written,
              'sft_examples_by_source_and_split':{source+':'+split:n for (source,split),n in counts.items()},
              'filter_counters':dict(counters),'domains':len(domains),'top_domains':domains.most_common(15),
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload},
              'split_policy':'FineWeb: domínio inteiro no mesmo split; SFT: componentes conectados de árvores/prompts inteiros; 3% por hash fixo',
              'dedup_policy':'normalização NFC/whitespace/case, hash exato, candidatos SimHash+Jaccard>=.85 em FineWeb; prompts conflitantes removidos; sobreposição textual SFT/FineWeb de 20 palavras removida',
              'privacy':'user_id/demographics não armazenados; e-mail, telefone, CPF/CNPJ, endereço pessoal e segredos filtrados por regras; nomes próprios públicos não são automaticamente PII',
              'limitations':['Filtros heurísticos não certificam ausência de toda PII, erro factual ou paráfrase duplicada.',
                             'Aya pode conter reanotações humanas de textos gerados anteriormente; nenhum modelo/API foi usado por esta execução.',
                             'Amostra de shards em streaming não é amostra uniforme de todo o universo FineWeb2.',
                             'Volume português de Aya/OASST2 é limitado pelo que as fontes contêm; não se inflaciona com exercícios antigos.'],
              'old_weights_deleted':False,'old_exercises_used':False,'external_model_or_inference_api_used':False,'reserved_tests_opened':False}
    save(BASE/'corpus_ready.json',manifest)
    print('CORPUS PREPARADO',json.dumps({k:manifest[k] for k in ('pretrain_docs','sft_examples_by_source_and_split','acquisition_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
