"""Bounded, attributed public corpus for a separately initialized TAC v3."""
import gzip
import hashlib
import heapq
import json
import os
import re
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = ROOT / 'data/tac_v3'
OUT = ROOT / 'reports/tac_v3/rodada_20261018_01'
BASE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
for k, v in {'HF_HOME': str(BASE/'cache'), 'HF_HUB_DISABLE_TELEMETRY': '1',
             'HF_HUB_DISABLE_XET': '1', 'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ[k] = v

from tac.config import Config, atomic_json
from tac.neural.v2_data import clean, canonical, fingerprint, problem, load_jsonl, prompt_text, PT
from tac.neural.v3 import V3Tokenizer


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(2**20): h.update(block)
    return h.hexdigest()


def check():
    cfg = Config(ROOT)
    if not cfg.get('learning_enabled'):
        raise RuntimeError('Aprendizado desligado: preparação suspensa')
    if (BASE/'PARAR_TREINO').exists():
        raise RuntimeError('Parada solicitada')


def write_line(f, row): f.write(json.dumps(row, ensure_ascii=False)+'\n')


def family(text):
    words = canonical(text).split()
    prefix = ' '.join(re.sub(r'\d+', '#', w) for w in words[:5])
    return 'question-family:'+prefix


def split(group):
    return 'validation' if int(hashlib.sha256(('TAC-v3-dev:'+group).encode()).hexdigest()[:8], 16)%100<4 else 'train'


class ApproxDedup:
    """Exact hashes plus bottom-32 word-shingle overlap; an estimate, not a proof."""
    def __init__(self): self.exact=set();self.bands=defaultdict(list);self.sketches=[]

    def duplicate(self, text):
        normalized = canonical(text)
        h=hashlib.sha256(normalized.encode()).digest()
        if h in self.exact:return True
        words=normalized.split()
        grams={int.from_bytes(hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=4).digest(),'little')
               for i in range(max(0,len(words)-4))}
        sketch=tuple(heapq.nsmallest(32,grams))
        candidates=set()
        for b in range(4):candidates.update(self.bands[(b,sketch[b*4:b*4+4])])
        for i in candidates:
            other=self.sketches[i]
            if len(set(sketch).intersection(other))/max(1,min(len(sketch),len(other)))>=.84:
                return True
        index=len(self.sketches);self.sketches.append(array('I',sketch));self.exact.add(h)
        for b in range(4):self.bands[(b,sketch[b*4:b*4+4])].append(index)
        return False


def source(repo, rev=None):
    import requests
    from huggingface_hub import HfApi
    revision=rev or HfApi().dataset_info(repo).sha
    response=requests.get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md',timeout=60)
    response.raise_for_status()
    folder=BASE/'licenses';folder.mkdir(exist_ok=True)
    path=folder/(repo.replace('/','__')+'_README.md')
    path.write_bytes(response.content)
    return {'repo':repo,'revision':revision,'card_sha256':sha(path),'url':'https://huggingface.co/datasets/'+repo}


def prepare_sft():
    from datasets import load_dataset
    info=source('Polygl0t/gigaverbo-v2-sft','86482227c6840198eb5ced5e5813bc23498e7c4a')
    info.update(license='Apache-2.0 as declared upstream',upstream_synthetic=True,
                external_inference_used_by_this_project=False,split='train')
    counts=Counter();seen={};conflicts=set()
    # New public examples, not the exercises used in the v2 continuation.
    for category, target in [('general',16000),('math',2000),('rewriting',1000),('summarization',1000)]:
        accepted=0
        stream=load_dataset(info['repo'],name=category,revision=info['revision'],split='train',streaming=True)
        for index,row in enumerate(stream.shuffle(seed=4317,buffer_size=512)):
            if index%200==0: check()
            counts['scanned_'+category]+=1
            messages=row['messages']
            if not messages or messages[-1]['role']!='assistant' or len(messages)%2:counts['roles']+=1;continue
            if any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):counts['roles']+=1;continue
            texts=[clean(m['content']) for m in messages]
            issue=next((problem(t) for t in texts if problem(t) not in (None,'too_short')),None)
            if issue:counts['rejected_'+issue]+=1;continue
            if not all(re.search(r'\w',t) for t in texts):continue
            combined=' '.join(texts)
            words=re.findall(r'\w+',combined.casefold())
            if len(words)>20 and sum(w in PT for w in words)/len(words)<.025:counts['language']+=1;continue
            if len(combined)>2800 or len(texts[-1])<12:counts['length']+=1;continue
            if any(s in combined.casefold() for s in ('<think>','</think>','as an ai','como uma ia','chatgpt','openai','escolha a opção','múltipla escolha','escolha entre')):
                counts['excluded_format']+=1;continue
            score=row.get('instruct_int_score')
            if score is not None and score<3:counts['quality_annotation']+=1;continue
            history=[[m['role'],t] for m,t in zip(messages[:-1],texts[:-1])]
            key=fingerprint(prompt_text(history));answer=texts[-1]
            if key in seen:
                if canonical(seen[key]['answer'])!=canonical(answer):conflicts.add(key)
                counts['duplicate_prompt']+=1;continue
            group=family(texts[0])
            seen[key]={'messages':history,'answer':answer,'source':info['repo'],'source_revision':info['revision'],
                       'source_id':category+':'+str(index),'skill':category,'group':group,'split':split(group),
                       'prompt_hash':key,'annotated_score':score}
            accepted+=1
            if accepted>=target:break
        print('SFT',category,accepted,flush=True)
    files={s:(BASE/f'sft_{s}.jsonl').open('x',encoding='utf-8') for s in ('train','validation')}
    distribution=Counter()
    try:
        for key,row in seen.items():
            if key in conflicts:continue
            write_line(files[row['split']],row);distribution[row['split']+':'+row['skill']]+=1
    finally:
        for f in files.values():f.close()
    atomic_json(BASE/'sft_ready.json',{'source':info,'counts':dict(counts),'distribution':dict(distribution),'conflicting_prompts_removed':len(conflicts)})
    return info


def prepare_language():
    from datasets import load_dataset
    import tldextract
    domain_of=tldextract.TLDExtract(suffix_list_urls=(),cache_dir=None)
    sources=[source('HuggingFaceFW/fineweb-2','af9c13333eb981300149d5ca60a8e9d659b276b9'),source('wikimedia/wikipedia')]
    sources[0].update(license='ODC-By-1.0; Common Crawl terms; underlying content rights remain',subset='por_Latn')
    sources[1].update(license='CC-BY-SA-3.0 and GFDL as declared in card',subset='20231101.pt')
    stats=Counter();domains=Counter();dedup=ApproxDedup();old=set()
    for p in (ROOT/'data/tac_v2').glob('pretrain_*.jsonl'):
        for row in load_jsonl(p):old.add(row['text_sha256'])
    files={s:gzip.open(BASE/f'pretrain_{s}.jsonl.gz','xt',encoding='utf-8',compresslevel=1) for s in ('train','validation')}
    started=time.monotonic()
    try:
        for info, target_mib in zip(sources,[768,256]):
            total=0;repo=info['repo']
            stream=load_dataset(repo,name=info['subset'],revision=info['revision'],split='train',streaming=True)
            for index,row in enumerate(stream.shuffle(seed=7319,buffer_size=128)):
                if index%200==0:check()
                stats[repo+':scanned']+=1
                text=clean(row['text']);issue=problem(text,document=True)
                if issue:stats['reject:'+issue]+=1;continue
                if fingerprint(text) in old:stats['reject:old_v2_document']+=1;continue
                if repo.startswith('HuggingFaceFW'):
                    if float(row.get('language_score',0))<.94:stats['reject:language_score']+=1;continue
                    parsed=urlsplit(row.get('url',''));host=parsed.hostname or ''
                    domain=domain_of(host).top_domain_under_public_suffix
                    if not domain or domain.endswith('wikipedia.org') or domains[domain]>=600:
                        stats['reject:domain_cap_or_wikipedia']+=1;continue
                    group='domain:'+domain
                    origin=urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'',''))
                else:
                    domain='wikipedia.org';group='wikipedia-title:'+canonical(row['title']);origin=row['url']
                if dedup.duplicate(text):stats['reject:duplicate_or_near']+=1;continue
                part=split(group);size=len(text.encode('utf-8'))
                if total+size>target_mib*2**20:break
                record={'text':text,'source':repo,'source_revision':info['revision'],'document_id':str(row['id']),
                        'url':origin,'group':group,'split':part,'text_sha256':fingerprint(text)}
                write_line(files[part],record);total+=size;domains[domain]+=1
                stats[repo+':'+part+':docs']+=1;stats[repo+':'+part+':utf8_bytes']+=size
                if stats[repo+':scanned']%10000==0:
                    atomic_json(BASE/'acquisition_live.json',dict(stats))
                    print('CORPUS',repo,round(total/2**20,1),'MiB',round(time.monotonic()-started),'s',flush=True)
            print('CORPUS_SOURCE_FINISHED',repo,round(total/2**20,2),'MiB',flush=True)
    finally:
        for f in files.values():f.close()
    atomic_json(BASE/'language_ready.json',{'sources':sources,'stats':dict(stats),'domains':len(domains),
                'top_domains':domains.most_common(20),'seconds':time.monotonic()-started,
                'split_policy':'All registrable domains in one split (offline PSL); Wikimedia normalized title groups; 4% hash holdout',
                'dedup':'Canonical SHA256 plus bottom-32 five-word shingle similarity >=.84 with four candidate bands; global across sources/splits',
                'limitations':['Heuristic filters do not guarantee absence of all PII, factual errors or duplicates.',
                    'Streaming shard prefix with bounded shuffle is not a uniform sample of the entire corpus.',
                    'Domain/title separation does not certify semantic independence of all facts.']})
    return sources


def documents(part):
    with gzip.open(BASE/f'pretrain_{part}.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def encode():
    def tokenizer_texts():
        total=0
        for row in documents('train'):
            yield row['text'];total+=len(row['text'].encode())
            if total>=128*2**20:break
        for row in load_jsonl(BASE/'sft_train.jsonl'):yield prompt_text(row['messages'])+' '+row['answer']
    print('TRAINING_LOCAL_TOKENIZER',flush=True)
    tokenizer=V3Tokenizer.fit(tokenizer_texts())
    if tokenizer.vocab_size!=8192:raise ValueError('Corpus insuficiente para vocabulário planejado')
    atomic_json(BASE/'tokenizer.json',tokenizer.state())
    counts=Counter();groups={};questions={}
    for part in ('train','validation'):
        with (BASE/f'language_{part}.u16').open('xb') as f:
            buffer=array('H')
            for i,row in enumerate(documents(part)):
                if i%1000==0:check()
                ids=tokenizer.encode(row['text']);buffer.extend(ids);counts['language_'+part+'_tokens']+=len(ids)
                if len(buffer)>2**20:buffer.tofile(f);buffer=array('H')
            buffer.tofile(f)
        groups[part]=set();questions[part]=set()
        with (BASE/f'instruction_{part}.jsonl').open('x',encoding='utf-8') as f:
            for row in load_jsonl(BASE/f'sft_{part}.jsonl'):
                prefix=[1]+tokenizer.encode(prompt_text(row['messages']),special=False)
                answer=tokenizer.encode(' '+row['answer'],special=False)+[2]
                if len(prefix)+len(answer)-1>1024:counts['sft_too_long_'+part]+=1;continue
                row.update(sequence=prefix+answer,prefix_length=len(prefix))
                write_line(f,row);groups[part].add(row['group']);questions[part].add(row['prompt_hash'])
                counts['sft_'+part+'_examples']+=1;counts['sft_'+part+'_target_tokens']+=len(answer)
    if groups['train']&groups['validation'] or questions['train']&questions['validation']:
        raise ValueError('Vazamento entre treino/validação')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in BASE.iterdir()
           if p.is_file() and (p.name.endswith('.u16') or p.name.startswith('instruction_') or p.name=='tokenizer.json')}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    atomic_json(BASE/'encoded_ready.json',{'generation':'TAC-v3','dataset_hash':digest,'counts':dict(counts),'files':files,
                'context':1024,'tokenizer_trained_on':'training partitions only','train_validation_prompt_overlap':0,
                'train_validation_family_overlap':0,'reserved_tests_opened':False,'external_inference_used':False})
    print('V3_READY',json.dumps(dict(counts)),flush=True)


def main():
    check()
    if (BASE/'encoded_ready.json').exists():print('Already prepared');return
    if not (BASE/'sft_ready.json').exists():prepare_sft()
    if not (BASE/'language_ready.json').exists():prepare_language()
    encode()


if __name__=='__main__':main()
