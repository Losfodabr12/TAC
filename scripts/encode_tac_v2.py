"""Tokenizer próprio novo e tokens locais; nenhum peso antigo é carregado."""
import array
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.v2_data import load_jsonl,prompt_text,fingerprint
BASE=ROOT/'data/tac_v2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()


def save(path,value):
    with path.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    if (BASE/'encoded_ready.json').exists():print('Tokens novos já preparados; não repetidos.');return
    ready=json.loads((BASE/'corpus_ready.json').read_text(encoding='utf-8'))
    for name,meta in ready['files'].items():assert sha(BASE/name)==meta['sha256']
    rng=random.Random(2026101702);texts=[];chars=0
    for row in load_jsonl(BASE/'pretrain_train.jsonl'):
        if rng.random()<.04 and chars<1800000:texts.append(row['text']);chars+=len(row['text'])
    sft_chars=0
    for row in load_jsonl(BASE/'sft_train.jsonl'):
        if sft_chars<1000000:
            text=prompt_text(row['messages'])+' '+row['answer'];texts.append(text);sft_chars+=len(text)
    print('Treinando BPE local novo:',chars+sft_chars,'caracteres de treino, nenhum de validação...',flush=True)
    started=time.time();tok=LocalBPETokenizer.fit(texts,vocab_size=1024)
    assert tok.vocab_size==1024
    save(BASE/'tokenizer.json',tok.state());del texts
    print('BPE pronto em',round(time.time()-started,1),'segundos',flush=True)
    stats={'generation':'TAC-v2','context':1024,'tokenizer_from_scratch':True,
           'tokenizer_training_characters':chars+sft_chars,'tokenizer_validation_used':False,
           'tokenizer_sha256':sha(BASE/'tokenizer.json'),'pretrain':{},'sft':{},'files':{},
           'old_dataset_or_weights_loaded':False}
    for split in ('train','validation'):
        tokens=0;documents=0
        path=BASE/f'language_{split}.u16'
        with path.open('wb') as out:
            for row in load_jsonl(BASE/f'pretrain_{split}.jsonl'):
                encoded=tok.encode(row['text']);array.array('H',encoded).tofile(out)
                documents+=1;tokens+=len(encoded)
                if documents%10000==0:print('Tokens',split,documents,'documentos',tokens,'tokens',flush=True)
        stats['pretrain'][split]={'documents':documents,'tokens_including_bos_eos':tokens,'dtype':'uint16 little-endian','bytes':path.stat().st_size}
        assert sys.byteorder=='little' and path.stat().st_size==tokens*2
        stats['files'][path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
        counts=Counter();sources=Counter();uniqueids=[];lengths=[]
        target=BASE/f'instruction_{split}.jsonl'
        with target.open('w',encoding='utf-8') as out:
            for row in load_jsonl(BASE/f'sft_{split}.jsonl'):
                prefix=[tok.bos]+tok.encode(prompt_text(row['messages']),special=False)
                answer=tok.encode(' '+row['answer'],special=False)+[tok.eos]
                sequence=prefix+answer
                if len(sequence)-1>1024:
                    counts['excluded_over_1024_without_truncating']+=1;continue
                row={**row,'sequence':sequence,'prefix_length':len(prefix),'supervised_tokens':len(answer)}
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts['examples']+=1;counts['supervised_tokens']+=len(answer);counts['input_tokens']+=len(sequence)-1
                sources[row['source']]+=1;uniqueids.append(row['source']+':'+row['source_id']);lengths.append(len(sequence)-1)
        stats['sft'][split]={**dict(counts),'by_source':dict(sources),'maximum_input_length':max(lengths),
                            'mean_input_length':sum(lengths)/len(lengths),'unique_source_examples':len(set(uniqueids)),
                            'truncations':0}
        stats['files'][target.name]={'sha256':sha(target),'bytes':target.stat().st_size}
    assert stats['sft']['train']['examples']>0 and stats['sft']['validation']['examples']>0
    # Confere componentes, hashes de documento e domínios sem decodificar testes.
    train_domains={r['domain'] for r in load_jsonl(BASE/'pretrain_train.jsonl')}
    val_domains={r['domain'] for r in load_jsonl(BASE/'pretrain_validation.jsonl')}
    train_hashes={r['text_sha256'] for r in load_jsonl(BASE/'pretrain_train.jsonl')}
    val_hashes={r['text_sha256'] for r in load_jsonl(BASE/'pretrain_validation.jsonl')}
    tr=list(load_jsonl(BASE/'sft_train.jsonl'));va=list(load_jsonl(BASE/'sft_validation.jsonl'))
    train_components={r['split_component'] for r in tr};val_components={r['split_component'] for r in va}
    train_prompts={fingerprint(prompt_text(r['messages'])) for r in tr}
    val_prompts={fingerprint(prompt_text(r['messages'])) for r in va}
    stats['split_checks']={'domain_overlap':len(train_domains&val_domains),'document_hash_overlap':len(train_hashes&val_hashes),
                           'sft_component_overlap':len(train_components&val_components),'sft_prompt_overlap':len(train_prompts&val_prompts)}
    assert not any(stats['split_checks'].values())
    stats['dataset_hash']=hashlib.sha256(json.dumps(stats['files'],sort_keys=True).encode()).hexdigest()
    save(BASE/'encoded_ready.json',stats)
    print('ENCODED READY',json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
