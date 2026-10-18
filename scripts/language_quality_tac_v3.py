"""Same unseen raw passages, likelihood per UTF-8 byte across tokenizers."""
import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tac.core import Core
from tac.config import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--label',required=True)
    args=parser.parse_args();core=Core(ROOT);torch.set_num_threads(4)
    target=ROOT/'reports/tac_v3/rodada_20261018_01'/('linguagem_'+args.label+'.json')
    if target.exists():return
    pools=defaultdict(list)
    with gzip.open(core.config.data/'tac_v3/pretrain_validation.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);pool=pools[row['source']]
            if len(pool)<128:pool.append(row)
            if len(pools)==2 and all(len(v)>=128 for v in pools.values()):break
    samples=[]
    for source,pool in pools.items():
        samples.extend(sorted(pool,key=lambda r:hashlib.sha256(r['document_id'].encode()).hexdigest())[:16])
    torch.cuda.set_per_process_memory_fraction(.5)
    model,cp=core.conversation.load(args.checkpoint,'cuda');tokenizer=model.text_tokenizer
    outputs=[];total_nll=total_bytes=0
    with torch.inference_mode():
        for row in samples:
            text=row['text'][:400].rsplit(' ',1)[0]
            ids=[1]+tokenizer.encode(text,special=False)+[2]
            if len(ids)-1>model.cfg.context:raise ValueError('Unexpected passage truncation')
            x=torch.tensor([ids[:-1]],device='cuda');y=torch.tensor([ids[1:]],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):_,loss=model(x,y)
            nll=float(loss)*(len(ids)-1);nbytes=len(text.encode('utf-8'))
            total_nll+=nll;total_bytes+=nbytes
            outputs.append({'source':row['source'],'document_id':row['document_id'],'url':row['url'],
                'text':text,'targets':len(ids)-1,'utf8_bytes':nbytes,'nll_sum':nll,'bits_per_byte':nll/math.log(2)/nbytes})
    atomic_json(target,{'checkpoint':args.checkpoint,'generation':cp['generation'],'bits_per_utf8_byte':total_nll/math.log(2)/total_bytes,
                       'passages':len(outputs),'truncation':False,'includes_eos':True,'outputs':outputs,
                       'limitation':'Small held-out development sample; likelihood does not establish conversational coherence.'})
    print(args.label,'bits per UTF-8 byte',total_nll/math.log(2)/total_bytes)


if __name__=='__main__':main()
