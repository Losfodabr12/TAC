"""Validação ampliada dos checkpoints já treinados, sem atualizações ou fontes."""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
from tac.neural.conversation import load_checkpoint
from tac.neural.model import ModelConfig,TACTransformer
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.decoding import decode_conversation,conversation_options
from tac.neural.conversation_training import score_output,token_f1
from tac.neural.v2_data import load_jsonl
from tac.neural.v2_training import model_hash
from tac.config import atomic_json
BASE=ROOT/'data/tac_v2_conversation_20261018';OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01'


def forbidden(*a,**kw):raise RuntimeError('Comparação somente de inferência')


@torch.inference_mode()
def main():
    torch.save=forbidden;torch.optim.AdamW.step=forbidden;torch.set_num_threads(4)
    run=json.loads((BASE/'result.json').read_text(encoding='utf-8'))
    candidates=[(0,run['reference'])]+[(e['step'],e['checkpoint']) for e in run['evaluations'] if e['step'] in (1000,4000)]
    pools=defaultdict(list)
    for r in load_jsonl(BASE/'curated_validation.jsonl'):
        category='arithmetic' if r['skill']=='arithmetic' else 'context' if r['source']=='TAC-extraction-v2' else 'reading' if r['skill']=='reading' else 'natural'
        pools[category].append(r)
    selected=[]
    for category,pool in sorted(pools.items()):
        random.Random(202610186).shuffle(pool)
        count=80 if category in ('arithmetic','context') else 60 if category=='reading' else len(pool)
        selected.extend((category,r) for r in pool[:count])
    retention=list(load_jsonl(BASE/'curated_retention.jsonl'))
    random.Random(202610187).shuffle(retention)
    selected.extend(('retention',r) for r in retention[:24])
    atomic_json(OUT/'expanded_suite.json',[{'category':category,'source':r['source'],'source_id':r['source_id'],'family':r.get('family')} for category,r in selected])
    summaries=[]
    for step,ident in candidates:
        path=ROOT/'data/checkpoints'/(ident+'.pt');file_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        cp=load_checkpoint(path);model=TACTransformer(ModelConfig(**cp['model_config'])).to('cuda')
        model.load_state_dict(cp['model']);model.text_tokenizer=LocalBPETokenizer.from_state(cp['tokenizer_state']);model.eval()
        before=model_hash(model);outputs=[];grouped=defaultdict(list)
        for i,(category,row) in enumerate(selected):
            result=decode_conversation(model,row['messages'],'cuda',max_tokens=96,generation_options=conversation_options('controlled',42))
            score=score_output(row,result['text'])
            score['loop']=result['stop_reason']=='repetition';score['length']=result['stop_reason']=='length'
            outputs.append({'category':category,'source':row['source'],'source_id':row['source_id'],'skill':row['skill'],
                'family':row.get('family'),'messages':row['messages'],'expected':row['answer'],'output':result,'score':score})
            grouped[category].append(score)
            if i%100==99:print(f'Validação ampliada: passo {step}, {i+1}/{len(selected)}',flush=True)
        summary={}
        for category,items in grouped.items():
            measured=[v['structured_correct'] for v in items if v['structured_correct'] is not None]
            summary[category]={'n':len(items),'structured_correct':sum(measured) if measured else None,
                'structured_accuracy':sum(measured)/len(measured) if measured else None,
                'reference_word_f1':sum(v['word_f1'] for v in items)/len(items),
                'loops':sum(v['loop'] for v in items),'length_stops':sum(v['length'] for v in items)}
        # Histórico produzido pela própria rede, como a interface, e sessão nova.
        sessions=[]
        for name,questions in [('topic_changes',['Opa, tudo bom?','O que é um lobo?','Quanto é 2 + 2?','Crie uma história curta.','Meu nome é Carlos. Qual é meu nome?']),
            ('name_update',['Meu nome é Arlete.','Qual é meu nome?','Agora meu nome é Noemi.','Qual é meu nome atual?'])]:
            history=[];turns=[]
            for q in questions:
                messages=[*history[-6:],['user',q]]
                result=decode_conversation(model,messages,'cuda',generation_options=conversation_options('controlled',42))
                turns.append({'question':q,'input_messages':messages,**result})
                history=[*messages,['assistant',result['text']]][-6:]
            sessions.append({'name':name,'turns':turns})
        fresh=decode_conversation(model,[['user','Qual é meu nome?']],'cuda',generation_options=conversation_options('controlled',42))
        assert model_hash(model)==before and hashlib.sha256(path.read_bytes()).hexdigest()==file_hash
        value={'step':step,'checkpoint':ident,'file_sha256':file_hash,'weights_hash':before,
               'summary':summary,'outputs':outputs,'self_generated_sessions':sessions,
               'new_session_without_previous_name':fresh,'parameters':model.parameter_count,
               'training_updates':0,'reserved_tests_opened':False,'inference_options':conversation_options('controlled',42)}
        atomic_json(OUT/f'expanded_{step:05d}.json',value)
        summaries.append({k:value[k] for k in ('step','checkpoint','summary','file_sha256','weights_hash')})
        atomic_json(OUT/'expanded_comparison.json',summaries)
        print(json.dumps(summaries[-1],ensure_ascii=False),flush=True)
        del model,cp;torch.cuda.empty_cache()


if __name__=='__main__':main()
