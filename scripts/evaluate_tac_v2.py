"""Geração livre de desenvolvimento após treino; não abre testes reservados."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
from tac.neural.model import ModelConfig,TACTransformer
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.v2_data import load_jsonl,prompt_text


def main():
    base=ROOT/'data/tac_v2';result=json.loads((base/'training_result.json').read_text(encoding='utf-8'))
    cp=torch.load(ROOT/'data/checkpoints'/(result['checkpoint']+'.pt'),map_location='cpu',weights_only=True)
    assert cp['generation']=='TAC-v2'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    model=TACTransformer(ModelConfig(**cp['model_config'])).to(device);model.load_state_dict(cp['model']);model.eval()
    tok=LocalBPETokenizer.from_state(cp['tokenizer_state']);probes=[]
    val=list(load_jsonl(base/'instruction_validation.jsonl'))
    from tac.neural.v2_training import V2Data
    from collections import defaultdict
    data=V2Data(base,device);totals=defaultdict(lambda:{'loss_times_tokens':0.,'supervised_tokens':0,'examples':0})
    for i,row in enumerate(val):
        x,y=data.instruction_batch([i],'validation')
        with torch.inference_mode():_,loss=model(x,y)
        n=int((y!=0).sum())
        for key in (row['source'],'all_validation','multi_turn' if len(row['messages'])>1 else 'single_turn'):
            totals[key]['loss_times_tokens']+=float(loss)*n;totals[key]['supervised_tokens']+=n;totals[key]['examples']+=1
    full_validation={k:{'loss':v['loss_times_tokens']/v['supervised_tokens'],'supervised_tokens':v['supervised_tokens'],'examples':v['examples']} for k,v in totals.items()}
    for source in ('CohereLabs/aya_dataset','OpenAssistant/oasst2'):
        selected=[r for r in val if r['source']==source][:4]
        for row in selected:probes.append({'kind':'held_out_public_validation','source':source,'source_id':row['source_id'],
            'messages':row['messages'],'reference':row['answer']})
    for messages,reference in [
        ([['user','Explique em duas frases por que uma roupa molhada seca no varal.']],None),
        ([['user','Escreva uma história curta sobre um faroleiro que encontra uma carta dentro de uma garrafa.']],None),
        ([['user','Liste três maneiras de organizar uma rotina de estudos.']],None),
        ([['user','A visita ao museu estava marcada para quarta-feira.'],['assistant','Entendi.'],
          ['user','Mudamos a visita para sábado. Em que dia vamos ao museu agora?']],'sábado'),
        ([['user','Minha mochila é verde.'],['assistant','Você informou que sua mochila é verde.'],
          ['user','Qual é a cor da minha mochila?']],'verde'),
        ([['user','Reescreva de forma educada: me mande esse arquivo logo.']],None),
        ([['user','Meu nome é Iolanda.'],['assistant','Olá, Iolanda.'],['user','Qual é meu nome?']],'Iolanda'),
        ([['user','Responda apenas o resultado: quanto é 8 mais 6?']],'14'),
    ]:probes.append({'kind':'new_manual_development_probe','messages':messages,'reference':reference})
    for i,p in enumerate(probes):
        prefix=tok.encode(prompt_text(p['messages']),special=False)
        assert len(prefix)+1<=model.cfg.context
        x=torch.tensor([[tok.bos,*prefix]],device=device)
        with torch.inference_mode():tokens=model.generate(x,max_new_tokens=192,greedy=True)
        p['output']=tok.decode(tokens[0,x.shape[1]:].tolist()).strip()
        p['input_tokens']=x.shape[1];p['generated_tokens']=tokens.shape[1]-x.shape[1]
        p['empty']=not bool(p['output']);p['unicode_replacement']='\ufffd' in p['output']
        print(i,p['kind'],p['output'][:200],flush=True)
    output={'generation':'TAC-v2','checkpoint':result['checkpoint'],'device':str(device),'greedy':True,
            'max_new_tokens':192,'model_parameters':model.parameter_count,'outputs':probes,
            'metrics_scope':'Raw outputs: open instruction quality is not certified by exact string match or loss.',
            'reserved_tests_used':False,'fallback_or_prepared_answer_used':False,'external_api_used':False,
            'full_validation_teacher_forced_fp32_by_source':full_validation,
            'full_validation_is_supplement_not_replacement_for_fixed_training_metrics':True,
            'empty_outputs':sum(p['empty'] for p in probes),'unicode_replacement_outputs':sum(p['unicode_replacement'] for p in probes)}
    out=ROOT/'reports/reconstrucao_zero/execucao_v2';out.mkdir(exist_ok=True)
    with (out/'geracao_livre.json').open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
