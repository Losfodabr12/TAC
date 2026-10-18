"""Sonda de EOS em respostas ouro AUTORAIS de treino; não treina nem seleciona modelos."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from tac.config import atomic_json
from tac.diagnostics import check_resources
from tac.fluency_lessons import lessons
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import load_checkpoint
from tac.neural.training import weight_hash
from scripts.run_controlled_experiments import LiveSettings,PROTOCOL,ARMS,model_from,OUT


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def probes(tokenizer):
    rows=[]
    for scene,turns,*_ in lessons()['train']:
        text='\n'.join('Usuário: '+u+'\nTAC: '+a for u,a in turns)
        full=tokenizer.encode(text);cursor=0
        for turn,(user,answer) in enumerate(turns):
            cursor+=len(('\n' if turn else '')+'Usuário: '+user+'\nTAC: '+answer)
            end=1+len(tokenizer.encode(text[:cursor],special=False))
            if turn==len(turns)-1:continue
            assert full[end]!=tokenizer.eos
            rows.append({'scene':scene,'turn':turn,'ids':full[:end],'original_next':full[end]})
    return rows


@torch.inference_mode()
def measure(model,rows,device,context):
    result=[]
    # Sem padding: o último índice é sempre a fronteira de resposta.
    for row in rows:
        ids=row['ids'][-context:]
        x=torch.tensor([ids],device=device);logits,_=model(x)
        logits=logits[0,-1].float();p=logits.softmax(-1)
        result.append({'scene':row['scene'],'turn':row['turn'],
            'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'p_eos':float(p[2]),'p_original_newline':float(p[row['original_next']]),
            'eos_logit_minus_newline':float(logits[2]-logits[row['original_next']]),
            'eos_is_argmax':int(logits.argmax())==2,'input_tokens':len(ids)})
    return {'probes':len(result),'mean_p_eos':sum(r['p_eos'] for r in result)/len(result),
        'mean_p_original_newline':sum(r['p_original_newline'] for r in result)/len(result),
        'eos_argmax_count':sum(r['eos_is_argmax'] for r in result),'rows':result}


def main():
    config=LiveSettings(ROOT);torch.set_num_threads(config.get('cpu_threads'));check_resources(config)
    device=torch.device('cuda' if torch.cuda.is_available() and config.get('device')!='cpu' else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(min(.75,config.get('vram_mb')*2**20/torch.cuda.get_device_properties(0).total_memory))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    paths={'active':config.data/'checkpoints'/(PROTOCOL['parent']+'.pt')}
    paths.update({arm:OUT/(arm+'_s42')/'candidate.pt' for arm in ARMS})
    results={}
    for label,path in paths.items():
        cp=load_checkpoint(path);tokenizer=LocalBPETokenizer.from_state(cp['tokenizer_state'])
        model=model_from(cp,tokenizer,device).eval();actual_hash=weight_hash(model)
        expected=read(OUT/'active.json')['weight_hash'] if label=='active' else read(path.parent/'result.json')['final_weight_hash']
        if actual_hash!=expected:raise AssertionError('Pesos não correspondem ao resultado: '+label)
        rows=probes(tokenizer);assert len(rows)==120
        results[label]={'checkpoint_load_and_hash_verified':True,'parameters':model.parameter_count,
            'state_hash':actual_hash,'context_256':measure(model,rows,device,256),'context_512':measure(model,rows,device,512)}
        check_resources(config);print(label,'checkpoint verificado e fronteiras avaliadas',flush=True)
        del model,cp
        if device.type=='cuda':torch.cuda.empty_cache()
    atomic_json(OUT/'sondas_eos_ouro.json',{'results':results,'scope':'120 fronteiras NÃO finais, 40 cenas autorais de TREINO. Somente seed 42 por disponibilidade dos pesos. Contextos ouro, não geração livre. Não é validação ou teste.',
        'training_started':False,'optimizer_created':False,'final_test_used':False,'promoted':False})


if __name__=='__main__':main()
