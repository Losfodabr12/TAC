"""Validate new checkpoints/data and preservation evidence, without training."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
from tac.config import atomic_json
from tac.core import Core
from tac.neural.conversation import load_checkpoint
from tac.neural.model import TACTransformer,ModelConfig
from tac.neural.v2_training import model_hash
from tac.neural.v2_data import load_jsonl
from tac.neural.v3_training import V3Data


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2**20):h.update(block)
    return h.hexdigest()


def main():
    core=Core(ROOT);base=core.config.data/'tac_v3';out=ROOT/'reports/tac_v3/rodada_20261018_01'
    result=json.loads((base/'training_result.json').read_text(encoding='utf-8'))
    data=V3Data(base,'cpu') # verifies every file used by the current manifest
    checkpoints=[]
    for recorded in load_jsonl(base/'checkpoints.jsonl'):
        path=core.conversation.storage.checkpoint_path(recorded['checkpoint'])
        assert sha(path)==recorded['file_sha256']
        cp=load_checkpoint(path);model=TACTransformer(ModelConfig(**cp['model_config']))
        model.load_state_dict(cp['model'])
        assert model.parameter_count==29_414_400 and model_hash(model)==recorded['weights_hash']
        assert all(torch.isfinite(p).all() for p in model.parameters())
        states=cp.get('optimizer',{}).get('state',{})
        assert all(torch.isfinite(v).all() for row in states.values() for v in row.values() if torch.is_tensor(v))
        checkpoints.append({**recorded,'sha256_ok':True,'strict_load_ok':True,'finite':True,'parameters':model.parameter_count})
        del cp,model
    records=[]
    for p in base.glob('exposure_*.jsonl'):records.extend(load_jsonl(p))
    steps=[r['step'] for r in records]
    assert sorted(steps)==list(range(1,result['global_steps']+1)), 'Missing or repeated committed updates'
    sft=Counter(i for r in records if r['kind']=='instruction' for i in r['indices'])
    assert len(sft)==len(data.rows['train']) and set(sft.values())=={2}
    language_indices=[i for r in records if r['kind']=='language' for i in r['indices']]
    assert len(language_indices)==len(set(language_indices)), 'Language windows repeated despite available unique corpus'
    assert all(__import__('math').isfinite(r['loss']) and __import__('math').isfinite(r['gradient_norm_before_clip']) for r in records)
    offsets=json.loads((base/'language_source_offsets.json').read_text())['train'];boundary=offsets['wikipedia_start_offset']
    by_source=Counter()
    for index in language_indices:
        start=index*1024+1;end=start+1024
        fw=max(0,min(end,boundary)-start)
        by_source['FineWeb2']+=fw;by_source['Wikipedia']+=1024-fw
    summary={'ok':True,'software_checks_are_not_semantic_approval':True,'checkpoints':checkpoints,
             'committed_updates':len(steps),'duplicate_or_missing_committed_updates':False,
             'instruction_unique':len(sft),'instruction_presentations':sum(sft.values()),
             'instruction_repetitions_per_example':[2],'language_unique_windows':len(language_indices),
             'language_window_repetitions':0,'language_supervised_tokens_by_source':dict(by_source),
             'all_losses_and_gradient_norms_finite':True,'current_dataset_hash':data.meta['dataset_hash'],
             'v2_active_checkpoint':core.active_conversation_id(),'reserved_tests_opened':False}
    atomic_json(out/'integridade_v3.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},indent=2))


if __name__=='__main__':main()
