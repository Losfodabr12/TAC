"""Read-only random-weight reference on the corrected, still-unused SFT validation."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tac.core import Core
from tac.config import atomic_json
from tac.neural.v3_training import V3Trainer
from tac.neural.v2_training import model_hash


def forbidden(*args,**kwargs):raise RuntimeError('Read-only evaluation cannot update or save model weights')


core=Core(ROOT);torch.save=forbidden;torch.optim.AdamW.step=forbidden
trainer=V3Trainer(core.config,core.db,core.gate)
record=json.loads((trainer.base/'checkpoints.jsonl').read_text().splitlines()[0])
assert record['global_step']==0
trainer.model,cp=core.conversation.load(record['checkpoint'],'cuda')
before=model_hash(trainer.model);value=trainer.evaluate()
assert before==model_hash(trainer.model)==record['weights_hash']
atomic_json(trainer.out/'initial_validation_revised_sft.json',{'checkpoint':record['checkpoint'],
            'dataset_hash':trainer.data.meta['dataset_hash'],'weights_hash':before,'validation':value,
            'optimizer_steps':0,'original_initial_evaluation_preserved':True})
print(json.dumps(value,indent=2))
