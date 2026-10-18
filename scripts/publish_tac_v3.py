"""Expose a reviewed v3 candidate in the selector; keep the active v2 unchanged."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.core import Core
from tac.neural.v2_data import load_jsonl
from tac.neural.v2_training import model_hash

core=Core(ROOT);base=core.config.data/'tac_v3';out=ROOT/'reports/tac_v3/rodada_20261018_01'
training=json.loads((base/'training_result.json').read_text(encoding='utf-8'))
review=json.loads((out/'REVISAO_GERACOES.json').read_text(encoding='utf-8'))
label=review['selected_label']
if label not in ('v3_epoca1','v3_final'):raise ValueError('Disponibilizar somente candidatos que passaram por instruções')
generation=json.loads((out/('geracoes_'+label+'.json')).read_text(encoding='utf-8'))
candidate=generation['checkpoint'];model,cp=core.conversation.load(candidate)
assert cp['generation']=='TAC-v3' and model.parameter_count==29_414_400
assert cp['v3_state']['presentations']>0 and cp['v3_state']['phase'] in ('instruction','complete')
record=next(r for r in load_jsonl(base/'checkpoints.jsonl') if r['checkpoint']==candidate)
assert model_hash(model)==record['weights_hash']
previous=core.active_conversation_id()
assert previous!=candidate,'Publication must not silently promote v3'
value={'generation':'TAC-v3','checkpoint':candidate,'selected_label':label,'parameters':model.parameter_count,
       'final_training_checkpoint':training['checkpoint'],'v2_active_preserved':previous,
       'quality_approved':False,'automatic_promotion':False,'experimental_comparison_only':True,
       'selection_reason':review['selection_reason'],'dataset_hash':training['dataset_hash']}
with core.gate.writing():
    core.gate.check()
    core.db.execute("INSERT INTO meta(key,value) VALUES('experimental_conversation_v3',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(candidate,))
    core.db.execute('UPDATE runs SET dataset_hash=? WHERE id=?',(training['dataset_hash'],training['run_id']))
    atomic_json(base/'deployment.json',value)
assert core.active_conversation_id()==previous
print(json.dumps(value,indent=2))
