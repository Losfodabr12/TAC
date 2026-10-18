"""Auditoria da entrega: só confere dados reais; não importa nem treina."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tac.core import Core
from tac.config import atomic_json

REPORT=ROOT/'reports'/'importador_100'
baseline=json.loads((REPORT/'baseline.json').read_text(encoding='utf-8'))
core=Core(ROOT)
active=[]
for row in baseline['active']:
    actual=core.db.query('SELECT value FROM meta WHERE key=?',(row['key'],))[0]['value']
    sha=hashlib.sha256((core.config.data/'checkpoints'/row['filename']).read_bytes()).hexdigest()
    assert actual==row['id'] and sha==row['sha256'],row['key']
    active.append({**row,'preserved':True})
sources=core.db.query('SELECT id,origin,content_hash FROM sources ORDER BY id')
assert sources==baseline['sources']
assert core.config.values==baseline['config']
runs=core.db.query('SELECT id,status FROM runs ORDER BY id')
assert runs==baseline['runs']
assert not core.db.query('SELECT * FROM dialogue_imports')

def assignments(source):
    tree=ast.parse(source)
    checks=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='checks' for t in n.targets)]
    selection=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AugAssign)) and (
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='selection_loss' for t in n.targets))
        or (isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='selection_loss'))]
    return [ast.dump(n,include_attributes=False) for n in [*checks,*selection]]

assert assignments((REPORT/'neural_conversation.py.antes').read_text(encoding='utf-8'))==assignments(
    (ROOT/'tac/neural/conversation.py').read_text(encoding='utf-8'))
assert (REPORT/'fluency_lessons.py.antes').read_bytes()==(ROOT/'tac/fluency_lessons.py').read_bytes()
previews={}
for suffix in ('.jsonl','.txt'):
    preview=core.conversation.preview_import(ROOT/'examples'/('conversas_exemplo'+suffix))
    assert preview['report']['accepted_examples']==7 and not preview['report']['rejected']
    previews[suffix]=preview['report']
prepared=core.conversation.prepare_training(profile='10m',curriculum='fluency')
result={'active_models':active,'sources_preserved':len(sources),'config_preserved':True,
        'runs_preserved':len(runs),'approval_expressions_unchanged':True,'selection_expression_unchanged':True,
        'reviewed_rules_unchanged':True,'schema':core.db.query("SELECT value FROM meta WHERE key='schema_version'")[0]['value'],
        'imports_saved_to_personal_data':0,'example_previews':previews,'next_training_counts':prepared['counts']}
atomic_json(REPORT/'auditoria.json',result)
print(json.dumps(result,ensure_ascii=True,indent=2))
