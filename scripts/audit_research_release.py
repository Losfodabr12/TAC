"""Confere preservação da instalação real após a fase 0.12; não treina."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.core import Core
from tac.config import atomic_json
from tac.neural.conversation import ConversationData
from tac.neural.data import Corpus


def approval_expressions(source):
    tree=ast.parse(source);result=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('checks','accepted') for x in node.targets):
            result.append(ast.dump(node,include_attributes=False))
    return result


def main():
    folder=ROOT/'reports/research_0120'
    baseline=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
    core=Core(ROOT); weights=[]
    for row in baseline['active']:
        current=core.db.query('SELECT value FROM meta WHERE key=?',(row['key'],))[0]['value']
        digest=hashlib.sha256((core.config.data/'checkpoints'/row['filename']).read_bytes()).hexdigest()
        weights.append({**row,'current_id':current,'current_sha256':digest,
                        'unchanged':current==row['id'] and digest==row['sha256']})
    sources=core.db.query('SELECT id,content_hash FROM sources ORDER BY id')
    runs=core.db.query('SELECT id,status FROM runs ORDER BY id')
    config=json.loads(core.config.path.read_text(encoding='utf-8'))
    old_changed=[key for key,value in baseline['config'].items() if config.get(key)!=value]
    datasets=[]
    for path in sorted((core.config.data/'datasets').glob('*.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        if path.name.startswith('conversation_') and 'sets' in value:
            data=ConversationData.load(path);kind='conversation'
        elif all(key in value for key in ('train','validation','provenance','hash')):
            data=Corpus.load(path);kind='text'
        else:continue
        datasets.append({'filename':path.name,'hash':data.digest,'kind':kind,'valid':True})
    previous=(folder/'tac_neural_conversation.py.antes').read_text(encoding='utf-8')
    current=(ROOT/'tac/neural/conversation.py').read_text(encoding='utf-8')
    sealed=folder/'sealed_transfer.json'
    manifest=json.loads(sealed.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    checks={
        'active_ids_and_weight_files_unchanged':all(row['unchanged'] for row in weights),
        'sources_ids_hashes_unchanged':sources==baseline['sources'],
        'production_runs_unchanged':runs==baseline['runs'],
        'old_configuration_values_preserved':not old_changed,
        'approval_expressions_identical':approval_expressions(previous)==approval_expressions(current),
        'existing_dataset_digests_valid':all(row['valid'] for row in datasets),
        'sealed_final_test_unchanged':hashlib.sha256(sealed.read_bytes()).hexdigest()==manifest['sha256'],
        'sealed_final_test_unused':not sealed.with_suffix('.consumed.json').exists(),
    }
    result={'checks':checks,'ok':all(checks.values()),'active_weights':weights,'sources_count':len(sources),
        'production_runs_count':len(runs),'registered_checkpoints':core.db.query('SELECT count(*) AS n FROM checkpoints')[0]['n'],
        'datasets':datasets,'changed_old_config_keys':old_changed,
        'added_config_keys':sorted(set(config)-set(baseline['config'])),
        'scope':'Compara hashes iniciais dos três modelos ativos, identidades de fontes/execuções e valores de configuração. Valida hashes internos dos snapshots; não havia hash inicial de cada arquivo histórico.'}
    atomic_json(folder/'preservacao.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('active_weights','datasets')},ensure_ascii=True,indent=2))
    if not result['ok']:raise SystemExit(1)


if __name__=='__main__':main()
