"""Content manifest: verify preservation in place, without creating weight backups."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.core import Core


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(2**20):h.update(chunk)
    return h.hexdigest()


def snapshot():
    core=Core(ROOT);files={}
    for p in (core.config.data/'checkpoints').glob('*.pt'):
        files[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    for folder in ('tac_v2','tac_v2_conversation_20261018'):
        for p in (core.config.data/folder).iterdir():
            if p.is_file():files[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    documents={r['origin']:r['content_hash'] for r in core.db.query('SELECT origin,content_hash FROM sources')}
    return {'active_conversation':core.active_conversation_id(),'files':files,'library_sources':documents}


out=ROOT/'reports/tac_v3/rodada_20261018_01';path=out/'v2_preservacao_antes.json'
if '--verify' not in sys.argv:
    if path.exists():raise RuntimeError('Preservation manifest already exists')
    before=snapshot();atomic_json(path,before)
    print('Original files hashed in place:',len(before['files']))
else:
    before=json.loads(path.read_text(encoding='utf-8'));after=snapshot()
    differences=[k for k,v in before['files'].items() if after['files'].get(k)!=v]
    lost_docs=[k for k,v in before['library_sources'].items() if after['library_sources'].get(k)!=v]
    result={'original_files':len(before['files']),'changed_or_missing_files':differences,
            'lost_or_changed_library_documents':lost_docs,'v2_active_before':before['active_conversation'],
            'active_now':after['active_conversation'],'ok':not differences and not lost_docs}
    atomic_json(out/'v2_preservacao_depois.json',result);print(json.dumps(result,indent=2))
    if not result['ok']:raise RuntimeError('Original content changed')
