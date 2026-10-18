"""Confere o ZIP entregue e a preservação dos dados; não executa treinamento."""
import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.diagnostic_export import NAMES

folder=ROOT/'reports/diagnostico_110'
baseline=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
assert config==baseline['config']
data=ROOT/config['data_dir']
with sqlite3.connect(data/'tac.sqlite3') as c:
    c.row_factory=sqlite3.Row
    for row in baseline['active']:
        assert c.execute('SELECT value FROM meta WHERE key=?',(row['key'],)).fetchone()[0]==row['id']
        path=data/'checkpoints'/row['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    assert [dict(r) for r in c.execute('SELECT id,status FROM runs ORDER BY id')]==baseline['runs']
    assert [dict(r) for r in c.execute('SELECT id,content_hash FROM sources ORDER BY id')]==baseline['sources']
    assert c.execute('SELECT count(*) FROM checkpoints').fetchone()[0]==baseline['checkpoints']
with zipfile.ZipFile(ROOT/'TAC_Diagnostico.zip') as z:
    assert z.testzip() is None
    assert set(z.namelist())==set(NAMES)|{'manifesto.json'}
    blobs={n:z.read(n) for n in z.namelist()}
    manifest=json.loads(blobs['manifesto.json'])
    for name,entry in manifest['arquivos'].items():
        assert hashlib.sha256(blobs[name]).hexdigest()==entry['sha256']
        if name.endswith('.json'):
            json.loads(blobs[name],parse_constant=lambda _:(_ for _ in ()).throw(ValueError('JSON não finito')))
    for blob in blobs.values():
        text=blob.decode('utf-8')
        assert 'C:\\\\Users\\\\' not in text and 'C:\\Users\\' not in text
        assert 'Joaof' not in text
    architecture=json.loads(blobs['arquitetura.json'])
    models={m['ativo_em'][0]:m['parametros_unicos_observados'] for m in architecture['modelos_ativos']}
    assert models=={'active_checkpoint':495808,'active_conversation':10004544,'active_dialogue':264723}
    assert all(m.get('contagem_confere',True) for m in architecture['modelos_ativos'])
    datasets=json.loads(blobs['datasets_estatisticas.json'])['conjuntos']
    checkpoints=json.loads(blobs['checkpoints.json'])['checkpoints']
    evaluations=json.loads(blobs['avaliacoes.json'])
    report={'zip_valido':True,'json_validos':True,'sha256_manifesto_conferem':True,
            'arquivos':len(blobs),'bytes_zip':(ROOT/'TAC_Diagnostico.zip').stat().st_size,
            'bytes_descompactados':sum(len(b) for b in blobs.values()),'parametros_ativos':models,
            'checkpoints_e_reforco':len(checkpoints),'datasets':len(datasets),
            'execucoes':len(evaluations['execucoes']),
            'avaliacoes_externas_registradas':len(evaluations['avaliacoes_externas_registradas']),
            'ids_e_hashes_dos_modelos_preservados':True,'config_preservada':True,
            'fontes_preservadas':len(baseline['sources']),'execucoes_preservadas':len(baseline['runs']),
            'caminhos_pessoais_e_login_nao_encontrados':True,
            'privacidade':'Verificação adicional de sentinelas e anonimização na suíte de testes; não substitui revisão humana de material que se escolha compartilhar.'}
atomic_json(folder/'auditoria.json',report)
print(json.dumps(report,ensure_ascii=True,indent=2))
