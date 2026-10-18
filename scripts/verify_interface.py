"""Audita preservação da entrega de interface. Não treina nem acessa a rede."""
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports' / 'interface_090'


def chunk_digest(connection):
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute('SELECT id,source_id,ordinal,section,text FROM chunks ORDER BY id'):
        digest.update(json.dumps(list(row),ensure_ascii=False).encode('utf-8'))
        count += 1
    return count, digest.hexdigest()


def main():
    baseline = json.loads((FOLDER/'baseline.json').read_text(encoding='utf-8'))
    original = sqlite3.connect((FOLDER/'antes_interface.sqlite3').as_uri()+'?mode=ro',uri=True)
    current = sqlite3.connect((ROOT/'data/tac.sqlite3').as_uri()+'?mode=ro',uri=True)
    current.row_factory = sqlite3.Row
    checks = {}
    checks['codigo_dos_treinadores_e_nucleo_preservado'] = all(
        hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
        for name,digest in baseline['protected_code'].items())
    sources = [dict(r) for r in current.execute('SELECT id,origin,content_hash FROM sources ORDER BY id')]
    checks['fontes_preservadas'] = sources == baseline['sources']
    checks['trechos_e_indices_de_origem_preservados'] = chunk_digest(original) == chunk_digest(current)
    active = [dict(r) for r in current.execute("SELECT m.key,c.id,c.filename FROM meta m JOIN checkpoints c ON c.id=m.value WHERE m.key LIKE 'active_%'")]
    for row in active:
        row['sha256'] = hashlib.sha256((ROOT/'data/checkpoints'/row['filename']).read_bytes()).hexdigest()
    checks['modelos_ativos_e_arquivos_de_pesos_preservados'] = sorted(active,key=lambda r:r['key']) == sorted(baseline['active'],key=lambda r:r['key'])
    checks['configuracoes_preservadas'] = json.loads((ROOT/'config.json').read_text(encoding='utf-8')) == baseline['config']
    previous_runs = dict(original.execute('SELECT id,status FROM runs ORDER BY id'))
    current_runs = dict(current.execute('SELECT id,status FROM runs ORDER BY id'))
    checks['historico_de_treinos_anterior_preservado'] = all(current_runs.get(key) == value for key,value in previous_runs.items())
    additions = [dict(r) for r in current.execute('SELECT id,kind,status,started,finished FROM runs ORDER BY started') if r['id'] not in previous_runs]
    original.close()
    current.close()
    result = {'checks':checks,'sources':len(sources),'active_models':active,'protected_modules':len(baseline['protected_code']),
              'additional_runs_during_work':additions,
              'note':'Houve atividade da janela preexistente com treino automático permitido; consulte o relatório.' if additions else 'Sem novas execuções na pasta de dados real.',
              'status':'ok' if all(checks.values()) else 'falhou'}
    (FOLDER/'verificacao.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
