"""Validação dos artefatos reais, sem carregar modelo ou executar treinamento."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json

OUT=ROOT/'reports/experimentos_0130'


def read(path):
    def invalid(value):raise ValueError('Valor JSON não finito: '+value)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    protocol=read(OUT/'protocol.json');summary=read(OUT/'summary.json');active=read(OUT/'active.json')
    paths=sorted(OUT.glob('*_s*/result.json'));runs=[read(p) for p in paths]
    assert {(r['arm'],r['seed']) for r in runs}=={(a,s) for a in protocol['arms'] for s in protocol['seeds']}
    assert len(runs)==15 and summary['completed_pilots']==15
    rates=None
    for r,path in zip(runs,paths):
        assert r['status']=='completed' and r['steps_completed']==120
        assert [s['step'] for s in r['history']]==list(range(1,121))
        current=[s['learning_rate'] for s in r['history']]
        if rates is None:rates=current
        assert current==rates
        assert r['initial_weight_hash']==active['weight_hash'] and r['initial_weight_hash']!=r['final_weight_hash']
        assert not r['promoted'] and not r['test_final_evaluated']
        assert r['passed_original_checks']==all(r['protection_checks'].values())
        checkpoint=path.parent/'candidate.pt'
        if r['checkpoint_kept']:assert r['seed']==42 and sha(checkpoint)==r['checkpoint_sha256']
        else:assert not checkpoint.exists()
        assert r['plan']==read(path.parent/'plan.json')
        for name in ('basic','context','natural','imported'):
            e=r['evaluation'][name];rows=e['outputs']
            assert len(rows)==e['examples']
            assert e['exact_matches']==sum(row['exact'] for row in rows)
            assert all(row['exact']==(row['actual'] in row['acceptable_answers']) for row in rows)
            assert all(row['semantic_equivalence'] is None for row in rows)
        for part in r['evaluation']['novel']['by_skill'].values():
            assert part['semantic_equivalence']['total']==0 and part['semantic_equivalence']['accuracy'] is None
    assert all(all(v for k,v in r.items() if type(v)is bool) for r in summary['fairness'])
    coverage=read(OUT/'cobertura_reproduzida.json')['results']
    assert len(coverage)==15 and all(v['schedule_and_exposure_reproduced'] for v in coverage.values())
    assert all(v['imported_unique_row_indices']==528 for k,v in coverage.items() if k.startswith('B_balanced_'))
    eos=read(OUT/'sondas_eos_ouro.json')['results']
    assert len(eos)==6 and all(v['checkpoint_load_and_hash_verified'] and v['parameters']==10004544 for v in eos.values())
    preserved=read(OUT/'preservacao_verificada.json');assert all(preserved.values())
    suites=ET.parse(OUT/'testes_final.xml').getroot().findall('testsuite')
    tests=sum(int(s.attrib['tests']) for s in suites)
    assert tests==217 and all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    assert (OUT/'capturas/bat_inicio.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    for name in ('comparacao','todos_os_pilotos','detalhes','fatos_verificados','treinamento_e_versoes'):
        assert (OUT/f'capturas/{name}.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    files=[p for p in OUT.glob('*.json') if p.name not in ('manifesto_relatorios.json','validacao_final.json')]+paths+[p.parent/'plan.json' for p in paths]
    for path in files:read(path)
    atomic_json(OUT/'validacao_final.json',{'status':'ok','pilots':15,'steps_each':120,'seeds':[42,417,903],
        'initial_checkpoint_identical':True,'all_weights_changed':True,'all_datasets_and_exposure_reproduced':True,
        'unchanged_task_streams_verified':True,'retention_target_token_budgets_equal':True,
        'retained_checkpoints_load_and_hash_verified':6,'saved_experimental_weight_files':5,
        'software_tests_passed':tests,'software_test_failures':0,'bat_smoke_screenshot_valid':True,
        'native_ui_screenshots_valid':True,'reports_json_validated':len(files),'preservation':preserved,
        'models_promoted':0,'final_test_used':False,'semantic_equivalence_general':'não avaliada'})
    inventory={p.relative_to(ROOT).as_posix():sha(p) for p in [*files,OUT/'validacao_final.json'] if p.exists()}
    atomic_json(OUT/'manifesto_relatorios.json',{'sha256':inventory,'note':'Pesos não são incluídos neste manifesto textual; hashes estão nos resultados. Fixtures de testes são excluídas.'})
    print(json.dumps({'status':'ok','pilots':15,'tests_passed':tests,'json_files_validated':len(files)}))


if __name__=='__main__':main()
