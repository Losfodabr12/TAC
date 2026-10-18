"""Audita a rodada pronta, sem treinar e sem interpretar arquivos de teste final."""
from __future__ import annotations

import hashlib
import argparse
import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.neural.conversation import load_checkpoint
from tac.neural.model import ModelConfig, count_parameters
from scripts.run_name_generalization import OUT, PROTOCOL, current_state, read
from scripts.run_controlled_experiments import sha


def main():
    # Guardas do executável, sem preparar dataset novo nem instanciar otimizador.
    cli = ROOT / 'scripts/run_name_generalization.py'
    blocked = subprocess.run([sys.executable, str(cli), '--prepare', '--output', str(ROOT / 'data')], capture_output=True)
    duplicate = subprocess.run([sys.executable, str(cli), '--prepare', '--output', str(OUT)], capture_output=True)
    assert blocked.returncode != 0 and duplicate.returncode != 0
    before = read(OUT / 'preservacao_antes.json')
    after = current_state()
    assert before == after, 'Alteração em originais detectada.'
    atomic_json(OUT / 'preservacao_final.json', after)
    protocol = read(OUT / 'protocolo.json')
    result = read(OUT / 'resultado.json')
    summary = read(OUT / 'resumo_metricas.json')
    assert all(protocol[k] == v for k, v in PROTOCOL.items())
    assert sha(OUT / 'dados_desenvolvimento.json') == protocol['dataset_sha256']
    assert sha(OUT / 'teste_selado.json') == protocol['final_sha256']
    assert not result['new_final_test_used'] and not result['old_final_tests_used'] and not result['promoted']
    assert not (OUT / 'teste_final_unica_avaliacao.json').exists()
    assert count_parameters(ModelConfig(**protocol['model_config'])) == result['parameter_count'] == 10004544
    records = [read(OUT / f'{arm}_s{seed}/result.json') for arm in PROTOCOL['arms'] for seed in PROTOCOL['seeds']]
    assert len(records) == 6 and all(r['steps'] == 120 and r['status'] == 'completed' for r in records)
    assert len({r['initial_weight_hash'] for r in records}) == 1
    assert all(r['initial_weight_hash'] != r['final_weight_hash'] for r in records)
    for seed in PROTOCOL['seeds']:
        pair = [r for r in records if r['seed'] == seed]
        assert pair[0]['batch_plan_sha256'] == pair[1]['batch_plan_sha256']
        assert pair[0]['exposure'] == pair[1]['exposure']
    for r in records:
        assert r['exposure']['unique_examples'] == 928 and r['exposure']['presentations'] == 1920
        assert r['exposure']['repetitions'] == 992
        steps = read(OUT / f"{r['arm']}_s{r['seed']}/passos.json")
        assert len(steps) == 120
        assert sum(s['supervised_tokens'] for s in steps) == sum(r['exposure']['supervised_tokens_by_skill'].values())
    candidate_path = OUT / 'candidato_nomes.pt'
    assert sha(candidate_path) == result['checkpoint_sha256']
    cp = load_checkpoint(candidate_path)
    selected = next(r for r in records if (r['arm'], r['seed']) == (result['chosen'], result['median_seed']))
    h = hashlib.sha256()
    for value in cp['model'].values():
        h.update(value.contiguous().numpy().tobytes())
    assert h.hexdigest() == selected['final_weight_hash']
    suite = ET.parse(OUT / 'testes.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests']) == 45
    assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
    # Verifica os conteúdos anteriores, inclusive após a compactação NTFS sem perda.
    old_manifest = read(ROOT / 'reports/experimentos_0130/manifesto_relatorios.json')['sha256']
    for name, expected in old_manifest.items():
        assert sha(ROOT / name) == expected, name
    direct_json = [p for p in OUT.glob('*.json') if p.name not in ('teste_selado.json', 'manifesto.json')]
    for path in direct_json:
        read(path)
    for name in ('curvas_validacao.png', 'generalizacao.png'):
        assert (OUT / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    final = {'status': 'ok', 'pilots': 6, 'steps_each': 120, 'seeds': PROTOCOL['seeds'],
             'all_weights_changed': True, 'same_initial_checkpoint': True,
             'matched_batches_and_tokens_between_arms': True, 'training_unique_examples': 928,
             'software_tests_passed': 45, 'parameter_count': 10004544,
             'active_and_originals_preserved': before == after,
             'old_report_manifest_files_verified': len(old_manifest),
             'old_reserved_used': False, 'new_final_test_used': False,
             'sealed_final_hash_preserved': True, 'promoted': False,
             'checkpoint_reload_verified': True, 'main_validation_template_overlap': summary['main_validation_template_overlap'],
             'diagnostic_generation_only': True, 'long_training_performed': False,
             'cli_refuses_production_output': True, 'cli_refuses_overwriting_dataset': True}
    atomic_json(OUT / 'validacao_final.json', final)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'manifesto.json']
    files += [p for p in OUT.glob('*_s*/*.json')]
    atomic_json(OUT / 'manifesto.json', {'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)},
                'scope': 'relatórios, dados congelados e um candidato; teste selado apenas hash; temporários do pytest excluídos'})
    print(final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Pasta de rodada inválida; use TAC/reports.')
    import scripts.run_name_generalization as runner
    runner.OUT = OUT
    main()
