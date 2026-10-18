"""Auditoria final dos pilotos temporais; sem treino ou leitura de testes selados."""
from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import load_checkpoint
from tac.neural.model import count_parameters, ModelConfig
import scripts.run_temporal_memory as runner
from scripts.run_temporal_memory import OUT, BASE, PROTOCOL, ARMS, read
from scripts.run_controlled_experiments import sha


def main():
    before = read(OUT / 'preservacao_antes.json')
    after = runner.state()
    assert before == after, 'Dados/estados originais alterados.'
    atomic_json(OUT / 'preservacao_final.json', after)
    protocol = read(OUT / 'protocolo.json')
    assert all(protocol[k] == v for k, v in PROTOCOL.items())
    assert sha(BASE / 'dados_desenvolvimento.json') == protocol['base_dataset_sha256']
    assert sha(OUT / 'dados_derivados.json') == protocol['derived_dataset_sha256']
    assert sha(ROOT / 'tac/neural/name_generalization.py') == protocol['name_evaluator_sha256']
    source = read(BASE / 'dados_desenvolvimento.json')
    rows = source['panels']['train']
    tokenizer = LocalBPETokenizer.from_state(source['tokenizer_state'])
    # Independente do sampler: mede o orçamento pelo conteúdo das respostas.
    lengths = [len(tokenizer.encode(' ' + row['answer'], special=False)) + 1 for row in rows]
    records = [read(OUT / f'{arm}_s{seed}/result.json') for arm in ARMS for seed in PROTOCOL['seeds']]
    assert len(records) == 12 and all(r['logical_steps'] == 120 and r['status'] == 'completed' for r in records)
    assert len({r['initial_weight_hash'] for r in records}) == 1
    assert all(r['initial_weight_hash'] != r['final_weight_hash'] for r in records)
    assert sum(r['optimizer_steps_this_run'] for r in records) == 1320
    for seed in PROTOCOL['seeds']:
        runs = {r['arm']: r for r in records if r['seed'] == seed}
        plans = {arm: read(OUT / f'{arm}_s{seed}/plano.json')['indices'] for arm in ARMS}
        original = plans['original']
        assert Counter(plans['intercalado']) == Counter(original)
        assert [lengths[i] for i in plans['redistribuido']] == [lengths[i] for i in original]
        assert plans['temporal_explicito'] == original
        assert runs['temporal_explicito']['target_stream_sha256'] == runs['original']['target_stream_sha256']
        old = read(BASE / f'sft_s{seed}/result.json')
        assert runs['original']['final_weight_hash'] == old['final_weight_hash']
        assert runs['original']['batch_sha256'] == old['batch_plan_sha256']
        assert runs['original']['evaluation']['validation'] == old['evaluation']['validation']
        for arm in ARMS:
            plan = plans[arm]
            ex = runs[arm]['exposure']
            assert len(plan) == 1920 and set(plan) == set(range(928))
            assert max(Counter(plan).values()) <= 3
            assert ex['unique_examples'] == 928 and ex['repetitions'] == 992
            assert sum(lengths[i] for i in plan) == ex['supervised_tokens'] == runs['original']['exposure']['supervised_tokens']
            assert sum(ex['supervised_tokens_by_skill'].values()) == ex['supervised_tokens']
            if not runs[arm]['checkpoint_reused']:
                steps = read(OUT / f'{arm}_s{seed}/passos.json')
                assert len(steps) == 120 and sum(r['supervised_tokens'] for r in steps) == ex['supervised_tokens']
    derived = read(OUT / 'dados_derivados.json')
    assert len(derived['explicit_train']) == len(rows) == 928
    assert all((a['group'], a['answer'], a['target_names']) == (b['group'], b['answer'], b['target_names'])
               for a, b in zip(rows, derived['explicit_train']))
    assert len(derived['temporal_development']) == 48 and len(derived['copy_development']) == 32
    assert len(derived['basic_protection_evaluation_only']) == 142
    assert set(r['changes'] for r in derived['temporal_development']) == {2, 3}
    train_names = {n for r in rows for n in r['source_names']}
    assert not train_names & {n for r in derived['temporal_development'] for n in r['source_names']}
    final = read(OUT / 'resultado.json')
    assert not final['promoted'] and not final['reserved_tests_used']
    assert count_parameters(ModelConfig(**protocol['model_config'])) == final['parameter_count'] == 10004544
    path = OUT / 'candidato_temporal.pt'
    assert sha(path) == final['checkpoint_sha256']
    cp = load_checkpoint(path)
    selected = next(r for r in records if (r['arm'], r['seed']) == (final['chosen'], final['median_seed']))
    digest = hashlib.sha256()
    for tensor in cp['model'].values():
        digest.update(tensor.contiguous().numpy().tobytes())
    assert digest.hexdigest() == selected['final_weight_hash']
    suite = ET.parse(OUT / 'testes.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests']) == 51
    assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
    # Os manifestos incluem apenas hashes; nenhum teste final é interpretado.
    old_manifest = read(BASE / 'manifesto.json')['sha256']
    for name, expected in old_manifest.items():
        assert sha(ROOT / name) == expected, name
    compression = read(OUT / 'compactacao_datasets.json')
    assert compression['before'] == compression['after'] and len(compression['before']) == 27
    for name, expected in compression['before'].items():
        assert sha(ROOT / 'data/datasets' / name).lower() == expected.lower()
    coverage_folder = OUT / 'cobertura_temporal'
    coverage_protocol = read(coverage_folder / 'protocolo.json')
    assert coverage_protocol['planned_after_development_observations']
    assert coverage_protocol['previous_single_query_training_by_changes'] == {'2': 32, '3': 32}
    assert sha(coverage_folder / 'treino_derivado.json') == coverage_protocol['derived_train_sha256']
    coverage_rows = read(coverage_folder / 'treino_derivado.json')['rows']
    assert len(coverage_rows) == 928 and all((a['answer'], a['target_names']) == (b['answer'], b['target_names']) for a, b in zip(rows, coverage_rows))
    coverage_records = [read(coverage_folder / f'cobertura_temporal_s{seed}/result.json') for seed in PROTOCOL['seeds']]
    for r in coverage_records:
        control = next(c for c in records if (c['arm'], c['seed']) == ('original', r['seed']))
        assert r['initial_weight_hash'] == control['initial_weight_hash'] and r['logical_steps'] == 120
        assert r['target_stream_sha256'] == control['target_stream_sha256']
        assert r['exposure']['unique_examples'] == 928 and r['exposure']['supervised_tokens'] == control['exposure']['supervised_tokens']
        assert read(coverage_folder / f"cobertura_temporal_s{r['seed']}/plano.json")['indices'] == read(OUT / f"original_s{r['seed']}/plano.json")['indices']
    coverage_result = read(coverage_folder / 'resultado.json')
    recovery = read(coverage_folder / 'reconstrucao_checkpoint.json')
    assert recovery['exact_reconstruction'] and recovery['optimizer_updates'] == 120
    assert recovery['expected_weight_hash'] == recovery['actual_weight_hash']
    assert coverage_result['pilots'] == 3 and coverage_result['optimizer_updates_this_run'] == 480
    assert sha(coverage_folder / 'candidato_cobertura.pt') == coverage_result['checkpoint_sha256']
    coverage_checkpoint = load_checkpoint(coverage_folder / 'candidato_cobertura.pt')
    coverage_selected = next(r for r in coverage_records if r['seed'] == coverage_result['median_seed'])
    assert recovery['actual_weight_hash'] == coverage_selected['final_weight_hash']
    h = hashlib.sha256()
    for tensor in coverage_checkpoint['model'].values():
        h.update(tensor.contiguous().numpy().tobytes())
    assert h.hexdigest() == coverage_selected['final_weight_hash']
    runner.OUT = coverage_folder
    assert read(coverage_folder / 'preservacao_antes.json') == runner.state()
    runner.OUT = OUT
    for name in ('resultados.png', 'curvas_validacao.png'):
        assert (OUT / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    for p in OUT.glob('*.json'):
        if p.name != 'manifesto.json':
            read(p)
    evidence = {'status': 'ok', 'pilots': 15, 'first_stage_pilots': 12, 'coverage_pilots': 3, 'logical_steps_each': 120,
          'new_optimizer_updates': 1800, 'recovery_optimizer_updates': 120, 'control_seed42_weights_reused': True,
          'three_controls_match_previous_weight_hashes': True, 'same_initial_weights': True,
          'same_supervised_token_budget_per_seed': True, 'all_928_examples_seen': True,
          'no_example_used_more_than_3_times': True, 'original_answers_preserved_in_explicit_variant': True,
          'unchanged_name_evaluator': True, 'old_basic_criteria_unchanged': True,
          'basic_questions_not_trained': True, 'parameter_count': 10004544,
          'passed_software_tests': 51, 'active_and_original_files_preserved': before == after,
          'previous_name_manifest_files_verified': len(old_manifest), 'all_reserved_tests_closed': True,
          'checkpoint_reload_verified': True, 'promoted': False}
    atomic_json(OUT / 'validacao_final.json', evidence)
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'manifesto.json']
    files += list(OUT.glob('*_s*/*.json'))
    files += [p for p in coverage_folder.rglob('*') if p.is_file() and p.suffix in ('.json', '.pt')]
    atomic_json(OUT / 'manifesto.json', {'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)},
                'scope': 'arquivos diretos e resultados dos pilotos; exclui temporários do pytest'})
    print(evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Use uma pasta dentro de TAC/reports.')
    runner.OUT = OUT
    main()
