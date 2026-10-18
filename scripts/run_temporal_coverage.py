"""Segunda hipótese: cobrir recuperação isolada do anterior após várias trocas.

Motivada pela auditoria de desenvolvimento da primeira rodada. Mesmos alvos,
seed, passos e orçamento; modifica só históricos t0/t4. Não abre testes finais.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_temporal_memory as engine
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.config import atomic_json
from tac.dialogue_imports import digest
from tac.learning import LearningGate
from tac.diagnostics import check_resources
from tac.neural.conversation import ConversationData, TASK, load_checkpoint
from tac.neural.temporal_memory import audit_rows
from tac.neural.training import weight_hash
from scripts.run_controlled_experiments import model_from, sha

PARENT = ROOT / 'reports/memoria_temporal_0130'
OUT = PARENT / 'cobertura_temporal'
ARM = 'cobertura_temporal'


def render_coverage(rows):
    inventory = {n: k for r in rows for n, k in r['name_kinds'].items()}
    names = sorted(inventory)
    result = []
    for row in rows:
        if row['family'] not in ('t0', 't4'):
            result.append({**row, 'messages': [list(m) for m in row['messages']]})
            continue
        changes = 2 + int(row['group'].rsplit('_', 1)[1]) % 2
        pool = [n for n in names if n not in row['source_names']]
        offset = int(digest(row['group'])[:8], 16) % len(pool)
        prior = [pool[offset]]
        if changes == 3:
            prior.append(pool[(offset + 17) % len(pool)])
        text = f'Comecei usando {prior[0]}. '
        if changes == 3:
            text += f'Mais tarde passei a usar {prior[1]}. '
        text += 'Após outra mudança, passei a me apresentar assim: '
        messages = [list(m) for m in row['messages']]
        messages[0][1] = text + messages[0][1]
        result.append({**row, 'messages': messages, 'source_names': [*row['source_names'], *prior],
                       'name_kinds': {**row['name_kinds'], **{n: inventory[n] for n in prior}},
                       'coverage_changes_after_last_correction': changes,
                       'extra_earlier_names': prior})
    assert len(result) == len(rows)
    assert all((a['group'], a['answer'], a['target_names']) == (b['group'], b['answer'], b['target_names'])
               for a, b in zip(rows, result))
    return result


def prepare():
    if OUT.exists():
        raise FileExistsError('Hipótese já registrada; não sobrescrever.')
    config, active_id, path, cp, tokenizer = engine.setup()
    parent_protocol = engine.read(PARENT / 'protocolo.json')
    if parent_protocol['parent_sha256'] != sha(path):
        raise ValueError('Ativo alterado.')
    panels = engine.read(engine.BASE / 'dados_desenvolvimento.json')['panels']
    rows = render_coverage(panels['train'])
    data = ConversationData({'train': rows}, tokenizer)
    audit = audit_rows(rows, data)
    by_changes = {str(n): sum(r['skill'] == 'nome_anterior' and r.get('coverage_changes_after_last_correction') == n for r in rows) for n in (2, 3)}
    assert by_changes == {'2': 32, '3': 32}
    OUT.mkdir()
    atomic_json(OUT / 'treino_derivado.json', {'rows': rows})
    atomic_json(OUT / 'protocolo.json', {**engine.PROTOCOL, 'arms': [ARM],
        'hypothesis': 'A base só supervisiona nome_anterior em t0/t4 com uma correção. Adicionar história anterior a essas mesmas apresentações ensina a recuperar um intermediário, sem mudar respostas.',
        'planned_after_development_observations': True, 'final_test_used_to_design': False,
        'changed_factor': 'cobertura das trajetórias no histórico; sem pistas extras nas perguntas ou mudanças dos alvos',
        'source_first_protocol_sha256': sha(PARENT / 'protocolo.json'),
        'source_train_sha256': sha(engine.BASE / 'dados_desenvolvimento.json'),
        'derived_train_sha256': sha(OUT / 'treino_derivado.json'),
        'parent': active_id, 'parent_sha256': sha(path), 'audit': audit,
        'previous_single_query_training_by_changes': by_changes,
        'selection': 'mesma regra de mínimo atual/anterior/transição; seed mediana. Sem promoção.'})
    print('Hipótese de cobertura congelada antes do treino:', by_changes, audit, flush=True)


def run():
    if not (PARENT / 'resultado.json').exists():
        raise RuntimeError('Aguarde a primeira rodada terminar; não executar simultaneamente.')
    if (OUT / 'preservacao_antes.json').exists():
        raise FileExistsError('Rodada já iniciada; preservar resultados.')
    config, active_id, path, cp, tokenizer = engine.setup()
    protocol = engine.read(OUT / 'protocolo.json')
    if protocol['parent_sha256'] != sha(path) or protocol['derived_train_sha256'] != sha(OUT / 'treino_derivado.json'):
        raise ValueError('Fonte ou dados alterados após congelar.')
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC; não executar treinamentos simultâneos.')
    before = None
    engine.OUT = OUT
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Preserve 512 MiB livres além do novo candidato.')
        before = engine.state()
        atomic_json(OUT / 'preservacao_antes.json', before)
        torch.set_num_threads(config.get('cpu_threads'))
        torch.use_deterministic_algorithms(True)
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        panels = engine.read(engine.BASE / 'dados_desenvolvimento.json')['panels']
        derived = engine.read(PARENT / 'dados_derivados.json')
        rows = engine.read(OUT / 'treino_derivado.json')['rows']
        states, records = [], []
        for seed in protocol['seeds']:
            result, model = engine.pilot(ARM, seed, config, gate, cp, tokenizer, panels, derived, device, training_rows=rows)
            records.append(result)
            states.append((engine.temporal_rank(result['evaluation']), seed, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        states.sort(key=lambda x: (x[0], x[1]))
        _, seed, weights = states[1]
        target = OUT / 'candidato_cobertura.pt'
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Disco abaixo da margem antes de salvar o candidato; originais preservados.')
        torch.save({'format': 'TAC-chat-v2', 'task': TASK, 'model_config': cp['model_config'],
                    'tokenizer': tokenizer.version, 'tokenizer_state': tokenizer.state(), 'model': weights,
                    'parent': active_id, 'step': 120, 'research_only': True, 'promotion_eligible': False,
                    'dataset_hash': protocol['derived_train_sha256'], 'optimizer_saved': False,
                    'training_config': {**protocol, 'arm': ARM, 'seed': seed}}, target)
        saved = load_checkpoint(target)
        model = model_from(saved, tokenizer, device)
        chosen = next(r for r in records if r['seed'] == seed)
        assert weight_hash(model) == chosen['final_weight_hash']
        atomic_json(OUT / 'resultado.json', {'status': 'completed', 'pilots': 3, 'steps_each': 120,
            'optimizer_updates_this_run': 360, 'chosen': ARM, 'median_seed': seed,
            'checkpoint_sha256': sha(target), 'checkpoint_reload_verified': True,
            'parameter_count': model.parameter_count, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'promoted': False, 'reserved_tests_used': False})
    finally:
        if before is not None:
            after = engine.state()
            atomic_json(OUT / 'preservacao_depois.json', after)
            atomic_json(OUT / 'preservacao_verificada.json', {'all_preserved': before == after, 'checks': {k: before[k] == after[k] for k in before}})
            print('Originais preservados na hipótese de cobertura:', before == after, flush=True)
        lock.unlock()
        if before is not None and before != after:
            raise RuntimeError('Falha de preservação.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--parent', type=Path, default=PARENT, help='Pasta da primeira rodada dentro de TAC/reports.')
    args = parser.parse_args()
    PARENT = args.parent.resolve()
    if not PARENT.is_relative_to(ROOT / 'reports') or PARENT == ROOT / 'reports':
        parser.error('Use uma pasta de rodada dentro de TAC/reports.')
    OUT = PARENT / 'cobertura_temporal'
    prepare() if args.prepare else run()
