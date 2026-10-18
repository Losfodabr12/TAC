"""Pilotos locais de nomes: separação por nomes/roteiros, sem promoção."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.config import atomic_json
from tac.diagnostics import check_resources
from tac.learning import LearningGate
from tac.dialogue_imports import digest
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import ConversationData, TASK, load_checkpoint
from tac.neural.file_learning import EpochSampler, measured_loss
from tac.neural.name_generalization import (audit_sets, create_sets, evaluate_names, name_weights, weighted_loss)
from tac.neural.training import weight_hash
from scripts.learn_conversation_file import preservation
from scripts.run_controlled_experiments import LiveSettings, model_from, protected_step, sha

OUT = ROOT / 'reports/generalizacao_nomes_0130'
PROTOCOL = {'version': 'TAC-names-generalization-v1', 'seeds': [42, 417, 903], 'steps': 120,
            'arms': {'sft': 1., 'nomes_peso3': 3.}, 'lr': .00008, 'warmup': 10,
            'scheduler': 'constant_after_linear_warmup', 'batch': 8, 'accumulation': 2,
            'weight_decay': .01, 'clip_grad_norm': 1., 'optimizer': 'AdamW', 'betas': [.9, .999], 'epsilon': 1e-8,
            'dropout': 0., 'loss_probe_steps': [0, 40, 80, 120], 'precision': 'FP16 autocast CUDA + scaler; parâmetros/AdamW/eval FP32',
            'generation': 'greedy, 80 tokens; sem respostas preparadas, cópia externa, RAG ou pós-correção',
            'max_training_seconds_per_pilot': 120, 'internal_exercises': 0, 'replay': False, 'kl': False,
            'checkpoint_source': 'active_conversation original; nunca o candidato de 85 exemplos',
            'sampling': 'permutação de todos os exemplos antes de repetir; mesma sequência nos dois braços de cada seed',
            'selection': 'média por braço na validação principal: mínimo de contexto e caracteres; escolher seed mediana, sem escolher a melhor seed',
            'consistency_gate': {'mean_context': .8, 'mean_characters': .8, 'min_seed_context': .7, 'max_seed_range': .15},
            'final_test_policy': 'abrir uma única vez apenas se gate de consistência for atingido, após congelar seleção; nunca ajustar depois',
            'old_reserved_tests': 'somente hashes de preservação, nunca JSON/texto', 'automatic_promotion': False,
            'automatic_extension': False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def current_state():
    state = preservation(Path('C:/Users/Joaof/Desktop/TAC_Experimento_Aprendizado'))
    state['old_research_weights'] = {p.relative_to(ROOT).as_posix(): sha(p)
        for p in sorted((ROOT / 'reports').rglob('*.pt')) if not p.is_relative_to(OUT)}
    return state


def setup():
    config = LiveSettings(ROOT)
    with sqlite3.connect((config.data / 'tac.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        active_id = db.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
        filename = db.execute('SELECT filename FROM checkpoints WHERE id=?', (active_id,)).fetchone()[0]
    path = config.data / 'checkpoints' / filename
    cp = load_checkpoint(path)
    tokenizer = LocalBPETokenizer.from_state(cp['tokenizer_state'])
    return config, active_id, path, cp, tokenizer


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve a pasta existente; este protocolo não sobrescreve conjuntos ou resultados.')
    config, active_id, path, cp, tokenizer = setup()
    check_resources(config)
    panels, test, inventories = create_sets()
    audit = audit_sets(panels, test, inventories, tokenizer)
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / 'dados_desenvolvimento.json', {'panels': panels, 'tokenizer_state': tokenizer.state()})
    atomic_json(OUT / 'teste_selado.json', {'format': 'TAC-names-final-v1', 'rows': test})
    # Auditoria sem exportar as linhas/nomes finais para o relatório de desenvolvimento.
    final_tokens = audit['tokenizer_audit'].pop('test')
    audit['tokenizer_audit']['test_aggregate_only'] = {'names': len(final_tokens), 'roundtrip_exact': all(r['roundtrip_exact'] for r in final_tokens)}
    atomic_json(OUT / 'auditoria_dados.json', audit)
    protocol = {**PROTOCOL, 'parent': active_id, 'parent_sha256': sha(path), 'model_config': cp['model_config'],
                'dataset_sha256': sha(OUT / 'dados_desenvolvimento.json'), 'final_sha256': sha(OUT / 'teste_selado.json'),
                'tokenizer_hash': digest(tokenizer.state()), 'training_rows': len(panels['train']),
                'panels': {k: len(v) for k, v in panels.items()}}
    atomic_json(OUT / 'protocolo.json', protocol)
    print(json.dumps({'prepared': True, 'panels': protocol['panels'], 'test_sealed': len(test), 'audited': True}))


def pilot(config, gate, cp, tokenizer, panels, baseline, device, arm, seed):
    folder = OUT / f'{arm}_s{seed}'
    if folder.exists():
        raise FileExistsError('Piloto já existe; não sobrescrever nem repetir silenciosamente.')
    folder.mkdir()
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = model_from(cp, tokenizer, device)
    data = ConversationData({'train': panels['train']}, tokenizer)
    initial_hash = weight_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['lr'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    sampler = EpochSampler(panels['train'], seed)
    factor = PROTOCOL['arms'][arm]
    curve = [{'step': 0, **baseline['loss']}]
    history, resources = [], []
    elapsed = 0.
    batch_fingerprints = []
    import hashlib
    plan_hash = hashlib.sha256()
    try:
        for step in range(1, PROTOCOL['steps'] + 1):
            gate.wait_if_paused()
            if step % 10 == 1:
                resources.append({'step': step - 1, **check_resources(config)})
            if elapsed > PROTOCOL['max_training_seconds_per_pilot']:
                raise TimeoutError('Limite do piloto; nenhum treino longo foi iniciado.')
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = PROTOCOL['lr'] * min(1., step / PROTOCOL['warmup'])
            model.train()
            raw_losses, objectives, count = [], [], 0
            for _ in range(PROTOCOL['accumulation']):
                gate.check()
                rows = sampler.take(PROTOCOL['batch'])
                x, y = data.batch(rows, model.cfg.context, device)
                plan_hash.update(x.detach().cpu().numpy().tobytes())
                plan_hash.update(y.detach().cpu().numpy().tobytes())
                weights = torch.zeros_like(y, dtype=torch.float32)
                for index, row in enumerate(rows):
                    w = name_weights(data, row, model.cfg.context, factor)
                    weights[index, :len(w)] = w.to(device)
                    sampler.tokens[row['skill']] += int((w != 0).sum())
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits, ce = model(x, y)
                    objective = weighted_loss(logits.float(), y, weights)
                if not torch.isfinite(objective):
                    raise FloatingPointError('Objetivo não finito.')
                scaler.scale(objective / PROTOCOL['accumulation']).backward()
                raw_losses.append(float(ce.detach()))
                objectives.append(float(objective.detach()))
                count += int((y != 0).sum())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Gradiente não finito; atualização bloqueada.')
            protected_step(optimizer, scaler, gate)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed += time.monotonic() - started
            history.append({'step': step, 'nll_microbatch_mean': sum(raw_losses) / 2, 'objective_microbatch_mean': sum(objectives) / 2,
                            'lr': optimizer.param_groups[0]['lr'], 'supervised_tokens': count, 'grad_norm': float(norm), 'train_seconds': elapsed})
            if step % 40 == 0:
                point = {'step': step, 'train': measured_loss(model, data, panels['train'], device),
                         'validation': measured_loss(model, data, panels['validation'], device)}
                curve.append(point)
                atomic_json(folder / 'curva.json', curve)
                print(arm, seed, step, 'NLL treino', round(point['train']['loss'], 4), 'val', round(point['validation']['loss'], 4), flush=True)
        evaluation = {}
        for label in ('train_probe', 'dev_names', 'dev_structures', 'validation'):
            evaluation[label] = evaluate_names(model, panels[label], device, gate=gate)
        # Validação principal em sessão fechada: respostas anteriores são da própria rede.
        evaluation['validation_own'] = evaluate_names(model, panels['validation'], device, own_history=True, gate=gate)
        result = {'status': 'completed', 'arm': arm, 'seed': seed, 'steps': len(history),
                  'name_weight': factor, 'initial_weight_hash': initial_hash, 'final_weight_hash': weight_hash(model),
                  'batch_plan_sha256': plan_hash.hexdigest(), 'exposure': sampler.summary(),
                  'train_seconds': elapsed, 'resources': resources,
                  'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                  'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else None,
                  'curve': curve, 'evaluation': evaluation, 'promoted': False, 'final_test_used': False}
        atomic_json(folder / 'passos.json', history)
        atomic_json(folder / 'result.json', result)
        v = evaluation['validation']['rates']
        print('Concluído', arm, seed, 'contexto', v['context_correct'], 'caracteres', v['characters_exact'], flush=True)
        return result, model
    except BaseException as exc:
        atomic_json(folder / 'erro.json', {'error': str(exc), 'history': history, 'curve': curve, 'promoted': False})
        raise
    finally:
        del optimizer, data
        gc.collect()


def run():
    config, active_id, path, cp, tokenizer = setup()
    protocol = read(OUT / 'protocolo.json')
    if protocol['parent'] != active_id or protocol['parent_sha256'] != sha(path):
        raise ValueError('O ativo mudou desde a criação dos dados.')
    if protocol['dataset_sha256'] != sha(OUT / 'dados_desenvolvimento.json'):
        raise ValueError('Dados de desenvolvimento alterados.')
    if (OUT / 'preservacao_antes.json').exists():
        raise FileExistsError('Rodada já iniciada. Preserve resultados; não reiniciar o protocolo silenciosamente.')
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC para executar experimentos sem tarefas simultâneas.')
    before = None
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        before = current_state()
        atomic_json(OUT / 'preservacao_antes.json', before)
        # Um único artefato de pesos será salvo no fim; nenhum original é removido.
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Margem de disco insuficiente para o candidato e 512 MiB livres.')
        torch.set_num_threads(config.get('cpu_threads'))
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        panels = read(OUT / 'dados_desenvolvimento.json')['panels']
        data = ConversationData({'train': panels['train']}, tokenizer)
        model = model_from(cp, tokenizer, device)
        baseline = {'loss': {'train': measured_loss(model, data, panels['train'], device),
                             'validation': measured_loss(model, data, panels['validation'], device)},
                    'evaluation': {}, 'weight_hash': weight_hash(model)}
        print('Avaliando ativo original em desenvolvimento, sem abrir teste final.', flush=True)
        for label in ('train_probe', 'dev_names', 'dev_structures', 'validation'):
            baseline['evaluation'][label] = evaluate_names(model, panels[label], device, gate=gate)
        baseline['evaluation']['validation_own'] = evaluate_names(model, panels['validation'], device, own_history=True, gate=gate)
        atomic_json(OUT / 'ativo.json', baseline)
        del model, data
        gc.collect()
        records, states = [], {}
        # Mantém só a seed mediana de cada braço em RAM, e só UMA ao final em disco.
        for arm in PROTOCOL['arms']:
            arm_states = []
            for seed in PROTOCOL['seeds']:
                result, model = pilot(config, gate, cp, tokenizer, panels, baseline, device, arm, seed)
                records.append(result)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                arm_states.append((result['evaluation']['validation']['rates']['context_correct'], seed, state))
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            arm_states.sort(key=lambda item: (item[0], item[1]))
            _, median_seed, state = arm_states[1]
            states[arm] = (median_seed, state)
            del arm_states
        aggregates = {}
        for arm in PROTOCOL['arms']:
            selected = [r for r in records if r['arm'] == arm]
            rates = [r['evaluation']['validation']['rates'] for r in selected]
            mean = {k: sum(v[k] for v in rates) / 3 for k in rates[0]}
            contexts = [v['context_correct'] for v in rates]
            consistent = (mean['context_correct'] >= .8 and mean['characters_exact'] >= .8 and min(contexts) >= .7 and max(contexts) - min(contexts) <= .15)
            aggregates[arm] = {'mean_validation': mean, 'context_min': min(contexts), 'context_max': max(contexts),
                               'consistent': consistent, 'median_seed': states[arm][0]}
        chosen = max(aggregates, key=lambda a: (min(aggregates[a]['mean_validation']['context_correct'], aggregates[a]['mean_validation']['characters_exact']), a == 'sft'))
        seed, state = states[chosen]
        selection = {'chosen': chosen, 'seed': seed, 'aggregates': aggregates,
                     'final_test_authorized_by_protocol': aggregates[chosen]['consistent'],
                     'promoted': False, 'production_criteria_changed': False}
        atomic_json(OUT / 'selecao_congelada.json', selection)
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Disco abaixo da margem antes de salvar. Resultados preservados; não remover originais.')
        target = OUT / 'candidato_nomes.pt'
        torch.save({'format': 'TAC-chat-v2', 'task': TASK, 'model_config': cp['model_config'],
                    'tokenizer_state': tokenizer.state(), 'tokenizer': tokenizer.version, 'model': state,
                    'parent': active_id, 'step': 120, 'research_only': True, 'promotion_eligible': False,
                    'training_config': {**protocol, 'arm': chosen, 'seed': seed},
                    'dataset_hash': protocol['dataset_sha256'], 'optimizer_saved': False}, target)
        selected_result = next(r for r in records if r['arm'] == chosen and r['seed'] == seed)
        loaded = load_checkpoint(target)
        loaded_model = model_from(loaded, tokenizer, device)
        assert weight_hash(loaded_model) == selected_result['final_weight_hash']
        if aggregates[chosen]['consistent']:
            # Seleção já foi gravada. Não existe mais treinamento depois desta linha.
            final_rows = read(OUT / 'teste_selado.json')['rows']
            candidate_final = evaluate_names(loaded_model, final_rows, device, own_history=True, gate=gate)
            del loaded_model
            original = model_from(cp, tokenizer, device)
            original_final = evaluate_names(original, final_rows, device, own_history=True, gate=gate)
            atomic_json(OUT / 'teste_final_unica_avaliacao.json', {'active': original_final, 'candidate': candidate_final,
                        'selection_sha256': sha(OUT / 'selecao_congelada.json'), 'no_further_training': True})
            del original
        final = {'completed_pilots': len(records), 'steps_each': 120, 'chosen': chosen, 'median_seed': seed,
                 'checkpoint_sha256': sha(target), 'checkpoint_reload_verified': True,
                 'parameter_count': sum(p.numel() for p in loaded_model.parameters()) if not aggregates[chosen]['consistent'] else 10004544,
                 'active_changed': False, 'promoted': False, 'old_final_tests_used': False,
                 'new_final_test_used': aggregates[chosen]['consistent'], 'torch_version': str(torch.__version__),
                 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None, 'device': str(device),
                 'independent_seeds': PROTOCOL['seeds'], 'automatic_extension_performed': False}
        atomic_json(OUT / 'resultado.json', final)
    finally:
        if before is not None:
            after = current_state()
            atomic_json(OUT / 'preservacao_depois.json', after)
            atomic_json(OUT / 'preservacao_verificada.json', {'all_preserved': before == after,
                        'checks': {k: before[k] == after[k] for k in before},
                        'new_final_sha256_unchanged': sha(OUT / 'teste_selado.json') == protocol['final_sha256']})
            print('Originais preservados:', before == after, flush=True)
        lock.unlock()
        if before is not None and before != after:
            raise RuntimeError('Falha de preservação dos originais; verifique os hashes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT, help='Pasta da rodada dentro de TAC/reports; --prepare exige pasta nova.')
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Escolha uma pasta dentro de TAC/reports. Não gravar fora do projeto ou sobre dados de produção.')
    prepare() if args.prepare else run()
