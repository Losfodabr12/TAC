"""Pilotos de memória temporal com orçamento fixo e testes finais fechados."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
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
from tac.neural.conversation import ConversationData, ConversationEngine, TASK, load_checkpoint
from tac.neural.file_learning import measured_loss
from tac.neural.name_generalization import evaluate_names, weighted_loss
from tac.neural.temporal_memory import (ARMS, audit_rows, copy_probe, exposure, interleaved_plan,
    original_plan, redistributed_plan, temporal_probe, temporal_render)
from tac.neural.training import weight_hash
from scripts.run_controlled_experiments import model_from, protected_step, sha
from scripts.run_name_generalization import setup, read
from scripts.learn_conversation_file import preservation

BASE = ROOT / 'reports/generalizacao_nomes_0130'
OUT = ROOT / 'reports/memoria_temporal_0130'
PROTOCOL = {'format': 'TAC-temporal-memory-v1', 'arms': list(ARMS), 'seeds': [42, 417, 903],
            'steps': 120, 'batch': 8, 'accumulation': 2, 'learning_rate': .00008,
            'warmup': 10, 'schedule': 'constant_after_linear_warmup', 'weight_decay': .01,
            'optimizer': 'AdamW', 'betas': [.9, .999], 'epsilon': 1e-8, 'optimizer_reset': True,
            'clip_grad_norm': 1., 'dropout': 0., 'name_weight': 1., 'max_training_seconds': 120,
            'precision': 'fp16 autocast CUDA + scaler; pesos/AdamW/avaliação FP32',
            'generation': 'greedy; 80 tokens nomes e 128 proteção básica; sem cópia externa/RAG/filtros',
            'sampler_cap': 3, 'unique_examples': 928, 'presentations': 1920,
            'supervision_budget': 'mesmo total de tokens-alvo por seed; redistribuído também igual por posição',
            'explicit_variant': 'mesmos grupos e respostas; pistas temporais no contexto e terceira mudança em t5; dois fatores de apresentação não isolados entre si',
            'explanation_metric': 'transição anterior→atual, com a mesma gramática de contraste; motivo semântico livre não pontuado',
            'selection': 'melhor média do mínimo entre atual, anterior e transição no desenvolvimento temporal; seed mediana. Retenção e regressões sempre reportadas; nenhuma promoção.',
            'final_tests': 'todos fechados, independentemente das notas desta rodada',
            'production_criteria_changed': False, 'automatic_promotion': False,
            'internal_training_examples': 0, 'replay': False, 'kl': False,
            'primary_name_evaluator': 'tac.neural.name_generalization.evaluate_names, sem alterações',
            'control_reuse': 'seed42 carrega pesos já existentes; seeds417/903 reconstituem 120 passos e exigem hash idêntico antes de reutilizar métricas'}


def state():
    result = preservation(Path('C:/Users/Joaof/Desktop/TAC_Experimento_Aprendizado'))
    result['old_research_weights'] = {p.relative_to(ROOT).as_posix(): sha(p)
        for p in sorted((ROOT / 'reports').rglob('*.pt')) if not p.is_relative_to(OUT)}
    result['name_source_dataset'] = sha(BASE / 'dados_desenvolvimento.json')
    result['name_sealed_final'] = sha(BASE / 'teste_selado.json')
    return result


def prepare():
    if OUT.exists():
        raise FileExistsError('Pasta já existente; não sobrescrever a rodada.')
    config, active_id, path, cp, tokenizer = setup()
    previous = read(BASE / 'protocolo.json')
    if previous['parent'] != active_id or previous['parent_sha256'] != sha(path):
        raise ValueError('Ativo mudou desde o experimento anterior.')
    if sha(BASE / 'dados_desenvolvimento.json') != previous['dataset_sha256']:
        raise ValueError('Dataset anterior alterado.')
    panels = read(BASE / 'dados_desenvolvimento.json')['panels']
    rendered = temporal_render(panels['train'])
    probes = temporal_probe(panels)
    copy = copy_probe(panels)
    public_basic = read(ROOT / 'reports/experimentos_0130/active.json')['evaluation']['basic']['outputs']
    basic = [{'messages': r['messages'], 'answer': r['expected'], 'acceptable_answers': r['acceptable_answers'],
              'skill': r['skill'], 'group': f'protecao_{i}'} for i, r in enumerate(public_basic)]
    data = ConversationData({'train': panels['train']}, tokenizer)
    audits = {label: audit_rows(rows, data) for label, rows in [('original', panels['train']),
              ('explicit', rendered), ('temporal_development', probes), ('copy_development', copy)]}
    train_names = {n for r in panels['train'] for n in r['source_names']}
    dev_names = {n for r in probes for n in r['source_names']}
    if train_names & dev_names or not {n for r in rendered for n in r['source_names']} <= train_names:
        raise ValueError('Vazamento de nomes entre treino e desenvolvimento.')
    OUT.mkdir(parents=True, exist_ok=False)
    if os.name == 'nt':
        subprocess.run(['compact.exe', '/C', '/I', '/Q', str(OUT)], capture_output=True, check=True)
    frozen = {'explicit_train': rendered, 'temporal_development': probes, 'copy_development': copy,
              'basic_protection_evaluation_only': basic}
    atomic_json(OUT / 'dados_derivados.json', frozen)
    protocol = {**PROTOCOL, 'parent': active_id, 'parent_sha256': sha(path),
                'base_dataset_sha256': sha(BASE / 'dados_desenvolvimento.json'),
                'derived_dataset_sha256': sha(OUT / 'dados_derivados.json'),
                'name_evaluator_sha256': sha(ROOT / 'tac/neural/name_generalization.py'),
                'model_config': cp['model_config'], 'tokenizer_hash': digest(tokenizer.state()),
                'audits': audits, 'basic_protection_examples': len(basic),
                'temporal_probe_examples': len(probes), 'copy_probe_examples': len(copy),
                'explicit_three_change_conversations': len({r['group'] for r in rendered if r['temporal_changes'] == 3})}
    atomic_json(OUT / 'protocolo.json', protocol)
    atomic_json(OUT / 'compactacao_datasets.json', read(ROOT / 'reports/compactacao_datasets_temporal.json'))
    print('Preparado:', {k: v['examples'] for k, v in audits.items()}, 'proteção', len(basic), flush=True)


def plan_for(arm, original, rows, lengths, seed):
    if arm == 'intercalado':
        return interleaved_plan(original, rows)
    if arm == 'redistribuido':
        return redistributed_plan(original, rows, lengths, seed)
    return list(original)


def make_batches(rows, data, plan):
    hashes = hashlib.sha256()
    targets = hashlib.sha256()
    batches = []
    for start in range(0, len(plan), 8):
        chosen = [rows[i] for i in plan[start:start + 8]]
        x, y = data.batch(chosen, 512, 'cpu')
        hashes.update(x.numpy().tobytes())
        hashes.update(y.numpy().tobytes())
        for row in chosen:
            answer = data.encode(row, 512)[1]
            targets.update(answer[answer != 0].numpy().tobytes())
        batches.append((x, y))
    return batches, hashes.hexdigest(), targets.hexdigest()


def evaluate(model, panels, derived, tokenizer, config, gate, device, old_evaluation=None):
    result = {}
    if old_evaluation is not None:
        result.update(old_evaluation)
    else:
        for label in ('train_probe', 'dev_names', 'dev_structures', 'validation'):
            result[label] = evaluate_names(model, panels[label], device, gate=gate)
        result['validation_own'] = evaluate_names(model, panels['validation'], device, own_history=True, gate=gate)
    result['copy'] = evaluate_names(model, derived['copy_development'], device, gate=gate)
    result['temporal'] = evaluate_names(model, derived['temporal_development'], device, gate=gate)
    guard_data = ConversationData({'train': [], 'basic_validation': derived['basic_protection_evaluation_only']}, tokenizer)
    result['basic'] = ConversationEngine(config, None, gate).evaluate(model, guard_data, device,
                            split='basic_validation', generation=True, interruptible=True)
    return result


def pilot(arm, seed, config, gate, cp, tokenizer, panels, derived, device, training_rows=None):
    folder = OUT / f'{arm}_s{seed}'
    folder.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    rows = training_rows if training_rows is not None else (derived['explicit_train'] if arm == 'temporal_explicito' else panels['train'])
    data = ConversationData({'train': rows}, tokenizer)
    lengths = [int((data.encode(row, 512)[1] != 0).sum()) for row in rows]
    original = original_plan(panels['train'], seed)
    plan = plan_for(arm, original, rows, lengths, seed)
    stats = exposure(plan, rows, lengths)
    if stats['supervised_tokens'] != sum(lengths[i] for i in original):
        raise ValueError('Orçamento de tokens supervisionados alterado.')
    batches, batch_hash, target_hash = make_batches(rows, data, plan)
    stats['padded_input_tokens'] = sum(x.numel() for x, _ in batches)
    stats['unpadded_input_tokens'] = sum(len(data.encode(rows[i], 512)[0]) for i in plan)
    stats['max_sequence_tokens'] = max(x.shape[1] for x, _ in batches)
    atomic_json(folder / 'plano.json', {'indices': plan, 'exposure': stats, 'batch_sha256': batch_hash,
                                       'target_stream_sha256': target_hash})
    old = read(BASE / f'sft_s{seed}/result.json') if arm == 'original' else None
    if old is not None and batch_hash != old['batch_plan_sha256']:
        raise ValueError('Plano original não foi reconstituído exatamente.')
    model = model_from(cp, tokenizer, device)
    initial_hash = weight_hash(model)
    history, curves, resources = [], [], []
    curves.append({'step': 0, 'train_objective_dataset': measured_loss(model, data, rows, device),
                   'validation': measured_loss(model, data, panels['validation'], device)})
    elapsed = 0.
    reused = arm == 'original' and seed == 42
    optimizer = None
    try:
        if reused:
            source = BASE / 'candidato_nomes.pt'
            saved = load_checkpoint(source)
            model.load_state_dict(saved['model'])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'], weight_decay=.01)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            for index in range(120):
                gate.wait_if_paused()
                if index % 10 == 0:
                    resources.append({'step': index, **check_resources(config)})
                if elapsed > PROTOCOL['max_training_seconds']:
                    raise TimeoutError('Limite curto do piloto; nenhum treino longo foi iniciado.')
                started = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr = PROTOCOL['learning_rate'] * min(1., (index + 1) / 10)
                optimizer.param_groups[0]['lr'] = lr
                losses, tokens = [], 0
                for micro in range(2):
                    gate.check()
                    x, y = batches[index * 2 + micro]
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                        logits, _ = model(x, y)
                        loss = weighted_loss(logits.float(), y, (y != 0).float())
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Loss não finita.')
                    scaler.scale(loss / 2).backward()
                    losses.append(float(loss.detach()))
                    tokens += int((y != 0).sum())
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Gradiente não finito; pesos não atualizados.')
                protected_step(optimizer, scaler, gate)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed += time.monotonic() - started
                history.append({'step': index + 1, 'loss': sum(losses) / 2, 'lr': lr,
                                'supervised_tokens': tokens, 'grad_norm': float(norm), 'training_seconds': elapsed})
                if (index + 1) % 40 == 0:
                    point = {'step': index + 1, 'train_objective_dataset': measured_loss(model, data, rows, device),
                             'validation': measured_loss(model, data, panels['validation'], device)}
                    curves.append(point)
                    atomic_json(folder / 'curvas.json', curves)
                    print(arm, seed, index + 1, 'NLL val', round(point['validation']['loss'], 5), flush=True)
        final_hash = weight_hash(model)
        if old is not None and final_hash != old['final_weight_hash']:
            raise ValueError('Pesos do controle diferem do piloto original; não reutilizar avaliações antigas.')
        print('Avaliando', arm, seed, 'geração direta e proteção, sem treino nessas avaliações.', flush=True)
        evaluation = evaluate(model, panels, derived, tokenizer, config, gate, device,
                              old_evaluation=old['evaluation'] if old is not None else None)
        result = {'arm': arm, 'seed': seed, 'status': 'completed', 'logical_steps': 120,
                  'optimizer_steps_this_run': 0 if reused else 120, 'checkpoint_reused': reused,
                  'old_name_metrics_reused_after_hash_check': old is not None,
                  'initial_weight_hash': initial_hash, 'final_weight_hash': final_hash,
                  'batch_sha256': batch_hash, 'target_stream_sha256': target_hash,
                  'exposure': stats, 'curves': curves if not reused else old['curve'],
                  'training_seconds_this_run': elapsed, 'resources': resources,
                  'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                  'evaluation': evaluation, 'promoted': False, 'reserved_tests_used': False}
        atomic_json(folder / 'passos.json', history)
        atomic_json(folder / 'result.json', result)
        print('Concluído', arm, seed, 'temporal', evaluation['temporal']['context_correct'], '/', evaluation['temporal']['total'],
              'básicas', evaluation['basic']['exact_matches'], '/', evaluation['basic']['examples'], flush=True)
        return result, model
    except BaseException as exc:
        atomic_json(folder / 'erro.json', {'error': str(exc), 'history': history, 'curves': curves, 'promoted': False})
        raise
    finally:
        del optimizer, batches, data
        gc.collect()


def temporal_rank(evaluation):
    skills = evaluation['temporal']['by_skill']
    return min(skills[s]['rates']['context_correct'] for s in ('atual', 'anterior', 'explicar_mudanca'))


def run():
    protocol = read(OUT / 'protocolo.json')
    if (OUT / 'preservacao_antes.json').exists():
        raise FileExistsError('Rodada já iniciada; não sobrescrever resultados.')
    config, active_id, path, cp, tokenizer = setup()
    if protocol['parent_sha256'] != sha(path) or protocol['name_evaluator_sha256'] != sha(ROOT / 'tac/neural/name_generalization.py'):
        raise ValueError('Ativo ou avaliador alterado desde o congelamento.')
    if protocol['base_dataset_sha256'] != sha(BASE / 'dados_desenvolvimento.json') or protocol['derived_dataset_sha256'] != sha(OUT / 'dados_derivados.json'):
        raise ValueError('Dados congelados alterados.')
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC antes dos pilotos isolados.')
    before = None
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        before = state()
        atomic_json(OUT / 'preservacao_antes.json', before)
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Preserve 512 MiB livres além do candidato.')
        torch.set_num_threads(config.get('cpu_threads'))
        torch.use_deterministic_algorithms(True)
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        panels = read(BASE / 'dados_desenvolvimento.json')['panels']
        derived = read(OUT / 'dados_derivados.json')
        original_model = model_from(cp, tokenizer, device)
        print('Avaliando ativo original no desenvolvimento e nas perguntas básicas públicas.', flush=True)
        baseline = evaluate(original_model, panels, derived, tokenizer, config, gate, device,
                            old_evaluation=read(BASE / 'ativo.json')['evaluation'])
        assert baseline['basic']['exact_matches'] == 131 and baseline['basic']['examples'] == 142
        atomic_json(OUT / 'ativo.json', {'weight_hash': weight_hash(original_model), 'evaluation': baseline})
        del original_model
        records, arm_states = [], {}
        for arm in ARMS:
            states = []
            for seed in PROTOCOL['seeds']:
                result, model = pilot(arm, seed, config, gate, cp, tokenizer, panels, derived, device)
                records.append(result)
                states.append((temporal_rank(result['evaluation']), seed, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            states.sort(key=lambda item: (item[0], item[1]))
            arm_states[arm] = (states[1][1], states[1][2])
            del states
        aggregates = {}
        for arm in ARMS:
            runs = [r for r in records if r['arm'] == arm]
            mean_skills = {s: sum(r['evaluation']['temporal']['by_skill'][s]['rates']['context_correct'] for r in runs) / 3
                           for s in ('atual', 'anterior', 'explicar_mudanca')}
            aggregates[arm] = {'temporal_skill_means': mean_skills, 'rank': min(mean_skills.values()),
                'basic_mean': sum(r['evaluation']['basic']['exact_accuracy'] for r in runs) / 3,
                'copy_mean': sum(r['evaluation']['copy']['rates']['characters_exact'] for r in runs) / 3,
                'validation_mean': sum(r['evaluation']['validation']['rates']['context_correct'] for r in runs) / 3,
                'median_seed': arm_states[arm][0]}
        chosen = max(ARMS, key=lambda a: (aggregates[a]['rank'], aggregates[a]['validation_mean'], a == 'original'))
        seed, selected_state = arm_states[chosen]
        selection = {'chosen_for_temporal_diagnosis': chosen, 'median_seed': seed, 'aggregates': aggregates,
                     'not_production_approval': True, 'promoted': False, 'reserved_tests_used': False}
        atomic_json(OUT / 'selecao_congelada.json', selection)
        target = OUT / 'candidato_temporal.pt'
        torch.save({'format': 'TAC-chat-v2', 'task': TASK, 'tokenizer_state': tokenizer.state(),
                    'tokenizer': tokenizer.version, 'model_config': cp['model_config'], 'model': selected_state,
                    'parent': active_id, 'step': 120, 'research_only': True, 'promotion_eligible': False,
                    'dataset_hash': protocol['derived_dataset_sha256'], 'optimizer_saved': False,
                    'training_config': {**protocol, 'arm': chosen, 'seed': seed}}, target)
        saved = load_checkpoint(target)
        saved_model = model_from(saved, tokenizer, device)
        selected_run = next(r for r in records if (r['arm'], r['seed']) == (chosen, seed))
        assert weight_hash(saved_model) == selected_run['final_weight_hash']
        atomic_json(OUT / 'resultado.json', {'status': 'completed', 'pilots': len(records), 'steps_each': 120,
            'optimizer_updates_this_run': sum(r['optimizer_steps_this_run'] for r in records),
            'chosen': chosen, 'median_seed': seed, 'checkpoint_sha256': sha(target), 'checkpoint_reload_verified': True,
            'parameter_count': saved_model.parameter_count, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'device': str(device), 'torch': str(torch.__version__), 'promoted': False, 'reserved_tests_used': False,
            'full_production_approval_evaluated': False})
    finally:
        if before is not None:
            after = state()
            atomic_json(OUT / 'preservacao_depois.json', after)
            atomic_json(OUT / 'preservacao_verificada.json', {'all_preserved': before == after,
                        'checks': {k: before[k] == after[k] for k in before}})
            print('Originais preservados:', before == after, flush=True)
        lock.unlock()
        if before is not None and before != after:
            raise RuntimeError('Falha de preservação detectada.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Use uma pasta dentro de TAC/reports.')
    prepare() if args.prepare else run()
