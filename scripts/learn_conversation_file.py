"""Treina cópia isolada somente com o JSONL fornecido; nunca ativa candidatos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
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
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import load_checkpoint, TASK
from tac.neural.file_learning import adapt_records, EpochSampler, evaluate_file, measured_loss
from tac.neural.training import weight_hash
from scripts.run_controlled_experiments import LiveSettings, model_from, protected_step, sha

PROTOCOL = {'seed': 42, 'max_steps': 360, 'block_steps': 120, 'learning_rate': .00008,
            'warmup_steps': 10, 'schedule': 'constant_after_linear_warmup', 'batch_size': 8,
            'accumulation': 2, 'weight_decay': .01, 'clip_grad_norm': 1., 'optimizer': 'AdamW',
            'betas': [.9, .999], 'epsilon': 1e-8, 'optimizer_reset': True, 'dropout': 0.,
            'stop': '100% de respostas literais no material estudado, com histórico original e próprio; ou limite de 360 passos/180s de treino',
            'training_seconds_limit': 180, 'generation': 'greedy, 128 novos tokens, sem filtros ou respostas prontas',
            'precision': 'fp16 autocast e GradScaler em CUDA; parâmetros e AdamW fp32; avaliação fp32',
            'loss': 'cross entropy somente sobre a resposta assistant e EOS; prompt mascarado',
            'sampling': 'permutação sem reposição até completar cada passagem pelo arquivo',
            'replay': False, 'kl': False, 'internal_examples': 0, 'rag': False, 'documents': False,
            'promotion': False, 'test_reserved_read': False, 'generalization_evaluated': False}


def preservation(source_folder):
    # Auditoria existente: catálogo e configurações somente leitura, teste só SHA.
    spec = importlib.util.spec_from_file_location('file_readonly_audit', ROOT / 'reports/investigacao_0130/audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    state = audit.state()
    state['supplied_files'] = {p.name: sha(p) for p in sorted(source_folder.iterdir()) if p.is_file()}
    # Preserva todos os pesos originais, não apenas os três modelos ativos.
    state['all_original_checkpoints'] = {p.name: sha(p) for p in sorted((ROOT / 'data/checkpoints').glob('*.pt'))}
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/aprendizado_conversas_0130')
    args = parser.parse_args()
    source, out = args.input.resolve(), args.output.resolve()
    if source.suffix.lower() != '.jsonl' or source.name != 'TAC_TREINO_CONVERSAS.jsonl':
        parser.error('Este protocolo aceita apenas TAC_TREINO_CONVERSAS.jsonl; não abra o teste reservado.')
    if not out.is_relative_to(ROOT / 'reports') or out == ROOT / 'reports' or out.exists():
        parser.error('Escolha uma pasta NOVA dentro de TAC/reports; não sobrescrever resultados.')
    config = LiveSettings(ROOT)
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('A TAC está aberta. Feche-a para isolar o experimento de qualquer treino automático.')
    before = None
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        if source.stat().st_size > config.get('max_file_mb') * 2**20:
            raise ValueError('Arquivo maior que o limite de importação.')
        out.mkdir(parents=True, exist_ok=False)
        before = preservation(source.parent)
        atomic_json(out / 'preservacao_antes.json', before)
        with sqlite3.connect((config.data / 'tac.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
            active_id = db.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
            filename = db.execute('SELECT filename FROM checkpoints WHERE id=?', (active_id,)).fetchone()[0]
        parent = config.data / 'checkpoints' / filename
        cp = load_checkpoint(parent)
        tokenizer = LocalBPETokenizer.from_state(cp['tokenizer_state'])
        records, data, validation = adapt_records(source.read_bytes(), tokenizer, cp['model_config']['context'])
        atomic_json(out / 'validacao_importacao.json', validation)
        if validation['rejected'] or not data.sets['train']:
            raise ValueError('Importação inválida; veja validacao_importacao.json. Nenhum treino iniciado.')
        (out / 'TAC_TREINO_COMPATIVEL.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        data.save(out / 'dataset.json')
        estimated = sum(t.numel() * t.element_size() for t in cp['model'].values())
        if shutil.disk_usage(out).free < estimated + 512 * 2**20:
            raise OSError('É necessário manter 512 MiB livres além dos pesos do candidato.')
        torch.set_num_threads(config.get('cpu_threads'))
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(PROTOCOL['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(PROTOCOL['seed'])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
            torch.cuda.reset_peak_memory_stats()
        protocol = {**PROTOCOL, 'parent': active_id, 'parent_sha256': sha(parent), 'input_sha256': sha(source),
                    'dataset_hash': data.digest, 'model_config': cp['model_config'], 'tokenizer': tokenizer.version,
                    'vocab_size': tokenizer.vocab_size, 'device': str(device), 'torch': str(torch.__version__),
                    'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                    'ram_limit_mb': config.get('ram_mb'), 'vram_limit_mb': config.get('vram_mb'),
                    'cpu_threads': config.get('cpu_threads'), 'examples': len(data.sets['train']),
                    'assessment': 'aquisição no próprio treino; nenhum teste inédito/final'}
        atomic_json(out / 'configuracao.json', protocol)
        model = model_from(cp, tokenizer, device)
        protocol['parameter_count'] = model.parameter_count
        atomic_json(out / 'configuracao.json', protocol)
        initial_hash = weight_hash(model)
        curve, losses, resources = [], [], []

        def evaluate(step):
            gate.check()
            print(f'Avaliando passo {step}: geração neural direta, sem consultas.', flush=True)
            measured = measured_loss(model, data, data.sets['train'], device)
            gold = evaluate_file(model, data, device)
            own = evaluate_file(model, data, device, own_history=True)
            result = {'step': step, **measured, 'gold_history': gold, 'own_history': own}
            atomic_json(out / f'avaliacao_{step:04d}.json', result)
            curve.append({'step': step, **measured, 'exact_gold': gold['exact_literal'],
                          'exact_own': own['exact_literal'], 'normalized_gold': gold['exact_normalized'],
                          'eos_gold': gold['eos'], 'eos_own': own['eos']})
            atomic_json(out / 'curva.json', curve)
            print(f"passo {step}: loss {measured['loss']:.5f}; originais {gold['exact_literal']}/{gold['examples']}; próprio histórico {own['exact_literal']}/{own['examples']}", flush=True)
            return result

        baseline = evaluate(0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'], weight_decay=PROTOCOL['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        sampler = EpochSampler(data.sets['train'], PROTOCOL['seed'])
        elapsed_training, completed, reason = 0., 0, 'limite_de_passos'
        current = baseline
        for index in range(PROTOCOL['max_steps']):
            gate.wait_if_paused()
            if index % 10 == 0:
                resources.append({'step': index, **check_resources(config)})
            if elapsed_training >= PROTOCOL['training_seconds_limit']:
                reason = 'limite_de_tempo'
                break
            start = time.monotonic()
            lr = PROTOCOL['learning_rate'] * min(1., (index + 1) / PROTOCOL['warmup_steps'])
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            model.train()
            ce_values, tokens = [], 0
            for _ in range(PROTOCOL['accumulation']):
                gate.check()
                rows = sampler.take(PROTOCOL['batch_size'])
                x, y = data.batch(rows, model.cfg.context, device)
                for r in rows:
                    sampler.tokens[r['skill']] += int((data.encode(r, model.cfg.context)[1] != 0).sum())
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss não finita: interrompido antes de atualizar os pesos.')
                scaler.scale(loss / PROTOCOL['accumulation']).backward()
                ce_values.append(float(loss.detach()))
                tokens += int((y != 0).sum())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['clip_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Gradiente não finito: atualização cancelada.')
            protected_step(optimizer, scaler, gate)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_training += time.monotonic() - start
            completed = index + 1
            losses.append({'step': completed, 'loss': sum(ce_values) / len(ce_values), 'lr': lr,
                           'tokens': tokens, 'gradient_norm_before_clip': float(norm), 'train_seconds': elapsed_training})
            if completed % 20 == 0:
                print(f"Treino {completed}/{PROTOCOL['max_steps']}, loss do lote {losses[-1]['loss']:.5f}", flush=True)
                atomic_json(out / 'passos.json', losses)
            if completed % PROTOCOL['block_steps'] == 0:
                current = evaluate(completed)
                n = len(data.sets['train'])
                if current['gold_history']['exact_literal'] == current['own_history']['exact_literal'] == n:
                    reason = 'todos_os_exemplos_reproduzidos'
                    break
                print('Ainda há respostas estudadas incorretas; próximo bloco curto, sem mudar hiperparâmetros.', flush=True)
        if current['step'] != completed:
            current = evaluate(completed)
        gate.check()
        final_hash = weight_hash(model)
        checkpoint = {'format': 'TAC-chat-v2', 'task': TASK, 'tokenizer': tokenizer.version,
                      'tokenizer_state': tokenizer.state(), 'model_config': cp['model_config'],
                      'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      'dataset_hash': data.digest, 'step': completed, 'parent': active_id,
                      'training_config': protocol, 'research_only': True,
                      'promotion_eligible': False, 'optimizer_saved': False}
        candidate_path = out / 'candidato_experimental.pt'
        torch.save(checkpoint, candidate_path)
        result = {'status': 'completed', 'steps': completed, 'stop_reason': reason, 'train_seconds': elapsed_training,
                  'initial_weight_hash': initial_hash, 'final_weight_hash': final_hash,
                  'weights_changed': initial_hash != final_hash, 'checkpoint_sha256': sha(candidate_path),
                  'exposure': sampler.summary(), 'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                  'peak_cuda_reserved_mb': torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else None,
                  'resources': resources, 'active_preserved': None, 'promoted': False, 'generalization_evaluated': False,
                  'reserved_test_used': False, 'baseline_step': baseline['step'], 'final_step': current['step'],
                  'optimizer_resume_available': False, 'notes': 'Aquisição no treino. Sem replay/KL por isolamento solicitado; não elegível para produção.'}
        atomic_json(out / 'passos.json', losses)
        atomic_json(out / 'resultado.json', result)
        with (out / 'curva.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        # Carrega de novo o artefato real; não confunde estado em RAM com pesos salvos.
        loaded = load_checkpoint(candidate_path)
        model.load_state_dict(loaded['model'])
        if weight_hash(model) != final_hash:
            raise ValueError('Checkpoint gravado não reproduz o hash do candidato.')
        result['checkpoint_reload_verified'] = True
        atomic_json(out / 'resultado.json', result)
    except BaseException as exc:
        if out.exists():
            atomic_json(out / 'erro.json', {'type': type(exc).__name__, 'message': str(exc), 'promoted': False})
        raise
    finally:
        if before is not None:
            after = preservation(source.parent)
            atomic_json(out / 'preservacao_depois.json', after)
            verified = {'all_preserved': before == after,
                        'checks': {k: before[k] == after[k] for k in before},
                        'reserved_files_content_used': False, 'reserved_files_operation': 'SHA-256 somente', 'promoted': False}
            atomic_json(out / 'preservacao_verificada.json', verified)
            result_path = out / 'resultado.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                result['active_preserved'] = verified['checks']['active']
                result['all_original_files_preserved'] = verified['all_preserved']
                atomic_json(result_path, result)
            print('Preservação integral:', verified['all_preserved'], flush=True)
        lock.unlock()
        if before is not None and not verified['all_preserved']:
            raise RuntimeError('A auditoria detectou alteração nos originais; veja preservacao_verificada.json.')


if __name__ == '__main__':
    main()
