"""Reconstrói só os pesos perdidos após bloqueio de disco; exige hash já medido."""
from __future__ import annotations

import gc
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_temporal_memory as engine
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.config import atomic_json
from tac.diagnostics import check_resources
from tac.learning import LearningGate
from tac.neural.conversation import ConversationData, TASK, load_checkpoint
from tac.neural.name_generalization import weighted_loss
from tac.neural.training import weight_hash
from scripts.run_controlled_experiments import model_from, protected_step, sha

OUT = ROOT / 'reports/memoria_temporal_0130/cobertura_temporal'


def main():
    if (OUT / 'resultado.json').exists() or (OUT / 'candidato_cobertura.pt').exists():
        raise FileExistsError('Artefato já existe; não sobrescrever.')
    records = [engine.read(OUT / f'cobertura_temporal_s{seed}/result.json') for seed in (42, 417, 903)]
    ordered = sorted(records, key=lambda r: (engine.temporal_rank(r['evaluation']), r['seed']))
    selected = ordered[1]
    seed = selected['seed']
    config, active_id, path, cp, tokenizer = engine.setup()
    protocol = engine.read(OUT / 'protocolo.json')
    if sha(path) != protocol['parent_sha256'] or sha(OUT / 'treino_derivado.json') != protocol['derived_train_sha256']:
        raise ValueError('Fontes alteradas; não reconstruir pesos diferentes.')
    rows = engine.read(OUT / 'treino_derivado.json')['rows']
    plan = engine.read(OUT / f'cobertura_temporal_s{seed}/plano.json')['indices']
    data = ConversationData({'train': rows}, tokenizer)
    batches, batch_hash, _ = engine.make_batches(rows, data, plan)
    if batch_hash != selected['batch_sha256']:
        raise ValueError('Plano não corresponde à execução medida.')
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC para recuperar o artefato isolado.')
    engine.OUT = OUT
    before = engine.state()
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Margem insuficiente; não iniciar a reconstrução.')
        torch.set_num_threads(config.get('cpu_threads'))
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        model = model_from(cp, tokenizer, device)
        assert weight_hash(model) == selected['initial_weight_hash']
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00008, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        history = []
        for index in range(120):
            gate.check()
            if index % 10 == 0:
                check_resources(config)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr'] = .00008 * min(1., (index + 1) / 10)
            values = []
            for micro in range(2):
                x, y = batches[index * 2 + micro]
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits, _ = model(x, y)
                    loss = weighted_loss(logits.float(), y, (y != 0).float())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss não finita.')
                scaler.scale(loss / 2).backward()
                values.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Gradiente não finito.')
            protected_step(optimizer, scaler, gate)
            history.append({'step': index + 1, 'loss': sum(values) / 2})
        final_hash = weight_hash(model)
        if final_hash != selected['final_weight_hash']:
            raise ValueError('Reconstrução diferente; não reutilizar resultados nem guardar como o mesmo candidato.')
        if shutil.disk_usage(OUT).free < 512 * 2**20 + 42_000_000:
            raise OSError('Disco sem margem; não gravar candidato.')
        target = OUT / 'candidato_cobertura.pt'
        torch.save({'format': 'TAC-chat-v2', 'task': TASK, 'model_config': cp['model_config'],
                    'tokenizer': tokenizer.version, 'tokenizer_state': tokenizer.state(),
                    'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'parent': active_id, 'step': 120, 'research_only': True, 'promotion_eligible': False,
                    'dataset_hash': protocol['derived_train_sha256'], 'optimizer_saved': False,
                    'training_config': {**protocol, 'arm': 'cobertura_temporal', 'seed': seed}}, target)
        loaded = load_checkpoint(target)
        digest = hashlib.sha256()
        for value in loaded['model'].values():
            digest.update(value.contiguous().numpy().tobytes())
        assert digest.hexdigest() == final_hash
        atomic_json(OUT / 'reconstrucao_checkpoint.json', {'reason': 'primeira gravação bloqueada por margem de disco; métricas já preservadas',
                    'seed': seed, 'optimizer_updates': 120, 'expected_weight_hash': selected['final_weight_hash'],
                    'actual_weight_hash': final_hash, 'exact_reconstruction': True, 'batch_sha256': batch_hash,
                    'evaluation_rerun': False, 'metrics_reused_only_after_exact_weight_match': True,
                    'history': history, 'promoted': False})
        atomic_json(OUT / 'resultado.json', {'status': 'completed', 'pilots': 3, 'steps_each': 120,
            'optimizer_updates_this_run': 480, 'recovery_optimizer_updates': 120,
            'chosen': 'cobertura_temporal', 'median_seed': seed, 'checkpoint_sha256': sha(target),
            'checkpoint_reload_verified': True, 'parameter_count': model.parameter_count,
            'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'promoted': False, 'reserved_tests_used': False})
        print('Checkpoint reconstruído exatamente:', seed, final_hash, flush=True)
    finally:
        after = engine.state()
        atomic_json(OUT / 'preservacao_reconstrucao.json', {'all_preserved': before == after,
                    'checks': {k: before[k] == after[k] for k in before}})
        lock.unlock()
        if before != after:
            raise RuntimeError('Falha de preservação na reconstrução.')


if __name__ == '__main__':
    main()
