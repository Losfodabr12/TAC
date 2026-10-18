"""Sondas pós-seleção, somente desenvolvimento: tokenizer, cópia e contexto."""
from __future__ import annotations

import gc
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.config import atomic_json
from tac.diagnostics import check_resources
from tac.learning import LearningGate
from tac.neural.conversation import ConversationData, load_checkpoint
from tac.neural.name_generalization import evaluate_names, name_weights
from scripts.run_name_generalization import OUT, read, setup
from scripts.run_controlled_experiments import model_from, sha


@torch.inference_mode()
def token_probe(model, data, rows, device):
    model.eval()
    count, hits, first_count, first_hits, nll = 0, 0, 0, 0, 0.
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        x, y = data.batch(batch, 512, device)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for index, row in enumerate(batch):
            w = name_weights(data, row, 512, 2.)
            mask[index, :len(w)] = (w > 1).to(device)
        logits, _ = model(x, y)
        predicted = logits.argmax(-1)
        first = mask & ~torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none', ignore_index=0).view_as(y)
        count += int(mask.sum())
        hits += int(((predicted == y) & mask).sum())
        first_count += int(first.sum())
        first_hits += int(((predicted == y) & first).sum())
        nll += float(ce[mask].sum())
    return {'name_target_tokens': count, 'name_token_accuracy_teacher_forced': hits / count,
            'name_token_nll_teacher_forced': nll / count, 'first_name_tokens': first_count,
            'first_name_token_accuracy': first_hits / first_count,
            'warning': 'Usa a resposta anterior correta como contexto: não é geração livre nem mede generalização sozinho.'}


def main():
    if (OUT / 'sondas_diagnosticas.json').exists():
        raise FileExistsError('Sondas já registradas; não sobrescrever.')
    result = read(OUT / 'resultado.json')
    config, _, _, active, tokenizer = setup()
    if sha(OUT / 'candidato_nomes.pt') != result['checkpoint_sha256']:
        raise ValueError('Candidato não corresponde à seleção congelada.')
    panels = read(OUT / 'dados_desenvolvimento.json')['panels']
    validation = panels['validation']
    ablated = [{**r, 'messages': [[role, text if role == 'user' else 'Entendido.'] for role, text in r['messages']]}
               for r in validation]
    names = {n: kind for row in validation for n, kind in row['name_kinds'].items()}
    copy_rows = [{'messages': [['user', f'Repita exatamente, sem mudar as letras, este nome: {n}.']],
                  'answer': n + '.', 'skill': 'copia_isolada', 'group': f'copy_{i}', 'episode': f'copy_{i}',
                  'turn': 0, 'family': 'copy_probe', 'imported': True, 'target_names': {'current': n},
                  'source_names': [n], 'name_kinds': {n: kind}} for i, (n, kind) in enumerate(sorted(names.items()))]
    data = ConversationData({'train': panels['train']}, tokenizer)
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC para executar as sondas.')
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        torch.set_num_threads(config.get('cpu_threads'))
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        measurements = {}
        for label in ('active', 'candidate'):
            cp = active if label == 'active' else load_checkpoint(OUT / 'candidato_nomes.pt')
            model = model_from(cp, tokenizer, device)
            measurements[label] = {'train_tokens': token_probe(model, data, panels['train_probe'], device),
                                   'validation_tokens': token_probe(model, data, validation, device),
                                   'assistant_name_ablated': evaluate_names(model, ablated, device, gate=gate),
                                   'isolated_copy': evaluate_names(model, copy_rows, device, gate=gate)}
            print('Sonda', label, 'nomes sem eco da TAC:', measurements[label]['assistant_name_ablated']['context_correct'],
                  'cópia isolada:', measurements[label]['isolated_copy']['characters_exact'], '/', len(copy_rows), flush=True)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        atomic_json(OUT / 'sondas_diagnosticas.json', {'scope': 'desenvolvimento pós-seleção; não escolhe modelo nem treina',
                    'selection_sha256': sha(OUT / 'selecao_congelada.json'), 'measurements': measurements,
                    'ablation': 'Substitui só falas anteriores da TAC por Entendido.; nomes informados pelo usuário permanecem no contexto.',
                    'old_reserved_used': False, 'new_final_used': False, 'optimizer_created': False})
    finally:
        lock.unlock()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Pasta de rodada inválida; use TAC/reports.')
    main()
