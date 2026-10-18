"""Sonda final: recuperação do anterior nos próprios exemplos realmente treinados.

Não escolhe configuração, não ajusta pesos e não lê testes finais.
"""
from __future__ import annotations

import gc
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
from tac.neural.file_learning import measured_loss
from tac.neural.name_generalization import evaluate_names
from scripts.run_temporal_memory import OUT, BASE, read, setup
from scripts.run_controlled_experiments import model_from, sha


def main():
    target = OUT / 'sonda_memorizacao_anterior.json'
    if target.exists():
        raise FileExistsError('Sonda já registrada; preservar resultados.')
    phase1 = read(OUT / 'resultado.json')
    phase2 = read(OUT / 'cobertura_temporal/resultado.json')
    original_rows = read(BASE / 'dados_desenvolvimento.json')['panels']['train']
    explicit = read(OUT / 'dados_derivados.json')['explicit_train']
    temporal_previous = [r for r in read(OUT / 'dados_derivados.json')['temporal_development'] if r['skill'] == 'anterior']
    coverage = read(OUT / 'cobertura_temporal/treino_derivado.json')['rows']
    phase1_rows = explicit if phase1['chosen'] == 'temporal_explicito' else original_rows
    cases = [('controle_original_seed42', BASE / 'candidato_nomes.pt', original_rows),
             (f"primeira_rodada_{phase1['chosen']}_seed{phase1['median_seed']}", OUT / 'candidato_temporal.pt', phase1_rows),
             (f"cobertura_seed{phase2['median_seed']}", OUT / 'cobertura_temporal/candidato_cobertura.pt', coverage)]
    config, _, _, _, tokenizer = setup()
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a TAC antes das sondas.')
    try:
        gate = LearningGate(config, None)
        gate.begin()
        check_resources(config)
        torch.set_num_threads(config.get('cpu_threads'))
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        outputs = {}
        for label, checkpoint, training in cases:
            model = model_from(load_checkpoint(checkpoint), tokenizer, device)
            rows = [r for r in training if r['skill'] == 'nome_anterior']
            data = ConversationData({'train': rows}, tokenizer)
            loss = measured_loss(model, data, rows, device)
            evaluation = evaluate_names(model, rows, device, gate=gate)
            # Intervenção apenas na formulação da pergunta; mesmos fatos e nomes
            # inéditos. Esta sonda não altera a avaliação principal nem o chat.
            known_question = next(r['messages'][-1][1] for r in rows if r['family'] == 't0')
            familiar = [{**r, 'messages': [*r['messages'][:-1], ['user', known_question]]} for r in temporal_previous]
            question_probe = evaluate_names(model, familiar, device, gate=gate)
            outputs[label] = {'checkpoint_sha256': sha(checkpoint), 'loss': loss, 'evaluation': evaluation,
                              'familiar_question_probe': question_probe, 'familiar_question': known_question,
                              'scope': '64 perguntas nome_anterior nos históricos efetivamente usados no treino deste checkpoint; sonda de 16 históricos novos com formulação familiar'}
            print('Sonda de memorização', label, evaluation['context_correct'], '/', evaluation['total'], flush=True)
            del model, data
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        atomic_json(target, {'models': outputs, 'training_performed': False, 'reserved_tests_used': False,
                            'selection_changed': False, 'limitation': 'Checkpoints guardados e seeds identificadas; sonda descritiva, não média de três seeds nem teste independente.'})
    finally:
        lock.unlock()


if __name__ == '__main__':
    main()
