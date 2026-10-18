"""Verifica a entrega de diálogos/correções usando pesos reais, sem treinar."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from tac.config import atomic_json
from tac.core import Core
from tac.neural.conversation import ConversationData
from tac.neural.training import weight_hash


def main():
    core = Core()
    torch.set_num_threads(2)
    baseline = json.loads((ROOT / "reports/baseline_060.json").read_text(encoding="utf-8"))
    run = json.loads((ROOT / "reports/conversation_latest.json").read_text(encoding="utf-8"))
    checkpoint = core.conversation.active_id()
    assert run['status'] == 'aprovado' and checkpoint == run['checkpoint']
    assert all(run['promotion_checks'].values())
    model, cp = core.conversation.load(checkpoint, 'cpu')
    assert model.parameter_count == 10_004_544
    assert weight_hash(model) == run['final_hash'] != run['initial_hash']
    assert core.trainer.active_id() == baseline['active_reader']
    assert core.dialogue.active_id() == baseline['active_dialogue']
    sources = {r['id']: r for r in core.memory.sources()}
    assert len(sources) == len(baseline['sources'])
    for old in baseline['sources']:
        assert all(sources[old['id']][key] == old[key] for key in ('content_hash', 'origin', 'chunks'))
    assert core.config.get('device') == baseline['config']['device']
    assert core.config.get('auto_on_start') == baseline['config']['auto_on_start']
    assert core.config.get('save_interactions') == baseline['config']['save_interactions']
    data = ConversationData.load(core.config.data / 'datasets' / f"conversation_{cp['dataset_hash']}.json")
    transcripts = []
    # Exercícios conhecidos: teste de integração, não evidência de generalização.
    cases = [
        ['Minha cor favorita é azul.', 'Agora prefiro verde.', 'Qual é minha cor favorita?'],
        ['Meu nome é Pedro.', 'Na verdade, me chame de Ana.', 'Qual é meu nome?'],
        ['Vamos falar de jogos.', 'Mudei de ideia, vamos falar de música.', 'Qual é o assunto agora?'],
    ]
    for inputs in cases:
        core.neural_history.clear()
        turns = []
        for text in inputs:
            reply = core.chat(text, conversation=True)
            assert reply['mode'] == 'diálogo neural experimental' and not reply['sources']
            turns.append({'user': text, 'assistant': reply['text']})
        transcripts.append(turns)
    assert transcripts[0][1]['assistant'] == 'Agora sua cor favorita é verde.'
    assert transcripts[0][2]['assistant'] == 'Agora sua cor favorita é verde.'
    assert weight_hash(core.conversation._cached_model) == run['final_hash']
    result = {
        'status': 'ok', 'checkpoint': checkpoint, 'parameters': model.parameter_count,
        'hash': run['final_hash'], 'dataset_hash': data.digest,
        'examples': {s: len(rows) for s, rows in data.sets.items()},
        'sources_preserved': len(sources), 'reader_and_classifier_preserved': True,
        'inference_does_not_change_weights': True, 'integration_sessions': transcripts,
        'checkpoint_file_sha256': hashlib.sha256(core.trainer.checkpoint_path(checkpoint).read_bytes()).hexdigest(),
    }
    with sqlite3.connect(core.db.path) as source, sqlite3.connect(ROOT / 'reports/dialogos_060.sqlite3') as target:
        source.backup(target)
    atomic_json(ROOT / 'reports/verification_060.json', result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
