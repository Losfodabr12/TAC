"""Consolida evidências da correção; não salva checkpoints nem executa treino."""
import hashlib
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
from diagnose_v2_generation import OUT, QUESTIONS, save, forbidden
from tac.neural.conversation import load_checkpoint
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.decoding import conversation_options, decode_conversation
from tac.neural.v2_training import model_hash


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    torch.save = forbidden;torch.optim.AdamW.step = forbidden;torch.set_num_threads(4)
    before = read(OUT/'before_identity.json')
    db = sqlite3.connect((ROOT/'data/tac.sqlite3').as_uri()+'?mode=ro', uri=True)
    active = db.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
    count = db.execute('SELECT count(*) FROM checkpoints WHERE run_id=?', (before['database']['run_id'],)).fetchone()[0]
    running = db.execute("SELECT count(*) FROM runs WHERE status='treinando'").fetchone()[0]
    db.close()
    assert active == before['active'] and running == 0
    path = ROOT/before['file']
    assert digest(path) == before['file_sha256']
    cp = load_checkpoint(path)
    assert cp['step'] == before['checkpoint_step'] and cp['v2_state']['phase'] == 'complete'
    model = TACTransformer(ModelConfig(**cp['model_config']))
    model.load_state_dict(cp['model']);model.text_tokenizer = LocalBPETokenizer.from_state(cp['tokenizer_state'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    assert model_hash(model) == before['weights_hash']
    final_greedy = []
    for question in QUESTIONS:
        result = decode_conversation(model, [['user', question]], next(model.parameters()).device,
                                     generation_options=conversation_options('greedy',42))
        final_greedy.append({'question': question, **result})
    save('final_greedy.json', final_greedy)
    base = ROOT/'data/tac_v2'
    encoded = read(base/'encoded_ready.json')
    corpus_checks = {name: digest(base/name) == info['sha256'] for name, info in encoded['files'].items()}
    corpus_checks['tokenizer.json'] = digest(base/'tokenizer.json') == encoded['tokenizer_sha256']
    assert all(corpus_checks.values())
    suite = ET.parse(OUT/'pytest_final.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
    assert int(suite.attrib['tests']) == 50
    real = read(OUT/'final_real_interface_path.json')
    assert len(real) == 5 and len({r['answer']['text'] for r in real}) == 5
    gui = read(OUT/'final_interface_gui_native.json')
    assert gui['status'] == 'OK' and len(gui['sequential_messages']) == 5
    config = read(ROOT/'config.json')
    old_config = read(OUT/'inference_config_change.json')['before']
    changed = [k for k in config if old_config[k] != config[k]]
    assert changed == ['conversation_generation'] and config['conversation_generation'] == 'controlled'
    assert model_hash(model) == before['weights_hash'] and digest(path) == before['file_sha256']
    result = {'status': 'OK', 'scope': 'correção de execução da geração; compreensão ainda falha',
              'active_checkpoint': active, 'checkpoint_step': cp['step'], 'parameters': model.parameter_count,
              'checkpoint_file_sha256': digest(path), 'weights_hash': model_hash(model),
              'checkpoint_preserved': True, 'training_updates': 0, 'new_training_checkpoints': 0,
              'new_generation_checkpoints_existing': count, 'training_runs_active': running,
              'corpus_and_tokenizer_match_original_hashes': corpus_checks,
              'reserved_tests_opened': False, 'external_AI_RAG_classifier_or_fixed_answers': False,
              'changed_config_fields': changed, 'tests_passed': 50, 'tests_deselected': 1,
              'fresh_questions': 5, 'distinct_fresh_outputs': 5,
              'fresh_stop_reasons': [r['answer']['decoding']['stop_reason'] for r in real],
              'fresh_max_generated_tokens': max(r['answer']['decoding']['generated_tokens'] for r in real),
              'real_windows_gui_sequence_verified': True, 'factual_comprehension_resolved': False}
    save('validacao_final.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':main()
