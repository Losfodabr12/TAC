"""Chat manual com pesos experimentais; sem consulta, treino ou gravação de falas."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PySide6.QtCore import QCoreApplication, QLockFile
from tac.diagnostics import check_resources
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import load_checkpoint
from tac.neural.controlled_eval import trace_reply
from scripts.run_controlled_experiments import LiveSettings, model_from, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('experimental', 'ativo'), default='experimental')
    args = parser.parse_args()
    config = LiveSettings(ROOT)
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(config.data / 'tac.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError('Feche a janela principal da TAC antes de abrir este teste separado.')
    try:
        folder = ROOT / 'reports/aprendizado_conversas_0130'
        run = json.loads((folder / 'resultado.json').read_text(encoding='utf-8'))
        if args.model == 'experimental':
            path = folder / 'candidato_experimental.pt'
            expected = run['checkpoint_sha256']
        else:
            with sqlite3.connect((config.data / 'tac.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
                active = db.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
                filename = db.execute('SELECT filename FROM checkpoints WHERE id=?', (active,)).fetchone()[0]
            path = config.data / 'checkpoints' / filename
            expected = None
        if expected and sha(path) != expected:
            raise ValueError('O arquivo experimental não corresponde ao resultado registrado.')
        check_resources(config)
        torch.set_num_threads(config.get('cpu_threads'))
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(min(.75, config.get('vram_mb') * 2**20 / torch.cuda.get_device_properties(0).total_memory))
        cp = load_checkpoint(path)
        model = model_from(cp, LocalBPETokenizer.from_state(cp['tokenizer_state']), device).eval()
        print(f'\nTAC — teste da rede {args.model}: {model.parameter_count:,} parâmetros.')
        print('Respostas geradas pelos pesos. Sem biblioteca, Wikipédia ou respostas prontas.')
        print('Nada será gravado; o modelo ativo não será alterado.')
        print('Digite /novo para limpar a conversa ou /sair para fechar.\n')
        history = []
        while True:
            try:
                text = input('Você: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text == '/sair':
                break
            if text == '/novo':
                history.clear()
                print('Nova conversa.\n')
                continue
            if not text:
                continue
            check_resources(config)
            history.append(['user', text])
            answer, trace = trace_reply(model, history[-7:], device, max_tokens=128)
            print('TAC:', answer or '[A rede encerrou sem produzir texto.]')
            if trace['hit_limit_without_eos']:
                print('[A resposta atingiu o limite de tamanho sem encerrar.]')
            history.append(['assistant', answer])
            print()
    finally:
        lock.unlock()


if __name__ == '__main__':
    main()
