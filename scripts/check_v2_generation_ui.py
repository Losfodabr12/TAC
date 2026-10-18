"""Sonda real da janela Windows e do envio assíncrono, sem treinar."""
import json
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['QT_QPA_PLATFORM'] = 'windows'

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from tac.core import Core
from tac.ui import Window, apply_theme
from diagnose_v2_generation import forbidden, save, OUT, QUESTIONS
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='')
    parser.add_argument('--full-sequence', action='store_true')
    args = parser.parse_args()
    torch.save = forbidden
    torch.optim.AdamW.step = forbidden
    app = QApplication([])
    core = Core(ROOT)
    assert core.config.get('conversation_generation') == 'controlled'
    assert not core.config.get('auto_train')
    apply_theme(app, core.config)
    w = Window(core);w.timer.stop();w.open_chat(2);w.show()
    app.processEvents()
    w.clear_conversation()
    questions = QUESTIONS if args.full_sequence else QUESTIONS[:1]
    sequence = []
    w.input.setText(questions[0])
    w.send_chat()
    def finish():
        if w.job is not None:
            QTimer.singleShot(100, finish)
            return
        assert core.last_answer is not None
        question, answer = core.last_answer
        assert answer['checkpoint'] == '64f45f52fee54053a8f0c6b16d9df172'
        assert answer['generation'] == 'TAC-v2' and answer['sources'] == []
        assert answer['text'] in w.transcript.toPlainText()
        assert answer['decoding']['max_new_tokens'] == 128
        assert question == questions[len(sequence)]
        sequence.append({'question': question, 'answer': answer})
        if len(sequence) < len(questions):
            w.input.setText(questions[len(sequence)])
            w.send_chat()
            QTimer.singleShot(100, finish)
            return
        assert w.grab().save(str(OUT/(args.prefix + 'interface_corrigida.png')))
        result = {'status': 'OK', 'platform': app.platformName(), 'question': question,
                  'answer': answer, 'chat_mode': w.chat_mode.currentText(),
                  'window_title': w.windowTitle(), 'real_send_chat_job': True,
                  'text_rendered': True, 'training_updates': 0, 'sequential_messages': sequence}
        save(args.prefix + 'interface_gui_native.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        w.close()
    QTimer.singleShot(100, finish)
    return app.exec()


if __name__ == '__main__':raise SystemExit(main())
