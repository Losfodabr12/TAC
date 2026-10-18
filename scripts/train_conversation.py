"""Execução manual de geração de respostas, com progresso e cancelamento visíveis."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import psutil
import torch
from PySide6.QtCore import QCoreApplication, QLockFile, QTimer
from tac.config import atomic_json
from tac.core import Core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--resume")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--profile", choices=("10m",))
    parser.add_argument("--text-corpus", help="SHA-256 do snapshot local autorizado")
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--from-checkpoint", help="Inicia nova etapa a partir de pesos próprios, mantendo a versão ativa")
    parser.add_argument("--retention-weight", type=float, default=2., help="Retenção da própria versão anterior em exemplos antigos de treino")
    parser.add_argument('--curriculum', choices=('fluency',), help='55% diálogos revisados, 30% linguagem, 15% exercícios antigos naturais')
    args = parser.parse_args()
    if args.show:
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
    else:
        app = QCoreApplication([])
    core = Core()
    original_device = core.config.get("device")
    if args.device:
        core.config.values["device"] = args.device
    lock = QLockFile(str(core.config.data / "tac.lock"))
    if not lock.tryLock(0):
        raise RuntimeError("Encerre a janela atual para iniciar esta execução exclusiva.")
    if args.resume:
        from tac.neural.conversation import load_checkpoint
        saved = load_checkpoint(core.trainer.checkpoint_path(args.resume))
        args.steps = saved["target_steps"]
        del saved
    process = psutil.Process()
    peak_ram = 0
    started = time.monotonic()
    def progress(step, message):
        nonlocal peak_ram
        peak_ram = max(peak_ram, process.memory_info().rss / 2**20)
        print(message, flush=True)
        atomic_json(ROOT / "reports/conversation_progress.json", {"step": step, "target": args.steps,
                    "message": message, "seconds": time.monotonic() - started, "peak_process_ram_mb": peak_ram})
    def train(update):
        def report(step, message):
            progress(step, message)
            update(step, message)
        return core.conversation.train(report, steps=args.steps, learning_rate=args.learning_rate, seconds=args.seconds, resume=args.resume,
                                       profile=args.profile, text_corpus=args.text_corpus, pretrain_steps=args.pretrain_steps,
                                       initial_checkpoint=args.from_checkpoint, retention_weight=args.retention_weight, curriculum=args.curriculum)
    def complete(result):
        result["peak_process_ram_mb"] = peak_ram
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_initialized() else None
        result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_initialized() else None
        atomic_json(ROOT / "reports" / f"conversation_{result['run_id']}.json", result)
        atomic_json(ROOT / "reports/conversation_latest.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("validation", "baseline", "context_validation", "reference_validation", "natural_baseline", "natural_validation", "session_baseline", "session_validation",'fluency_baseline','fluency_validation')}, ensure_ascii=True), flush=True)
        print(json.dumps({k: v for k, v in result["validation"].items() if k != "outputs"}, ensure_ascii=True), flush=True)
        if result.get("context_validation"):
            print(json.dumps({k: v for k, v in result["context_validation"].items() if k != "outputs"}, ensure_ascii=True), flush=True)
        if result.get("natural_validation"):
            print(json.dumps({k: v for k, v in result["natural_validation"].items() if k != "outputs"}, ensure_ascii=True), flush=True)
        if result.get('session_validation'):
            print(json.dumps({k:v for k,v in result['session_validation'].items() if k != 'sessions'}), flush=True)
        if result.get('fluency_validation'):
            print(json.dumps({k:v for k,v in result['fluency_validation'].items() if k != 'sessions'},ensure_ascii=True),flush=True)
    try:
        if args.show:
            from tac.ui import Window, apply_theme
            app.setStyle("Fusion")
            apply_theme(app, core.config)
            window = Window(core)
            window.auto_armed = False
            window.advanced_check.setChecked(True)
            window.nav.setCurrentRow(3)
            window.show()
            def finished(result):
                complete(result)
                window.conversation_result(result)
            QTimer.singleShot(150, lambda: window.start_job("Treinamento neural", train, finished, learning=True))
            app.exec()
        else:
            complete(core.perform("Treinamento de diálogo", lambda: train(lambda *_: None), learning=True))
    finally:
        if args.device:
            core.config.values["device"] = original_device
            core.config.save()
        lock.unlock()


if __name__ == "__main__":
    main()
