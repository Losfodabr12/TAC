"""Treinamento manual autorizado, com expansão conservadora até 500 mil pesos."""
from __future__ import annotations

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
from tac.neural.data import Corpus
from tac.neural.model import ModelConfig, TACTransformer, expand_model
from tac.neural.tokenizer import ByteTokenizer
from tac.neural.training import safe_load, weight_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--retention-weight", type=float, default=20)
    parser.add_argument("--teacher-checkpoint", help="Referência local fixa para preservar uma versão anterior")
    parser.add_argument("--show", action="store_true", help="Exibe progresso e controles reais na interface")
    args = parser.parse_args()
    if not 200 <= args.steps <= 30000 or args.steps % 200:
        parser.error("Use de 200 a 30000 passos, múltiplo de 200")
    if args.show:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
    else:
        app = QCoreApplication.instance() or QCoreApplication([])
    core = Core()
    lock = QLockFile(str(core.config.data / "tac.lock"))
    if not lock.tryLock(0):
        raise RuntimeError("Encerre a janela atual para que somente um processo treine os mesmos dados.")
    core.gate.check()  # Nunca concede permissão caso o usuário tenha desligado o aprendizado.
    original_values = dict(core.config.values)
    parent_id = core.trainer.active_id()
    parent = safe_load(core.trainer.checkpoint_path(parent_id))
    target = ModelConfig(context=256, width=96, heads=4, layers=2, ff_width=896)
    model = TACTransformer(ModelConfig(**parent["model_config"]))
    model.load_state_dict(parent["model"])
    old_hash = weight_hash(model)
    expansion_needed = model.cfg != target
    expanded = expand_model(model, target) if expansion_needed else model
    report = {"parent": parent_id, "parent_hash": old_hash, "parameters_before": model.parameter_count,
              "parameters_after": expanded.parameter_count, "requested_steps": args.steps,
              "learning_rate": args.learning_rate, "retention_weight": args.retention_weight,
              "architecture": target.__dict__, "started": time.time()}
    if expanded.parameter_count > 500000:
        raise ValueError("A arquitetura ultrapassa a autorização de 500 mil parâmetros")
    torch.set_num_threads(core.config.get("cpu_threads"))
    torch.manual_seed(42)
    test_tokens = torch.randint(3, 259, (2, min(model.cfg.context, target.context)))
    with torch.no_grad():
        delta = (model.eval()(test_tokens)[0] - expanded.eval()(test_tokens)[0]).abs().max().item()
    if delta > 1e-5:
        raise RuntimeError("A expansão alterou as previsões antes do treino")
    report["maximum_logit_difference_before_training"] = delta
    del model, expanded
    core.config.values = {**original_values, "batch_size": 8, "accumulation": 2,
                          "learning_rate": args.learning_rate, "retention_weight": args.retention_weight,
                          "epochs": args.steps // 200,
                          "steps_per_epoch": 200, "train_seconds": 1800, "max_corpus_chars": 8000000}
    if args.steps // 200 > 100:
        core.config.values.update(epochs=100, steps_per_epoch=args.steps // 100)
    core.config.validate(core.config.values)
    corpus = Corpus.from_memory(core.db, core.config)
    report["dataset"] = {"hash": corpus.digest, "train_examples": len(corpus.train),
                         "validation_examples": len(corpus.validation),
                         "characters": sum(map(len, corpus.train_text)) + sum(map(len, corpus.val_text)),
                         "sources": sorted({p["source_id"] for p in corpus.provenance if "source_id" in p})}
    progress_path = ROOT / "reports/training_500k_progress.json"
    process = psutil.Process()
    peak_ram = process.memory_info().rss / 2**20
    temporary_keys = {"batch_size", "accumulation", "learning_rate", "retention_weight", "epochs", "steps_per_epoch", "train_seconds", "max_corpus_chars"}
    def restore_preferences():
        current = dict(core.config.values)
        core.config.values = {key: (original_values[key] if key in temporary_keys else value)
                              for key, value in current.items()}
    def progress(step, message):
        nonlocal peak_ram
        peak_ram = max(peak_ram, process.memory_info().rss / 2**20)
        if step % 100 == 0 or "Época" in message:
            print(message, flush=True)
            atomic_json(progress_path, {"step": step, "target": args.steps, "message": message,
                                        "elapsed_seconds": time.time() - report["started"], "peak_process_ram_mb": peak_ram})
    def work(update):
        def on_update(step, message):
            progress(step, message)
            update(step, message)
        return core.trainer.train(on_update, corpus=corpus, model_config=target if expansion_needed else None,
                                  teacher_checkpoint=args.teacher_checkpoint)
    def finish(result):
        report["result"] = result
        report["seconds"] = time.time() - report["started"]
        report["peak_process_ram_mb"] = peak_ram
        report["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        report["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
        # As preferências de duração/automático continuam as do usuário.
        restore_preferences()
        if result["status"] == "aprovado":
            core.config.update(context=target.context, ff_width=target.ff_width, learning_rate=args.learning_rate)
        else:
            core.config.save()
        atomic_json(ROOT / "reports" / f"training_500k_{result['run_id']}.json", report)
        atomic_json(ROOT / "reports/training_500k_latest.json", report)
        print(json.dumps({"status": result["status"], "selected_step": result["selected_step"],
                          "checkpoint": result["checkpoint"], "parameters": result["parameters"],
                          "validation_before": result["baseline"], "validation_after": result["validation"],
                          "detail": result["detail"]}, ensure_ascii=True), flush=True)
    def failed(message):
        report["error"] = message
        report["seconds"] = time.time() - report["started"]
        restore_preferences()
        core.config.save()
        atomic_json(ROOT / "reports" / f"training_500k_error_{int(report['started'])}.json", report)
    try:
        if args.show:
            from tac.ui import Window, apply_theme
            app.setApplicationName("TAC")
            app.setStyle("Fusion")
            apply_theme(app, core.config)
            window = Window(core)
            window.auto_armed = False
            window.advanced_check.setChecked(True)
            window.nav.setCurrentRow(3)
            window.show()
            def completed(value):
                finish(value)
                window.training_result(value)
                window.grab().save(str(ROOT / "reports/captures/040_treinamento.png"))
            def begin():
                window.start_job("Treinamento neural", work, completed, learning=True)
                window.job.failed.connect(failed)
            QTimer.singleShot(150, begin)
            app.exec()
        else:
            finish(core.perform("Treino autorizado até 500 mil parâmetros", lambda: work(lambda *_: None), learning=True))
    finally:
        if "result" not in report:
            restore_preferences()
            core.config.save()
        lock.unlock()


if __name__ == "__main__":
    main()
