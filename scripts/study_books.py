"""Lote explicitamente autorizado: baixar, indexar e treinar 20 livros públicos.

Não é executado na inicialização. Mantém as preferências persistentes do usuário.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import psutil
import torch
from PySide6.QtCore import QCoreApplication, QLockFile

from tac.book_study import study_books
from tac.config import atomic_json
from tac.core import Core
from tac.neural.data import Corpus
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import safe_load, weight_hash


@torch.inference_mode()
def evaluate_books(model, corpus, books, device):
    """16 janelas de validação por livro, separadas do treino e reprodutíveis."""
    model.eval()
    results = []
    for book in books:
        items = [corpus.validation[i] for i, p in enumerate(corpus.provenance)
                 if p.get("source_id") == book["source_id"]]
        if not items:
            raise ValueError(f"Livro sem dados de validação: {book['title']}")
        rng = torch.Generator().manual_seed(2026 + book["gutenberg_id"])
        loss_sum, count, correct = 0.0, 0, 0
        for _ in range(4):
            x = torch.zeros((4, model.cfg.context), dtype=torch.long)
            y = torch.zeros_like(x)
            for i in range(4):
                row = items[int(torch.randint(len(items), (), generator=rng))]
                start = int(torch.randint(max(1, len(row) - model.cfg.context), (), generator=rng))
                window = row[start:start + model.cfg.context + 1]
                x[i, :len(window) - 1] = window[:-1]
                y[i, :len(window) - 1] = window[1:]
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            n = int((y != 0).sum())
            count += n
            loss_sum += float(loss) * n
            correct += int(((logits.argmax(-1) == y) & (y != 0)).sum())
        results.append({"gutenberg_id": book["gutenberg_id"], "title": book["title"],
                        "windows": 16, "bytes": count, "loss": loss_sum / count,
                        "byte_accuracy": correct / count})
    total = sum(r["bytes"] for r in results)
    return {"loss": sum(r["loss"] * r["bytes"] for r in results) / total,
            "byte_accuracy": sum(r["byte_accuracy"] * r["bytes"] for r in results) / total,
            "books": results, "windows": 16 * len(results), "bytes": total}


def load_model(core, checkpoint, device):
    value = safe_load(core.trainer.checkpoint_path(checkpoint))
    model = TACTransformer(ModelConfig(**value["model_config"])).to(device)
    model.load_state_dict(value["model"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True,
                        help="Autoriza explicitamente os downloads e o treino deste lote")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    args = parser.parse_args()
    if not 20 <= args.steps <= 6000 or args.steps % 20:
        parser.error("Use 20 a 6000 passos, múltiplo de 20.")
    core = Core()
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(core.config.data / "tac.lock"))
    if not lock.tryLock(0):
        raise RuntimeError("Feche a janela da TAC antes de executar este lote.")
    original_values = dict(core.config.values)
    # Ajustes somente nesta execução; não altera as preferências salvas.
    values = {**original_values, "learning_enabled": True, "max_text_chars": 2000000,
              "max_corpus_chars": 6000000, "epochs": args.steps // 20,
              "steps_per_epoch": 20, "train_seconds": 600,
              "learning_rate": args.learning_rate}
    # Poucos checkpoints: 6 épocas para 1200 passos, sem exceder 100 épocas.
    values["epochs"] = 6
    values["steps_per_epoch"] = args.steps // 6 if args.steps % 6 == 0 else args.steps // 2
    if args.steps % 6:
        values["epochs"] = 2
    core.config.validate(values)
    core.config.values = values
    signal.signal(signal.SIGINT, lambda *_: core.gate.cancel.set())
    report_path = ROOT / "reports" / "treino_20_livros.json"
    started = time.monotonic()
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "iniciando",
              "prior_active": core.trainer.active_id(), "settings": values,
              "limitations": "Predição do próximo byte. Loss e acerto de bytes não medem compreensão de livros nem capacidade de conversar."}
    max_rss = 0
    def progress(step, message):
        nonlocal max_rss
        max_rss = max(max_rss, psutil.Process().memory_info().rss // 2**20)
        print(message, flush=True)
        atomic_json(core.config.data / "books" / "progress.json",
                    {"step": step, "message": message, "elapsed_seconds": time.monotonic() - started})
    try:
        manifest = core.perform("Lote de 20 livros", lambda: study_books(core, progress), learning=True)
        books = manifest["books"]
        if len(books) != 20 or len({b["sha256"] for b in books}) != 20:
            raise ValueError("O lote não contém 20 obras distintas completas.")
        corpus = Corpus.from_memory(core.db, core.config)
        source_ids = {p.get("source_id") for p in corpus.provenance}
        if any(b["source_id"] not in source_ids for b in books):
            raise ValueError("A seleção neural deixou livros de fora.")
        report.update(status="treinando", books=books, dataset_hash=corpus.digest,
                      training_characters=sum(map(len, corpus.train_text)),
                      validation_characters=sum(map(len, corpus.val_text)),
                      represented_sources=len(source_ids))
        device = core.trainer.device()
        torch.set_num_threads(core.config.get("cpu_threads"))
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        old_model = load_model(core, report["prior_active"], device)
        report["baseline_books"] = evaluate_books(old_model, corpus, books, device)
        report["parameters"] = old_model.parameter_count
        del old_model
        atomic_json(report_path, report)
        result = core.perform("Treinamento dos 20 livros", lambda: core.trainer.train(progress, corpus=corpus), learning=True)
        report["training"] = result
        model = load_model(core, result["checkpoint"], device)
        report["candidate_books"] = evaluate_books(model, corpus, books, device)
        report["checkpoint_verified"] = weight_hash(model) == result["final_hash"]
        del model
        report["active_after"] = core.trainer.active_id()
        report["status"] = result["status"]
        report["peak_torch_vram_mb"] = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0
        report["max_observed_process_ram_mb"] = max_rss
        report["elapsed_seconds"] = time.monotonic() - started
        report["sources_after"] = core.db.query("SELECT COUNT(*) AS n FROM sources")[0]["n"]
        report["samples"] = [core.trainer.generate(p)["text"] for p in ["O Brasil ", "Olá, meu nome é "]]
        atomic_json(report_path, report)
        atomic_json(core.config.data / "books" / "training_report.json", report)
        atomic_json(ROOT / "reports" / f"treino_20_livros_{result['run_id']}.json", report)
        print(json.dumps({"status": report["status"], "parameters": report["parameters"],
                          "steps": result["steps"], "checkpoint": result["checkpoint"],
                          "books_loss_before": report["baseline_books"]["loss"],
                          "books_loss_after": report["candidate_books"]["loss"],
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=True, indent=2))
    except BaseException as exc:
        report.update(status="erro", error=f"{type(exc).__name__}: {exc}")
        atomic_json(report_path, report)
        raise
    finally:
        core.config.values = original_values
        lock.unlock()


if __name__ == "__main__":
    main()
