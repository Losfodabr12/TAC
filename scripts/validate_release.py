"""Validação real de release. Rede somente no exemplo fornecido no pedido do projeto."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tac.config import atomic_json
from tac.core import Core
from tac.neural.data import Corpus
from tac.neural.reinforcement import run_experiment


def main():
    workspace = ROOT / "reports" / ("validation_" + time.strftime("%Y%m%d_%H%M%S"))
    core = Core(workspace)
    report = {"workspace": str(workspace), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "fresh_auto_train": core.config.get("auto_train"), "fresh_learning": core.gate.enabled}
    core.enqueue_file(ROOT / "examples" / "primeiro_estudo.txt", "TAC / demonstração")
    core.gate.set_enabled(True)
    report["document_study"] = core.perform("Estudo de demonstração", core.run_queue, learning=True)
    report["document_answer"] = core.chat("Como o Python e o SQLite são usados?")
    print("TXT estudado e recuperado com fonte", flush=True)
    core.config.update(epochs=1, steps_per_epoch=50, train_seconds=120, reserve_ram_mb=512)
    corpus = Corpus(["abcde " * 80, "0123456789 " * 50, "sim nao " * 70], ["abcde " * 30, "0123456789 " * 30, "sim nao " * 30], [{"fixture": "cyclic-pattern-task-v1"}])
    samples = []
    def progress(step, message):
        info = core.stats()
        if torch.cuda.is_available():
            info.update(vram_allocated_mb=torch.cuda.memory_allocated() / 2**20,
                        vram_reserved_mb=torch.cuda.memory_reserved() / 2**20)
        samples.append(info)
        print(message, flush=True)
    started = time.monotonic()
    report["training"] = core.perform("Treino de padrões reproduzíveis", lambda: core.trainer.train(progress, corpus=corpus), learning=True)
    report["training_seconds"] = time.monotonic() - started
    report["resource_samples"] = samples
    report["neural_sample"] = core.trainer.generate("abcde ")
    report["reinforcement"] = core.perform("Reforço verificável", lambda: run_experiment(core.config, core.db, core.gate, steps=200), learning=True)
    print("Experimento de reforço concluído", flush=True)
    # URL expressamente fornecida como exemplo pelo usuário. Nenhum outro domínio é estudado.
    core.config.update(web_pages=2, web_depth=1)
    core.enqueue_url("https://docs.python.org/3/", True, "Python", terms=True)
    report["web_study"] = core.perform("Teste real do site fornecido", lambda: core.run_queue(progress), learning=True)
    report["web_history"] = core.db.query("SELECT * FROM study_history WHERE origin LIKE 'https://%' ORDER BY id")
    core.gate.set_enabled(False)
    report["final_stats"] = core.stats()
    atomic_json(ROOT / "reports" / "release_validation.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
