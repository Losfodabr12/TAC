"""Atualização manual autorizada: contexto, exercícios e treino local verificável."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from PySide6.QtCore import QCoreApplication, QLockFile

from tac.config import atomic_json
from tac.core import Core
from tac.dialogue_lessons import LESSONS, examples
from tac.language_styles import written_material
from tac.neural.data import Corpus
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import safe_load, weight_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--phase", choices=("prepare", "dialogue", "neural", "all"), default="all")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    args = parser.parse_args()
    if not 20 <= args.steps <= 6000 or args.steps % 20:
        parser.error("Use de 20 a 6000 passos, múltiplo de 20")
    core = Core()
    app = QCoreApplication.instance() or QCoreApplication([])
    lock = QLockFile(str(core.config.data / "tac.lock"))
    if not lock.tryLock(0):
        raise RuntimeError("A TAC precisa encerrar a sessão atual antes desta atualização.")
    old_values = dict(core.config.values)
    core.config.values = {**old_values, "learning_enabled": True}
    report_path = ROOT / "reports/upgrade_030.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    def progress(step, message):
        if step % 25 == 0 or "Época" in message or "Exercícios" in message:
            print(message, flush=True)
        atomic_json(ROOT / "reports/upgrade_030_progress.json", {"phase": args.phase, "step": step, "message": message})
    def persist():
        atomic_json(report_path, report)
    try:
        if args.phase in ("prepare", "all"):
            results = []
            for i, source in enumerate(core.memory.sources()):
                result = core.perform("Preservar contexto", lambda s=source: core.memory.reindex_source(s["id"]), learning=True)
                results.append(result)
                progress(i, f"Contexto #{source['id']}: {result['before']} -> {result['after']} trechos")
            report["reindex"] = results
            folder = ROOT / "materials"
            folder.mkdir(exist_ok=True)
            path = folder / "exercicios_conversa_e_escrita_tac.txt"
            material = written_material(LESSONS, examples)
            with core.gate.writing():
                path.write_text(material, encoding="utf-8")
            queue_id = core.enqueue_file(path, "Exercícios autorais / linguagem formal e informal")
            result = core.perform("Estudar exercícios autorais", lambda: core.run_queue(progress, queue_ids=[queue_id]), learning=True)
            if result["errors"] or not result["source_ids"]:
                raise RuntimeError("Não foi possível preparar o material autoral.")
            report["language_source_id"] = result["source_ids"][0]
            report["material_characters"] = len(material)
            report["material_sha256"] = hashlib.sha256(material.encode()).hexdigest()
            report["classifier_examples"] = {s: len(examples(s)) for s in ("train", "validation", "test")}
            persist()
        if args.phase in ("dialogue", "all"):
            report["dialogue"] = core.perform("Praticar diferentes jeitos de falar", lambda: core.dialogue.train(progress, epochs=100), learning=True)
            report["classifier_examples"] = {s: len(examples(s)) for s in ("train", "validation", "test")}
            persist()
            atomic_json(ROOT / "reports" / f"upgrade_030_dialogue_{report['dialogue']['run_id']}.json", report["dialogue"])
        if args.phase in ("neural", "all"):
            # A história é localizada pela origem aprovada, sem buscar arquivos privados novos.
            story = core.db.query("SELECT id FROM sources WHERE id=40 AND kind='file'")
            if not story or "language_source_id" not in report:
                raise ValueError("Material focado ausente; execute prepare primeiro.")
            focus_ids = [story[0]["id"], report["language_source_id"]]
            core.config.values = {**core.config.values, "learning_rate": args.learning_rate,
                                  "epochs": args.steps // 20, "steps_per_epoch": 20, "train_seconds": 600}
            if args.steps // 20 > 100:
                core.config.values.update(epochs=args.steps // 100, steps_per_epoch=100)
            core.config.validate(core.config.values)
            corpus = Corpus.from_memory(core.db, core.config, source_ids=focus_ids)
            individual = {str(sid): Corpus.from_memory(core.db, core.config, source_ids=[sid]) for sid in focus_ids}
            device = core.trainer.device()
            cp = safe_load(core.trainer.checkpoint_path(core.trainer.active_id()))
            model = TACTransformer(ModelConfig(**cp["model_config"])).to(device)
            model.load_state_dict(cp["model"])
            before = {sid: core.trainer.evaluate(model, data, "validation", device) for sid, data in individual.items()}
            del model
            started = time.monotonic()
            result = core.perform("Treinar história e estilos", lambda: core.trainer.train(progress, corpus=corpus), learning=True)
            selected = safe_load(core.trainer.checkpoint_path(result["checkpoint"]))
            model = TACTransformer(ModelConfig(**selected["model_config"])).to(device)
            model.load_state_dict(selected["model"])
            after = {sid: core.trainer.evaluate(model, data, "validation", device) for sid, data in individual.items()}
            report["neural"] = {**result, "focus_source_ids": focus_ids, "focus_before": before,
                                "focus_after": after, "seconds": time.monotonic() - started,
                                "checkpoint_hash_verified": weight_hash(model) == result["final_hash"]}
            report["active_after"] = core.trainer.active_id()
            persist()
            atomic_json(ROOT / "reports" / f"upgrade_030_neural_{result['run_id']}.json", report["neural"])
        print(json.dumps({"phase": args.phase, "dialogue_status": report.get("dialogue", {}).get("status"),
                          "neural_status": report.get("neural", {}).get("status"),
                          "language_source_id": report.get("language_source_id")}, ensure_ascii=True))
    finally:
        core.config.values = old_values
        lock.unlock()


if __name__ == "__main__":
    main()
