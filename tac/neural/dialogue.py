"""Rede local de intenções para conversa guiada, independente do Transformer.

As respostas são exemplos autorais selecionados pelo classificador treinado.
Não há geração livre nem afirmação de compreensão geral.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import uuid

import torch
from torch import nn

from ..config import atomic_json
from ..db import now
from ..diagnostics import check_resources
from ..dialogue_lessons import LEGACY_TEST, LESSONS, examples
from ..learning import Cancelled, LearningDisabled
from .training import Trainer, weight_hash

VERSION = "TAC-dialogue-v1"
DIMENSION = 2048


def normalized(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)).split())


def features(text):
    text = normalized(text)[:500]
    # Nomes pessoais são dados de sessão, não classes que a rede deva memorizar.
    text = re.sub(r"\b(meu nome e|eu me chamo|me chamo|pode me chamar de|me chama de|sou o|sou a|eu sou)\s+[^.!?,]+", r"\1 pessoa", text)
    words = re.findall(r"\w+", text)
    parts = [("w:" + word, 2.0) for word in words]
    parts += [("b:" + a + "_" + b, 1.0) for a, b in zip(words, words[1:])]
    padded = " " + text + " "
    for size in (2, 3, 4):
        parts += [("c:" + padded[i:i + size], .25 if size == 2 else .5) for i in range(max(0, len(padded) - size + 1))]
    output = torch.zeros(DIMENSION)
    for part, weight in parts:
        digest = hashlib.blake2b(part.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % DIMENSION
        output[index] += weight if digest[4] & 1 else -weight
    return output / output.norm().clamp(min=1e-6)


class DialogueNetwork(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIMENSION, 128), nn.GELU(), nn.Linear(128, classes))

    def forward(self, x):
        return self.net(x)


class DialogueEngine:
    def __init__(self, config, db, gate):
        self.config, self.db, self.gate = config, db, gate
        self._loaded = None
        self._model = None
        self._checkpoint = None

    def active_id(self):
        rows = self.db.query("SELECT value FROM meta WHERE key='active_dialogue'")
        return rows[0]["value"] if rows else None

    def load(self, checkpoint_id=None):
        checkpoint_id = checkpoint_id or self.active_id()
        if not checkpoint_id:
            return None
        rows = self.db.query("SELECT filename FROM checkpoints WHERE id=?", (checkpoint_id,))
        if not rows:
            raise ValueError("Checkpoint de conversa não encontrado.")
        folder = (self.config.data / "checkpoints").resolve()
        path = (folder / rows[0]["filename"]).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size > 20 * 2**20:
            raise ValueError("Checkpoint de conversa inválido.")
        cp = torch.load(path, map_location="cpu", weights_only=True)
        labels = cp.get("labels")
        if (cp.get("format") != VERSION or cp.get("dimension") != DIMENSION
                or not isinstance(labels, list) or not 2 <= len(labels) <= 64
                or len(set(labels)) != len(labels) or "unknown" not in labels
                or any(label not in LESSONS for label in labels)):
            raise ValueError("Versão dos exercícios de conversa incompatível.")
        torch.set_num_threads(self.config.get("cpu_threads"))
        model = DialogueNetwork(len(cp["labels"]))
        model.load_state_dict(cp["model"])
        model.eval()
        return model, cp

    @torch.inference_mode()
    def predict(self, text):
        active = self.active_id()
        if not active:
            return None
        if active != self._loaded:
            self._model, self._checkpoint = self.load(active)
            self._loaded = active
        probabilities = self._model(features(text)[None]).softmax(-1)[0]
        index = int(probabilities.argmax())
        return self._checkpoint["labels"][index], float(probabilities[index]), self._checkpoint["threshold"]

    def reply(self, text, session_name=None):
        prediction = self.predict(text)
        if prediction is None:
            return None
        intent, confidence, threshold = prediction
        if intent == "unknown" or confidence < threshold:
            return None
        plain = normalized(text)
        # Preserva a distinção entre identidade do assistente e entidades nos documentos.
        if intent == "identity" and not re.search(r"\b(voce|vc|ce|seu|teu|tua|te|tac|se apresente)\b", plain):
            return None
        if intent == "practice" and not re.search(r"trein|aprend|estud|pratic|pesos|rede|modelo|convers|dialog", plain):
            return None
        reply = LESSONS[intent]["reply"]
        new_name = None
        if intent == "introduce_name":
            match = re.search(r"(?:meu nome [ée]|me chamo|me chamar de|me chama de|sou [oa]|eu sou)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '-]{0,35})", text, re.I)
            if not match:
                return None
            new_name = match.group(1).strip(" .!?,")
            if len(new_name.split()) > 4:
                return None
            reply = reply.format(name=new_name)
        elif intent == "recall_name":
            reply = reply.format(name=session_name) if session_name else "Você ainda não me disse seu nome nesta conversa. Se quiser, pode me contar."
        return {"mode": "conversa guiada", "text": reply, "display_text": reply,
                "sources": [], "method": "intenção escolhida por rede neural local; resposta autoral",
                "intent": intent, "confidence": confidence, "session_name": new_name}

    def _sets(self):
        labels = list(LESSONS)
        rows = {split: examples(split) for split in ("train", "validation", "test")}
        seen = [{normalized(r["text"]) for r in rows[split]} for split in rows]
        if seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]:
            raise ValueError("Os exercícios contêm frases repetidas entre os conjuntos.")
        tensors = {}
        for split, samples in rows.items():
            tensors[split] = (torch.stack([features(r["text"]) for r in samples]), torch.tensor([labels.index(r["intent"]) for r in samples]))
        tensors["legacy"] = (torch.stack([features(r["text"]) for r in LEGACY_TEST]), torch.tensor([labels.index(r["intent"]) for r in LEGACY_TEST]))
        return labels, rows, tensors

    @torch.inference_mode()
    def evaluate(self, model, dataset, labels, threshold=.75):
        model.eval()
        x, y = dataset
        logits = model(x.to(next(model.parameters()).device))
        probs = logits.softmax(-1)
        confidence, predicted = probs.max(-1)
        unknown = labels.index("unknown")
        gated = torch.where(confidence < threshold, unknown, predicted)
        target = y.to(gated.device)
        correct = gated == target
        negative = target == unknown
        return {"loss": float(nn.functional.cross_entropy(logits, target)),
                "accuracy": float(correct.float().mean()),
                "known_accuracy": float(correct[~negative].float().mean()),
                "unknown_false_accepts": int((~correct[negative]).sum()),
                "total": len(y), "correct": int(correct.sum()),
                "predictions": gated.cpu().tolist()}

    def train(self, progress=lambda *_: None, epochs=100):
        self.gate.check()
        check_resources(self.config)
        torch.set_num_threads(self.config.get("cpu_threads"))
        torch.manual_seed(self.config.get("seed"))
        labels, rows, data = self._sets()
        digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.gate.writing():
            atomic_json(self.config.data / "datasets" / f"dialogue_{digest}.json", {"format": VERSION, "rows": rows, "provenance": [{"authored": "TAC exercises v2 — linguagem informal e emojis"}]})
        device = Trainer(self.config, self.db, self.gate).device()
        model = DialogueNetwork(len(labels)).to(device)
        loaded = self.load()
        if loaded:
            if loaded[1]["labels"] == labels:
                model.load_state_dict(loaded[0].state_dict())
            else:
                # Novas intenções recebem linhas próprias; os pesos antigos são copiados.
                with torch.no_grad():
                    model.net[0].load_state_dict(loaded[0].net[0].state_dict())
                    for old_index, label in enumerate(loaded[1]["labels"]):
                        index = labels.index(label)
                        model.net[2].weight[index].copy_(loaded[0].net[2].weight[old_index])
                        model.net[2].bias[index].copy_(loaded[0].net[2].bias[old_index])
        initial_hash = weight_hash(model)
        baseline = self.evaluate(model, data["validation"], labels)
        baseline_test = self.evaluate(model, data["test"], labels, loaded[1]["threshold"] if loaded else .75)
        legacy_base = self.evaluate(model, data["legacy"], labels, loaded[1]["threshold"] if loaded else .75)
        if loaded:
            old_labels = loaded[1]["labels"]
            old_targets = torch.tensor([old_labels.index(r["intent"]) for r in LEGACY_TEST])
            legacy_base = self.evaluate(loaded[0], (data["legacy"][0], old_targets), old_labels, loaded[1]["threshold"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
        run_id = uuid.uuid4().hex
        self.db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash,baseline) VALUES(?,?,?,?,?,?,?)", (run_id, "dialogue", "treinando", now(), json.dumps({"epochs": epochs, "seed": self.config.get("seed"), "version": VERSION}), digest, baseline["accuracy"]))
        x, y = data["train"]
        generator = torch.Generator().manual_seed(self.config.get("seed"))
        started, step, interrupted, train_loss = time.monotonic(), 0, False, 0.0
        progress(0, f"Preparando {len(LESSONS)-1} grupos de exercícios; {len(x)} exemplos de treino.")
        try:
            for epoch in range(1, epochs + 1):
                check_resources(self.config)
                self.gate.wait_if_paused()
                if time.monotonic() - started > self.config.get("train_seconds"):
                    raise Cancelled("Limite de duração dos exercícios atingido.")
                model.train()
                for indices in torch.randperm(len(x), generator=generator).split(64):
                    self.gate.check()
                    logits = model(x[indices].to(device))
                    loss = nn.functional.cross_entropy(logits, y[indices].to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    with self.gate.writing():
                        self.gate.check()
                        optimizer.step()
                        step += 1
                    train_loss = float(loss.detach())
                if epoch % 10 == 0 or epoch == epochs:
                    evaluation = self.evaluate(model, data["validation"], labels)
                    self.db.execute("INSERT INTO metrics(run_id,step,epoch,train_loss,val_loss,accuracy,created) VALUES(?,?,?,?,?,?,?)", (run_id, step, epoch, train_loss, evaluation["loss"], evaluation["accuracy"], now()))
                    progress(round(epoch / epochs * 100), f"Exercícios {epoch}/{epochs} · acerto nas frases de validação: {evaluation['accuracy']:.0%}")
        except (Cancelled, LearningDisabled):
            interrupted = True
        except Exception as exc:
            self.db.execute("UPDATE runs SET status='erro',finished=?,detail=? WHERE id=?", (now(), str(exc), run_id))
            raise
        # O limiar usa apenas validação. O teste fixo só é avaliado depois dessa escolha.
        trials = [(self.evaluate(model, data["validation"], labels, t), t) for t in (.65, .75, .85, .9, .95)]
        eligible = [trial for trial in trials if trial[0]["unknown_false_accepts"] == 0]
        validation, threshold = max(eligible or trials, key=lambda p: (p[0]["accuracy"] - p[0]["unknown_false_accepts"] / len(data["validation"][1]), p[1]))
        test = self.evaluate(model, data["test"], labels, threshold)
        legacy = self.evaluate(model, data["legacy"], labels, threshold)
        accepted = (not interrupted and validation["accuracy"] >= max(.75, baseline["accuracy"] - .02)
                    and validation["unknown_false_accepts"] == 0 and test["unknown_false_accepts"] == 0
                    and test["accuracy"] >= max(.70, baseline_test["accuracy"] - .02)
                    and legacy["accuracy"] >= max(.70, legacy_base["accuracy"] - .02)
                    and legacy["unknown_false_accepts"] == 0)
        status = "interrompido" if interrupted else ("aprovado" if accepted else "rejeitado")
        checkpoint_id = uuid.uuid4().hex
        filename = f"dialogue_{checkpoint_id}.pt"
        path = self.config.data / "checkpoints" / filename
        report = {"kind": "dialogue", "run_id": run_id, "checkpoint": checkpoint_id, "status": status,
                  "groups": len(LESSONS)-1, "train_examples": len(rows["train"]), "validation_examples": len(rows["validation"]), "test_examples": len(rows["test"]),
                  "baseline_accuracy": baseline["accuracy"], "validation_accuracy": validation["accuracy"],
                  "test_accuracy": test["accuracy"], "test_correct": test["correct"], "unknown_false_accepts": test["unknown_false_accepts"],
                  "steps": step, "threshold": threshold, "initial_hash": initial_hash, "final_hash": weight_hash(model),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "legacy_test_examples": legacy["total"], "legacy_test_correct": legacy["correct"],
                  "legacy_test_accuracy": legacy["accuracy"], "legacy_baseline_accuracy": legacy_base["accuracy"],
                  "seconds": time.monotonic()-started, "scope": "intenções guiadas, respostas autorais; não é geração livre",
                  "test_results": [{**r, "prediction": labels[p]} for r, p in zip(rows["test"], test["predictions"])]}
        value = {"format": VERSION, "dimension": DIMENSION, "labels": labels, "threshold": threshold,
                 "model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "sample_rng": generator.get_state(), "step": step, "report": report}
        torch.save(value, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        self.db.execute("INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)", (checkpoint_id, run_id, filename, step, validation["loss"], status, now()))
        if accepted:
            try:
                with self.gate.writing():
                    self.gate.check()
                    self.db.execute("INSERT INTO meta(key,value) VALUES('active_dialogue',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (checkpoint_id,))
            except (LearningDisabled, Cancelled):
                report["status"] = status = "interrompido"
        self.db.execute("UPDATE runs SET status=?,finished=?,final=?,detail=? WHERE id=?", (status, now(), test["accuracy"], json.dumps(report, ensure_ascii=False), run_id))
        atomic_json(self.config.data / "dialogue_report.json", report)
        self._loaded = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return report
