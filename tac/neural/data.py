from __future__ import annotations

import hashlib
import json

import torch

from ..config import atomic_json
from .tokenizer import ByteTokenizer


def fixed_examples():
    # Tarefa estreita e reproduzível de continuação de ciclos. Não mede conversação.
    return [("abcde "[i % 6:] + "abcde " * 12)[:64] for i in range(6)] + ["0123456789 " * 6, "sim nao " * 8]


class Corpus:
    def __init__(self, train, validation, provenance=None):
        if not train or not validation:
            raise ValueError("Dados insuficientes para treino e validação separados.")
        self.train_text, self.val_text = train, validation
        self.provenance = provenance or []
        self.sampled_sources = {}
        self.sampled_training_bytes = 0
        self.digest = hashlib.sha256(json.dumps({"train": train, "validation": validation, "provenance": self.provenance}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        tokenizer = ByteTokenizer()
        self.train = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in train if t.strip()]
        self.validation = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in validation if t.strip()]
        self.fixed = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in fixed_examples()]

    @classmethod
    def from_memory(cls, db, config, source_ids=None):
        train, val, provenance = [], [], []
        remaining = int(config.get("max_corpus_chars"))
        # Cotas iguais para fontes grandes; textos pequenos liberam sua cota.
        # Antes o limite era gasto nas fontes mais antigas, excluindo as novas.
        args = ()
        condition = ""
        if source_ids is not None:
            args = tuple(dict.fromkeys(int(value) for value in source_ids))
            if not args or len(args) > 500:
                raise ValueError("Selecione de 1 a 500 fontes para o treino focado.")
            condition = " AND source_id IN (" + ",".join("?" for _ in args) + ")"
        groups = db.query("SELECT source_id, SUM(length(text)) AS chars FROM chunks WHERE length(text)>=40" + condition + " GROUP BY source_id", args)
        # Consulta automática à web não autoriza incorporar o conteúdo aos pesos.
        research_ids = {r['id'] for r in db.query('SELECT id,metadata FROM sources')
                        if json.loads(r['metadata']).get('research_web')}
        groups = [g for g in groups if g['source_id'] not in research_ids]
        corrections = db.query("SELECT COALESCE(SUM(length(correction)),0) AS chars FROM corrections WHERE validated=1 AND length(correction)>=40")[0]["chars"] if source_ids is None else 0
        if corrections:
            groups.append({"source_id": 0, "chars": corrections})
        groups.sort(key=lambda g: (g["chars"], g["source_id"]))
        for index, group in enumerate(groups):
            quota = min(group["chars"], remaining // (len(groups) - index))
            remaining -= quota
            if group["source_id"]:
                rows = db.iterate("SELECT c.text,c.ordinal,s.id AS source_id,s.content_hash FROM chunks c JOIN sources s ON c.source_id=s.id WHERE s.id=? AND length(c.text)>=40 ORDER BY ((c.ordinal * 1103515245 + ?) % 2147483647),c.ordinal", (group["source_id"], int(config.get("seed"))))
            else:
                rows = db.iterate("SELECT id AS correction_id,correction AS text FROM corrections WHERE validated=1 AND length(correction)>=40 ORDER BY id")
            # Ordem determinística dispersa as amostras ao longo de cada livro.
            for row in rows:
                if quota < 40:
                    break
                text = row["text"][:quota]
                cut = int(len(text) * 0.8)
                # Janelas não cruzam a fronteira entre treino e validação.
                train.append(text[:cut])
                val.append(text[cut:])
                quota -= len(text)
                if group["source_id"]:
                    provenance.append({"source_id": row["source_id"], "hash": row["content_hash"], "ordinal": row["ordinal"]})
                else:
                    provenance.append({"correction_id": row["correction_id"]})
        return cls(train, val, provenance)

    def save(self, path):
        atomic_json(path, {"train": self.train_text, "validation": self.val_text,
                           "provenance": self.provenance, "hash": self.digest, "version": 1})

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text(encoding="utf-8"))
        corpus = cls(value["train"], value["validation"], value["provenance"])
        if corpus.digest != value["hash"]:
            raise ValueError("O snapshot de treinamento foi alterado.")
        return corpus

    def batch(self, split, batch_size, context, generator, device):
        items = getattr(self, split)
        x = torch.zeros((batch_size, context), dtype=torch.long)
        y = torch.zeros_like(x)
        for i in range(batch_size):
            row_index = int(torch.randint(len(items), (), generator=generator))
            row = items[row_index]
            if split == "train" and row_index < len(self.provenance):
                source = self.provenance[row_index].get("source_id")
                if source is not None:
                    self.sampled_sources[source] = self.sampled_sources.get(source, 0) + 1
            max_start = max(1, len(row) - context)
            start = int(torch.randint(max_start, (), generator=generator))
            window = row[start:start + context + 1]
            x[i, :len(window) - 1] = window[:-1]
            y[i, :len(window) - 1] = window[1:]
        if split == "train":
            self.sampled_training_bytes += int((y != 0).sum())
        return x.to(device), y.to(device)
