"""Replay textual BPE de snapshots locais, sem misturar fronteiras reservadas."""
import hashlib
import json

import torch


class TextReplay:
    def __init__(self, path, tokenizer):
        value = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps({key: value[key] for key in ("train", "validation", "provenance")},
                                         ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest != value["hash"] or path.stem != digest:
            raise ValueError("Corpus textual alterado")
        self.digest = digest
        self.rows = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in value["train"] if text.strip()]
        self.validation_rows = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in value["validation"] if text.strip()]
        self.provenance = value["provenance"]
        self.samples = 0
        self.tokens = 0
        self.sources = {}

    def batch(self, size, context, rng, device, validation=False):
        rows = self.validation_rows if validation else self.rows
        x = torch.zeros((size, context), dtype=torch.long)
        y = torch.zeros_like(x)
        for i in range(size):
            row_index = int(torch.randint(len(rows), (), generator=rng))
            row = rows[row_index]
            if not validation and row_index < len(self.provenance):
                source = self.provenance[row_index].get("source_id")
                if source is not None:
                    self.sources[source] = self.sources.get(source, 0) + 1
            start = int(torch.randint(max(1, len(row) - context), (), generator=rng))
            seq = row[start:start + context + 1]
            x[i, :len(seq)-1], y[i, :len(seq)-1] = seq[:-1], seq[1:]
        if not validation:
            self.samples += size
            self.tokens += int((y != 0).sum())
        return x.to(device), y.to(device)

    @torch.inference_mode()
    def evaluate(self, model, device, seed=4129):
        model.eval()
        rng = torch.Generator().manual_seed(seed)
        loss_sum = count = 0
        for _ in range(8):
            x, y = self.batch(4, min(256, model.cfg.context), rng, device, validation=True)
            _, loss = model(x, y)
            n = int((y != 0).sum())
            loss_sum += float(loss) * n
            count += n
        return {"loss": loss_sum / max(1, count), "tokens": count, "batches": 8, "seed": seed}
