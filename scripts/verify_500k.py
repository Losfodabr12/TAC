"""Audita os pesos ativos, os dados preservados e métricas reais da expansão."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from tac.config import atomic_json
from tac.core import Core
from tac.documents import normalize, text_hash
from tac.neural.data import Corpus
from tac.neural.inventory import active_models
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.tokenizer import ByteTokenizer
from tac.neural.training import safe_load, weight_hash


@torch.inference_mode()
def measure(model, corpus, device, split="validation", batches=24):
    rng = torch.Generator().manual_seed(8102026)
    loss_sum = correct = count = 0
    for _ in range(batches):
        x, y = corpus.batch(split, 4, 128, rng, device)
        logits, loss = model(x, y)
        valid = y != 0
        n = int(valid.sum())
        loss_sum += float(loss) * n
        correct += int(((logits.argmax(-1) == y) & valid).sum())
        count += n
    return {"loss": loss_sum / count, "byte_accuracy": correct / count, "targets": count,
            "seed": 8102026, "batches": batches, "context": 128}


def main():
    c = Core()
    torch.set_num_threads(c.config.get("cpu_threads"))
    before = json.loads((ROOT / "reports/baseline_040.json").read_text(encoding="utf-8"))
    latest = json.loads((ROOT / "reports/training_500k_latest.json").read_text(encoding="utf-8"))
    assert latest["result"]["status"] == "aprovado"
    sources = {s["id"]: s for s in c.memory.sources()}
    assert len(sources) == len(before["sources"])
    assert all(sources[s["id"]]["content_hash"] == s["content_hash"] and sources[s["id"]]["origin"] == s["origin"] for s in before["sources"])
    assert c.dialogue.active_id() == before["active_dialogue"]
    expected_changes = {"context", "ff_width", "learning_rate"}
    assert all(c.config.get(key) == value for key, value in before["config"].items() if key not in expected_changes)
    assert c.trainer.active_id() == latest["result"]["checkpoint"]
    cp = safe_load(c.trainer.checkpoint_path(c.trainer.active_id()))
    original = safe_load(c.trainer.checkpoint_path(before["active_checkpoint"]))
    device = c.trainer.device()
    current = TACTransformer(ModelConfig(**cp["model_config"])).to(device).eval()
    current.load_state_dict(cp["model"])
    old = TACTransformer(ModelConfig(**original["model_config"])).to(device).eval()
    old.load_state_dict(original["model"])
    assert current.parameter_count == 495808 and current.parameter_count < 500000
    assert weight_hash(current) == latest["result"]["final_hash"]
    assert weight_hash(current) != weight_hash(old)
    backup = next((c.config.data / "backups").glob("antes_expansao_500k_*.zip"))
    with zipfile.ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    old_path = c.trainer.checkpoint_path(before["active_checkpoint"])
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == manifest["files"]["checkpoints/" + old_path.name]
    added_output_norm = sum(float(block.ff[2].weight[:, 384:].abs().sum().detach()) for block in current.blocks)
    assert added_output_norm > 0
    assert all(torch.isfinite(p).all() for p in current.parameters())
    corpus = Corpus.load(c.config.data / "datasets" / f"{cp['dataset_hash']}.json")
    # Janelas mais numerosas com outra semente. Não foram usadas para selecionar épocas.
    report = {"models": active_models(c.config, c.db), "active_checkpoint": c.trainer.active_id(),
              "architecture": cp["model_config"], "weights_hash_verified": True,
              "previous_checkpoint_unchanged_against_backup": True,
              "added_connections_l1_norm": added_output_norm,
              "sources_preserved": len(sources), "preferences_preserved_except_architecture_and_lr": True,
              "global_validation": {"before": measure(old, corpus, device), "after": measure(current, corpus, device)}}
    report["development_fixed"] = {"before": c.trainer.evaluate(old, corpus, "fixed", device, context=128),
                                    "after": c.trainer.evaluate(current, corpus, "fixed", device, context=128)}
    assert report["development_fixed"]["after"][0] <= report["development_fixed"]["before"][0] * 1.02
    for sid, label in ((40, "story"), (41, "authored_dialogues")):
        data = Corpus.from_memory(c.db, c.config, source_ids=[sid])
        report[label] = {"before": measure(old, data, device), "after": measure(current, data, device)}
    story = sources[40]
    original_file = c.config.data / "inbox" / json.loads(story["metadata"])["filename"]
    canonical = normalize(original_file.read_text(encoding="utf-8-sig"))
    assert canonical == "\n".join(chunk["text"] for chunk in c.memory.chunks(40))
    assert text_hash(canonical) == story["content_hash"]
    report["story_text_and_emojis_preserved"] = True
    report["chat_checks"] = []
    for prompt in ("eae", "me diz teu nome", "tô feliz", "qual a capital do Brasil?"):
        answer = c.chat(prompt)
        assert answer["mode"] != "neural experimental"
        if "capital" in prompt:
            assert "Brasília" in answer.get("display_text", answer["text"])
        report["chat_checks"].append({"prompt": prompt, "answer": answer})
    tokenizer = ByteTokenizer()
    samples = []
    for prompt in ("Usuário: eae\nTAC:", "Usuário: qual é seu nome?\nTAC:", "Ren recebeu dos deuses", "O Brasil é"):
        item = {"prompt": prompt, "seed": 42, "temperature": .7, "mode": "neural experimental"}
        for label, model in (("before", old), ("after", current)):
            tokens = torch.tensor([[tokenizer.bos, *tokenizer.encode(prompt, special=False)]], device=device)
            generated = model.generate(tokens, max_new_tokens=100, seed=42, temperature=.7)
            item[label] = tokenizer.decode(generated[0, tokens.shape[1]:].tolist())
        samples.append(item)
    report["samples"] = samples
    report["run_history"] = c.db.query("SELECT id,status,started,detail FROM runs WHERE kind='neural' AND started>(SELECT started FROM runs WHERE id=(SELECT run_id FROM checkpoints WHERE id=?)) ORDER BY started", (before["active_checkpoint"],))
    atomic_json(ROOT / "reports/training_500k_validation.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "chat_checks", "run_history")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
