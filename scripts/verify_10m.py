"""Auditoria local dos pesos de diálogo, sem treinar ou acessar a internet."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from tac.config import atomic_json
from tac.core import Core
from tac.neural.conversation import ConversationData, decode_reply
from tac.neural.model import ModelConfig, TACTransformer, count_parameters
from tac.neural.training import safe_load, weight_hash


def main():
    core = Core()
    torch.set_num_threads(core.config.get("cpu_threads"))
    model, cp = core.conversation.load(core.conversation.active_id())
    assert 10_000_000 <= model.parameter_count <= 10_100_000
    assert model.parameter_count == count_parameters(model.cfg)
    assert cp["step"] > 0 and cp["optimizer"]["state"]
    data = ConversationData.load(core.config.data / "datasets" / f"conversation_{cp['dataset_hash']}.json")
    heldout = {tuple(tuple(m) for m in row["messages"]) for s in data.sets if s != "train" for row in data.sets[s]}
    assert not any(tuple(tuple(m) for m in row["messages"]) in heldout for row in data.sets["train"])
    before = weight_hash(model)
    reply = decode_reply(model, [["user", "eae"]], "cpu")
    assert weight_hash(model) == before
    baseline = json.loads((ROOT / "reports/baseline_050.json").read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in core.memory.sources()}
    assert all(sources[s["id"]]["content_hash"] == s["content_hash"] and sources[s["id"]]["origin"] == s["origin"] for s in baseline["sources"])
    assert core.dialogue.active_id() == baseline["active_dialogue"]
    # Leitor anterior continua independente e intacto.
    reader_cp = safe_load(core.trainer.checkpoint_path(core.trainer.active_id()))
    reader = TACTransformer(ModelConfig(**reader_cp["model_config"]))
    reader.load_state_dict(reader_cp["model"])
    reader_hash = weight_hash(reader)
    assert reader_hash == "933c122760693f647dabfdc11fed336c67150385836da19d517a29b6e9c83373"
    result = {"status": "ok", "checkpoint": core.conversation.active_id(), "parameters": model.parameter_count,
              "model_config": vars(model.cfg), "hash": before, "source_count": core.db.query("SELECT count(*) n FROM sources")[0]["n"],
              "reader_hash": reader_hash, "dataset_hash": data.digest, "examples": {k: len(v) for k, v in data.sets.items()},
              "inference_does_not_change_weights": True, "sample_eae": reply,
              "auto_on_start": core.config.get("auto_on_start"), "save_interactions": core.config.get("save_interactions")}
    result["source_hashes_and_origins_preserved"] = True
    assert result["source_count"] == 41
    atomic_json(ROOT / "reports/verification_10m.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
