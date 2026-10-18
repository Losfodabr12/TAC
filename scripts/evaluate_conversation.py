"""Avalia uma versão própria em snapshots reservados; não treina nem ativa pesos."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("validation", "test", "validation_context", "test_context", "all"), default="test")
    args = parser.parse_args()
    core = Core()
    torch.set_num_threads(core.config.get("cpu_threads"))
    checkpoint = args.checkpoint or core.conversation.active_id()
    model, cp = core.conversation.load(checkpoint, args.device)
    data = ConversationData.load(core.config.data / "datasets" / f"conversation_{cp['dataset_hash']}.json")
    report = {"checkpoint": checkpoint, "parameters": model.parameter_count, "dataset_hash": data.digest,
              "device": args.device, "splits": {}}
    splits = [s for s in data.sets if s != "train"] if args.split == "all" else [args.split]
    for split in splits:
        metrics = core.conversation.evaluate(model, data, args.device, split=split, generation=True)
        report["splits"][split] = metrics
        print(split, json.dumps({k: v for k, v in metrics.items() if k != "outputs"}, ensure_ascii=True), flush=True)
    probes = [
        [["user", "eae"]],
        [["user", "qual seu nome?"]],
        [["user", "Meu nome é Joaquim."], ["assistant", "Prazer, Joaquim."], ["user", "qual é meu nome?"]],
        [["user", "Minha cor favorita é azul."], ["assistant", "Entendido."], ["user", "Agora prefiro verde."], ["assistant", "Certo."], ["user", "Qual é minha cor favorita?"]],
        [["user", "Estou escrevendo uma história de aventura."], ["assistant", "Sobre o que é a história?"], ["user", "O protagonista perdeu o irmão. O que você acha?"]],
        [["user", "Explique por que o céu é azul."]],
    ]
    report["exploratory_samples"] = [{"messages": m, "actual": decode_reply(model, m, args.device)} for m in probes]
    atomic_json(ROOT / "reports" / f"evaluation_conversation_{checkpoint}_{args.split}.json", report)
    print(json.dumps(report["exploratory_samples"], ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
