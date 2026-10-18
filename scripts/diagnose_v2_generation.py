"""Sondas direcionadas de inferência; não treina nem grava checkpoints/datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.conversation import decode_reply, load_checkpoint, prompt_text
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.v2_data import prompt_text as training_prompt
from tac.neural.v2_training import V2Data, model_hash

OUT = ROOT / "reports/correcao_geracao_v2/rodada_20261018_01"
QUESTIONS = ["Opa, tudo bom?", "O que é um lobo?", "Quanto é 2 + 2?",
             "Crie uma história curta.", "Meu nome é Carlos. Qual é meu nome?"]


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def forbidden(*args, **kwargs):
    raise RuntimeError("Esta investigação proíbe treino e gravação de pesos.")


def top(distribution, tokenizer, n=10):
    values, ids = distribution.topk(n)
    return [{"id": i, "text": "<EOS>" if i == 2 else tokenizer.decode([i]), "p": p}
            for i, p in zip(ids.tolist(), values.tolist())]


def inputs(model, messages):
    text = prompt_text(messages)
    encoded = model.text_tokenizer.encode(text, special=False)
    ids = [1, *encoded[-model.cfg.context + 1:]]
    return text, torch.tensor([ids], device=next(model.parameters()).device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    torch.save = forbidden
    torch.optim.AdamW.step = forbidden
    torch.optim.Adam.step = forbidden
    torch.optim.SGD.step = forbidden
    torch.set_num_threads(4)
    db = sqlite3.connect((ROOT / "data/tac.sqlite3").as_uri() + "?mode=ro", uri=True)
    active = db.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
    row = db.execute("SELECT filename,step,status,run_id FROM checkpoints WHERE id=?", (active,)).fetchone()
    db.close()
    path = ROOT / "data/checkpoints" / row[0]
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cp = load_checkpoint(path)
    assert cp["generation"] == "TAC-v2"
    model = TACTransformer(ModelConfig(**cp["model_config"]))
    model.load_state_dict(cp["model"])
    model.text_tokenizer = LocalBPETokenizer.from_state(cp["tokenizer_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tokenizer = model.text_tokenizer
    weights_before = model_hash(model)
    identity = {"active": active, "file": str(path.relative_to(ROOT)), "file_sha256": before,
                "weights_hash": weights_before, "database": {"step": row[1], "status": row[2], "run_id": row[3]},
                "checkpoint_step": cp["step"], "parameters": model.parameter_count,
                "model_config": cp["model_config"], "device": device,
                "precision": str(next(model.parameters()).dtype), "optimizer_steps": 0,
                "reserved_tests_opened": False, "external_models_or_answers": False}
    save(args.phase + "_identity.json", identity)
    print(json.dumps(identity, ensure_ascii=False), flush=True)

    if args.phase == "before":
        data = V2Data(ROOT / "data/tac_v2", device)
        masks = Counter()
        errors = []
        for split in ("train", "validation"):
            for i, r in enumerate(data.rows[split]):
                prefix = [tokenizer.bos, *tokenizer.encode(training_prompt(r["messages"]), special=False)]
                answer = tokenizer.encode(" " + r["answer"], special=False) + [tokenizer.eos]
                x, y = data.instruction_batch([i], split)
                expected = torch.tensor(answer, device=device)
                ok = (prompt_text(r["messages"]) == training_prompt(r["messages"])
                      and prefix == r["sequence"][:r["prefix_length"]]
                      and r["sequence"] == prefix + answer
                      and len(r["sequence"]) - 1 <= model.cfg.context
                      and torch.equal(y[y != 0], expected)
                      and y[0, -1].item() == 2
                      and torch.equal(x[0], torch.tensor(r["sequence"][:-1], device=device)))
                if not ok:
                    errors.append([split, i])
                masks[split + "_examples"] += 1
                masks[split + "_answer_targets"] += len(answer)
                masks[split + "_eos_targets"] += answer.count(2)
        eos = []
        # Toda a validação nova da v2, sem consultar testes antigos/reservados.
        for i, r in enumerate(data.rows["validation"]):
            x, y = data.instruction_batch([i], "validation")
            logits, loss = model(x, y)
            end = logits[0, -1].float()
            prob = end.softmax(-1)
            eos.append({"index": i, "source": r.get("source"), "loss": float(loss),
                        "eos_probability": float(prob[2]),
                        "eos_rank": int((end > end[2]).sum().item()) + 1,
                        "top_token": int(end.argmax()), "supervised_tokens": int((y != 0).sum())})
        save("training_format_and_eos.json", {"checked": dict(masks), "errors": errors,
              "loss_ignore_index": 0, "training_format": "BOS + Usuário: ...\\nTAC: + ' ' + resposta + EOS",
              "inference_format_identical": not errors, "roles_are_plain_text_not_special_tokens": True,
              "eos_validation_teacher_forced": eos,
              "eos_top1": sum(r["eos_rank"] == 1 for r in eos), "eos_examples": len(eos),
              "eos_mean_probability": sum(r["eos_probability"] for r in eos) / len(eos)})
        print("Máscaras e EOS verificados: " + str(dict(masks)) + "; erros=" + str(errors), flush=True)

        conditions = []
        for q in QUESTIONS:
            messages = [["user", q]]
            text, ids = inputs(model, messages)
            assert text == training_prompt(messages)
            distribution = model(ids)[0][0, -1].float().softmax(-1)
            # Mesmo prefixo de resposta para separar condicionamento de amostragem.
            shared = torch.cat([ids, torch.tensor([tokenizer.encode(" O bom", special=False)], device=device)], 1)
            continuation = model(shared)[0][0, -1].float().softmax(-1)
            conditions.append({"question": q, "transformer_text": text, "input_ids": ids[0].tolist(),
                               "roundtrip_exact": tokenizer.decode(ids[0].tolist()) == text,
                               "truncated_tokens": max(0, len(tokenizer.encode(text, special=False)) + 1 - len(ids[0])),
                               "top_next": top(distribution, tokenizer), "p_next": distribution.cpu().tolist(),
                               "top_after_shared_prefix": top(continuation, tokenizer),
                               "p_after_shared_prefix": continuation.cpu().tolist()})
        pairs = []
        for i, a in enumerate(conditions):
            for j in range(i + 1, len(conditions)):
                b = conditions[j]
                d = {"a": a["question"], "b": b["question"]}
                for key in ("p_next", "p_after_shared_prefix"):
                    p, q = torch.tensor(a[key]), torch.tensor(b[key])
                    m = (p + q) / 2
                    js = ((p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()
                          + (q * (q.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()) / 2
                    d[key] = {"total_variation": float((p - q).abs().sum() / 2), "js_nats": float(js)}
                pairs.append(d)
        save("input_sensitivity.json", {"conditions": conditions, "pairs": pairs})

    profiles = ({"legacy_greedy": {"greedy": True},
                 "penalized_greedy": {"greedy": True, "repetition_penalty": 1.12},
                 "conservative_sampling": {"greedy": False, "temperature": .55, "top_k": 30,
                                           "top_p": .9, "repetition_penalty": 1.12, "seed": 42}}
                if args.phase == "before" else
                {"production_greedy": {"greedy": True, "repetition_penalty": 1.12},
                 "production_controlled": {"greedy": False, "temperature": .55, "top_k": 30,
                                           "top_p": .9, "repetition_penalty": 1.12, "seed": 42}})
    results = []
    for name, options in profiles.items():
        for q in QUESTIONS:
            messages = [["user", q]]
            budget = 256 if name == "legacy_greedy" else 128
            if args.phase == "after":
                from tac.neural.decoding import decode_conversation
                result = decode_conversation(model, messages, device, max_tokens=128, generation_options=options)
                text = result.pop("text")
            else:
                result = {}
                text = decode_reply(model, messages, device, max_tokens=budget, generation_options=options)
            results.append({"profile": name, "question": q, "messages": messages, "options": options,
                            "max_new_tokens": budget, "text": text, **result})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    # Respostas próprias voltam ao histórico: comparar com as sondas limpas acima.
    history = []
    for q in QUESTIONS:
        messages = [*history[-6:], ["user", q]]
        if args.phase == "after":
            from tac.neural.decoding import decode_conversation
            result = decode_conversation(model, messages, device, max_tokens=128,
                                         generation_options=profiles["production_controlled"])
            text = result.pop("text")
            history = [*messages, ["assistant", text]][-6:]
        else:
            result = {}
            text = decode_reply(model, messages, device, max_tokens=256, generation_options={"greedy": True})
            history = [*messages, ["assistant", text]][-6:]
        results.append({"profile": "sequential", "question": q, "messages": messages, "text": text, **result})
        print(json.dumps({k: v for k, v in results[-1].items() if k != "messages"}, ensure_ascii=False), flush=True)
    save(args.phase + "_generations.json", results)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    weights_after = model_hash(model)
    assert after == before and weights_before == weights_after
    save(args.phase + "_preservation.json", {"file_sha256_before": before, "file_sha256_after": after,
            "weights_hash_before": weights_before, "weights_hash_after": weights_after,
            "optimizer_steps": 0, "reserved_tests_opened": False, "status": "OK"})
    print("PRESERVAÇÃO OK", flush=True)


if __name__ == "__main__":
    main()
