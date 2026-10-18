"""Experimento separado: REINFORCE em adição finita (a,b em 0..3)."""
import json
import os
import time
import uuid

import torch
from torch import nn

from ..db import now
from ..diagnostics import check_resources
from ..learning import Cancelled, LearningDisabled
from .training import weight_hash


class ArithmeticPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 48), nn.Tanh(), nn.Linear(48, 7))

    def forward(self, x):
        return self.net(x)


def accuracy(model, inputs, targets):
    with torch.no_grad():
        return float((model(inputs).argmax(-1) == targets).float().mean())


def run_experiment(config, db, gate, progress=lambda *_: None, steps=300):
    check_resources(config)
    gate.check()
    torch.set_num_threads(config.get("cpu_threads"))
    torch.manual_seed(config.get("seed"))
    rng = torch.Generator().manual_seed(config.get("seed"))
    model = ArithmeticPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    pairs = [(a, b) for a in range(4) for b in range(4)]
    inputs = torch.stack([torch.cat((torch.nn.functional.one_hot(torch.tensor(a), 4), torch.nn.functional.one_hot(torch.tensor(b), 4))).float() for a, b in pairs])
    targets = torch.tensor([a + b for a, b in pairs])
    baseline = accuracy(model, inputs, targets)
    initial_hash = weight_hash(model)
    run_id = uuid.uuid4().hex
    db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash,baseline) VALUES(?,?,?,?,?,?,?)", (run_id, "reforco", "treinando", now(), json.dumps({"steps": steps, "seed": config.get("seed")}), "addition-0-3-v1", baseline))
    started = time.monotonic()
    status, step, attempts = "concluido", 0, []
    try:
        for step in range(1, steps + 1):
            gate.wait_if_paused()
            check_resources(config)
            if time.monotonic() - started > config.get("train_seconds"):
                raise Cancelled("Limite de duração do experimento.")
            logits = model(inputs)
            probabilities = logits.softmax(-1)
            actions = torch.multinomial(probabilities.detach(), 1, generator=rng).squeeze(-1)
            # Recompensa objetiva: 1 apenas quando a soma escolhida está correta.
            rewards = (actions == targets).float()
            advantage = rewards - probabilities.detach().gather(1, targets[:, None]).squeeze(-1)
            log_prob = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(-1)
            entropy = -(probabilities * logits.log_softmax(-1)).sum(-1).mean()
            loss = -(log_prob * advantage).mean() - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            with gate.writing():
                gate.check()
                optimizer.step()
            attempts.extend((run_id, step, a, b, int(actions[i]), float(rewards[i]), now()) for i, (a, b) in enumerate(pairs))
            if step % 20 == 0:
                score = accuracy(model, inputs, targets)
                db.execute("INSERT INTO metrics(run_id,step,epoch,train_loss,accuracy,created) VALUES(?,?,?,?,?,?)", (run_id, step, 0, float(loss.detach()), score, now()))
                with db.connect() as c:
                    c.executemany("INSERT INTO rl_attempts(run_id,step,a,b,action,reward,created) VALUES(?,?,?,?,?,?,?)", attempts)
                attempts.clear()
                progress(step, f"Reforço: {step}/{steps} • acerto nas 16 somas: {score:.1%}")
    except (Cancelled, LearningDisabled):
        status = "interrompido"
    except Exception:
        db.execute("UPDATE runs SET status='erro',finished=? WHERE id=?", (now(), run_id))
        raise
    finally:
        if attempts:
            with db.connect() as c:
                c.executemany("INSERT INTO rl_attempts(run_id,step,a,b,action,reward,created) VALUES(?,?,?,?,?,?,?)", attempts)
    final = accuracy(model, inputs, targets)
    path = config.data / "checkpoints" / f"rl_{run_id}.pt"
    torch.save({"format": "TAC-RL-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "seed": config.get("seed"), "baseline": baseline, "final": final}, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    db.execute("UPDATE runs SET status=?,finished=?,final=?,detail=? WHERE id=?", (status, now(), final, "Avaliação nas mesmas 16 somas treinadas; não mede generalização.", run_id))
    return {"run_id": run_id, "status": status, "baseline": baseline, "accuracy": final, "initial_hash": initial_hash, "final_hash": weight_hash(model), "steps": step, "checkpoint": str(path), "scope": "16 somas finitas; sem evidência de raciocínio geral"}
