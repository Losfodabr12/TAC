from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import atomic_json
from ..db import now
from ..diagnostics import check_resources
from ..learning import Cancelled, LearningDisabled
from .data import Corpus
from .model import ModelConfig, TACTransformer, expand_model
from .tokenizer import ByteTokenizer


def weight_hash(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def safe_load(path):
    if path.stat().st_size > 300 * 2**20:
        raise ValueError("Checkpoint excede o limite permitido.")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "TAC-v1" or value.get("tokenizer") != ByteTokenizer.version:
        raise ValueError("Checkpoint incompatível.")
    cfg = ModelConfig(**value["model_config"])
    if not (16 <= cfg.context <= 256 and 32 <= cfg.width <= 256 and 1 <= cfg.layers <= 4 and cfg.vocab_size == 259 and 1 <= cfg.heads <= 8 and cfg.width % cfg.heads == 0 and isinstance(cfg.ff_width, int) and 0 <= cfg.ff_width <= 2048):
        raise ValueError("Arquitetura de checkpoint fora dos limites.")
    return value


def retention_prefixes(generator, context, batch_size=4):
    """Prefixos sintéticos independentes dos testes, para repetição funcional.

    Mistura ruído de caracteres e motivos aleatórios repetidos. Não contém
    respostas corretas embutidas: os alvos são as previsões da rede anterior.
    """
    alphabet = torch.tensor(list(b"abcdefghijklmnopqrstuvwxyz0123456789 .,!?\n"), dtype=torch.long) + 3
    indices = torch.randint(len(alphabet), (batch_size, context), generator=generator)
    prefixes = alphabet[indices]
    for row in range(batch_size // 2, batch_size):
        length = int(torch.randint(3, min(13, context + 1), (), generator=generator))
        motif = prefixes[row, :length].clone()
        prefixes[row] = motif.repeat((context + length - 1) // length)[:context]
    prefixes[::2, 0] = ByteTokenizer.bos
    return prefixes


class Trainer:
    def __init__(self, config, db, gate):
        self.config, self.db, self.gate = config, db, gate
        self.last_result = None

    @property
    def folder(self):
        return self.config.data / "checkpoints"

    def checkpoint_path(self, checkpoint_id):
        rows = self.db.query("SELECT filename,status FROM checkpoints WHERE id=?", (checkpoint_id,))
        if not rows:
            raise ValueError("Checkpoint não encontrado.")
        if rows[0]['status'] == 'descartado_v2':
            raise ValueError('Checkpoint da geração antiga descartado na reconstrução. Use uma versão da TAC v2.')
        path = (self.folder / rows[0]["filename"]).resolve()
        if not path.is_relative_to(self.folder.resolve()) or path.suffix != ".pt":
            raise ValueError("Caminho de checkpoint inválido.")
        return path

    def active_id(self):
        rows = self.db.query("SELECT value FROM meta WHERE key='active_checkpoint'")
        return rows[0]["value"] if rows else None

    def activate(self, checkpoint_id):
        with self.gate.writing():
            cp = safe_load(self.checkpoint_path(checkpoint_id))
            if cp["step"] < 1:
                raise ValueError("Checkpoint ainda sem passos de treinamento.")
            model = TACTransformer(ModelConfig(**cp["model_config"]))
            model.load_state_dict(cp["model"])
            self.db.execute("INSERT INTO meta(key,value) VALUES('active_checkpoint',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (checkpoint_id,))
            self.db.event("restauracao_modelo", checkpoint_id)

    def device(self):
        preferred = self.config.get("device")
        if preferred != "cpu" and torch.cuda.is_available():
            available, total = torch.cuda.mem_get_info()
            if available < 512 * 2**20:
                if preferred == "cuda":
                    raise MemoryError("Pouca VRAM disponível. Selecione CPU ou libere a GPU.")
                return torch.device("cpu")
            torch.cuda.set_per_process_memory_fraction(min(0.75, self.config.get("vram_mb") * 2**20 / total))
            return torch.device("cuda")
        if preferred == "cuda":
            raise RuntimeError("CUDA indisponível. Escolha automático ou CPU nas configurações.")
        return torch.device("cpu")

    @torch.inference_mode()
    def evaluate(self, model, corpus, split, device, context=None):
        model.eval()
        rng = torch.Generator().manual_seed(1977)
        total_loss, count, correct = 0.0, 0, 0
        # Amostra fixa reproduzível por split; não substitui avaliação humana.
        for _ in range(4):
            x, y = corpus.batch(split, 4, context or model.cfg.context, rng, device)
            logits, loss = model(x, y)
            n = int((y != 0).sum().item())
            total_loss += float(loss.item()) * n
            count += n
            correct += int(((logits.argmax(-1) == y) & (y != 0)).sum().item())
        return total_loss / max(1, count), correct / max(1, count)

    def train(self, progress=lambda *_: None, resume=None, corpus=None, model_config=None, teacher_checkpoint=None):
        self.gate.check()
        if resume and model_config is not None:
            raise ValueError("Retomar um checkpoint preserva sua arquitetura.")
        check_resources(self.config)
        torch.set_num_threads(self.config.get("cpu_threads"))
        torch.manual_seed(self.config.get("seed"))
        device = self.device()
        cp = safe_load(self.checkpoint_path(resume)) if resume else None
        active = self.active_id()
        if not cp and active:
            cp = safe_load(self.checkpoint_path(active))
        if resume:
            path = self.config.data / "datasets" / f"{cp['dataset_hash']}.json"
            if not path.exists():
                raise ValueError("Snapshot de dados ausente/removido. Inicie um novo treinamento.")
            corpus = Corpus.load(path)
        elif corpus is None:
            corpus = Corpus.from_memory(self.db, self.config)
        snapshot = self.config.data / "datasets" / f"{corpus.digest}.json"
        with self.gate.writing():
            corpus.save(snapshot)
        cfg = ModelConfig(**cp["model_config"]) if cp else ModelConfig(context=self.config.get("context"), width=self.config.get("width"), heads=self.config.get("heads"), layers=self.config.get("layers"), ff_width=self.config.get("ff_width"))
        model = TACTransformer(cfg)
        if cp:
            model.load_state_dict(cp["model"])
        expansion = None
        if model_config is not None:
            if cp:
                model = expand_model(model, model_config)
                expansion = {"parent_checkpoint": active, "parent_config": asdict(cfg), "new_config": asdict(model_config)}
            else:
                model = TACTransformer(model_config)
            cfg = model_config
        model = model.to(device)
        # Compara as arquiteturas nas mesmas janelas, inclusive após uma retomada.
        evaluation_context = min(cfg.context, cp.get("training_config", {}).get("evaluation_context", cp["model_config"]["context"])) if cp else cfg.context
        def evaluate(model, data, split, device):
            if evaluation_context == model.cfg.context:
                return self.evaluate(model, data, split, device)
            return self.evaluate(model, data, split, device, context=evaluation_context)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.get("learning_rate"), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        rng = torch.Generator().manual_seed(self.config.get("seed"))
        step = 0
        if resume:
            optimizer.load_state_dict(cp["optimizer"])
            scaler.load_state_dict(cp["scaler"])
            rng.set_state(cp["sample_rng"])
            torch.set_rng_state(cp["torch_rng"])
            if device.type == "cuda" and cp.get("cuda_rng"):
                torch.cuda.set_rng_state_all(cp["cuda_rng"])
            step = cp["step"]
        initial_hash = weight_hash(model)
        baseline, baseline_accuracy = evaluate(model, corpus, "validation", device)
        fixed_base, _ = evaluate(model, corpus, "fixed", device)
        # Mantém avaliação de dados anteriores para medir esquecimento no mesmo conjunto.
        previous = None
        if resume and cp["training_config"].get("replay_dataset_hash"):
            prior_path = self.config.data / "datasets" / f"{cp['training_config']['replay_dataset_hash']}.json"
            if not prior_path.exists():
                raise ValueError("Snapshot de repetição anterior removido; não é possível retomar o mesmo treino.")
            previous = Corpus.load(prior_path)
        elif active and not resume:
            prior = safe_load(self.checkpoint_path(active))
            prior_path = self.config.data / "datasets" / f"{prior['dataset_hash']}.json"
            if prior_path.exists():
                previous = Corpus.load(prior_path)
        previous_base = evaluate(model, previous, "validation", device)[0] if previous else None
        if resume and cp.get("sampling"):
            corpus.sampled_sources = dict(cp["sampling"]["sources"])
            corpus.sampled_training_bytes = cp["sampling"]["training_bytes"]
            if previous:
                previous.sampled_training_bytes = cp["sampling"].get("replay_bytes", 0)
        run_id = uuid.uuid4().hex
        params = dict(self.config.values)
        params["replay_dataset_hash"] = previous.digest if previous else None
        params["evaluation_context"] = evaluation_context
        if teacher_checkpoint is not None:
            if resume:
                raise ValueError("A retomada mantém o professor salvo no checkpoint.")
            params["teacher_checkpoint"] = teacher_checkpoint
        if expansion:
            params["model_expansion"] = expansion
        if resume:
            # A retomada usa os parâmetros efetivos do checkpoint, não preferências novas.
            params.update(cp["training_config"])
            steps_epoch = int(cp["training_config"]["steps_per_epoch"])
            target_steps = int(cp["target_steps"])
            params["steps_per_epoch"] = steps_epoch
            if step >= target_steps:
                raise ValueError("Este treino terminou. Inicie um novo treinamento incremental.")
        else:
            steps_epoch = params["steps_per_epoch"]
            target_steps = params["epochs"] * steps_epoch
        # A referência da expansão permanece fixa entre rodadas: a tolerância não
        # se acumula em sucessivas atualizações de 2%.
        reference_id = (cp or {}).get("training_config", {}).get("fixed_reference_checkpoint")
        reference_id = reference_id or (cp or {}).get("training_config", {}).get("model_expansion", {}).get("parent_checkpoint")
        reference_id = reference_id or (expansion or {}).get("parent_checkpoint")
        fixed_reference = fixed_base
        if reference_id:
            reference_cp = safe_load(self.checkpoint_path(reference_id))
            reference = TACTransformer(ModelConfig(**reference_cp["model_config"])).to(device).eval()
            reference.load_state_dict(reference_cp["model"])
            fixed_reference = self.evaluate(reference, corpus, "fixed", device, context=evaluation_context)[0]
            params["fixed_reference_checkpoint"] = reference_id
            del reference
        fixed_limit_baseline = min(fixed_base, fixed_reference)
        # Repetição funcional opcional: conserva previsões da própria rede anterior
        # em prefixos aleatórios, sem consultar os exemplos ou respostas dos testes.
        teacher = None
        retention_weight = float(params.get("retention_weight", 0))
        if retention_weight:
            sampler_version = "mixed-ascii-v2"
            if resume and params.get("retention_sampler") != sampler_version:
                raise ValueError("Este experimento de retenção não registra a versão do amostrador. Restaure seus pesos e inicie um novo treino incremental.")
            params["retention_sampler"] = sampler_version
            teacher_id = params.get("teacher_checkpoint") or active
            if not teacher_id:
                raise ValueError("A retenção funcional requer um modelo anterior treinado.")
            teacher_cp = safe_load(self.checkpoint_path(teacher_id))
            teacher = TACTransformer(ModelConfig(**teacher_cp["model_config"])).to(device).eval()
            teacher.load_state_dict(teacher_cp["model"])
            teacher.requires_grad_(False)
            params["teacher_checkpoint"] = teacher_id
        precision_skips = 0
        self.db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash,baseline) VALUES(?,?,?,?,?,?,?)", (run_id, "neural", "treinando", now(), json.dumps(params), corpus.digest, baseline))
        self.db.execute("INSERT INTO metrics(run_id,step,epoch,val_loss,fixed_loss,accuracy,created) VALUES(?,?,?,?,?,?,?)", (run_id, step, 0, baseline, fixed_base, baseline_accuracy, now()))
        started = time.monotonic()
        interrupted, detail = False, ""
        candidates = []
        def acceptable(value, fixed_value, previous_value):
            return (value <= baseline * 1.02 and fixed_value <= fixed_limit_baseline * 1.02
                    and (previous is None or previous_value <= previous_base * 1.02))
        val, fixed, accuracy, train_loss = baseline, fixed_base, 0.0, None
        progress(step, f"{device}: {model.parameter_count:,} parâmetros; loss inicial {baseline:.4f}")
        try:
            while step < target_steps:
                self.gate.wait_if_paused()
                check_resources(self.config)
                if time.monotonic() - started >= params["train_seconds"]:
                    raise Cancelled("Limite de duração atingido. Retome pelo checkpoint.")
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for micro in range(params["accumulation"]):
                    self.gate.check()
                    # Metade dos microlotes preserva exemplos anteriores. Sem treinar testes.
                    chosen = previous if previous and (step * params["accumulation"] + micro) % 2 == 0 else corpus
                    x, y = chosen.batch("train", params["batch_size"], cfg.context, rng, device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        _, loss = model(x, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Loss não finita; atualização rejeitada.")
                    loss_sum += float(loss.detach())
                    scaler.scale(loss / params["accumulation"]).backward()
                if teacher is not None:
                    prefixes = retention_prefixes(rng, min(64, cfg.context, teacher.cfg.context)).to(device)
                    with torch.no_grad():
                        teacher_logits, _ = teacher(prefixes)
                    # FP32 evita instabilidade no cálculo da divergência.
                    student_logits, _ = model(prefixes)
                    log_prob = torch.log_softmax(student_logits.float(), dim=-1)
                    target_prob = torch.softmax(teacher_logits.float(), dim=-1)
                    retention = torch.nn.functional.kl_div(log_prob, target_prob, reduction="none").sum(-1).mean()
                    scaler.scale(retention_weight * retention).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
                if not torch.isfinite(norm):
                    # Overflow de FP16 é recuperável: reduz a escala e não toca pesos.
                    precision_skips += 1
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    progress(step, f"Precisão mista: lote descartado; escala reduzida para {scaler.get_scale():g}. Pesos preservados.")
                    if scaler.get_scale() < 1:
                        raise FloatingPointError("Gradientes inválidos persistentes; reduza a taxa de aprendizado ou use CPU.")
                    continue
                with self.gate.writing():
                    self.gate.check()
                    scaler.step(optimizer)
                    scaler.update()
                    step += 1
                train_loss = loss_sum / params["accumulation"]
                if step % steps_epoch == 0 or step == target_steps:
                    val, accuracy = evaluate(model, corpus, "validation", device)
                    fixed, _ = evaluate(model, corpus, "fixed", device)
                    self.db.execute("INSERT INTO metrics(run_id,step,epoch,train_loss,val_loss,fixed_loss,accuracy,created) VALUES(?,?,?,?,?,?,?,?)", (run_id, step, math.ceil(step / steps_epoch), train_loss, val, fixed, accuracy, now()))
                    epoch_checkpoint = self.save_checkpoint(model, optimizer, scaler, rng, corpus, run_id, step, target_steps, params, val, "epoca", previous)
                    prior_val = evaluate(model, previous, "validation", device)[0] if previous else None
                    if acceptable(val, fixed, prior_val):
                        candidates.append((val, step, epoch_checkpoint))
                    progress(step, f"Época {math.ceil(step / steps_epoch)} • treino {train_loss:.4f} • validação {val:.4f} • bytes corretos {accuracy:.1%}")
                elif step % 5 == 0:
                    progress(step, f"Passo {step}/{target_steps} • loss {train_loss:.4f}")
        except (Cancelled, LearningDisabled) as exc:
            interrupted, detail = True, str(exc)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                detail = "VRAM insuficiente. Reduza lote/contexto nas configurações ou use CPU."
            self.save_checkpoint(model, optimizer, scaler, rng, corpus, run_id, step, target_steps, params, val, "erro", previous)
            self.db.execute("UPDATE runs SET status='erro',finished=?,detail=? WHERE id=?", (now(), detail, run_id))
            raise RuntimeError(detail) from exc
        # Checkpoint de interrupção salva o estado exato, sem executar novos optimizer.step.
        val, accuracy = evaluate(model, corpus, "validation", device)
        fixed, _ = evaluate(model, corpus, "fixed", device)
        previous_final = evaluate(model, previous, "validation", device)[0] if previous else None
        accepted = not interrupted and acceptable(val, fixed, previous_final)
        status = "interrompido" if interrupted else ("aprovado" if accepted else "rejeitado")
        checkpoint_id = self.save_checkpoint(model, optimizer, scaler, rng, corpus, run_id, step, target_steps, params, val, status, previous)
        selected_sampling = {"sources": dict(corpus.sampled_sources), "training_bytes": corpus.sampled_training_bytes,
                             "replay_bytes": previous.sampled_training_bytes if previous else 0}
        last_checkpoint_id, selected_step = checkpoint_id, step
        if not accepted and not interrupted:
            failures = []
            for label, before, after in (("validação atual", baseline, val), ("teste fixo de referência", fixed_limit_baseline, fixed), ("validação anterior", previous_base, previous_final)):
                if before is not None and after > before * 1.02:
                    failures.append(f"{label}: piora de {(after / before - 1) * 100:.1f}% (limite 2%)")
            detail = "Atualização rejeitada: " + "; ".join(failures)
        # Se o final regredir, uma época anterior precisa passar novamente nos mesmos critérios.
        # O checkpoint final rejeitado fica preservado para inspeção.
        if not interrupted and not accepted:
            for _, candidate_step, candidate_id in sorted(candidates):
                candidate = safe_load(self.checkpoint_path(candidate_id))
                candidate_model = TACTransformer(cfg).to(device)
                candidate_model.load_state_dict(candidate["model"])
                candidate_val, candidate_accuracy = evaluate(candidate_model, corpus, "validation", device)
                candidate_fixed, _ = evaluate(candidate_model, corpus, "fixed", device)
                candidate_previous = evaluate(candidate_model, previous, "validation", device)[0] if previous else None
                if acceptable(candidate_val, candidate_fixed, candidate_previous):
                    model.load_state_dict(candidate["model"])
                    val, accuracy, fixed, previous_final = candidate_val, candidate_accuracy, candidate_fixed, candidate_previous
                    checkpoint_id, selected_step = candidate_id, candidate_step
                    selected_sampling = candidate["sampling"]
                    accepted, status = True, "aprovado"
                    detail = f"O passo final {step} foi rejeitado. A época do passo {selected_step} passou novamente na validação e foi selecionada."
                    del candidate_model
                    break
                del candidate_model
        if accepted:
            # Desligamento entre avaliação e promoção também impede alteração do modelo ativo.
            try:
                with self.gate.writing():
                    self.gate.check()
                    self.db.execute("INSERT INTO meta(key,value) VALUES('active_checkpoint',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (checkpoint_id,))
                    self.db.execute("UPDATE checkpoints SET status='aprovado' WHERE id=?", (checkpoint_id,))
            except (LearningDisabled, Cancelled):
                accepted = False
                status = "interrompido"
        self.db.execute("INSERT INTO metrics(run_id,step,epoch,train_loss,val_loss,fixed_loss,accuracy,created) VALUES(?,?,?,?,?,?,?,?)", (run_id, selected_step, math.ceil(selected_step / steps_epoch), train_loss if selected_step == step else None, val, fixed, accuracy, now()))
        self.last_result = {"run_id": run_id, "checkpoint": checkpoint_id, "status": status, "steps": step, "baseline": baseline, "validation": val, "fixed_baseline": fixed_base, "fixed_final": fixed, "previous_baseline": previous_base, "previous_final": previous_final, "initial_hash": initial_hash, "final_hash": weight_hash(model), "parameters": model.parameter_count, "device": str(device), "detail": detail}
        self.last_result.update(selected_step=selected_step, last_checkpoint=last_checkpoint_id)
        self.last_result.update(evaluation_context=evaluation_context, precision_skips=precision_skips,
                                retention_weight=retention_weight, teacher_checkpoint=params.get("teacher_checkpoint"))
        self.last_result.update(fixed_reference_baseline=fixed_reference, fixed_reference_checkpoint=reference_id)
        self.last_result["sampled_sources"] = selected_sampling["sources"]
        self.last_result["sampled_training_bytes"] = selected_sampling["training_bytes"]
        self.last_result["replay_training_bytes"] = selected_sampling["replay_bytes"]
        self.last_result["executed_source_samples"] = dict(corpus.sampled_sources)
        self.db.execute("UPDATE runs SET status=?,finished=?,final=?,detail=? WHERE id=?", (status, now(), val, json.dumps(self.last_result, ensure_ascii=False), run_id))
        atomic_json(self.config.data / "last_training.json", self.last_result)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self.last_result

    def save_checkpoint(self, model, optimizer, scaler, rng, corpus, run_id, step, target_steps, params, val, status, previous=None):
        checkpoint_id = uuid.uuid4().hex
        path = self.folder / f"{checkpoint_id}.pt"
        value = {"format": "TAC-v1", "tokenizer": ByteTokenizer.version, "model_config": asdict(model.cfg),
                 "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "sample_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                 "step": step, "target_steps": target_steps, "dataset_hash": corpus.digest,
                 "training_config": params, "run_id": run_id}
        value["sampling"] = {"sources": dict(corpus.sampled_sources), "training_bytes": corpus.sampled_training_bytes,
                             "replay_bytes": previous.sampled_training_bytes if previous else 0}
        temp = path.with_suffix(".tmp")
        torch.save(value, temp)
        os.replace(temp, path)
        self.db.execute("INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)", (checkpoint_id, run_id, path.name, step, val, status, now()))
        return checkpoint_id

    def generate(self, prompt):
        active = self.active_id()
        if not active:
            raise ValueError("Nenhum modelo treinado aprovado. Estude documentos e autorize um treinamento.")
        check_resources(self.config)
        cp = safe_load(self.checkpoint_path(active))
        model = TACTransformer(ModelConfig(**cp["model_config"]))
        model.load_state_dict(cp["model"])
        torch.set_num_threads(self.config.get("cpu_threads"))
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([[tokenizer.bos, *tokenizer.encode(prompt, special=False)[-model.cfg.context + 1:]]])
        generated = model.generate(tokens, max_new_tokens=100)
        return {"mode": "neural experimental", "text": "Amostra da rede neural local. Não é uma resposta factual validada; o modelo inicial não conversa de forma confiável.\n\n" + tokenizer.decode(generated[0].tolist()), "sources": []}
