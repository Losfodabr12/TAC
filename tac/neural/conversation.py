"""Geração de respostas treinada localmente, separada do classificador e do leitor."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch

from ..config import atomic_json
from ..conversation_lessons import canonical, make_sets, prompt_text
from ..context_lessons import make_context_sets
from ..natural_dialogues import make_natural_sets
from ..db import now
from ..diagnostics import check_resources
from ..learning import Cancelled, LearningDisabled
from .model import ModelConfig, TACTransformer, count_parameters
from .bpe import LocalBPETokenizer
from .tokenizer import ByteTokenizer
from .training import Trainer, safe_load, weight_hash

LEGACY_TASK = "TAC-conversation-sft-v1"
TASK = "TAC-conversation-sft-v2"
MAX_PARAMETERS = 10_100_000


def conversation_10m_config(vocab_size=1024):
    # 10.004.544 parâmetros com o BPE local de 1024 tokens e RoPE.
    return ModelConfig(vocab_size=vocab_size, context=512, width=384, heads=6,
                       layers=5, ff_width=1728, tie_embeddings=True, rope=True)


def initialize_rotary_model(cp, tokenizer):
    """Transfere somente pesos próprios. RoPE altera a função e exige reavaliação."""
    model = TACTransformer(conversation_10m_config(tokenizer.vocab_size))
    source, target = cp["model"], model.state_dict()
    with torch.no_grad():
        for name, value in target.items():
            original = source[name]
            if original.shape == value.shape:
                value.copy_(original)
            elif name.endswith("ff.0.weight") or name.endswith("ff.0.bias"):
                value[:original.shape[0]].copy_(original)
            elif name.endswith("ff.2.weight"):
                value.zero_()
                value[:, :original.shape[1]].copy_(original)
            else:
                raise ValueError("Transferência incompatível: " + name)
    model.load_state_dict(target)
    return model


def context_ready(metrics):
    if metrics is None:
        return True
    groups = metrics.get("by_skill", {})
    skills = ("context_color", "context_update", "context_fact", "context_location")
    scores = [groups.get(s, {}).get("correct", 0) / max(1, groups.get(s, {}).get("total", 0)) for s in skills]
    logic = groups.get("logic_condition", {})
    # Quatro tarefas de contexto têm peso igual. Aritmética é reportada à parte;
    # não é usada como substituto da capacidade de manter uma conversa.
    return (min(scores) >= .65 and sum(scores) / len(scores) >= .8
            and logic.get("correct", 0) / max(1, logic.get("total", 0)) >= .8)


def load_checkpoint(path):
    if path.stat().st_size > 512 * 2**20:
        raise ValueError("Checkpoint de diálogo muito grande")
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if cp.get("format") == "TAC-v1":
        return safe_load(path)
    if cp.get("format") != "TAC-chat-v2" or cp.get("task") != TASK:
        raise ValueError("Formato de diálogo incompatível")
    cfg = ModelConfig(**cp["model_config"])
    if cp.get('generation') == 'TAC-v3':
        from .v3 import V3Tokenizer, v3_config
        tokenizer = V3Tokenizer.from_state(cp['tokenizer_state'])
        if vars(cfg) != vars(v3_config()) or tokenizer.vocab_size != cfg.vocab_size:
            raise ValueError('Arquitetura TAC v3 incompatível')
        return cp
    if path.stat().st_size > 300 * 2**20:
        raise ValueError('Checkpoint v2 de diálogo muito grande')
    tokenizer = LocalBPETokenizer.from_state(cp["tokenizer_state"])
    max_context = 1024 if cp.get('generation') == 'TAC-v2' else 512
    if not (cfg.vocab_size == tokenizer.vocab_size <= 2048 and 16 <= cfg.context <= max_context
            and 32 <= cfg.width <= 384 and 1 <= cfg.layers <= 5 and 1 <= cfg.heads <= 8
            and cfg.width % cfg.heads == 0 and 0 <= cfg.ff_width <= 2048
            and count_parameters(cfg) <= MAX_PARAMETERS):
        raise ValueError("Arquitetura de diálogo fora dos limites")
    return cp


def initialize_bpe_model(cp, tokenizer):
    old_cfg = ModelConfig(**cp["model_config"])
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, width=old_cfg.width, heads=old_cfg.heads,
                      layers=old_cfg.layers, context=old_cfg.context,
                      ff_width=min(old_cfg.ff_width or 4 * old_cfg.width, 768), tie_embeddings=True)
    model = TACTransformer(cfg)
    source, target = cp["model"], model.state_dict()
    with torch.no_grad():
        for name, value in target.items():
            if name in ("tokens.weight", "output.weight"):
                continue
            original = source[name]
            if original.shape == value.shape:
                value.copy_(original)
            elif name.endswith("ff.0.weight") or name.endswith("ff.0.bias"):
                value.copy_(original[:value.shape[0]])
            elif name.endswith("ff.2.weight"):
                value.copy_(original[:, :value.shape[1]])
            else:
                raise ValueError("Camada não transferível: " + name)
        # A origem é a própria rede de bytes da TAC. Tokens novos recebem a média
        # das representações dos seus bytes, sem pesos externos.
        embeddings = (source["tokens.weight"] + source["output.weight"]) / 2
        for index in range(cfg.vocab_size):
            ids = [index] if index < 259 else [byte + 3 for byte in tokenizer.tokens[index]]
            target["tokens.weight"][index].copy_(embeddings[ids].mean(0))
        target["output.weight"].copy_(target["tokens.weight"])
    model.load_state_dict(target)
    return model


class ConversationData:
    def __init__(self, sets, tokenizer=None):
        self.sets = sets
        self.tokenizer = tokenizer or ByteTokenizer()
        content = {"sets": sets, "tokenizer": self.tokenizer.state()} if isinstance(self.tokenizer, LocalBPETokenizer) else sets
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
        self.digest = hashlib.sha256(payload).hexdigest()
        self._encoded = {}
        self.groups = defaultdict(list)
        for index, row in enumerate(sets["train"]):
            self.groups[row["skill"]].append(index)
        self.skills = sorted(self.groups)
        self.pools = {}
        for index, row in enumerate(sets['train']):
            pool = row.get('mixture', 'legacy')
            grouping = row['group'] if pool == 'imported' else row['skill']
            self.pools.setdefault(pool, defaultdict(list))[grouping].append(index)

    def save(self, path):
        atomic_json(path, {"format": TASK if isinstance(self.tokenizer, LocalBPETokenizer) else LEGACY_TASK,
                           "sets": self.sets, "hash": self.digest,
                           "tokenizer_state": self.tokenizer.state() if isinstance(self.tokenizer, LocalBPETokenizer) else None})

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = LocalBPETokenizer.from_state(value["tokenizer_state"]) if value.get("tokenizer_state") else None
        data = cls(value["sets"], tokenizer)
        if value.get("format") not in (TASK, LEGACY_TASK) or data.digest != value["hash"]:
            raise ValueError("Snapshot de diálogo incompatível ou alterado")
        return data

    def encode(self, row, context):
        key = (id(row), context)
        if key in self._encoded:
            return self._encoded[key]
        messages = row['messages'][-7:] if row.get('imported') else row['messages']
        prefix = self.tokenizer.encode(prompt_text(messages), special=False)
        answer = self.tokenizer.encode(" " + row["answer"], special=False) + [self.tokenizer.eos]
        if len(answer) >= context - 4:
            raise ValueError("Resposta maior que o contexto do exercício")
        prefix = [self.tokenizer.bos] + prefix[-(context - len(answer)):]
        sequence = torch.tensor(prefix + answer, dtype=torch.long)
        x, y = sequence[:-1], sequence[1:].clone()
        y[:len(prefix) - 1] = 0  # Somente a resposta e EOS contribuem à perda.
        self._encoded[key] = (x, y)
        return x, y

    def batch(self, rows, context, device):
        encoded = [self.encode(row, context) for row in rows]
        length = max(len(x) for x, _ in encoded)
        x = torch.zeros((len(rows), length), dtype=torch.long)
        y = torch.zeros_like(x)
        for i, (a, b) in enumerate(encoded):
            x[i, :len(a)], y[i, :len(b)] = a, b
        return x.to(device), y.to(device)

    def sample(self, size, rng):
        result = []
        for _ in range(size):
            skill = self.skills[int(torch.randint(len(self.skills), (), generator=rng))]
            indices = self.groups[skill]
            result.append(self.sets["train"][indices[int(torch.randint(len(indices), (), generator=rng))]])
        return result

    def sample_pool(self, size, rng, pool):
        groups = self.pools.get(pool)
        if not groups:
            raise ValueError('Mistura de treino vazia: ' + pool)
        skills = sorted(groups)
        result = []
        for _ in range(size):
            indices = groups[skills[int(torch.randint(len(skills), (), generator=rng))]]
            result.append(self.sets['train'][indices[int(torch.randint(len(indices), (), generator=rng))]])
        return result


def prepare_data(cp, large=False, curriculum=None):
    sets = make_sets()
    tokenizer = LocalBPETokenizer.from_state(cp["tokenizer_state"]) if cp.get("tokenizer_state") else None
    if large:
        fragments = []
        if tokenizer:
            for token in tokenizer.tokens.values():
                text = token.decode("utf-8", errors="replace")
                if text.isalpha() and 2 <= len(text) <= 8:
                    fragments.append(text)
        fragments = tuple(sorted(set(fragments)))
        for suffix, additions in (("context", make_context_sets(fragments)), ("natural", make_natural_sets(fragments))):
            sets["train"] += additions["train"]
            sets["validation_" + suffix] = additions["validation"]
            sets["test_" + suffix] = additions["test"]
    if curriculum == 'fluency':
        from ..fluency_lessons import make_fluency_sets, natural_legacy
        reviewed = make_fluency_sets()
        sets['train'] = [{**r, 'mixture':'legacy'} for r in sets['train'] if natural_legacy(r)] + reviewed['train']
        sets['validation_fluency'], sets['test_fluency'] = reviewed['validation'], reviewed['test']
    forbidden = {canonical(prompt_text(row["messages"])) for split, rows in sets.items() if split != "train" for row in rows}
    seen, clean = set(), []
    for row in sets["train"]:
        key = canonical(prompt_text(row["messages"]))
        if key not in forbidden and key not in seen:
            seen.add(key)
            clean.append(row)
    sets["train"] = clean
    tokenizer = tokenizer or LocalBPETokenizer.fit(prompt_text(r["messages"]) + " " + r["answer"] for r in sets["train"])
    return ConversationData(sets, tokenizer)


def natural_ready(metrics, baseline):
    if metrics is None:
        return True
    groups = metrics.get("by_skill", {})
    critical = [f"natural_{slot}_{stage}" for slot in ("color", "name", "topic") for stage in ("correction", "recall")]
    scores = [groups.get(s, {}).get("correct", 0) / max(1, groups.get(s, {}).get("total", 0)) for s in critical]
    return (min(scores) >= .8 and metrics.get("exact_accuracy", 0) >= .65
            and metrics["exact_accuracy"] >= (baseline or {}).get("exact_accuracy", 0) - .02)


@torch.inference_mode()
def decode_reply(model, messages, device, max_tokens=128, generation_options=None):
    tokenizer = getattr(model, "text_tokenizer", ByteTokenizer())
    prompt = tokenizer.encode(prompt_text(messages), special=False)
    tokens = torch.tensor([[tokenizer.bos, *prompt[-model.cfg.context + 1:]]], device=device)
    generated = model.generate(tokens, max_new_tokens=max_tokens, **({'greedy':True} if generation_options is None else generation_options))
    return tokenizer.decode(generated[0, tokens.shape[1]:].tolist()).strip()


class ConversationEngine:
    def __init__(self, config, db, gate):
        self.config, self.db, self.gate = config, db, gate
        self.storage = Trainer(config, db, gate)
        self._cached_id, self._cached_model = None, None

    def import_store(self):
        from ..dialogue_imports import ConversationImports
        return ConversationImports(self.config,self.db,self.gate)

    def preview_import(self, path, progress=lambda *_:None, cancel=None):
        from ..dialogue_imports import prompt_key
        parent = self.active_id()
        if not parent:
            raise ValueError('Prepare uma versão de conversa aprovada com os exemplos do programa antes de importar.')
        cp = load_checkpoint(self.storage.checkpoint_path(parent))
        if not cp.get('tokenizer_state'):
            raise ValueError('A importação exige uma versão de conversa com o vocabulário local BPE.')
        data = prepare_data(cp,large=cp['model_config']['width']>=384,curriculum='fluency')
        reserved = {prompt_key(r) for key,rows in data.sets.items() if key!='train' for r in rows}
        trained = {prompt_key(r) for r in data.sets['train']}
        old_path = self.config.data/'datasets'/f"conversation_{cp['dataset_hash']}.json"
        if old_path.is_file():
            historical = ConversationData.load(old_path)
            trained.update(prompt_key(r) for r in historical.sets['train'])
        result = self.import_store().preview(path,data.tokenizer,cp['model_config']['context'],reserved,trained,progress,cancel)
        result['reference_checkpoint'] = parent
        return result

    def commit_import(self, preview):
        # O vocabulário/modelo não pode mudar entre conferir e confirmar.
        if preview.get('reference_checkpoint') != self.active_id():
            raise ValueError('A versão de conversa mudou. Confira o arquivo novamente.')
        return self.import_store().commit(preview)

    def prepare_training(self, profile=None, curriculum=None, resume=None):
        from ..dialogue_imports import digest, merge_imports, training_counts
        parent = resume or self.active_id() or self.storage.active_id()
        if not parent:
            raise ValueError('Prepare primeiro uma versão inicial com os exercícios do programa.')
        cp = load_checkpoint(self.storage.checkpoint_path(parent))
        store = self.import_store()
        skipped = []
        if resume:
            data = ConversationData.load(self.config.data/'datasets'/f"conversation_{cp['dataset_hash']}.json")
        else:
            data = prepare_data(cp,large=profile=='10m' or cp['model_config']['width']>=384,curriculum=curriculum)
            if cp.get('tokenizer_state'):
                rows = store.snapshot(digest(data.tokenizer.state()))
                data, skipped = merge_imports(data,rows)
            if self.config.get('conversation_transfer_practice'):
                from ..transfer_practice import augment
                data=augment(data)
        return {'data':data,'source_checkpoint':parent,'revision':store.revision(),'counts':training_counts(data),
                'profile':profile,'curriculum':curriculum,'resume':resume,'skipped':skipped,
                'document_dataset':self.config.get('conversation_document_dataset'),
                'transfer_practice':self.config.get('conversation_transfer_practice')}

    def active_id(self):
        rows = self.db.query("SELECT value FROM meta WHERE key='active_conversation'")
        return rows[0]["value"] if rows else None

    def activate(self, checkpoint_id):
        with self.gate.writing():
            _, cp = self.load(checkpoint_id)
            if cp["step"] < 1:
                raise ValueError("Checkpoint ainda sem treino")
            self.db.execute("INSERT INTO meta(key,value) VALUES('active_conversation',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (checkpoint_id,))
            self.db.event("restauracao_dialogo_neural", checkpoint_id)

    def load(self, checkpoint_id, device="cpu"):
        cp = load_checkpoint(self.storage.checkpoint_path(checkpoint_id))
        if cp.get("task") not in (TASK, LEGACY_TASK):
            raise ValueError("Este checkpoint não é de diálogo gerativo")
        model = TACTransformer(ModelConfig(**cp["model_config"])).to(device)
        model.load_state_dict(cp["model"])
        if cp.get('generation') == 'TAC-v3':
            from .v3 import V3Tokenizer
            model.text_tokenizer = V3Tokenizer.from_state(cp['tokenizer_state'])
        else:
            model.text_tokenizer = LocalBPETokenizer.from_state(cp["tokenizer_state"]) if cp.get("tokenizer_state") else ByteTokenizer()
        model.eval()
        return model, cp

    @torch.inference_mode()
    def evaluate(self, model, data, device, split="validation", generation=False, interruptible=False):
        model.eval()
        loss_sum = correct = count = 0
        rows = data.sets[split]
        for start in range(0, len(rows), 8):
            if interruptible:
                self.gate.check()
            x, y = data.batch(rows[start:start + 8], model.cfg.context, device)
            logits, loss = model(x, y)
            n = int((y != 0).sum())
            loss_sum += float(loss) * n
            correct += int(((logits.argmax(-1) == y) & (y != 0)).sum())
            count += n
        result = {"loss": loss_sum / max(1, count), "token_accuracy": correct / max(1, count), "targets": count,
                  "examples": len(rows), "tokenizer": data.tokenizer.version}
        if generation:
            outputs = []
            for row in rows:
                if interruptible:
                    self.gate.check()
                answer = decode_reply(model, row["messages"], device)
                accepted_answers = row.get("acceptable_answers", [row["answer"]])
                outputs.append({"messages": row["messages"], "expected": row["answer"], "acceptable_answers": accepted_answers,
                                "actual": answer, "skill": row["skill"], "exact": answer in accepted_answers})
            result["exact_matches"] = sum(row["exact"] for row in outputs)
            result["exact_accuracy"] = result["exact_matches"] / max(1, len(outputs))
            result["by_skill"] = {skill: {"correct": sum(row["exact"] for row in outputs if row["skill"] == skill),
                                                  "total": sum(row["skill"] == skill for row in outputs)} for skill in sorted({row["skill"] for row in rows})}
            result["outputs"] = outputs
        return result

    @torch.inference_mode()
    def evaluate_sessions(self, model, data, device, split="validation_natural", interruptible=False):
        """Usa respostas geradas pela própria rede no histórico, sem respostas ouro."""
        rows = [row for row in data.sets[split] if row['skill'].endswith('_recall') and len(row['messages']) == 5]
        sessions = []
        for row in rows:
            history = []
            for index in (0, 2, 4):
                if interruptible:
                    self.gate.check()
                user = row['messages'][index]
                answer = decode_reply(model, [*history[-6:], user], device)
                history.extend([user, ['assistant', answer]])
            sessions.append({'skill': row['skill'], 'history': history,
                             'expected_immediate': row['messages'][3][1], 'expected_recall': row['answer'],
                             'immediate_correct': history[3][1] == row['messages'][3][1],
                             'recall_correct': history[5][1] in row.get('acceptable_answers', [row['answer']])})
        count = len(sessions)
        return {'examples': count, 'immediate_correct': sum(r['immediate_correct'] for r in sessions),
                'recall_correct': sum(r['recall_correct'] for r in sessions),
                'immediate_accuracy': sum(r['immediate_correct'] for r in sessions) / max(1, count),
                'recall_accuracy': sum(r['recall_correct'] for r in sessions) / max(1, count), 'sessions': sessions}

    @torch.inference_mode()
    def evaluate_fluency(self, model, data, device, split='validation_fluency', interruptible=False):
        from ..fluency_lessons import score_reply
        episodes = defaultdict(list)
        for row in data.sets[split]:
            episodes[row['episode']].append(row)
        sessions, scores = [], []
        for episode, rows in episodes.items():
            history, turns = [], []
            for row in sorted(rows,key=lambda r:r['turn']):
                if interruptible:
                    self.gate.check()
                user = row['messages'][-1]
                actual = decode_reply(model,[*history[-6:],user],device)
                score = score_reply(actual,row['relevance_rules'])
                scores.append(score)
                turns.append({'user':user[1], 'actual':actual, 'reference':row['answer'], 'rules':row['relevance_rules'], **score})
                history.extend([user,['assistant',actual]])
            sessions.append({'episode':episode,'turns':turns})
        count = len(scores)
        return {'episodes':len(sessions),'turns':count,'rule_matches':sum(s['passes_simple_rules'] for s in scores),
                'rule_accuracy':sum(s['passes_simple_rules'] for s in scores)/max(1,count),
                'slot_drift_rate':sum(s['slot_drift'] for s in scores)/max(1,count),
                'method':'critérios lexicais simples em histórico próprio; não mede fluência geral', 'sessions':sessions}

    @torch.inference_mode()
    def evaluate_imported(self, model, data, device, split):
        """Avaliação informativa por habilidade, com respostas próprias no histórico."""
        model.eval()
        episodes = defaultdict(list)
        for row in data.sets.get(split,[]):
            episodes[row['episode']].append(row)
        outputs, skills = [], defaultdict(lambda:{'correct':0,'total':0})
        for episode, rows in sorted(episodes.items()):
            last = max(rows,key=lambda r:r['turn'])
            targets = {r['turn']:r for r in rows}
            history = []
            # Mantém a conversa inteira para reconstruir turnos deduplicados;
            # respostas de referência nunca entram no prompt desta avaliação.
            for turn, user in enumerate(last['messages'][::2]):
                self.gate.check()
                answer = decode_reply(model,[*history[-6:],user],device)
                history.extend([user,['assistant',answer]])
                if turn not in targets:
                    continue
                row = targets[turn]
                exact = answer == row['answer']
                skills[row['skill']]['total'] += 1
                skills[row['skill']]['correct'] += int(exact)
                outputs.append({'group':row['group'],'episode':episode,'turn':turn,'skill':row['skill'],
                                'user':user[1],'expected':row['answer'],'actual':answer,'exact':exact})
        return {'examples':len(outputs),'groups':len({r['group'] for r in outputs}),
                'by_skill':dict(skills),'exact_matches':sum(r['exact'] for r in outputs),'outputs':outputs,
                'method':'reprodução textual exata com histórico gerado; não mede compreensão geral'}

    def save_checkpoint(self, model, optimizer, scaler, rng, data, run_id, step, params, validation, status):
        checkpoint_id = uuid.uuid4().hex
        path = self.config.data / "checkpoints" / (checkpoint_id + ".pt")
        value = {"format": "TAC-chat-v2" if isinstance(data.tokenizer, LocalBPETokenizer) else "TAC-v1",
                 "tokenizer": data.tokenizer.version, "task": TASK if isinstance(data.tokenizer, LocalBPETokenizer) else LEGACY_TASK,
                 "tokenizer_state": data.tokenizer.state() if isinstance(data.tokenizer, LocalBPETokenizer) else None,
                 "model_config": vars(model.cfg), "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "sample_rng": rng.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                 "dataset_hash": data.digest, "step": step, "target_steps": params["steps"], "training_config": params,
                 "run_id": run_id, "validation": validation}
        temp = path.with_suffix(".tmp")
        torch.save(value, temp)
        os.replace(temp, path)
        self.db.execute("INSERT INTO checkpoints(id,run_id,filename,step,val_loss,status,created) VALUES(?,?,?,?,?,?,?)",
                        (checkpoint_id, run_id, path.name, step, validation["loss"], status, now()))
        return checkpoint_id

    def train(self, progress=lambda *_: None, steps=6000, learning_rate=None, seconds=1800, resume=None,
              profile=None, text_corpus=None, pretrain_steps=0, initial_checkpoint=None, retention_weight=2., curriculum=None,
              prepared=None, stability_policy=None, sampler_mode=None):
        self.gate.check()
        check_resources(self.config)
        progress(0, "Preparando dados locais e a configuração de treino...")
        torch.set_num_threads(self.config.get("cpu_threads"))
        torch.manual_seed(self.config.get("seed"))
        device = self.storage.device()
        learning_rate = self.config.get('conversation_learning_rate') if learning_rate is None else learning_rate
        stability_policy = self.config.get('conversation_stability') if stability_policy is None else stability_policy
        sampler_mode = self.config.get('conversation_sampler') if sampler_mode is None else sampler_mode
        if stability_policy not in ('off','anchor_v1') or sampler_mode not in ('replacement','deck_v1'):
            raise ValueError('Estratégia de treino de conversa inválida')
        if not 1 <= steps <= 60000 or not .00001 <= learning_rate <= .003 or not 1 <= seconds <= 3600:
            raise ValueError("Limites do treino: 1..60000 passos, taxa 0,00001..0,003 e duração 1..3600s")
        if profile not in (None, "10m"):
            raise ValueError("Perfil de conversa inválido")
        if curriculum not in (None, 'fluency'):
            raise ValueError('Currículo inválido')
        if not 0 <= retention_weight <= 10:
            raise ValueError("Peso de retenção deve estar entre 0 e 10")
        parent = initial_checkpoint or self.active_id() or self.storage.active_id()
        if initial_checkpoint and resume:
            raise ValueError("Escolha iniciar uma etapa ou retomar, não ambos")
        if not parent:
            raise ValueError("Treine primeiro a rede de texto para preparar o ponto de partida")
        cp = load_checkpoint(self.storage.checkpoint_path(resume or parent))
        if resume and cp.get("task") not in (TASK, LEGACY_TASK):
            raise ValueError("A retomada exige um checkpoint de diálogo")
        if resume:
            data = ConversationData.load(self.config.data / "datasets" / f"conversation_{cp['dataset_hash']}.json")
        else:
            if prepared is not None:
                if (prepared['source_checkpoint'] != parent or prepared['resume'] is not None
                        or prepared['profile'] != profile or prepared['curriculum'] != curriculum
                        or prepared['revision'] != self.import_store().revision()):
                    raise ValueError('O material do treino mudou. Confira novamente antes de iniciar.')
                if prepared.get('document_dataset',self.config.get('conversation_document_dataset'))!=self.config.get('conversation_document_dataset'):
                    raise ValueError('A seleção de textos mudou; confira novamente.')
                if prepared.get('transfer_practice',self.config.get('conversation_transfer_practice'))!=self.config.get('conversation_transfer_practice'):
                    raise ValueError('A prática experimental mudou; confira novamente.')
                data = prepared['data']
            else:
                data = prepare_data(cp, large=profile == "10m" or cp["model_config"]["width"] >= 384, curriculum=curriculum)
                if cp.get('tokenizer_state'):
                    from ..dialogue_imports import digest, merge_imports
                    rows = self.import_store().snapshot(digest(data.tokenizer.state()))
                    data, _ = merge_imports(data,rows)
                if self.config.get('conversation_transfer_practice'):
                    from ..transfer_practice import augment
                    data=augment(data)
        params = {"steps": steps, "learning_rate": learning_rate, "seconds": seconds, "batch_size": 16,
                  "seed": self.config.get("seed"), "parent": parent, "warmup": min(100, max(1, steps // 10)), "task": TASK,
                  "position_jitter": 64, "accumulation": 1, "evaluation_interval": 1000,
                  "profile": profile, "text_corpus": text_corpus, "pretrain_steps": pretrain_steps,
                  "replay_interval": 10, "promotion_policy": "context-macro-v1"}
        fresh = not resume and profile == "10m" and cp["model_config"]["width"] < 384
        adapted = not resume and profile == "10m" and not fresh and not cp["model_config"].get("rope")
        params["architecture_adapted"] = adapted
        large = profile == "10m" or cp["model_config"]["width"] >= 384
        if "validation_natural" in data.sets:
            params["promotion_policy"] = "context-natural-v2"
        retainable = cp.get("tokenizer_state") and cp.get("task") in (TASK, LEGACY_TASK)
        params["retention_weight"] = retention_weight if retainable and not (fresh or adapted) else 0.
        params["retention_source"] = parent
        params['curriculum'] = curriculum
        params['stability_policy'] = stability_policy
        params['sampler_mode'] = sampler_mode
        if self.config.get('conversation_document_dataset'):
            params['text_corpus']=self.config.get('conversation_document_dataset')
        if curriculum == 'fluency':
            params['retention_source'] = self.active_id() or parent
            params['mixture_counts'] = {'reviewed':0,'language':0,'legacy':0}
        if large:
            params.update(batch_size=min(8, self.config.get("batch_size")), accumulation=self.config.get("accumulation"), position_jitter=256,
                          evaluation_interval=max(1000, steps // 3), warmup=min(300, max(1, steps // 10)))
        if resume:
            params = cp["training_config"]
        stable = params.get('stability_policy') == 'anchor_v1'
        if (stable or params.get('sampler_mode')=='deck_v1') and params.get('curriculum')!='fluency':
            raise ValueError('A estratégia experimental exige o currículo de fluência.')
        if stable and not params.get('retention_weight'):
            raise ValueError('A âncora exige uma versão anterior e retenção autorizada.')
        from ..dialogue_imports import training_counts
        counts = training_counts(data)
        has_imported = counts['user_examples'] > 0
        if has_imported and (params.get('curriculum') != 'fluency' or not params.get('retention_weight')):
            raise ValueError('Treino com conversas importadas exige a mistura de conversa com retenção da versão anterior.')
        if has_imported and ('reviewed' not in data.pools or 'legacy' not in data.pools):
            raise ValueError('Faltam exercícios do programa para misturar às suas conversas.')
        if has_imported:
            params.setdefault('imported_mixture','TAC-imported-mix-v1')
            if not resume:params['selection_policy']='protected-imported-v1'
            params['mixture_counts'].setdefault('imported',0)
        params.setdefault('imported_counts',counts)
        if 'validation_imported' in data.sets:
            params.setdefault('imported_reference',self.active_id())
        with self.gate.writing():
            data.save(self.config.data / "datasets" / f"conversation_{data.digest}.json")
            if params.get('curriculum') == 'fluency' and not params.get('text_corpus'):
                from ..fluency_lessons import make_language_corpus
                language = make_language_corpus()
                language.save(self.config.data/'datasets'/f'{language.digest}.json')
                params['text_corpus'] = language.digest
                del language
        if fresh:
            model = TACTransformer(conversation_10m_config(data.tokenizer.vocab_size)).to(device)
        elif adapted:
            model = initialize_rotary_model(cp, data.tokenizer).to(device)
        elif isinstance(data.tokenizer, LocalBPETokenizer) and not cp.get("tokenizer_state"):
            model = initialize_bpe_model(cp, data.tokenizer).to(device)
        else:
            model = TACTransformer(ModelConfig(**cp["model_config"])).to(device)
            model.load_state_dict(cp["model"])
        if model.parameter_count > MAX_PARAMETERS:
            raise ValueError("A rede excede o limite de 10,1 milhões de parâmetros")
        model.text_tokenizer = data.tokenizer
        replay = None
        if params.get("text_corpus"):
            from .conversation_corpus import TextReplay
            digest = params["text_corpus"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Identificador de corpus inválido")
            replay = TextReplay(self.config.data / "datasets" / f"{digest}.json", data.tokenizer)
        if not 0 <= params.get("pretrain_steps", 0) < params["steps"]:
            raise ValueError("A etapa textual deve deixar passos para os exercícios de conversa")
        import shutil
        estimated_checkpoint = model.parameter_count * 12 + 2**20
        if shutil.disk_usage(self.config.data).free < estimated_checkpoint * 6 + 512 * 2**20:
            raise OSError("Espaço insuficiente para checkpoints e margem de segurança")
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        teacher = previous_data = None
        if params.get("retention_weight", 0):
            teacher, teacher_cp = self.load(params["retention_source"], device)
            if teacher_cp.get("tokenizer_state") != data.tokenizer.state():
                raise ValueError("Retenção exige o mesmo vocabulário local")
            teacher.requires_grad_(False)
            if params.get('curriculum') == 'fluency':
                previous_data = data
            else:
                previous_data = ConversationData.load(self.config.data / "datasets" / f"conversation_{teacher_cp['dataset_hash']}.json")
                reserved = {canonical(prompt_text(row['messages'])) for split, rows in data.sets.items() if split != 'train' for row in rows}
                previous_data = ConversationData({**previous_data.sets, 'train': [row for row in previous_data.sets['train']
                    if canonical(prompt_text(row['messages'])) not in reserved]}, previous_data.tokenizer)
            del teacher_cp
        rng = torch.Generator().manual_seed(params["seed"])
        sampler = None
        if params.get('sampler_mode')=='deck_v1':
            from .research import DeckSampler
            sampler = DeckSampler(data,params.setdefault('deck_state',{'groups':{},'rows':{}}))
        step = 0
        if resume:
            optimizer.load_state_dict(cp["optimizer"])
            if cp["scaler"]:
                scaler.load_state_dict(cp["scaler"])
            rng.set_state(cp["sample_rng"])
            torch.set_rng_state(cp["torch_rng"])
            if device.type == "cuda" and cp.get("cuda_rng"):
                torch.cuda.set_rng_state_all(cp["cuda_rng"])
            step = cp["step"]
            if step >= params["steps"]:
                raise ValueError("Este treino já terminou; inicie uma nova etapa incremental")
        initial_hash = weight_hash(model)
        progress(step, f"Avaliando a versão inicial de {model.parameter_count:,} parâmetros; nenhum passo novo executado nesta etapa.")
        baseline = self.evaluate(model, data, device, generation=not (fresh or adapted) and cp.get("task") in (TASK, LEGACY_TASK), interruptible=True)
        baseline["evaluated_step"] = step
        natural_baseline = self.evaluate(model, data, device, split="validation_natural", generation=True, interruptible=True) if "validation_natural" in data.sets else None
        session_baseline = self.evaluate_sessions(model, data, device, interruptible=True) if natural_baseline else None
        fluency_baseline = self.evaluate_fluency(model, data, device, interruptible=True) if 'validation_fluency' in data.sets else None
        if fluency_baseline:
            params.setdefault('fluency_reference',{k:v for k,v in fluency_baseline.items() if k!='sessions'})
        # Uma retomada não deve adotar a versão já degradada como nova referência.
        def compact_reference(value):
            return {k:v for k,v in value.items() if k not in ('outputs','sessions')} if value else None
        if not resume:
            params['original_baseline']=compact_reference(baseline)
            params['original_natural_baseline']=compact_reference(natural_baseline)
        text_baseline = replay.evaluate(model, device) if replay else None
        reference = None
        active_reference = self.active_id()
        if (fresh or initial_checkpoint or resume) and active_reference:
            previous, _ = self.load(active_reference, device)
            reference = self.evaluate(previous, data, device, generation=True, interruptible=True)
            del previous
        run_id = uuid.uuid4().hex
        self.db.execute("INSERT INTO runs(id,kind,status,started,config,dataset_hash,baseline) VALUES(?,?,?,?,?,?,?)",
                        (run_id, "conversation", "treinando", now(), json.dumps(params), data.digest, baseline["loss"]))
        self.db.execute("INSERT INTO metrics(run_id,step,epoch,val_loss,accuracy,created) VALUES(?,?,?,?,?,?)", (run_id, step, 0, baseline["loss"], baseline["token_accuracy"], now()))
        started = time.monotonic()
        best_loss, best_id = float("inf"), None
        best_rank = None
        best_step = None
        interrupted, detail, precision_skips = False, "", 0
        validation = baseline
        objective_history=[]
        progress(step, f"Diálogo real • {model.parameter_count:,} parâmetros • loss inicial {baseline['loss']:.4f}")
        if params.get("pretrain_steps"):
            progress(step, f"Preparação linguística: até o passo {params['pretrain_steps']}; depois exercícios de diálogo/contexto/lógica com replay textual.")
        try:
            while step < params["steps"]:
                self.gate.wait_if_paused()
                if step % 10 == 0:
                    check_resources(self.config)
                if time.monotonic() - started >= params["seconds"]:
                    raise Cancelled("Limite de duração; retome o checkpoint de diálogo")
                fraction = (step + 1) / params["steps"]
                lr_scale = min(1., (step + 1) / params["warmup"]) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
                optimizer.param_groups[0]["lr"] = params["learning_rate"] * lr_scale
                model.train()
                optimizer.zero_grad(set_to_none=True)
                objectives=[]
                textual = replay is not None and (step < params.get("pretrain_steps", 0) or step % params.get("replay_interval", 10) == 0)
                for micro in range(params.get("accumulation", 1)):
                    self.gate.check()
                    kind = None
                    if params.get('curriculum') == 'fluency':
                        from ..fluency_lessons import batch_kind
                        if params.get('imported_mixture'):
                            from ..dialogue_imports import imported_kind
                            kind = imported_kind(step,micro,params['accumulation'])
                        else:
                            kind = batch_kind(step, micro, params['accumulation'])
                        params['mixture_counts'][kind] += 1
                        textual = kind == 'language'
                    retaining = (kind == 'legacy' and teacher is not None) if kind else (not textual and previous_data is not None and micro == 0
                                 and (params.get("accumulation", 1) > 1 or step % 2 == 0))
                    if textual:
                        x, y = replay.batch(params["batch_size"], min(256, model.cfg.context), rng, device)
                    else:
                        rows = (sampler.sample(kind,params['batch_size'],rng) if sampler else data.sample_pool(params['batch_size'],rng,kind)) if kind else (previous_data if retaining else data).sample(params["batch_size"], rng)
                        x, y = data.batch(rows, model.cfg.context, device)
                    jitter = min(params.get("position_jitter", 0), model.cfg.context - x.shape[1])
                    offset = int(torch.randint(jitter + 1, (), generator=rng)) if jitter else 0
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits, loss = model(x, y, position_offset=offset)
                        cross_entropy=loss
                        retention=torch.zeros((),device=device)
                        if retaining:
                            with torch.no_grad():
                                reference_logits, _ = teacher(x, position_offset=offset)
                            mask = y != 0
                            retention = torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),
                                reference_logits[mask].float().softmax(-1), reduction="batchmean")
                            loss = loss + params["retention_weight"] * retention
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Perda de diálogo inválida")
                    objectives.append({'pool':kind or ('replay' if retaining else 'dialogue'),
                        'cross_entropy':float(cross_entropy.detach()),'kl':float(retention.detach())})
                    scaler.scale(loss * (.5 if stable else 1.) / params.get("accumulation", 1)).backward()
                if stable:
                    from .research import anchor_loss
                    self.gate.check()
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        anchor,anchor_ce,anchor_kl=anchor_loss(model,teacher,data,rng,device,params['batch_size'],params['retention_weight'])
                    if not torch.isfinite(anchor):
                        raise FloatingPointError('Perda da âncora inválida')
                    scaler.scale(anchor*.5).backward()
                    params['mixture_counts'].setdefault('anchor',0)
                    params['mixture_counts']['anchor']+=1
                    objectives.append({'pool':'anchor','cross_entropy':float(anchor_ce.detach()),'kl':float(anchor_kl.detach())})
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
                if not torch.isfinite(norm):
                    precision_skips += 1
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() < 1:
                        raise FloatingPointError("Gradientes inválidos persistentes")
                    continue
                with self.gate.writing():
                    self.gate.check()
                    scaler.step(optimizer)
                    scaler.update()
                    step += 1
                if step==1 or step%100==0 or step==params['steps']:
                    objective_history.append({'step':step,'learning_rate':optimizer.param_groups[0]['lr'],'objectives':objectives})
                if step % params.get("evaluation_interval", 1000) == 0 or step == params["steps"]:
                    validation = self.evaluate(model, data, device, interruptible=True)
                    validation["evaluated_step"] = step
                    if "validation_context" in data.sets:
                        validation["context_loss"] = self.evaluate(model, data, device, split="validation_context", interruptible=True)["loss"]
                    if "validation_natural" in data.sets:
                        validation["natural_loss"] = self.evaluate(model, data, device, split="validation_natural", interruptible=True)["loss"]
                    if 'validation_fluency' in data.sets:
                        validation['fluency_loss'] = self.evaluate(model,data,device,split='validation_fluency',interruptible=True)['loss']
                    self.db.execute("INSERT INTO metrics(run_id,step,epoch,train_loss,val_loss,accuracy,created) VALUES(?,?,?,?,?,?,?)", (run_id, step, math.ceil(step / 1000), float(loss.detach()), validation["loss"], validation["token_accuracy"], now()))
                    selection_loss = validation["loss"] + validation.get("context_loss", 0) + .25 * validation.get("natural_loss", 0)
                    selection_loss += validation.get('fluency_loss',0)
                    rank=(selection_loss,)
                    if params.get('selection_policy')=='protected-imported-v1' and data.sets.get('validation_imported'):
                        from .research import protected_selection_key
                        imported_loss=self.evaluate(model,data,device,split='validation_imported',interruptible=True)['loss']
                        validation['imported_selection_loss']=imported_loss
                        rank=protected_selection_key(validation,params.get('original_baseline') or baseline,selection_loss,imported_loss)
                    if best_rank is None or rank < best_rank:
                        best_id = self.save_checkpoint(model, optimizer, scaler, rng, data, run_id, step, params, validation, "epoca")
                        best_loss, best_step = selection_loss, step
                        best_rank = rank
                    extra = f" • novos diálogos {validation['natural_loss']:.4f}" if 'natural_loss' in validation else ""
                    if 'fluency_loss' in validation:
                        extra += f" • conversas revisadas {validation['fluency_loss']:.4f}"
                    progress(step, f"Diálogo {step}/{params['steps']} • loss resposta {validation['loss']:.4f} • tokens corretos {validation['token_accuracy']:.1%}" + extra)
                elif step % 100 == 0:
                    progress(step, f"Diálogo {step}/{params['steps']} • treino {float(loss.detach()):.4f}")
        except (Cancelled, LearningDisabled) as exc:
            interrupted, detail = True, str(exc)
        except Exception as exc:
            self.save_checkpoint(model, optimizer, scaler, rng, data, run_id, step, params, validation, "erro")
            self.db.execute("UPDATE runs SET status='erro',finished=?,detail=? WHERE id=?", (now(), str(exc), run_id))
            raise
        if not interrupted and best_id and best_step == step:
            last_id = best_id
            self.db.execute("UPDATE checkpoints SET status='candidato' WHERE id=?", (last_id,))
        else:
            last_id = self.save_checkpoint(model, optimizer, scaler, rng, data, run_id, step, params, validation, "interrompido" if interrupted else "candidato")
        selected_id, selected_step = last_id, step
        if not interrupted and best_id:
            selected = load_checkpoint(self.storage.checkpoint_path(best_id))
            model.load_state_dict(selected["model"])
            selected_id, selected_step = best_id, selected["step"]
        # Nunca promove uma rede só porque sua perda caiu: gera respostas completas.
        try:
            metrics = self.evaluate(model, data, device, generation=True, interruptible=True) if not interrupted else validation
            if not interrupted:
                metrics["evaluated_step"] = selected_step
        except (Cancelled, LearningDisabled) as exc:
            interrupted, detail, metrics = True, str(exc), validation
        unknown = metrics.get("by_skill", {}).get("unknown", {})
        unknown_ok = not unknown or unknown["correct"] / max(1, unknown["total"]) >= .8
        context_metrics = None
        if not interrupted and "validation_context" in data.sets:
            try:
                context_metrics = self.evaluate(model, data, device, split="validation_context", generation=True, interruptible=True)
                context_metrics["evaluated_step"] = selected_step
            except (Cancelled, LearningDisabled) as exc:
                interrupted, detail = True, str(exc)
        context_ok = (context_ready(context_metrics) if params.get("promotion_policy") in ("context-macro-v1", "context-natural-v2")
                      else context_metrics is None or context_metrics["exact_accuracy"] >= .65)
        natural_metrics = None
        session_metrics = None
        if not interrupted and natural_baseline is not None:
            try:
                natural_metrics = self.evaluate(model, data, device, split="validation_natural", generation=True, interruptible=True)
                natural_metrics["evaluated_step"] = selected_step
                session_metrics = self.evaluate_sessions(model, data, device, interruptible=True)
            except (Cancelled, LearningDisabled) as exc:
                interrupted, detail = True, str(exc)
        stage_baseline=baseline
        baseline=params.get('original_baseline') or (reference if resume and reference else baseline)
        natural_baseline=params.get('original_natural_baseline') or natural_baseline
        checks = {"treino_concluido": not interrupted, "loss_basica": metrics["loss"] < baseline["loss"] * 1.02,
                  "respostas_basicas": metrics.get("exact_accuracy", 0) >= max(.60, baseline.get("exact_accuracy", 0) - .02, (reference or {}).get("exact_accuracy", 0) - .02),
                  "desconhecimento": unknown_ok, "contexto_anterior": context_ok,
                  "dialogos_e_correcoes": natural_ready(natural_metrics, natural_baseline),
                  "sessoes_com_respostas_proprias": session_metrics is None or (session_metrics['immediate_accuracy'] >= .8 and session_metrics['recall_accuracy'] >= .8)}
        fluency_metrics = None
        if not interrupted and fluency_baseline is not None:
            from ..fluency_lessons import fluency_ready
            try:
                progress(step,'Avaliando conversas revisadas com respostas geradas pela própria TAC…')
                fluency_metrics = self.evaluate_fluency(model,data,device,interruptible=True)
                checks['conversas_revisadas'] = fluency_ready(fluency_metrics,params.get('fluency_reference',fluency_baseline))
            except (Cancelled,LearningDisabled) as exc:
                interrupted, detail = True, str(exc)
                checks['treino_concluido'] = False
        if interrupted:
            checks['treino_concluido'] = False
        accepted = all(checks.values())
        if not accepted and not interrupted:
            detail = "Critérios não atingidos: " + ", ".join(key for key, ok in checks.items() if not ok)
        imported_evaluation = None
        if 'validation_imported' in data.sets:
            imported_evaluation = {'reference_checkpoint':params.get('imported_reference'),
                                   'informational_only':True,'status':'pendente','splits':{}}
            if not interrupted:
                previous = None
                try:
                    reference_id = params.get('imported_reference')
                    if not reference_id:
                        raise ValueError('Não havia versão ativa de conversa para comparar no início desta etapa.')
                    if teacher is not None and reference_id == params.get('retention_source'):
                        previous = teacher
                    else:
                        previous, _ = self.load(reference_id,device)
                    for split in ('validation','test'):
                        progress(step,f'Verificando suas conversas de {"validação" if split=="validation" else "teste"}; comparação apenas informativa…')
                        imported_evaluation['splits'][split] = {
                            'active':self.evaluate_imported(previous,data,device,split+'_imported'),
                            'candidate':self.evaluate_imported(model,data,device,split+'_imported')}
                    imported_evaluation['status'] = 'concluido'
                except (Cancelled,LearningDisabled) as exc:
                    imported_evaluation['status'] = 'interrompido'
                    accepted, interrupted, detail = False, True, str(exc)
                    checks['treino_concluido'] = False
                except Exception as exc:
                    # Falha informativa não troca a decisão dos critérios existentes.
                    imported_evaluation.update(status='erro',error=str(exc))
                finally:
                    del previous
        if accepted:
            try:
                with self.gate.writing():
                    self.gate.check()
                    self.db.execute("INSERT INTO meta(key,value) VALUES('active_conversation',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (selected_id,))
                    self.db.execute("UPDATE checkpoints SET status='aprovado' WHERE id=?", (selected_id,))
            except (Cancelled, LearningDisabled):
                accepted, interrupted = False, True
        status = "interrompido" if interrupted else ("aprovado" if accepted else "rejeitado")
        text_final = replay.evaluate(model, device) if replay and not interrupted else None
        result = {"run_id": run_id, "checkpoint": selected_id, "last_checkpoint": last_id, "status": status, "steps": step,
                  "selected_step": selected_step, "baseline": baseline, "validation": metrics, "parameters": model.parameter_count,
                  "initial_hash": initial_hash, "final_hash": weight_hash(model), "device": str(device), "seconds": time.monotonic() - started,
                  "precision_skips": precision_skips, "dataset_hash": data.digest, "detail": detail,
                  "dataset_examples": {key: len(value) for key, value in data.sets.items()}, "context_validation": context_metrics,
                  "reference_validation": reference,
                  "natural_baseline": natural_baseline, "natural_validation": natural_metrics, "promotion_checks": checks,
                  "session_baseline": session_baseline, "session_validation": session_metrics,
                  'fluency_baseline':fluency_baseline, 'fluency_validation':fluency_metrics,
                  'mixture_counts':params.get('mixture_counts'), 'curriculum':params.get('curriculum'),
                  'imported_counts':counts,'imported_evaluation':imported_evaluation,
                  'objective_history':objective_history,'stability_policy':params.get('stability_policy','off'),
                  'sampler_mode':params.get('sampler_mode','replacement'),
                  'selection_policy':params.get('selection_policy','legacy-internal'),
                  'stage_baseline':stage_baseline,'promotion_reference':'original_execution' if params.get('original_baseline') else 'legacy_or_active_reference',
                  "resumed_from": resume, "initialized_from": initial_checkpoint,
                  "text_baseline": text_baseline, "text_final": text_final,
                  "text_replay": {"corpus": replay.digest, "samples": replay.samples, "tokens": replay.tokens,
                                  "sources": replay.sources} if replay else None}
        self.db.execute("UPDATE runs SET status=?,finished=?,final=?,detail=? WHERE id=?", (status, now(), metrics["loss"], json.dumps(result, ensure_ascii=False), run_id))
        atomic_json(self.config.data / "last_conversation_training.json", result)
        del model, optimizer, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def reply(self, messages):
        check_resources(self.config)
        active = self.active_id()
        if not active:
            raise ValueError("Ainda não há diálogo neural aprovado")
        torch.set_num_threads(self.config.get("cpu_threads"))
        if active != self._cached_id:
            record = load_checkpoint(self.storage.checkpoint_path(active))
            is_v2 = record.get('generation') in ('TAC-v2', 'TAC-v3')
            device = self.storage.device() if is_v2 else 'cpu'
            self._cached_model, _ = self.load(active, device)
            self._cached_generation = record.get('generation')
            self._cached_id = active
        options=None
        if self.config.get('conversation_generation')=='controlled':
            options={'greedy':False,'temperature':.65,'top_k':30,'top_p':.9,'repetition_penalty':1.08,'seed':self.config.get('seed')}
        device = next(self._cached_model.parameters()).device
        generation_info = None
        if getattr(self, '_cached_generation', None) in ('TAC-v2', 'TAC-v3'):
            from .decoding import conversation_options, decode_conversation
            generation_info = decode_conversation(self._cached_model, messages, device, max_tokens=128,
                generation_options=conversation_options(self.config.get('conversation_generation'), self.config.get('seed')))
            text = generation_info.pop('text')
        else:
            text = decode_reply(self._cached_model, messages, device, max_tokens=128, generation_options=options)
        return {"mode": "diálogo neural experimental", "text": text or "[A rede encerrou a resposta sem gerar texto.]", "sources": [],
                "method": "texto gerado token a token pelo Transformer local; não selecionado pelo classificador",
                "checkpoint": active, "context_tokens": self._cached_model.cfg.context,
                "tokenizer": self._cached_model.text_tokenizer.version,
                "generation": getattr(self,'_cached_generation',None), 'decoding': generation_info}
