from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, ROOT
from .db import Database, now
from .diagnostics import check_resources, resources, setup_logging
from .documents import SUPPORTED, extract
from .learning import Cancelled, LearningDisabled, LearningGate
from .memory import Memory


class Core:
    def __init__(self, root=ROOT, data_dir=None):
        self.config = Config(Path(root))
        if data_dir:
            self.config.update(data_dir=data_dir)
            for f in ("logs", "checkpoints", "inbox", "datasets", "backups"):
                (self.config.data / f).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.config.data)
        self.db = Database(self.config.data / "tac.sqlite3")
        self.gate = LearningGate(self.config, self.db)
        self.memory = Memory(self.db, self.gate)
        self.operation_lock = threading.Lock()
        self.operation = "Ociosa"
        self.last_answer = None
        self._trainer = None
        self._dialogue = None
        self._conversation = None
        self.neural_history = []
        self._neural_history_checkpoint = None
        self.session_name = None
        self.correction_target = None

    @property
    def trainer(self):
        if self._trainer is None:
            from .neural.training import Trainer
            self._trainer = Trainer(self.config, self.db, self.gate)
        return self._trainer

    def recover_interrupted(self):
        self.db.execute("UPDATE queue SET status='pendente',progress='Retomada após reinício' WHERE status='processando'")
        # A separately launched recoverable GPU session can still be alive when
        # the interface opens. Do not mislabel its run as interrupted.
        from PySide6.QtCore import QLockFile
        for path in (self.config.data/'training.lock', self.config.data/'tac_v2_conversation_20261018/training.lock'):
            if not path.exists():continue
            probe=QLockFile(str(path));probe.setStaleLockTime(0)
            if not probe.tryLock(0):return
            probe.unlock()
        self.db.execute("UPDATE runs SET status='interrompido',finished=?,detail='Processo anterior terminou sem finalização; retome o último checkpoint disponível.' WHERE status='treinando'", (now(),))

    @property
    def dialogue(self):
        if self._dialogue is None:
            from .neural.dialogue import DialogueEngine
            self._dialogue = DialogueEngine(self.config, self.db, self.gate)
        return self._dialogue

    @property
    def conversation(self):
        if self._conversation is None:
            from .neural.conversation import ConversationEngine
            self._conversation = ConversationEngine(self.config, self.db, self.gate)
        return self._conversation

    def active_conversation_id(self):
        rows = self.db.query("SELECT value FROM meta WHERE key='active_conversation'")
        return rows[0]["value"] if rows else None

    def perform(self, name, callback, learning=False):
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError("Aguarde a operação atual terminar ou solicite seu cancelamento.")
        try:
            self.operation = name
            if learning:
                self.gate.begin()
            self.db.event("inicio", name)
            result = callback()
            self.db.event("fim", name)
            return result
        except Exception as exc:
            self.logger.exception("Falha em %s", name)
            self.db.event("erro", f"{name}: {type(exc).__name__}: {exc}")
            raise
        finally:
            self.operation = "Ociosa"
            self.operation_lock.release()

    def enqueue_file(self, path, subject=""):
        path = Path(path).resolve()
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            raise ValueError("Escolha um documento PDF, TXT, DOCX, Markdown ou HTML.")
        if path.stat().st_size > self.config.get("max_file_mb") * 2**20:
            raise ValueError("Documento maior que o limite configurado.")
        # Cópia gerenciada; nunca altera nem apaga o arquivo original.
        name = uuid.uuid4().hex + path.suffix.lower()
        target = self.config.data / "inbox" / name
        shutil.copyfile(path, target)
        return self.db.execute("INSERT INTO queue(kind,target,options,created) VALUES('file',?,?,?)", (name, json.dumps({"origin": str(path), "title": path.stem, "subject": subject}, ensure_ascii=False), now()))

    def enqueue_url(self, url, documentation=False, subject="", terms=False):
        # Apenas valida sintaxe aqui: enfileirar não inicia acesso à rede.
        from urllib.parse import urlsplit
        parsed = urlsplit(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Informe uma URL HTTP/HTTPS pública sem credenciais.")
        if not terms:
            raise ValueError("Confirme que você tem autorização de acesso e verificou os termos do site.")
        url = parsed._replace(fragment="").geturl()
        with self.db.lock, self.db.connect() as c:
            existing = c.execute("SELECT id FROM queue WHERE kind='web' AND target=? AND status IN ('pendente','processando') ORDER BY id LIMIT 1", (url,)).fetchone()
            if existing:
                return existing["id"]
            return c.execute("INSERT INTO queue(kind,target,options,created) VALUES('web',?,?,?)", (url, json.dumps({"documentation": documentation, "subject": subject, "terms_confirmed": True}), now())).lastrowid

    def run_queue(self, progress=lambda *_: None, queue_ids=None):
        self.gate.check()
        if queue_ids is None:
            rows = self.db.query("SELECT * FROM queue WHERE status='pendente' ORDER BY id")
        else:
            ids = tuple(dict.fromkeys(int(i) for i in queue_ids))
            rows = self.db.query("SELECT * FROM queue WHERE status='pendente' AND id IN (" + ",".join("?" for _ in ids) + ") ORDER BY id", ids) if ids else []
        done, errors, interrupted, source_ids, processed_pages = 0, 0, False, [], 0
        for index, row in enumerate(rows):
            try:
                self.gate.check()
                check_resources(self.config)
            except (LearningDisabled, Cancelled):
                interrupted = True
                break
            self.db.execute("UPDATE queue SET status='processando',progress='' WHERE id=?", (row["id"],))
            options = json.loads(row["options"])
            try:
                if row["kind"] == "file":
                    target = (self.config.data / "inbox" / row["target"]).resolve()
                    if not target.is_relative_to((self.config.data / "inbox").resolve()):
                        raise ValueError("Caminho de fila inválido.")
                    def stage(message):
                        self.db.execute("UPDATE queue SET progress=? WHERE id=?", (message, row["id"]))
                        progress(index, message)
                    stage(f"Lendo documento: {options.get('title', target.name)}")
                    title, text, meta = extract(target, self.config, self.gate.check)
                    if target.suffix.lower() not in (".html", ".htm"):
                        title = options.get("title", title)
                    stage(f"Indexando e salvando: {title} ({len(text):,} caracteres)")
                    source_id, status = self.memory.ingest(title, options["origin"], "file", text, options.get("subject", ""), meta)
                    source_ids.append(source_id)
                    self.study_history(row["id"], options["origin"], status, f"Fonte #{source_id}")
                    detail = f"{title} • {len(text):,} caracteres • fonte #{source_id} • {status}"
                    row_status = "concluido"
                else:
                    from .web import PublicFetcher
                    pages, page_errors = 0, []
                    def web_progress(n, detail):
                        self.db.execute("UPDATE queue SET progress=? WHERE id=?", (detail, row["id"]))
                        progress(index, detail)
                        if detail.startswith("Erro"):
                            page_errors.append(detail)
                            self.study_history(row["id"], row["target"], "erro_pagina", detail)
                    fetcher = PublicFetcher(self.gate, self.config, web_progress)
                    for page in fetcher.crawl(row["target"], options.get("documentation", False), web_progress):
                        web_progress(pages, f"Indexando e salvando: {page['title']} ({len(page['text']):,} caracteres)")
                        source_id, status = self.memory.ingest(page["title"], page["origin"], "web", page["text"], options.get("subject", ""), {"root": row["target"], "depth": page["depth"]})
                        source_ids.append(source_id)
                        self.study_history(row["id"], page["origin"], status, f"Fonte #{source_id}")
                        pages += 1
                        processed_pages += 1
                    if pages == 0:
                        raise ValueError("Nenhuma página foi estudada. " + (page_errors[0] if page_errors else "Confira o endereço e as permissões do site."))
                    detail = f"{pages} página(s) disponível(is) na Biblioteca"
                    row_status = "parcial" if page_errors else "concluido"
                    if page_errors:
                        detail += f"; {len(page_errors)} página(s) com erro. Veja o histórico."
                self.db.execute("UPDATE queue SET status=?,progress=?,finished=? WHERE id=?", (row_status, detail, now(), row["id"]))
                done += 1
                progress(index + 1, f"Material {index + 1}/{len(rows)} concluído: {detail}")
            except (LearningDisabled, Cancelled):
                self.db.execute("UPDATE queue SET status='pendente',progress='Interrompido com segurança' WHERE id=?", (row["id"],))
                interrupted = True
                break
            except Exception as exc:
                self.logger.exception("Erro no material %s", row["id"])
                self.db.execute("UPDATE queue SET status='erro',progress=?,finished=? WHERE id=?", (str(exc), now(), row["id"]))
                self.study_history(row["id"], options.get("origin", row["target"]), "erro", str(exc))
                errors += 1
                progress(index, f"Erro recuperável: {exc}")
        return {"processed": done, "total": len(rows), "errors": errors, "interrupted": interrupted,
                "source_ids": list(dict.fromkeys(source_ids)), "pages": processed_pages}

    def study_history(self, queue_id, origin, status, detail):
        self.db.execute("INSERT INTO study_history(queue_id,origin,status,detail,created) VALUES(?,?,?,?,?)", (queue_id, origin, status, detail, now()))

    def chat(self, question, source_id=None, neural=False, conversation=False, progress=lambda *_: None,
             research=False, research_topic=None):
        if not question.strip():
            raise ValueError("Digite uma pergunta.")
        if len(question) > 5000:
            raise ValueError("Limite de 5.000 caracteres por pergunta.")
        online = None
        from .research import Researcher, public_topic, qualify_saved_answer
        if research:
            self.gate.check()
            online = Researcher(self).lookup(question, progress, topic=research_topic)
        elif (self.config.get('research_auto') and self.gate.enabled and not neural and not conversation
              and source_id is None and public_topic(question)):
            progress(0, 'Procurando primeiro na biblioteca local…')
            local = self.memory.answer(question)
            if local.get('sources'):
                online = local
            else:
                online = Researcher(self).lookup(question, progress)
        if online is not None:
            answer = online
            # Preserva as falas anteriores, sem inserir conteúdo externo como
            # instrução ou resposta neural no próximo prompt.
        elif conversation:
            active = self.active_conversation_id()
            changed_model = (self._neural_history_checkpoint is not None
                             and self._neural_history_checkpoint != active)
            if changed_model:
                self.neural_history.clear()
            messages = [*self.neural_history[-6:], ["user", question]]
            answer = self.conversation.reply(messages)
            self.neural_history = [*messages, ["assistant", answer["text"]]][-6:]
            self._neural_history_checkpoint = active
            if changed_model:
                answer['session_reset_reason'] = 'checkpoint_changed'
        elif neural:
            answer = self.trainer.generate(question)
        else:
            guided = None
            courtesy = False
            if source_id is not None and len(question) <= 120:
                from .answers import fold
                terms = re.findall(r"\w+", fold(question))
                allowed = set("oi ola oie eae eai opa salve tac amiga amigo parceira parceiro bom dia boa tarde noite obrigado obrigada obg vlw valeu muito mesmo tchau ate mais logo falou flw por tudo".split())
                courtesy = bool(terms) and all(t in allowed or re.fullmatch(r"oi+|oie+|ola+|eae+|eai+", t) for t in terms)
            if (source_id is None or courtesy) and self.db.query("SELECT 1 FROM meta WHERE key='active_dialogue'"):
                guided = self.dialogue.reply(question, self.session_name)
                if source_id is not None and guided and guided.get("intent") not in ("greeting", "thanks", "goodbye"):
                    guided = None
            answer = guided or self.memory.answer(question, source_id)
            if guided and guided.get("intent") == "correction" and self.last_answer:
                previous = self.last_answer[1]
                self.correction_target = previous.get("display_text", previous["text"])
            elif answer.get("mode") != "conversa guiada":
                self.correction_target = None
            if answer.get("session_name"):
                self.session_name = answer["session_name"]
        answer = qualify_saved_answer(self, answer)
        self.last_answer = (question, answer)
        if self.config.get("save_interactions") and self.gate.enabled:
            try:
                with self.gate.writing():
                    self.db.execute("INSERT INTO interactions(question,answer,mode,sources,created) VALUES(?,?,?,?,?)", (question, answer["text"], answer["mode"], json.dumps(answer["sources"], ensure_ascii=False), now()))
            except LearningDisabled:
                pass
        return answer

    def delete_source(self, source_id):
        self.memory.delete_source(source_id)
        # Retira snapshots derivados e impossibilita retomá-los. Pesos históricos são preservados.
        for path in (self.config.data / "datasets").glob("*.json"):
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if any(p.get("source_id") == source_id for p in snapshot.get("provenance", [])):
                path.unlink()

    def auto_due(self):
        if not (self.gate.enabled and self.config.get("auto_train")):
            return False
        rows = self.db.query("SELECT started FROM runs WHERE kind='neural' ORDER BY started DESC LIMIT 1")
        if rows:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(rows[0]["started"])).total_seconds()
            return elapsed >= self.config.get("auto_interval_minutes") * 60
        return True

    def backup_due(self):
        # A reconstrução usa checkpoints próprios para recuperação e não
        # reempacota automaticamente versões antigas ou todo o acervo.
        # A ação manual de criar cópia continua disponível.
        if (self.config.data/'tac_v2/progress.json').exists():
            return False
        rows = self.db.query("SELECT created FROM events WHERE kind='backup_exportado' ORDER BY id DESC LIMIT 1")
        return not rows or (datetime.now(timezone.utc) - datetime.fromisoformat(rows[0]["created"])).total_seconds() > self.config.get("backup_hours") * 3600

    def stats(self):
        info = resources()
        info["sources"] = self.db.query("SELECT count(*) AS n FROM sources")[0]["n"]
        info["chunks"] = self.db.query("SELECT count(*) AS n FROM chunks")[0]["n"]
        info["disk_mb"] = sum(p.stat().st_size for p in self.config.data.rglob("*") if p.is_file()) / 2**20
        info["operation"] = self.operation
        ids = tuple(r["value"] for r in self.db.query("SELECT value FROM meta WHERE key IN ('active_checkpoint','active_dialogue','active_conversation') ORDER BY key"))
        if getattr(self, "_inventory_ids", None) != ids:
            from .neural.inventory import active_models
            self._inventory_models = active_models(self.config, self.db)
            self._inventory_ids = ids
        info["models"] = getattr(self, "_inventory_models", [])
        return info
