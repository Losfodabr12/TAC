from __future__ import annotations

import json
import re
import unicodedata

from .db import now
from .documents import CHUNKING_VERSION, split_sections, text_hash

STOP = set("a o as os um uma uns umas de do da dos das e em no na nos nas por para com que qual quais como onde sobre tac você voce esse essa este esta documento documentos arquivo arquivos aprendeu explique conteúdo conteudo diferença diferenca entre informação informacao informações informacoes encontrou falam favor me seu sua há ha foi é sao são ao se eu".split())


def words(text):
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return [w for w in re.findall(r"[^\W_]+", plain, re.UNICODE) if len(w) > 1 and w not in STOP]


class Memory:
    def __init__(self, db, gate):
        self.db, self.gate = db, gate

    def ingest(self, title, origin, kind, text, subject="", metadata=None):
        digest = text_hash(text)
        metadata = {**(metadata or {}), "chunking_version": CHUNKING_VERSION}
        with self.gate.writing(), self.db.lock, self.db.connect() as c:
            old = c.execute("SELECT id,content_hash,metadata FROM sources WHERE origin=?", (origin,)).fetchone()
            if old:
                previous_meta = json.loads(old['metadata'])
                if previous_meta.get('research_web'):
                    metadata = {**previous_meta, **metadata, 'research_web': True, 'training_approved': False}
                    if old['content_hash'] != digest:
                        metadata.update(verification='atualizada_sem_revisao',
                                        verification_note='Conteúdo atualizado; a comparação anterior não se aplica.', checked_at=now())
            duplicate = c.execute("SELECT id FROM sources WHERE content_hash=?", (digest,)).fetchone()
            if old and old["content_hash"] == digest:
                return old["id"], "inalterado"
            if duplicate and (not old or old["id"] != duplicate["id"]):
                return duplicate["id"], "duplicado"
            if old:
                source_id = old["id"]
                c.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
                c.execute("UPDATE sources SET title=?,kind=?,content_hash=?,subject=?,updated=?,metadata=? WHERE id=?",
                          (title, kind, digest, subject, now(), json.dumps(metadata or {}, ensure_ascii=False), source_id))
                status = "atualizado"
            else:
                source_id = c.execute("INSERT INTO sources(title,origin,kind,content_hash,subject,created,updated,metadata) VALUES(?,?,?,?,?,?,?,?)",
                                      (title, origin, kind, digest, subject, now(), now(), json.dumps(metadata or {}, ensure_ascii=False))).lastrowid
                status = "estudado"
            c.executemany("INSERT INTO chunks(source_id,ordinal,section,text) VALUES(?,?,?,?)",
                          ((source_id, i, section, chunk) for i, (section, chunk) in enumerate(split_sections(text))))
            return source_id, status

    def reindex_source(self, source_id):
        """Reorganiza índices derivados sem alterar a origem ou os pesos."""
        with self.gate.writing(), self.db.lock, self.db.connect() as c:
            self.gate.check()
            source = c.execute("SELECT metadata FROM sources WHERE id=?", (source_id,)).fetchone()
            if source is None:
                raise ValueError("Fonte não encontrada.")
            metadata = json.loads(source["metadata"])
            rows = c.execute("SELECT text FROM chunks WHERE source_id=? ORDER BY ordinal", (source_id,)).fetchall()
            if metadata.get("chunking_version") == CHUNKING_VERSION:
                return {"source_id": source_id, "before": len(rows), "after": len(rows), "changed": False}
            text = "\n".join(row["text"] for row in rows)
            parts = list(split_sections(text))
            if "\n".join(chunk for _, chunk in parts) != text:
                raise ValueError("A reorganização alteraria o texto; índice anterior preservado.")
            c.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            c.executemany("INSERT INTO chunks(source_id,ordinal,section,text) VALUES(?,?,?,?)",
                          ((source_id, i, section, chunk) for i, (section, chunk) in enumerate(parts)))
            metadata["chunking_version"] = CHUNKING_VERSION
            c.execute("UPDATE sources SET metadata=? WHERE id=?", (json.dumps(metadata, ensure_ascii=False), source_id))
            self.gate.check()
            return {"source_id": source_id, "before": len(rows), "after": len(parts), "changed": True}

    def sources(self, subject=""):
        return self.db.query("SELECT s.*, (SELECT count(*) FROM chunks WHERE source_id=s.id) AS chunks FROM sources s WHERE subject LIKE ? ORDER BY updated DESC", (f"%{subject}%",))

    def chunks(self, source_id):
        return self.db.query("SELECT c.*,s.title,s.origin,s.subject,SUM(length(c.text)+1) OVER(ORDER BY c.ordinal ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)-length(c.text)-1 AS char_start FROM chunks c JOIN sources s ON s.id=c.source_id WHERE source_id=? ORDER BY ordinal", (source_id,))

    def search(self, query, limit=6, source_id=None):
        tokens = list(dict.fromkeys(words(query)))[:24]
        if not tokens:
            return self.chunks(source_id)[:limit] if source_id else []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        where = " AND c.source_id=?" if source_id else ""
        args = (match, source_id, limit) if source_id else (match, limit)
        return self.db.query("SELECT c.*,s.title,s.origin,s.subject,bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN sources s ON s.id=c.source_id WHERE chunks_fts MATCH ?" + where + " ORDER BY rank LIMIT ?", args)

    def answer(self, question, source_id=None):
        docs = self.sources()
        if not docs:
            return {"mode": "documental", "text": "A biblioteca ainda está vazia: nenhum documento ou site foi estudado. Abra a Central de Aprendizado, adicione um material e clique no botão verde Ativar e estudar fila. Depois você poderá consultar o conteúdo com suas fontes.", "sources": []}
        matched = [d for d in docs if re.search(r"(?<!\w)" + re.escape(d["title"]) + r"(?!\w)", question, re.I)]
        compare = any(w in question.lower() for w in ("diferen", "compare", "contrad", "conflit"))
        if source_id is None and len(matched) == 1:
            source_id = matched[0]["id"]
        overview = any(w in question.lower() for w in ("aprendeu", "explique", "resum", "conteúdo", "conteudo"))
        if overview and source_id is None and len(docs) == 1:
            source_id = docs[0]["id"]
        retrieval = bool(re.search(r"\b(pesquis|encontr|trechos|fontes|documentos|arquivos|busque)", question, re.I))
        if not compare and not overview and not retrieval:
            from .answers import answer_question
            return answer_question(self, question, source_id)
        if compare and len(matched) >= 2:
            hits = []
            for d in matched[:3]:
                hits.extend(self.chunks(d["id"])[:2])
            prefix = "Comparação documental: trechos das fontes indicadas. Diferenças ou contradições exigem leitura; a TAC não verifica a verdade das afirmações."
        elif overview and source_id:
            chunks = self.chunks(source_id)
            informative = [c for c in chunks if len(c["text"]) >= 160 and re.search(r"[.!?]", c["text"])]
            hits = (informative or chunks)[:5]
            prefix = "Visão extrativa do documento (seleção de trechos do texto; não é um resumo neural):"
        else:
            hits = self.search(question, source_id=source_id)
            prefix = "Trechos recuperados por pesquisa textual. Verifique o contexto nas fontes:"
            if compare and not hits and not source_id:
                hits = [chunk for document in docs[:10] for chunk in self.chunks(document["id"])[:2]]
                prefix = "Amostra documental para inspeção de diferenças/conflitos (até 10 fontes, dois trechos por fonte). A detecção é lexical e incompleta:"
        if not hits:
            return {"mode": "documental", "text": "Não encontrei trechos que fundamentem uma resposta. Adicione materiais, informe o assunto ou selecione um documento na Biblioteca.", "sources": []}
        citations, blocks = [], [prefix]
        for n, hit in enumerate(hits, 1):
            blocks.append(f"\n[{n}] {hit['title']} — {hit['section']}\n{hit['text']}\nOrigem: {hit['origin']} (fonte #{hit['source_id']}, trecho #{hit['id']})")
            citations.append({k: hit[k] for k in ("source_id", "id", "title", "origin", "section")})
        if "contrad" in question.lower():
            candidates = self.contradictions(hits)
            blocks.append("\nPossíveis conflitos lexicais (heurística, sem verificação factual):\n" + ("\n".join(candidates) if candidates else "Não detectei pares com negação explícita nos trechos recuperados. Isso não comprova ausência de contradições."))
        visible = "Encontrei estes trechos no material:\n\n" + "\n\n".join(f"{h['title']}\n{h['text']}" for h in hits)
        if "contrad" in question.lower():
            visible += "\n\n" + blocks[-1]
        return {"mode": "documental", "text": "\n".join(blocks), "display_text": visible, "sources": citations}

    @staticmethod
    def contradictions(hits):
        claims = {}
        found = []
        for h in hits:
            for sentence in re.split(r"[.!?\n]", h["text"]):
                key = " ".join(words(re.sub(r"\bnão\b", "", sentence, flags=re.I)))
                if len(key) < 8:
                    continue
                neg = bool(re.search(r"\b(não|nunca)\b", sentence, re.I))
                if key in claims and claims[key][0] != neg:
                    found.append(f"“{claims[key][1]}” / “{sentence.strip()}” (fontes {claims[key][2]} e {h['source_id']})")
                claims[key] = (neg, sentence.strip(), h["source_id"])
        return found[:10]

    def correct(self, original, correction, subject, source, validated=False):
        if not correction.strip() or not source.strip():
            raise ValueError("Informe a correção e sua fonte.")
        with self.gate.writing():
            return self.db.execute("INSERT INTO corrections(original,correction,subject,source,validated,created) VALUES(?,?,?,?,?,?)", (original, correction, subject, source, int(validated), now()))

    def set_subject(self, source_id, subject):
        with self.gate.writing():
            self.db.execute("UPDATE sources SET subject=? WHERE id=?", (subject.strip(), source_id))

    def delete_source(self, source_id):
        # Exclusão é uma ação explícita de administração, mesmo com aprendizado desligado.
        self.db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        self.db.event("exclusao_fonte", str(source_id))
