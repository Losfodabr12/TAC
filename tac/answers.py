"""Respostas documentais curtas: regras explícitas e seleção lexical de frases.

Não é geração neural nem interpretação semântica geral. Valores só são devolvidos
quando há uma relação explícita no texto; o trecho comprobatório é preservado.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def tokens(text):
    return set(re.findall(r"[^\W_]+", fold(text)))


def question_content(text):
    # Remove só um cumprimento inicial seguido de pergunta/pedido explícito.
    # "O que significa eae?" e nomes citados no restante continuam intactos.
    prefix = re.compile(
        r"^\s*(?:eae+|eai+|e\s+a[ií]+|oi+|oie+|ol[áa]+|opa|salve)\b"
        r"(?:[\W_]+tac\b)?[\W_]+"
        r"(?=(?:qual|quais|o\s+que|quem|onde|como|por\s+que|quanto|quantos|"
        r"explique|encontre|procure|pesquise|resuma|compare|mostre|me\s+(?:fale|diga)|voc[êe]\s+sabe)\b)",
        re.I,
    )
    cleaned = prefix.sub("", text, count=1)
    return re.sub(r'^(?:oq|o\s*q|oque)\s+(?:é|e)\s+', 'O que é ', cleaned, flags=re.I)


FILLER = tokens("a o as os um uma uns umas de do da dos das e em no na nos nas por para com que qual quais como quem onde sobre tac voce esse essa este esta documento documentos arquivo arquivos me seu sua seus suas favor diga dizer quero saber informe eh se completo completa atual verdadeiro exatamente quando fica pais cidade")
TITLE_FILLER = tokens("wikipedia enciclopedia livre wikisource fonte documento arquivo guia sobre")
PROPERTY_WORDS = {
    "name": {"nome", "oficial", "oficialmente", "chama", "denomina", "denominado", "denominada", "chamado", "chamada"},
    "capital": {"capital"},
}


def intent_of(question):
    plain = fold(question)
    if re.search(r"\bcapital\b", plain) and not re.search(r"\b(social|humano|humana|capitalismo)\b", plain):
        return "capital"
    if (re.search(r"\b(nome|se chama|chamad[oa]|denominad[oa])\b", plain)
            and not re.search(r"\b(origem|significa|etimologia|por que|porque|veio)\b", plain)):
        return "name"
    return None


def primary_title(source):
    return re.split(r"\s+[–—|]\s+|\s+-\s+", source["title"])[0].replace("_", " ")


@dataclass
class Evidence:
    hit: dict
    quote: str
    value: str | None = None
    score: float = 0.0


def citation(hit):
    return {k: hit[k] for k in ("source_id", "id", "title", "origin", "section")}


def render(items, direct=False, kind=None, conflict=False):
    if conflict:
        head = "As fontes apresentam valores diferentes. Não consigo escolher um como correto:"
    elif direct:
        head = ("Nome oficial informado pela fonte: " if kind == "name" else "Capital informada pela fonte: ") + items[0].value.rstrip(". ") + "."
    else:
        head = "Trecho mais específico que encontrei:" if len(items) == 1 else "Trechos mais específicos que encontrei:"
    blocks, refs = [head], []
    for index, item in enumerate(items, 1):
        quote = item.quote
        if len(quote) > 600:
            quote = quote[:597].rsplit(" ", 1)[0] + "…"
        blocks.append(f"\n“{quote}”\nFonte [{index}]: {item.hit['title']}\nOrigem: {item.hit['origin']} (fonte #{item.hit['source_id']}, trecho #{item.hit['id']})")
        refs.append(citation(item.hit))
    if direct and not conflict:
        visible = ("O nome oficial informado no material é " if kind == "name" else "A capital informada no material é ") + items[0].value.rstrip(". ") + "."
    else:
        visible = (head + "\n\n" if conflict else "") + "\n\n".join(item.quote[:600] for item in items)
    return {"mode": "documental", "text": "\n".join(blocks), "display_text": visible, "sources": refs,
            "method": "extração por regras" if direct else "seleção de frases",
            "evidence": [item.quote for item in items]}


def explicit_facts(hit, kind, topic, title_matches):
    # Esses padrões descrevem relações textuais; não contêm respostas/countries pré-definidos.
    patterns = {
        "name": [
            r"(?P<subject>[^\n.!?]{1,130}?)\s*,\s*oficialmente\s+(?:(?:denominad[oa]|chamad[oa])\s+)?(?P<value>[^,;.!?\n(]{2,120})",
            r"(?P<subject>[^\n.!?]{0,100}?)\bnome\s+oficial\s+(?:é|e|:|=)\s*(?P<value>[^,;.!?\n(]{2,120})",
            r"\bnome\s+oficial\s+(?:de|do|da)\s+(?P<subject>[^\n.!?]{1,70}?)\s+(?:é|e|:)\s+(?P<value>[^,;.!?\n(]{2,120})",
            r"(?P<subject>[^\n.!?]{1,100}?)\s+(?:é|e)\s+(?:oficialmente\s+)?(?:denominad[oa]|chamad[oa])\s+(?P<value>[^,;.!?\n(]{2,120})",
        ],
        "capital": [
            r"\bcapital\s+(?:de|do|da)\s+(?P<subject>[^\n.!?]{1,70}?)\s+(?:é|e|:)\s+(?P<value>[^,;.!?\n]{2,100})",
            r"(?P<value>[^\n.!?]{2,80}?)\s+(?:é|e)\s+a\s+capital\s+(?:de|do|da)\s+(?P<subject>[^,;.!?\n]{1,70})",
        ],
    }
    found = []
    # Perguntas de um roteiro de conversa não são declarações factuais.
    declarative = "\n".join(line for line in hit["text"].splitlines()
                            if not re.match(r"^\s*(?:usu[aá]rio|user|pergunta|q)\s*:", line, re.I)
                            and not re.match(r"^\s*(?:qual|quais|quem|onde|quanto|quantos)\b", line, re.I)
                            and not line.rstrip().endswith("?"))
    for pattern in patterns[kind]:
        for match in re.finditer(pattern, declarative, re.I):
            subject = tokens(match.group("subject"))
            # Um título não pode substituir um sujeito explícito sobre outra entidade.
            support = topic <= subject if topic else title_matches
            implicit = subject <= tokens("o a seu sua cujo cuja")
            if not support and not (title_matches and implicit):
                continue
            value = match.group("value").strip(" #\t")
            if tokens(value) & tokens("qual quais quem onde quanto pergunta usuario"):
                continue
            if value and len(value) <= 120:
                found.append(Evidence(hit, match.group(0).strip(), value))
    if kind == "capital" and title_matches and hit.get("char_start", 10**9) < 2000:
        offset = hit["char_start"]
        for line in hit["text"].splitlines():
            # Campo curto de uma tabela/infobox; não captura parágrafos sobre capitais.
            match = re.fullmatch(r"\s*Capital\s*[:–-]?\s+([\wÀ-ÿ][^\n.!?]{1,65})\s*", line, re.I)
            if offset < 2000 and match and not re.match(r"(?:de|do|da|é|e)\b", fold(match.group(1))):
                found.append(Evidence(hit, line.strip(), match.group(1).strip()))
            offset += len(line) + 1
    return found


def answer_question(memory, question, source_id=None):
    question = question_content(question)
    relation = re.search(r"\b(?:relacao|relacionamento|vinculo|ligacao)\s+entre\s+(.+?)\s+e\s+(.+?)[?.!]*$", fold(question))
    if relation:
        names = [tokens(name) - FILLER for name in relation.groups()]
        hits = memory.search(" ".join(" ".join(name) for name in names), limit=30, source_id=source_id)
        supported = [hit for hit in hits if all(name and name <= tokens(hit["text"]) for name in names)]
        if supported:
            items = [Evidence(hit, hit["text"]) for hit in supported[:1]]
            answer = render(items)
            quote = items[0].quote
            if len(quote) > 700:
                quote = quote[:700].rsplit("\n", 1)[0] + "\n…"
            answer["display_text"] = "Encontrei um trecho com os dois nomes:\n\n" + quote
            answer["method"] = "trechos com os nomes solicitados; sem inferir relações não declaradas"
            return answer
        return not_found(None)
    kind = intent_of(question)
    topic = tokens(question) - FILLER - {"é", "sao", "foi", "ser", "esta", "sabe"}
    if kind:
        topic -= PROPERTY_WORDS[kind]
    sources = memory.sources()
    preferred = []
    for source in sources:
        title_tokens = tokens(primary_title(source)) - TITLE_FILLER
        if source_id == source["id"] or (topic and topic <= title_tokens):
            preferred.append(source)
    if source_id:
        preferred = [s for s in preferred if s["id"] == source_id]
    if not topic and not source_id:
        return {"mode": "documental", "text": "De qual assunto você está falando? Informe o nome ou selecione um documento na Biblioteca.", "sources": []}
    search_query = question + (" " + " ".join(PROPERTY_WORDS[kind]) if kind else "")
    hits = memory.search(search_query, limit=24, source_id=source_id)
    # Permite encontrar uma relação em campo curto que não repete o título do artigo.
    if kind and preferred:
        full = [h for source in preferred[:3] for h in memory.chunks(source["id"])]
        seen = {h["id"] for h in full}
        hits = full + [h for h in hits if h["id"] not in seen]
    if kind:
        facts = []
        for hit in hits:
            title_matches = bool(topic and topic <= (tokens(primary_title(hit)) - TITLE_FILLER)) or (not topic and source_id == hit["source_id"])
            facts.extend(explicit_facts(hit, kind, topic, title_matches))
        unique = {}
        for fact in facts:
            key = fold(re.sub(r"\[[^\]]*\]", "", fact.value)).strip()
            unique.setdefault(key, fact)
        if unique:
            selected = list(unique.values())
            return render(selected[:2], direct=True, kind=kind, conflict=len(selected) > 1)
        return not_found(kind)
    candidates = []
    definition = re.match(r'^(?:o que (?:é|e|são|sao)|defina)\s+(.+?)[?.!]*$', question, re.I)
    if definition and preferred:
        intro = [h for source in preferred[:3] for h in memory.chunks(source['id'])[:8]]
        seen = {h['id'] for h in intro}
        hits = intro + [h for h in hits if h['id'] not in seen]
    for hit in hits:
        title_matches = bool(topic and topic <= (tokens(primary_title(hit)) - TITLE_FILLER))
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])|\n", hit["text"]):
            sentence = sentence.strip(" #\t")
            if len(sentence) < 15 or re.match(r"^(Ver artigo|Ver também|https?://)", sentence, re.I):
                continue
            terms = tokens(sentence)
            coverage = len(topic & terms) / max(1, len(topic))
            if coverage < 0.6:
                continue
            score = coverage * 10 + len(topic & terms) + (2 if title_matches else 0) - max(0, len(sentence) - 250) / 500
            if definition:
                lead = re.sub(r'^(?:o|a|os|as)\s+', '', fold(sentence))
                if re.match(re.escape(fold(definition.group(1).strip()))
                            + r'(?:\s*\([^)]{0,120}\))?\s+(?:e|sao|significa|consiste|designa|refere-se)\b', lead):
                    score += 6  # Prefere a definição do assunto à menção lateral.
            candidates.append(Evidence(hit, sentence, score=score))
    if candidates:
        candidates.sort(key=lambda item: -item.score)
        return render(candidates[:1])
    return not_found(None)


def not_found(kind):
    requested = "o nome solicitado" if kind == "name" else ("a capital solicitada" if kind == "capital" else "essa informação específica")
    return {"mode": "documental", "text": f"Não encontrei {requested} de forma explícita no material estudado. Não tenho uma resposta fundamentada para essa pergunta. Você pode informar o assunto com mais precisão ou consultar os trechos na Biblioteca.", "sources": []}
