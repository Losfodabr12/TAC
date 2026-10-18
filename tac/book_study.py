"""Seleção explícita de obras públicas; arquivos são dados, nunca instruções."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urljoin

from .config import atomic_json
from .db import now
from .documents import normalize
from .web import PublicFetcher, USER_AGENT, public_url


BOOKS = [
    (55752, "Dom Casmurro", "Machado de Assis"),
    (54829, "Memórias Póstumas de Brás Cubas", "Machado de Assis"),
    (67162, "Helena", "Machado de Assis"),
    (56737, "Esaú e Jacó", "Machado de Assis"),
    (33056, "Histórias sem Data", "Machado de Assis"),
    (53101, "A Mão e a Luva", "Machado de Assis"),
    (55797, "Memorial de Aires", "Machado de Assis"),
    (57001, "Papéis Avulsos", "Machado de Assis"),
    (44540, "Cinco Minutos", "José de Alencar"),
    (67740, "Iracema", "José de Alencar"),
    (67831, "A Pata da Gazela", "José de Alencar"),
    (38496, "Ubirajara", "José de Alencar"),
    (69187, "O Cortiço", "Aluísio Azevedo"),
    (67535, "Triste Fim de Policarpo Quaresma", "Lima Barreto"),
    (69229, "A Falência", "Júlia Lopes de Almeida"),
    (67594, "A Intrusa", "Júlia Lopes de Almeida"),
    (24401, "Viagens na Minha Terra", "Almeida Garrett"),
    (17591, "Frei Luís de Sousa", "Almeida Garrett"),
    (18220, "A Cidade e as Serras", "Eça de Queirós"),
    (31971, "O Crime do Padre Amaro", "Eça de Queirós"),
]


def book_text(raw, expected_id):
    """Exclui apenas o invólucro Gutenberg; preserva a obra e o original."""
    text = raw.decode("utf-8-sig", errors="strict")
    if not re.search(r"\b(?:eBook|EBook|ebook)\s*#?\s*" + str(expected_id) + r"\b", text[:5000]):
        raise ValueError("Identificador do livro não confere com o catálogo.")
    start = re.search(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*[^\n]*\n", text, re.M | re.I)
    end = re.search(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text, re.M | re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Marcadores da obra ausentes; arquivo não será truncado por adivinhação.")
    clean = normalize(text[start.end():end.start()])
    if len(clean) < 15000:
        raise ValueError("Arquivo curto demais para esta seleção de livros completos.")
    return clean


def download_text(fetcher, url):
    for _ in range(6):
        canonical, _, _, _ = public_url(url)
        policy, interval = fetcher.robot_policy(canonical)
        if not policy.can_fetch(USER_AGENT, canonical):
            raise ValueError("Download impedido por robots.txt.")
        status, headers, body, final = fetcher.raw(canonical, 8 * 2**20, interval)
        if status in (301, 302, 303, 307, 308):
            if not headers.get("Location"):
                raise ValueError("Redirecionamento sem destino.")
            url = urljoin(final, headers["Location"])
            continue
        if status != 200:
            raise ValueError(f"Download retornou HTTP {status}.")
        if "text/plain" not in headers.get("Content-Type", "").lower():
            raise ValueError("Download não é um arquivo textual.")
        return final, body
    raise ValueError("Muitos redirecionamentos no download.")


def study_books(core, progress=lambda *_: None):
    """Só é chamada por uma ação explícita; sem downloads ao abrir a TAC."""
    core.gate.check()
    folder = core.config.data / "books"
    folder.mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "version": 1, "authorization": "Lote de 20 livros solicitado explicitamente pelo usuário",
        "catalog": "https://www.gutenberg.org/browse/languages/pt",
        "mirror_list": "https://www.gutenberg.org/MIRRORS.ALL",
        "access_policy": "https://www.gutenberg.org/policy/robot_access.html",
        "copyright": "Obras originais de autores falecidos até 1934; catálogo Gutenberg informa domínio público nos EUA. Licenças e créditos integrais preservados nos originais.",
        "books": [],
    }
    fetcher = PublicFetcher(core.gate, core.config)
    for i, (number, title, author) in enumerate(BOOKS, 1):
        core.gate.check()
        progress(i - 1, f"Livro {i}/20: {title} — baixando/verificando")
        original = folder / f"{number}-original.txt"
        old = next((b for b in manifest["books"] if b["gutenberg_id"] == number), None)
        url = f"https://gutenberg.pglaf.org/cache/epub/{number}/pg{number}.txt"
        try:
            if original.exists():
                raw = original.read_bytes()
                if old and hashlib.sha256(raw).hexdigest() != old["sha256"]:
                    raise ValueError("Original local foi alterado; integridade não confere.")
            else:
                url, raw = download_text(fetcher, url)
            clean = book_text(raw, number)
            if len(clean) > core.config.get("max_text_chars"):
                raise ValueError("Livro excede o limite de texto configurado; não foi cortado.")
            metadata = {
                "gutenberg_id": number, "title": title, "author": author,
                "catalog_url": f"https://www.gutenberg.org/ebooks/{number}",
                "download_url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                "original_file": str(original.relative_to(core.config.root)),
                "characters": len(clean), "bytes": len(raw), "processed": now(),
                "batch": "20_livros_portugues", "training": "Indexado; treino neural registrado separadamente em runs e relatórios.",
            }
            with core.gate.writing():
                core.gate.check()
                original.write_bytes(raw)
                (folder / f"{number}.txt").write_text(clean, encoding="utf-8")
                source_id, status = core.memory.ingest(title, metadata["catalog_url"], "web", clean,
                                                      "Literatura em português / 20 livros", metadata)
                metadata.update(source_id=source_id, status=status)
                manifest["books"] = [b for b in manifest["books"] if b["gutenberg_id"] != number] + [metadata]
                atomic_json(manifest_path, manifest)
                core.study_history(None, metadata["catalog_url"], status, f"Livro completo: {title}; fonte #{source_id}; {len(clean)} caracteres")
            progress(i, f"Livro {i}/20 disponível: {title} ({len(clean):,} caracteres)")
        except Exception as exc:
            core.logger.exception("Livro %s não concluído", number)
            core.study_history(None, url, "erro", f"{title}: {exc}")
            raise
    return manifest
