import pytest

from tac.book_study import BOOKS, book_text, download_text
from tac.learning import Cancelled
from tac.neural.data import Corpus


def test_book_selection_and_strict_wrapper():
    assert len(BOOKS) == len({b[0] for b in BOOKS}) == 20
    text = "Texto da obra com ação e diálogo.\n" * 900
    raw = ("eBook #55752\n*** START OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***\n"
           + text + "\n*** END OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***\nLICENSE").encode()
    clean = book_text(raw, 55752)
    assert "ação" in clean and "LICENSE" not in clean
    with pytest.raises(ValueError, match="Identificador"):
        book_text(raw, 1)
    with pytest.raises(ValueError, match="Marcadores"):
        book_text(raw.replace(b"*** END", b"MISSING END"), 55752)


def test_balanced_corpus_keeps_new_sources_and_validated_corrections(core):
    core.gate.set_enabled(True)
    ids = []
    for i in range(5):
        sid, _ = core.memory.ingest(f"Book {i}", f"book:{i}", "file", (f"Material unico {i}. " * 500))
        ids.append(sid)
    correction_id = core.memory.correct("Erro", "Correção validada e autorizada para treinamento. " * 3,
                                        "Teste", "Autor do teste", validated=True)
    unvalidated_id = core.memory.correct("Erro", "Correção ainda pendente de validação. " * 3,
                                         "Teste", "Autor do teste", validated=False)
    core.config.update(max_corpus_chars=2000)
    a = Corpus.from_memory(core.db, core.config)
    b = Corpus.from_memory(core.db, core.config)
    assert {p["source_id"] for p in a.provenance if "source_id" in p} == set(ids)
    assert {p["correction_id"] for p in a.provenance if "correction_id" in p} == {correction_id}
    assert sum(map(len, a.train_text + a.val_text)) <= 2000
    assert a.digest == b.digest
    # O limite não recolhe apenas as primeiras fontes ou depende da ordem SQL.
    assert len({p["ordinal"] for p in a.provenance if "ordinal" in p}) >= 1


def test_disabled_learning_prevents_book_download(core):
    from tac.book_study import study_books
    assert not core.gate.enabled
    with pytest.raises(Cancelled):
        study_books(core)
    assert not (core.config.data / "books").exists()


def test_download_checks_robots_before_body(monkeypatch):
    monkeypatch.setattr("tac.book_study.public_url", lambda u: (u, "example.org", 443, ["1.1.1.1"]))
    class Policy:
        def can_fetch(self, *_):
            return False
    class Fetcher:
        def robot_policy(self, *_):
            return Policy(), 0
        def raw(self, *_):
            pytest.fail("Não deve baixar se robots impede acesso")
    with pytest.raises(ValueError, match="robots"):
        download_text(Fetcher(), "https://example.org/book.txt")
