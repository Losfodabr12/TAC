import json

import pytest
import torch

from tac.documents import CHUNKING_VERSION, normalize, split_sections
from tac.learning import LearningDisabled
from tac.neural.data import Corpus
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import safe_load, weight_hash
from test_training import setup_training


def story_text():
    return normalize("EPISÓDIO 1 — ENCONTRO\n" +
                     "\n".join(f"Alice:\n— Posso ajudar, Bento? 😊\nBento respira.\n— Pode sim, tentativa {i}.\nOs dois seguem o caminho." for i in range(60)))


def test_dialogue_chunks_preserve_context_and_every_character():
    text = story_text()
    chunks = list(split_sections(text))
    assert "\n".join(chunk for _, chunk in chunks) == text
    assert max(len(chunk) for _, chunk in chunks) <= 1200
    assert len(chunks) < 15
    assert any("Alice:" in chunk and "Bento respira." in chunk and "😊" in chunk for _, chunk in chunks)


def test_reindex_preserves_source_and_text_and_requires_permission(core):
    core.gate.set_enabled(True)
    text = story_text()
    source_id, _ = core.memory.ingest("Narrativa", "original.txt", "file", text)
    original = core.db.query("SELECT * FROM sources WHERE id=?", (source_id,))[0]
    with core.db.connect() as c:
        c.execute("UPDATE sources SET metadata='{}' WHERE id=?", (source_id,))
        c.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
        c.executemany("INSERT INTO chunks(source_id,ordinal,section,text) VALUES(?,?,?,?)",
                      ((source_id, i, "Legado", line) for i, line in enumerate(text.splitlines())))
    core.gate.set_enabled(False)
    with pytest.raises(LearningDisabled):
        core.memory.reindex_source(source_id)
    core.gate.set_enabled(True)
    result = core.perform("Reorganizar", lambda: core.memory.reindex_source(source_id), learning=True)
    assert result["after"] < result["before"] / 10
    after = core.db.query("SELECT * FROM sources WHERE id=?", (source_id,))[0]
    assert after["content_hash"] == original["content_hash"]
    assert after["origin"] == original["origin"]
    assert json.loads(after["metadata"])["chunking_version"] == CHUNKING_VERSION
    assert "\n".join(c["text"] for c in core.memory.chunks(source_id)) == text
    assert core.memory.search("Bento")
    assert not core.memory.reindex_source(source_id)["changed"]
    corpus = Corpus.from_memory(core.db, core.config)
    assert sum(len(v) >= 128 for v in corpus.val_text) > len(corpus.val_text) / 2


def test_explicit_document_cannot_be_overridden_by_guided_intent(core, monkeypatch):
    core.gate.set_enabled(True)
    sid, _ = core.memory.ingest("Conto", "conto.txt", "file", "Alice acompanha Bento na viagem.\nBento responde:\n— Obrigado por me acompanhar, Alice.")
    core.db.execute("INSERT INTO meta VALUES('active_dialogue','placeholder_in_test')")
    def wrong_route(*_):
        pytest.fail("Um documento explicitamente selecionado não deve passar pelo classificador")
    monkeypatch.setattr(core.dialogue, "reply", wrong_route)
    core.gate.set_enabled(False)
    answer = core.chat("Qual a relação entre Alice e Bento?", source_id=sid)
    assert answer["mode"] == "documental"
    assert answer["sources"][0]["source_id"] == sid
    assert any("Alice" in e and "Bento" in e for e in answer["evidence"])
    assert "sem inferir" in answer["method"]


def test_focus_corpus_and_selected_queue_do_not_include_unapproved_items(core, tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text(story_text(), encoding="utf-8")
    second.write_text("Outro material pendente que não deve ser processado nesta ação.", encoding="utf-8")
    a, b = core.enqueue_file(first), core.enqueue_file(second)
    core.gate.set_enabled(True)
    result = core.perform("Fila aprovada", lambda: core.run_queue(queue_ids=[a]), learning=True)
    assert result["processed"] == result["total"] == 1
    assert core.db.query("SELECT status FROM queue WHERE id=?", (b,))[0]["status"] == "pendente"
    sid = result["source_ids"][0]
    other, _ = core.memory.ingest("Outra", "outra", "file", "Conteúdo fora do grupo selecionado. " * 20)
    corpus = Corpus.from_memory(core.db, core.config, source_ids=[sid])
    assert {p["source_id"] for p in corpus.provenance} == {sid}
    assert all(p["source_id"] != other for p in corpus.provenance)


def test_best_epoch_is_rechecked_before_promotion(core, monkeypatch):
    corpus = setup_training(core, 2)
    core.config.update(epochs=2)
    values = {"validation": iter([10., 8., 9., 11., 8.]), "fixed": iter([10., 10., 20., 20., 10.])}
    # Avaliação controlada só para verificar seleção/rejeição, não para alegar progresso.
    monkeypatch.setattr(core.trainer, "evaluate", lambda model, data, split, device: (next(values[split]), .5))
    result = core.perform("Seleção controlada", lambda: core.trainer.train(corpus=corpus), learning=True)
    assert result["steps"] == 4 and result["selected_step"] == 2
    assert result["status"] == "aprovado"
    assert result["checkpoint"] != result["last_checkpoint"]
    cp = safe_load(core.trainer.checkpoint_path(result["checkpoint"]))
    model = TACTransformer(ModelConfig(**cp["model_config"]))
    model.load_state_dict(cp["model"])
    assert cp["step"] == 2 and weight_hash(model) == result["final_hash"]
    assert core.db.query("SELECT status FROM checkpoints WHERE id=?", (result["last_checkpoint"],))[0]["status"] == "rejeitado"


@pytest.mark.neural
def test_incremental_replay_resume_reproduces_weights(core, tmp_path):
    from tac.core import Core
    prior = setup_training(core, 8)
    core.perform("Base", lambda: core.trainer.train(corpus=prior), learning=True)
    new = Corpus(["abcde " * 90, "sim nao " * 50], ["abcde " * 35, "sim nao " * 20])
    core.config.update(steps_per_epoch=12)
    def stop(step, *_):
        if step == 5:
            core.gate.set_enabled(False)
    interrupted = core.perform("Parar", lambda: core.trainer.train(stop, corpus=new), learning=True)
    cp = safe_load(core.trainer.checkpoint_path(interrupted["checkpoint"]))
    assert cp["training_config"]["replay_dataset_hash"] == prior.digest
    core.gate.set_enabled(True)
    core.config.update(batch_size=1, accumulation=3)
    resumed = core.perform("Retomar", lambda: core.trainer.train(resume=interrupted["checkpoint"]), learning=True)
    other = Core(tmp_path / "uninterrupted_replay")
    other.config.update(reserve_ram_mb=256, ram_mb=4096, cpu_threads=2)
    original = setup_training(other, 8)
    other.perform("Base", lambda: other.trainer.train(corpus=original), learning=True)
    other.config.update(steps_per_epoch=12)
    complete = other.perform("Completo", lambda: other.trainer.train(corpus=new), learning=True)
    assert resumed["final_hash"] == complete["final_hash"]
