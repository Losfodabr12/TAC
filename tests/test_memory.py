import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.core import Core
from tac.learning import LearningDisabled


def study(core, path):
    core.enqueue_file(path, "programação")
    core.gate.set_enabled(True)
    return core.perform("Teste de estudo", core.run_queue, learning=True)


def test_new_install_permissions_and_persistence(core):
    assert core.config.get("auto_train") is False
    assert core.config.get("auto_on_start") is False
    assert core.gate.enabled is False
    core.gate.set_enabled(True)
    reopened = Core(core.config.root)
    assert reopened.gate.enabled is True
    assert reopened.config.get("auto_train") is False


def test_txt_retrieval_source_restart_and_duplicate(core, tmp_path):
    doc = tmp_path / "Guia Python.txt"
    text = "Python é uma linguagem de programação. O SQLite armazena dados locais de maneira persistente."
    doc.write_text(text, encoding="utf-8")
    result = study(core, doc)
    assert result["processed"] == 1
    assert doc.read_text(encoding="utf-8") == text
    answer = core.chat("Quais documentos falam sobre Python?")
    assert answer["mode"] == "documental"
    assert text in answer["text"]
    assert str(doc) in answer["text"]
    assert answer["sources"][0]["origin"] == str(doc)
    assert not core.db.query("SELECT * FROM runs")
    # Outro processo abre a memória sem reutilizar conexão ou objetos Python.
    script = "from pathlib import Path; from tac.core import Core; import sys; c=Core(Path(sys.argv[1])); a=c.chat('Python'); assert a['sources']; assert 'Python' in a['text']; assert a['sources'][0]['origin'].endswith('Guia Python.txt'); print('persisted')"
    done = subprocess.run([sys.executable, "-c", script, str(core.config.root)], cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert "persisted" in done.stdout
    study(core, doc)
    assert len(core.memory.sources()) == 1
    assert core.db.query("SELECT status FROM study_history ORDER BY id DESC LIMIT 1")[0]["status"] == "inalterado"


def test_disabled_queue_no_memory_or_weight_writes(core, tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("Conhecimento que ainda não foi autorizado para estudo.", encoding="utf-8")
    core.enqueue_file(path)
    assert len(core.db.query("SELECT * FROM queue")) == 1
    with pytest.raises(LearningDisabled):
        core.perform("Estudo bloqueado", core.run_queue, learning=True)
    assert core.memory.sources() == []
    with pytest.raises(LearningDisabled):
        core.memory.ingest("x", "x", "file", "texto")
    with pytest.raises(LearningDisabled):
        core.memory.correct("x", "correção", "tema", "fonte")
    assert not list((core.config.data / "checkpoints").glob("*.pt"))


def test_disable_before_commit_keeps_pending(core, tmp_path, monkeypatch):
    path = tmp_path / "x.txt"
    path.write_text("Texto local autorizado inicialmente.", encoding="utf-8")
    core.enqueue_file(path)
    core.gate.set_enabled(True)
    def extraction(*args):
        core.gate.set_enabled(False)
        return "x", "texto novo não deve ser gravado", {}
    monkeypatch.setattr("tac.core.extract", extraction)
    core.perform("Interrupção", core.run_queue, learning=True)
    assert core.memory.sources() == []
    assert core.db.query("SELECT status FROM queue")[0]["status"] == "pendente"


def test_update_delete_fts_and_unknown_answer(core, tmp_path):
    path = tmp_path / "conhecimento.txt"
    path.write_text("A palavra heliotropismo está presente neste documento científico.", encoding="utf-8")
    study(core, path)
    source_id = core.memory.sources()[0]["id"]
    assert core.memory.search("heliotropismo")
    path.write_text("A palavra fotossíntese agora substitui o assunto anterior neste documento.", encoding="utf-8")
    study(core, path)
    assert not core.memory.search("heliotropismo")
    assert core.memory.search("fotossíntese")
    core.delete_source(source_id)
    assert not core.memory.search("fotossíntese")
    assert not core.db.query("SELECT * FROM chunks")
    assert path.exists()
    assert not core.chat("astrobiologia")["sources"]


def test_interactions_and_corrections_require_permission(core):
    core.config.update(save_interactions=True)
    core.chat("oi")
    assert not core.db.query("SELECT * FROM interactions")
    core.gate.set_enabled(True)
    core.chat("oi")
    assert len(core.db.query("SELECT * FROM interactions")) == 1
    core.memory.correct("original", "Texto correto com contexto suficiente para treinamento posterior, quando aprovado.", "assunto", "fonte", False)
    assert core.db.query("SELECT validated FROM corrections")[0]["validated"] == 0
    assert not core.db.query("SELECT * FROM runs")


def test_comparison_and_conflicts_are_quoted(core):
    core.gate.set_enabled(True)
    core.memory.ingest("Relatorio A", "a", "file", "O sistema é compatível com Python.")
    core.memory.ingest("Relatorio B", "b", "file", "O sistema não é compatível com Python.")
    answer = core.chat("Quais contradições entre Relatorio A e Relatorio B?")
    assert len(answer["sources"]) == 2
    assert "heurística" in answer["text"]
    assert "fontes" in answer["text"]
    general = core.chat("Quais informações parecem contraditórias?")
    assert len(general["sources"]) == 2
    assert "Possíveis conflitos lexicais" in general["text"]


def test_recovery_is_explicit_and_preserves_pending_work(core):
    core.db.execute("INSERT INTO queue(kind,target,options,status,created) VALUES('web','https://example.org/','{}','processando','2026-01-01')")
    # Abrir um objeto Core para leitura não interfere no trabalho de outra instância.
    reopened = Core(core.config.root)
    assert reopened.db.query("SELECT status FROM queue")[0]["status"] == "processando"
    reopened.recover_interrupted()
    assert reopened.db.query("SELECT status FROM queue")[0]["status"] == "pendente"
