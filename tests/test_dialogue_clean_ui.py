import json

import pytest

from tac.dialogue_lessons import LESSONS, examples
from tac.learning import LearningDisabled
from test_backup_ui import pump


def train(core, epochs=100):
    core.config.update(device="cpu", cpu_threads=2, ram_mb=4096, reserve_ram_mb=256)
    core.gate.set_enabled(True)
    return core.perform("Praticar conversas", lambda: core.dialogue.train(epochs=epochs), learning=True)


def test_dialogue_examples_are_distinct_and_have_real_separate_tests():
    from tac.neural.dialogue import normalized
    sets = [{normalized(r["text"]) for r in examples(split)} for split in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(LESSONS) - 1 == 18
    assert len(examples("train")) > 600


@pytest.mark.neural
def test_guided_dialogue_trains_routes_facts_and_keeps_name_in_session(core):
    result = train(core)
    assert result["status"] == "aprovado"
    assert result["initial_hash"] != result["final_hash"]
    assert result["validation_accuracy"] > result["baseline_accuracy"]
    assert result["test_accuracy"] >= .70
    assert result["unknown_false_accepts"] == 0
    assert result["legacy_test_examples"] == 54
    assert result["legacy_test_accuracy"] >= .70
    assert core.dialogue.active_id() == result["checkpoint"]
    core.memory.ingest("Brasil", "fonte pública", "file", "Brasil, oficialmente República Federativa do Brasil, é um país da América do Sul. A capital do Brasil é Brasília.")
    core.gate.set_enabled(False)
    assert core.chat("oi")["mode"] == "conversa guiada"
    for greeting in ("eae", "EAE!", "eai", "eae tac", "eae 😄", "oi 😊", "oieee"):
        answer = core.chat(greeting)
        assert answer.get("intent") == "greeting", (greeting, answer)
    for text, expected in (("estou triste", "feeling_low"), ("tô muito feliz", "feeling_good"), ("tô irritado", "frustration")):
        assert core.chat(text).get("intent") == expected
    question = core.chat("eae, qual a capital do brasil?")
    assert question["mode"] == "documental"
    assert "Brasília" in question["display_text"] and question["sources"]
    source_id = core.memory.sources()[0]["id"]
    assert core.chat("eae", source_id=source_id)["intent"] == "greeting"
    assert core.chat("vlw", source_id=source_id)["intent"] == "thanks"
    assert "TAC" in core.chat("qual é seu nome")["text"]
    assert "TAC" in core.chat("me diz teu nome")["text"]
    assert "Gabriel" in core.chat("meu nome é Gabriel")["text"]
    assert "Gabriel" in core.chat("qual é meu nome")["text"]
    factual = core.chat("qual o nome oficial do brasil")
    assert factual["mode"] == "documental"
    assert factual["sources"]
    assert "República Federativa do Brasil" in factual["display_text"]
    assert not core.db.query("SELECT * FROM interactions")
    from tac.core import Core
    reopened = Core(core.config.root)
    assert reopened.session_name is None
    assert reopened.chat("oi")["mode"] == "conversa guiada"
    assert not reopened.gate.enabled


def test_greeting_cleanup_keeps_quoted_words_and_entity_names():
    from tac.answers import question_content
    assert question_content("eae, qual a capital do Brasil?") == "qual a capital do Brasil?"
    assert question_content("Oi 😊, qual é o nome do Brasil?") == "qual é o nome do Brasil?"
    assert question_content("oi tac, como funciona Python?") == "como funciona Python?"
    assert question_content("o que significa eae?") == "o que significa eae?"
    assert question_content("qual o nome da empresa Oi?") == "qual o nome da empresa Oi?"


@pytest.mark.neural
def test_dialogue_training_cannot_run_with_learning_disabled(core):
    with pytest.raises(LearningDisabled):
        core.perform("Praticar conversas", lambda: core.dialogue.train(epochs=1), learning=True)
    assert not core.db.query("SELECT * FROM runs WHERE kind='dialogue'")


def test_clean_navigation_and_sources_only_when_requested(core, qtapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QPlainTextEdit
    from tac.ui import Window
    core.gate.set_enabled(True)
    core.memory.ingest("Aurória", "https://example.org/auroria", "web", "Aurória, oficialmente União de Aurória, tem dados neste documento.")
    window = Window(core)
    window.timer.stop()
    assert [window.nav.item(i).text() for i in range(5)] == ["Início", "Conversar", "Meus materiais", "Melhorar a TAC", "Preferências"]
    assert all(window.nav.item(i).isHidden() for i in (5,6,7,8))
    assert not window.chat_tools.isHidden()
    assert window.activity_detail.isHidden()
    assert not window.practice_pause_button.isEnabled()
    assert not window.practice_stop_button.isEnabled()
    window.input.setText("qual o nome de Aurória")
    window.send_button.click()
    pump(qtapp, lambda: window.job is None)
    transcript = window.transcript.toPlainText()
    assert "União de Aurória" in transcript
    assert "https://example.org" not in transcript
    assert "Origem:" not in transcript
    assert not window.sources_button.isHidden()
    captured = {}
    def inspect():
        dialog = qtapp.activeModalWidget()
        captured["text"] = dialog.findChild(QPlainTextEdit).toPlainText()
        dialog.accept()
    QTimer.singleShot(30, inspect)
    window.sources_button.click()
    assert "https://example.org/auroria" in captured["text"]
    assert "Aurória, oficialmente União de Aurória" in captured["text"]
    window.advanced_check.setChecked(True)
    assert all(not window.nav.item(i).isHidden() for i in (5,6,7,8))
    window.close()
