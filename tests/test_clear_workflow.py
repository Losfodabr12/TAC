import pytest

from tac.presentation import explain_error, snapshot
from test_backup_ui import pump


def test_one_button_authorizes_study_and_opens_grounded_answer(core, qtapp, tmp_path):
    from tac.ui import Window
    path = tmp_path / "clareza.txt"
    path.write_text("Clareza: a TAC guarda texto com sua origem. Este exemplo permite verificar que o estudo terminou e a consulta funciona.", encoding="utf-8")
    core.enqueue_file(path)
    window = Window(core)
    window.timer.stop()
    assert not core.gate.enabled
    assert window.nav.currentRow() == 0
    assert "aguardando" in window.activity_title.text().lower()
    assert "Permitir e guardar" in window.study_button.text()
    assert not window.train_button.isEnabled()
    assert not window.chat_mode.model().item(1).isEnabled()
    window.study_button.click()
    pump(qtapp, lambda: window.job is None)
    assert core.gate.enabled
    assert "Pronta para consultar" == window.activity_title.text()
    assert "Material guardado" in window.study_results.text()
    assert window.study_progress.format() == "Material guardado"
    assert window.queue_table.item(0, 3).text() == "Disponível para consulta"
    assert window.train_button.isEnabled()
    window.study_query_button.click()
    pump(qtapp, lambda: window.job is None)
    assert "Clareza:" in window.transcript.toPlainText()
    assert "Origem:" not in window.transcript.toPlainText()
    assert core.last_answer[1]["sources"]
    assert not window.sources_button.isHidden()
    assert window.chat_mode.currentIndex() == 0
    window.close()


def test_failed_site_is_never_presented_as_success(core, qtapp, monkeypatch):
    import tac.web as web
    from tac.ui import Window
    monkeypatch.setattr(web, "public_url", lambda url: (url, "example.org", 443, ["8.8.8.8"]))
    def blocked(*args, **kwargs):
        raise ValueError("Site retornou HTTP 403. Nenhuma proteção será contornada.")
    monkeypatch.setattr(web.PublicFetcher, "fetch", blocked)
    core.enqueue_url("https://example.org/", terms=True)
    window = Window(core)
    window.timer.stop()
    window.study_button.click()
    pump(qtapp, lambda: window.job is None)
    assert window.last_study_result["errors"] == 1
    assert window.last_study_result["processed"] == 0
    assert "Nenhum material foi estudado" in window.study_results.text()
    assert "com erro" in window.activity_title.text()
    assert "bloqueou" in window.queue_detail.toPlainText()
    assert window.study_progress.format() == "Finalizado com erro(s)"
    assert not window.study_button.isEnabled()
    assert window.retry_button.isEnabled()
    assert not core.memory.sources()
    window.close()


def test_add_site_does_not_start_or_duplicate_it(core, qtapp):
    from tac.ui import Window
    window = Window(core)
    window.timer.stop()
    for _ in range(2):
        window.url.setText("https://example.org/docs/")
        window.terms.setChecked(True)
        window.add_url_button.click()
    assert not core.gate.enabled
    assert len(core.db.query("SELECT * FROM queue")) == 1
    assert window.job is None
    assert "ainda não começou" in window.notice.text()
    assert "Permitir e guardar" in window.study_button.text()
    window.close()


def test_empty_training_explains_next_action_without_loading_model(core, qtapp):
    from tac.ui import Window
    window = Window(core)
    window.timer.stop()
    assert not window.train_button.isEnabled()
    window.train()
    assert window.job is None
    assert core._trainer is None
    assert "primeiro" in window.notice.text().lower()
    assert "Não foi possível concluir" in window.activity_title.text()
    assert not core.db.query("SELECT * FROM runs")
    window.close()


def test_queue_selection_survives_live_refresh(core, qtapp):
    from tac.ui import Window
    first = core.enqueue_url("https://example.org/a", terms=True)
    core.enqueue_url("https://example.org/b", terms=True)
    window = Window(core)
    window.timer.stop()
    window.queue_table.selectRow(1)
    core.db.execute("UPDATE queue SET progress='Verificando permissão' WHERE id=?", (first,))
    window.refresh_queue()
    assert window.queue_selected()["id"] == first
    assert window.queue_table.item(1, 4).text() == "Verificando permissão"
    window.close()


def test_snapshot_distinguishes_knowledge_from_neural_model(core):
    assert snapshot(core)["state"] == "empty"
    core.gate.set_enabled(True)
    core.memory.ingest("x", "local", "file", "Conhecimento documental pode ser consultado mesmo sem qualquer modelo neural treinado.")
    state = snapshot(core)
    assert state["state"] == "ready"
    assert state["sources"] == 1
    assert state["active_checkpoint"] is None
    assert state["can_train"]
    core.gate.set_enabled(False)
    assert not snapshot(core)["can_train"]
    assert snapshot(core)["state"] == "ready"


def test_recent_document_can_be_opened_after_restarting_window(core, qtapp):
    from tac.ui import Window
    core.gate.set_enabled(True)
    core.memory.ingest("Fonte persistente", "origem local", "file", "Uma fonte persistente continua disponível depois de reiniciar a janela, sem exigir novo estudo e sem modelo neural treinado.")
    core.gate.set_enabled(False)
    window = Window(core)
    window.timer.stop()
    assert not window.study_query_button.isHidden()
    window.study_query_button.click()
    pump(qtapp, lambda: window.job is None)
    assert "origem local" not in window.transcript.toPlainText()
    assert core.last_answer[1]["sources"][0]["origin"] == "origem local"
    assert "Pronta para consultar" == window.activity_title.text()
    assert not core.gate.enabled
    window.close()


def test_single_source_overview_needs_no_extra_context_selection(core):
    core.gate.set_enabled(True)
    core.memory.ingest("Fonte única", "origem", "file", "Este é o conteúdo verificável da única fonte disponível na biblioteca. O usuário pode pedir uma visão geral sem precisar selecionar o documento novamente.")
    answer = core.chat("O que você aprendeu?")
    assert answer["sources"]
    assert "Fonte única" in answer["text"]


@pytest.mark.parametrize("message,expected", [
    ("Dados insuficientes para treino e validação separados.", "Primeiro estude"),
    ("Site retornou HTTP 429.", "limitou"),
    ("Acesso não permitido por robots.txt.", "não estudou"),
    ("A página maior que o limite.", "limite de tamanho"),
])
def test_errors_give_a_concrete_next_step(message, expected):
    assert expected in explain_error(message)
