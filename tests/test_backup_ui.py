import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from tac.backups import export_backup, import_backup, relocate_data


def test_backup_roundtrip_and_permissions(core):
    core.gate.set_enabled(True)
    core.memory.ingest("fonte", "teste", "file", "Memória persistente e verificável em backup local da TAC.")
    books = core.config.data / "books"
    books.mkdir()
    (books / "original.txt").write_text("Obra original e licença preservadas.", encoding="utf-8")
    (books / "manifest.json").write_text('{"books": []}', encoding="utf-8")
    core.config.update(auto_train=True, auto_on_start=True)
    backup = core.config.data / "backups" / "test.zip"
    core.perform("Backup", lambda: export_backup(core, backup))
    destination = core.perform("Restore", lambda: import_backup(core, backup))
    from tac.core import Core
    restored = Core(core.config.root)
    assert len(restored.memory.sources()) == 1
    assert restored.chat("verificável")["sources"]
    assert not restored.gate.enabled
    assert not restored.config.get("auto_train")
    assert Path(destination).is_dir()
    assert (restored.config.data / "books/original.txt").read_text(encoding="utf-8") == "Obra original e licença preservadas."
    assert (restored.config.data / "books/manifest.json").is_file()


def test_backup_rejects_path_traversal(core):
    backup = core.config.root / "malicioso.zip"
    with zipfile.ZipFile(backup, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "TAC-backup-v1", "files": {"../outside.txt": "x"}}))
        z.writestr("../outside.txt", "não extrair")
    with pytest.raises(ValueError):
        import_backup(core, backup)
    assert not (core.config.root.parent / "outside.txt").exists()


def test_data_relocation(core):
    core.gate.set_enabled(True)
    core.memory.ingest("movimento", "teste", "file", "Dados que persistem durante a migração local.")
    original = core.config.data
    core.perform("Migração", lambda: relocate_data(core, "dados_novos"))
    from tac.core import Core
    moved = Core(core.config.root)
    assert moved.memory.search("migração")
    assert original.exists()
    assert moved.config.data.name == "dados_novos"


def pump(app, until, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return
        time.sleep(.005)
    raise AssertionError("Timeout aguardando interface")


def test_gui_simple_pages_and_thread_responsiveness(core, qtapp):
    from PySide6.QtCore import QTimer
    from tac.ui import Window
    window = Window(core)
    window.timer.stop()
    window.show()
    assert window.pages.count() == 9
    assert not window.auto_check.isChecked()
    beats = []
    timer = QTimer()
    timer.timeout.connect(lambda: beats.append(1))
    timer.start(10)
    window.start_job("Carga de teste", lambda progress: time.sleep(.3))
    pump(qtapp, lambda: window.job is None)
    timer.stop()
    assert len(beats) >= 8
    window.toggle_learning()
    assert core.gate.enabled
    window.toggle_learning()
    assert not core.gate.enabled
    window.close()


def test_gui_txt_chat_workflow(core, qtapp, tmp_path):
    from tac.ui import Window
    path = tmp_path / "interface.txt"
    path.write_text("O identificador QUARTZO demonstra importação e recuperação pela interface desktop.", encoding="utf-8")
    window = Window(core)
    window.timer.stop()
    window.add_files([str(path)])
    pump(qtapp, lambda: window.job is None)
    assert len(core.db.query("SELECT * FROM queue")) == 1
    window.toggle_learning()
    window.process_queue()
    pump(qtapp, lambda: window.job is None)
    window.input.setText("QUARTZO")
    window.send_chat()
    pump(qtapp, lambda: window.job is None)
    assert "QUARTZO" in window.transcript.toPlainText()
    assert "Origem:" not in window.transcript.toPlainText()
    assert core.last_answer[1]["mode"] == "documental"
    assert core.last_answer[1]["sources"][0]["origin"] == str(path)
    assert not window.sources_button.isHidden()
    window.close()


def test_automatic_training_requires_startup_authorization(core, qtapp):
    from tac.ui import Window
    core.gate.set_enabled(True)
    core.memory.ingest("fonte", "local", "file", "Documento suficiente para uma execução automática autorizada.")
    core.config.update(auto_train=True, auto_on_start=False)
    window = Window(core)
    window.timer.stop()
    window.tick()
    assert window.job is None
    assert not core.db.query("SELECT * FROM runs")
    window.close()


@pytest.mark.skipif(os.name != "nt", reason="Arquivo BAT é exclusivo de Windows")
def test_real_bat_startup(tmp_path):
    root = Path(__file__).parents[1]
    screenshot = tmp_path / "ui.png"
    result = subprocess.run(["cmd.exe", "/d", "/c", str(root / "INICIAR_TAC.bat"), "--smoke-ui", "--test-root", str(tmp_path / "bat-test"), "--screenshot", str(screenshot)], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert screenshot.exists()
