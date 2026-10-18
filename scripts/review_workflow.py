"""Aceitação com UI real e o artigo Wikipédia fornecido pelo usuário."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from tac.config import atomic_json
from tac.core import Core
from tac.ui import Window, apply_theme


def main():
    folder = ROOT / "reports" / ("workflow_real_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    core = Core(folder)
    app = QApplication([])
    app.setStyle("Fusion")
    apply_theme(app, core.config)
    window = Window(core)
    window.timer.stop()
    window.show()
    events = []

    def wait():
        deadline = time.monotonic() + 90
        while window.job is not None:
            app.processEvents()
            time.sleep(.01)
            if time.monotonic() > deadline:
                core.gate.cancel.set()
                raise TimeoutError("A operação excedeu o limite do teste.")
        app.processEvents()

    try:
        window.go('materials')
        window.url.setText("https://pt.wikipedia.org/wiki/Brasil")
        window.terms.setChecked(True)
        window.add_url_button.click()
        app.processEvents()
        assert not core.gate.enabled
        assert "Permitir e guardar" in window.study_button.text()
        window.grab().save(str(folder / "01_aguardando.png"))
        window.study_button.click()
        def record(step, message):
            events.append(message)
            if len(events) == 3:
                window.grab().save(str(folder / "02_estudando.png"))
        window.job.update.connect(record)
        wait()
        assert window.last_study_result["processed"] == 1, window.study_results.text()
        assert window.last_study_result["errors"] == 0
        assert window.activity_title.text() == "Pronta para conversar"
        window.grab().save(str(folder / "03_concluido.png"))
        window.study_query_button.click()
        wait()
        overview = core.last_answer[1]
        window.grab().save(str(folder / "04_consulta.png"))
        window.input.setText("Brasil América do Sul")
        window.send_button.click()
        wait()
        answer = core.last_answer[1]
        assert answer["mode"] == "documental"
        assert answer["sources"]
        assert "https://pt.wikipedia.org/wiki/Brasil" in answer["text"]
        assert "Menu principal" not in answer["text"]
        result = {"status": "passed", "study": window.last_study_result,
                  "stages": events, "sources": core.memory.sources(),
                  "overview": overview, "answer": answer, "isolated_data": str(folder),
                  "user_data_modified": False}
        atomic_json(ROOT / "reports" / "revisao_workflow_real.json", result)
        print(json.dumps({"status": "passed", "stages": len(events), "sources": len(result["sources"]), "evidence": str(folder)}, ensure_ascii=True), flush=True)
    finally:
        core.gate.cancel.set()
        if window.job is not None:
            wait()
        core.gate.set_enabled(False)
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
