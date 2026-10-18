"""Verifica dados reais do lote e a interface, sem treinar nem acessar a rede."""
import hashlib
import contextlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from PySide6.QtWidgets import QApplication
from tac.config import atomic_json
from tac.core import Core
from tac.memory import words
from tac.neural.inventory import active_models
from tac.ui import Window, apply_theme

core = Core()
report = json.loads((ROOT / "reports/treino_20_livros.json").read_text(encoding="utf-8"))
backup = sorted((core.config.data / "backups").glob("antes_20_livros_*.zip"))[-1]
with zipfile.ZipFile(backup) as z:
    previous_path = ROOT / "reports/book_backup_check.sqlite3"
    previous_path.write_bytes(z.read("tac.sqlite3"))
with zipfile.ZipFile(backup) as z, contextlib.closing(sqlite3.connect(previous_path)) as previous:
    prior_sources = previous.execute("SELECT id,content_hash FROM sources ORDER BY id").fetchall()
    current_sources = {s["id"]: s["content_hash"] for s in core.db.query("SELECT id,content_hash FROM sources")}
    assert all(current_sources.get(i) == digest for i, digest in prior_sources)
    prior_dialogue = previous.execute("SELECT value FROM meta WHERE key='active_dialogue'").fetchone()[0]
    assert core.db.query("SELECT value FROM meta WHERE key='active_dialogue'")[0]["value"] == prior_dialogue
    assert json.loads(z.read("config.json")) == core.config.values

assert len(report["books"]) == 20
for book in report["books"]:
    raw = (ROOT / book["original_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == book["sha256"]
    assert str(book["source_id"]) in report["training"]["sampled_sources"]
    # Edições antigas podem escrever "fallencia" ou "Ayres"; verifica texto real.
    chunk = next(c for c in core.memory.chunks(book["source_id"]) if len(words(c["text"])) >= 3)
    query = " ".join(words(chunk["text"])[:3])
    assert core.memory.search(query, source_id=book["source_id"]), book["title"]
assert report["checkpoint_verified"]
assert core.trainer.active_id() == report["training"]["checkpoint"]
models = active_models(core.config, core.db)
assert [m["parameters"] for m in models] == [285888, 264336]

app = QApplication([])
app.setStyle("Fusion")
apply_theme(app, core.config)
window = Window(core)
window.timer.stop()
window.advanced_check.setChecked(True)
window.show()
app.processEvents()
assert "285.888" in window.model_label.text()
assert "aprovado" in window.train_label.text()
window.nav.setCurrentRow(3)
app.processEvents()
captures = ROOT / "reports/captures"
captures.mkdir(exist_ok=True)
window.grab().save(str(captures / "treino_20_livros.png"))
window.present_monitor(core.stats())
assert "285.888" in window.monitor.toPlainText()
assert "264.336" in window.monitor.toPlainText()
window.nav.setCurrentRow(4)
app.processEvents()
window.grab().save(str(captures / "parametros_021.png"))
result = {"prior_sources_preserved": len(prior_sources), "sources_after": len(current_sources),
          "books_integrity_and_training_coverage": 20, "preferences_preserved": True,
          "guided_conversation_preserved": True, "models": models,
          "gui_parameter_counts": True, "gui_training_metrics_restored": True,
          "checkpoint_verified": True, "training_status": report["status"]}
atomic_json(ROOT / "reports/validacao_20_livros.json", result)
window.close()
print(json.dumps(result, ensure_ascii=False, indent=2))
