"""Valida instalação real, preservação e interface. Sem treino e sem rede."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from PySide6.QtWidgets import QApplication
from tac.config import atomic_json
from tac.core import Core
from tac.dialogue_lessons import examples
from tac.documents import normalize, text_hash
from tac.neural.inventory import active_models
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import safe_load, weight_hash
from tac.ui import Window, apply_theme

core = Core()
before = json.loads((ROOT / "reports/baseline_030.json").read_text(encoding="utf-8"))
report_path = ROOT / "reports/upgrade_030.json"
upgrade = json.loads(report_path.read_text(encoding="utf-8"))
sources = {s["id"]: s for s in core.memory.sources()}
assert all(sources[s["id"]]["content_hash"] == s["content_hash"] and sources[s["id"]]["origin"] == s["origin"] for s in before["sources"])
assert core.config.values == before["config"]
assert core.trainer.active_id() == upgrade["neural"]["checkpoint"]
assert core.dialogue.active_id() == upgrade["dialogue"]["checkpoint"]
assert upgrade["neural"]["status"] == upgrade["dialogue"]["status"] == "aprovado"
model_info = active_models(core.config, core.db)
cp = safe_load(core.trainer.checkpoint_path(core.trainer.active_id()))
model = TACTransformer(ModelConfig(**cp["model_config"]))
model.load_state_dict(cp["model"])
assert weight_hash(model) == upgrade["neural"]["final_hash"]
assert {int(sid) for sid in cp["sampling"]["sources"]} == {40, upgrade["language_source_id"]}

story = sources[40]
original_copy = core.config.data / "inbox" / json.loads(story["metadata"])["filename"]
canonical = normalize(original_copy.read_text(encoding="utf-8-sig"))
recovered = "\n".join(c["text"] for c in core.memory.chunks(40))
assert recovered == canonical and text_hash(recovered) == story["content_hash"]

checks = []
for text, intent in [("eae", "greeting"), ("salve minha parceira", "greeting"), ("td bem contigo", "wellbeing"),
                     ("me diz teu nome", "identity"), ("vlw", "thanks"), ("não saquei", "clarify"),
                     ("tô triste", "feeling_low"), ("tô feliz", "feeling_good"), ("tô irritado", "frustration")]:
    answer = core.chat(text)
    assert answer.get("intent") == intent, (text, answer)
    checks.append({"question": text, "answer": answer})
answer = core.chat("eae, qual a capital do Brasil?")
assert answer["mode"] == "documental" and "Brasília" in answer["display_text"]
assert "valores diferentes" not in answer["text"]
checks.append({"question": "eae, qual a capital do Brasil?", "answer": answer})
for sid in (None, 40):
    answer = core.chat("Qual a relação entre Ren e Raphtalia?", source_id=sid)
    assert answer["mode"] == "documental" and answer["sources"]
    assert all(s["source_id"] == 40 for s in answer["sources"])
    assert all(any(proof in chunk["text"] for chunk in core.memory.chunks(40)) for proof in answer["evidence"])
    checks.append({"question": "Qual a relação entre Ren e Raphtalia?", "selected_source": sid, "answer": answer})
for text, intent in [("eae", "greeting"), ("vlw", "thanks")]:
    answer = core.chat(text, source_id=40)
    assert answer.get("intent") == intent
    checks.append({"question": text, "selected_source": 40, "answer": answer})
unknown = core.chat("Qual é o nome oficial de Território Inexistente XYZ?")
assert not unknown["sources"]
checks.append({"question": "Qual é o nome oficial de Território Inexistente XYZ?", "answer": unknown})

app = QApplication([])
app.setStyle("Fusion")
apply_theme(app, core.config)
window = Window(core)
window.timer.stop()
window.advanced_check.setChecked(True)
window.show()
app.processEvents()
window.refresh_training()
assert "aprovado" in window.train_label.text() and "40" in window.train_label.text()
assert window.checkpoint_rows[0]["id"] == core.trainer.active_id()
assert "ATIVO" in window.checkpoints_table.item(0, 4).text()
steps = [row["step"] for row in window.plot.rows]
assert steps == sorted(set(steps))
captures = ROOT / "reports/captures"
captures.mkdir(exist_ok=True)
window.nav.setCurrentRow(1)
window.learning_tabs.setCurrentIndex(1)
app.processEvents()
window.grab().save(str(captures / "030_conversa_guiada.png"))
window.nav.setCurrentRow(3)
app.processEvents()
window.grab().save(str(captures / "030_treinamento.png"))
window.nav.setCurrentRow(2)
app.processEvents()
window.grab().save(str(captures / "030_biblioteca.png"))
window.present_monitor(core.stats())
window.nav.setCurrentRow(4)
app.processEvents()
window.grab().save(str(captures / "030_monitoramento.png"))
window.close()

upgrade["classifier_examples"] = {s: len(examples(s)) for s in ("train", "validation", "test")}
atomic_json(report_path, upgrade)
result = {"sources_before": len(before["sources"]), "sources_after": len(sources), "original_sources_preserved": True,
          "preferences_preserved": True, "story_characters": len(canonical), "story_chunks": len(core.memory.chunks(40)),
          "story_text_and_emojis_preserved": True, "models": model_info,
          "selected_step": cp["step"], "selected_sampling": cp["sampling"],
          "neural_checkpoint_verified": True, "gui_metrics_after_reopen": True, "checks": checks}
atomic_json(ROOT / "reports/upgrade_030_validation.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=True, indent=2))
