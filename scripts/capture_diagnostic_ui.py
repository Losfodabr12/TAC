"""Captura nativa do botão instalado, sem treinar ou exportar novamente."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from tac.core import Core
from tac.ui import Window,apply_theme

app=QApplication([])
app.setStyle('Fusion')
core=Core(ROOT)
apply_theme(app,core.config)
window=Window(core)
window.timer.stop(); window.auto_armed=False
window.show(); window.advanced_check.setChecked(True); window.go('technical')
def capture():
    target=ROOT/'reports/diagnostico_110/treinamento_e_versoes.png'
    window.grab().save(str(target))
    window.close()
QTimer.singleShot(500,capture)
sys.exit(app.exec())
