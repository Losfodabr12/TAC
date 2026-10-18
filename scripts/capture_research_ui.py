"""Capturas nativas da fase 0.12, sem estudar, gerar respostas ou treinar."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from tac.core import Core
from tac.ui import Window,apply_theme

app=QApplication([]);app.setStyle('Fusion')
core=Core(ROOT);apply_theme(app,core.config)
window=Window(core);window.timer.stop();window.auto_armed=False
window.resize(1240,840);window.show();window.advanced_check.setChecked(True)
folder=ROOT/'reports/research_0120/capturas';folder.mkdir(exist_ok=True)


def capture():
    window.go('technical');app.processEvents()
    window.grab().save(str(folder/'treinamento_e_versoes.png'))
    def modal():
        dialog=app.activeModalWidget()
        if dialog is None:raise RuntimeError('Resultados não abriram.')
        dialog.grab().save(str(folder/'resultados.png'));dialog.reject()
    QTimer.singleShot(200,modal);window.show_conversation_research()
    window.go('settings');app.processEvents()
    scroll=window.pages.currentWidget()
    scroll.ensureWidgetVisible(window.settings_widgets['conversation_learning_rate'],0,180)
    app.processEvents();window.grab().save(str(folder/'ajustes_de_conversa.png'))
    window.close()


QTimer.singleShot(500,capture)
sys.exit(app.exec())
