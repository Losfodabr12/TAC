"""Capturas nativas do painel real, com automação e temporizador interrompidos."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QLockFile,QTimer
from PySide6.QtWidgets import QApplication
from tac.core import Core
from tac.ui import Window,apply_theme

app=QApplication([]);app.setStyle('Fusion');core=Core(ROOT)
lock=QLockFile(str(core.config.data/'tac.lock'))
if not lock.tryLock(0):raise RuntimeError('Aguarde os pilotos ou feche a outra janela antes de capturar.')
apply_theme(app,core.config);window=Window(core);window.timer.stop();window.auto_armed=False
window.resize(1240,840);window.show();window.advanced_check.setChecked(True)
folder=ROOT/'reports/experimentos_0130/capturas';folder.mkdir(exist_ok=True)


def capture():
    window.go('technical');app.processEvents();window.grab().save(str(folder/'treinamento_e_versoes.png'))
    def modal():
        dialog=app.activeModalWidget()
        if dialog is None:raise RuntimeError('Painel novo não abriu.')
        dialog.grab().save(str(folder/'comparacao.png'))
        dialog.tabs.setCurrentIndex(1);app.processEvents();dialog.grab().save(str(folder/'todos_os_pilotos.png'))
        dialog.show_run(0);app.processEvents();dialog.grab().save(str(folder/'detalhes.png'))
        dialog.selector.setCurrentIndex(dialog.selector.findData('closed_factual_verified_fraction'))
        dialog.tabs.setCurrentIndex(0);app.processEvents();dialog.grab().save(str(folder/'fatos_verificados.png'))
        dialog.reject()
    QTimer.singleShot(250,modal);window.show_conversation_research();window.close()


QTimer.singleShot(300,capture)
try:sys.exit(app.exec())
finally:lock.unlock()
