"""Capturas da interface no backend nativo do Windows, sem estudar ou treinar."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from tac.core import Core
from tac.ui import Window, apply_theme

app = QApplication([])
app.setStyle("Fusion")
core = Core(ROOT)
apply_theme(app, core.config)
window = Window(core)
window.timer.stop()
window.auto_armed = False
window.show()
folder = ROOT / "reports" / "interface_090" / "capturas"
folder.mkdir(parents=True, exist_ok=True)
screens = [('inicio','home'), ('conversar','chat'), ('meus_materiais','materials'),
           ('melhorar_tac','improve'), ('preferencias','preferences'), ('respostas_guiadas','guided'),
           ('treinamento_versoes','technical'), ('uso_computador','monitor'), ('ajustes','settings')]


def wait_for_answer():
    deadline = time.monotonic() + 45
    while window.job is not None:
        app.processEvents()
        time.sleep(.01)
        if time.monotonic() > deadline:
            window.cancel()
            raise TimeoutError('A resposta excedeu o limite da captura.')
    app.processEvents()


def capture(index=0):
    if index >= len(screens):
        window.advanced_check.setChecked(False)
        window.go('home')
        window.resize(880, 640)
        app.processEvents()
        window.grab().save(str(folder / 'inicio_janela_menor.png'))
        window.resize(1240, 840)
        def wizard():
            dialog = app.activeModalWidget()
            if dialog:
                dialog.grab().save(str(folder / 'como_comeco.png'))
                dialog.accept()
        QTimer.singleShot(200, wizard)
        window.explain_learning()
        window.close()
        return
    name, page = screens[index]
    window.go(page)
    if page == 'improve':
        recorded = core.db.query("SELECT detail FROM runs WHERE kind='conversation' AND status IN ('aprovado','rejeitado') ORDER BY started DESC LIMIT 1")
        if recorded:
            window.conversation_result(__import__('json').loads(recorded[0]['detail']))
    def save():
        if page == 'chat' and core.active_conversation_id():
            window.chat_mode.setCurrentIndex(2)
            window.input.setText('eae')
            window.send_chat()
            wait_for_answer()
        window.grab().save(str(folder / (name + '.png')))
        if page == 'chat':
            window.chat_mode.setCurrentIndex(0)
            local = core.db.query("SELECT id FROM sources WHERE title LIKE 'Brasil%' ORDER BY id LIMIT 1")
            if local:
                window.current_source = local[0]['id']
                window.input.setText('Qual é o nome oficial do Brasil?')
                window.send_chat()
                wait_for_answer()
            app.processEvents()
            window.grab().save(str(folder / 'consultar_materiais.png'))
        if page == 'materials':
            window.material_tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(folder / 'adicionar_arquivo.png'))
            window.material_tabs.setCurrentIndex(0)
            window.learning_tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(folder / 'materiais_guardados.png'))
            window.learning_tabs.setCurrentIndex(0)
        if page == 'improve':
            scroll = window.pages.currentWidget()
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            app.processEvents()
            window.grab().save(str(folder / 'resultado_treino.png'))
            scroll.verticalScrollBar().setValue(0)
        QTimer.singleShot(100, lambda: capture(index + 1))
    QTimer.singleShot(150, save)


QTimer.singleShot(300, capture)
sys.exit(app.exec())
