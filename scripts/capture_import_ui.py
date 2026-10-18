"""Capturas nativas; cancela as prévias. Nunca importa ou treina no acervo pessoal."""
import json
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
folder=ROOT/'reports/importador_100/capturas'
folder.mkdir(parents=True,exist_ok=True)
window.show()

def modal_capture(name):
    def take():
        dialog=app.activeModalWidget()
        if dialog is None:
            QTimer.singleShot(50,take)
            return
        dialog.grab().save(str(folder/(name+'.png')))
        dialog.reject()
    QTimer.singleShot(150,take)

def capture():
    for name,page in (('inicio','home'),('melhorar_tac','improve')):
        window.go(page); app.processEvents()
        window.grab().save(str(folder/(name+'.png')))
    modal_capture('formatos')
    window.show_conversation_examples()
    # Leitura real com o vocabulário ativo. Rejeita o diálogo de confirmação.
    preview=core.conversation.preview_import(ROOT/'examples/conversas_exemplo.jsonl')
    modal_capture('conferencia_importacao')
    window.show_import_preview(preview)
    prepared=core.conversation.prepare_training(profile='10m',curriculum='fluency')
    modal_capture('antes_de_treinar')
    window.confirm_dialogue_training(prepared)
    window.close()
    # Resultado REAL de teste CPU; o título e aviso distinguem o ambiente isolado.
    isolated_root=ROOT/'reports/importador_100/pytest_training/test_real_imported_steps_reten0/TAC'
    isolated=Core(isolated_root)
    demo=Window(isolated)
    demo.timer.stop(); demo.auto_armed=False
    demo.setWindowTitle('TAC — Resultado de teste isolado; não é o acervo pessoal')
    demo.show(); demo.go('improve')
    result=json.loads((isolated.config.data/'last_conversation_training.json').read_text(encoding='utf-8'))
    demo.conversation_result(result)
    demo.notice.setText('TESTE ISOLADO EM CPU: exercício pequeno com resposta “Oi.”. Não mede conversa livre. Seu modelo pessoal foi preservado.')
    demo.notice.show()
    app.processEvents()
    scroll=demo.pages.currentWidget()
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()
    demo.grab().save(str(folder/'resultado_teste_isolado.png'))
    demo.close()
    app.quit()

QTimer.singleShot(200,capture)
sys.exit(app.exec())
