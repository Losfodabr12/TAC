"""Percurso de importação na interface com processamento e banco reais isolados."""
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel,QPlainTextEdit,QPushButton,QMessageBox

from tac.ui import Window
from test_backup_ui import pump
from test_conversation_imports import record,write_jsonl
from test_imported_training import imported_training


def choose_modal(qtapp,caption,observed):
    def click():
        dialog=qtapp.activeModalWidget()
        if dialog is None:
            QTimer.singleShot(10,click)
            return
        observed.extend(w.toPlainText() for w in dialog.findChildren(QPlainTextEdit))
        next(b for b in dialog.findChildren(QPushButton) if b.text()==caption).click()
    QTimer.singleShot(10,click)


def test_gui_previews_then_imports_with_explicit_permission_and_no_training(core,qtapp,imported_training,monkeypatch):
    core.gate.set_enabled(False)
    path=write_jsonl(core.config.root/'nova_conversa.jsonl',[record('nova','train',question='Pode entrar?',answer='Oi.')])
    window=Window(core); window.timer.stop(); window.show()
    runs=len(core.db.query('SELECT * FROM runs'))
    old=len(core.db.query('SELECT * FROM dialogue_imports'))
    heartbeats=[]; timer=QTimer(); timer.timeout.connect(lambda:heartbeats.append(1)); timer.start(10)
    original=core.conversation.preview_import
    def slow_preview(*args):
        time.sleep(.2)  # Atraso controlado verifica que o trabalho não ocupa a interface.
        return original(*args)
    monkeypatch.setattr(core.conversation,'preview_import',slow_preview)
    observed=[]
    choose_modal(qtapp,'Permitir e importar registros válidos',observed)
    window.import_conversations(str(path))
    pump(qtapp,lambda:len(core.db.query('SELECT * FROM dialogue_imports'))==old+1 and window.job is None)
    timer.stop()
    assert core.gate.enabled and len(heartbeats)>=5
    assert 'Exemplos de resposta aceitos: 1' in observed[0]
    assert 'Nenhum treino foi iniciado' in window.notice.text()
    assert len(core.db.query('SELECT * FROM runs'))==runs
    assert core.conversation.active_id()==imported_training
    assert window.import_conversations_button.isEnabled()
    window.close()


def test_cancel_preview_preserves_permissions_and_cancel_training_does_not_start(core,qtapp,imported_training,monkeypatch):
    core.gate.set_enabled(False)
    window=Window(core); window.timer.stop()
    path=write_jsonl(core.config.root/'cancelar.jsonl',[record('cancelar','train',question='Posso cancelar?',answer='Oi.')])
    count=len(core.db.query('SELECT * FROM dialogue_imports'))
    choose_modal(qtapp,'Cancelar',[])
    window.import_conversations(str(path))
    pump(qtapp,lambda:window.job is None)
    assert not core.gate.enabled and len(core.db.query('SELECT * FROM dialogue_imports'))==count
    calls=[]
    monkeypatch.setattr(core.conversation,'train',lambda *a,**k:calls.append(k))
    def cancel_training():
        box=qtapp.activeModalWidget()
        if box is None:
            QTimer.singleShot(10,cancel_training); return
        assert 'Este treino vai usar 1 conversas suas e 8 exercícios do programa.' in box.text()
        next(b for b in box.buttons() if b.text()=='Cancelar').click()
    core.config.update(conversation_profile='current')
    QTimer.singleShot(10,cancel_training)
    window.train_conversation()
    pump(qtapp,lambda:window.job is None)
    assert not calls and not core.gate.enabled
    assert window.last_training_preview['counts']['user_groups']==1
    window.close()
