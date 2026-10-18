"""Aceitação da apresentação. Exemplos controlados aqui não medem inteligência."""
import json
import re
import time

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton

from tac.core import Core
from tac.presentation import answer_origin, training_summary
from tac.ui import Window
from test_backup_ui import pump


@pytest.fixture
def window(core, qtapp):
    value = Window(core)
    value.timer.stop()
    value.show()
    qtapp.processEvents()
    yield value
    if value.job:
        value.job.wait(10000)
        pump(qtapp, lambda: value.job is None)
    value.close()


def test_wikipedia_is_one_persistent_setting_in_both_directions(window, core, qtapp):
    assert not window.research_auto.isChecked()
    window.preferences_research_auto.setChecked(True)
    assert window.research_auto.isChecked()
    assert json.loads(core.config.path.read_text(encoding='utf-8'))['research_auto'] is True
    window.research_auto.setChecked(False)
    assert not window.preferences_research_auto.isChecked()
    reopened = Core(core.config.root)
    other = Window(reopened)
    other.timer.stop()
    assert not other.research_auto.isChecked() and not other.preferences_research_auto.isChecked()
    other.close()
    # Também acompanha uma alteração feita pela configuração já existente.
    core.config.update(research_auto=True)
    window.refresh_activity()
    assert window.research_auto.isChecked() and window.preferences_research_auto.isChecked()


def test_normal_navigation_import_card_and_all_advanced_functions(window):
    assert window.nav.currentRow() == 0
    assert window.pages.count() == 9
    assert all(window.nav.item(i).isHidden() for i in range(5, 9))
    assert window.import_conversations_button.isEnabled()
    assert window.import_conversations_button.text() == 'Importar conversas'
    assert window.chat_mode.view().isRowHidden(1)
    window.advanced_check.setChecked(True)
    assert all(not window.nav.item(i).isHidden() for i in range(9))
    for name in ('practice_button','auto_check','plot','checkpoints_table','monitor','device_setting','startup_check'):
        assert getattr(window, name).parent() is not None
    window.go('technical')
    window.advanced_check.setChecked(False)
    assert window.nav.currentRow() == 0


def test_consultation_hint_survives_refresh(window):
    window.chat_mode.setCurrentIndex(0)
    window.refresh_activity()
    assert window.chat_readiness.text() == 'Aqui a TAC busca nos seus textos. Isso não ensina nada a ela.'
    assert not window.chat_readiness.isHidden()


def test_existing_automatic_writing_permission_is_visible_without_changing_it(window, core):
    assert window.auto_study_notice.isHidden()
    core.config.update(auto_train=True)
    window.refresh_activity()
    assert not window.auto_study_notice.isHidden()
    assert 'automático de escrita' in window.auto_study_notice.text()
    assert not core.db.query('SELECT * FROM runs')


@pytest.mark.parametrize('suffix', ['.json','.jsonl','.exe'])
def test_unsupported_file_gives_immediate_guidance_without_queue(window, core, tmp_path, monkeypatch, suffix):
    path = tmp_path / ('dados' + suffix)
    path.write_text('{"messages":[]}', encoding='utf-8')
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda parent,title,text: messages.append(text))
    window.add_files([str(path)])
    assert not core.db.query('SELECT * FROM queue')
    assert window.job is None
    assert 'JSON/JSONL' in messages[0] and 'PDF, TXT, DOCX, Markdown e HTML' in messages[0]


def test_conversation_file_warning_cancel_and_explicit_text_import(window, core, qtapp, tmp_path):
    path = tmp_path / 'conversa.txt'
    path.write_text('Usuário: Oi.\nTAC: Como foi seu dia?\nUsuário: Foi tranquilo.', encoding='utf-8')
    captured = []
    def choose(caption):
        def click():
            box = qtapp.activeModalWidget()
            captured.append(box.text())
            next(b for b in box.buttons() if b.text() == caption).click()
        QTimer.singleShot(20, click)
    choose('Cancelar')
    window.add_files([str(path)])
    assert not core.db.query('SELECT * FROM queue')
    assert 'não entrará no treino de conversa' in captured[0]
    choose('Guardar como texto')
    window.add_files([str(path)])
    pump(qtapp, lambda: window.job is None)
    assert len(core.db.query('SELECT * FROM queue')) == 1
    assert not core.db.query('SELECT * FROM runs')
    assert not core.memory.sources()
    assert 'Nenhum treino' in window.notice.text()


def test_first_steps_has_direct_working_action(window, qtapp):
    observed = []
    def click():
        dialog = qtapp.activeModalWidget()
        texts = [w.text() for w in dialog.findChildren(QLabel)]
        observed.extend(texts)
        next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Adicionar site').click()
    QTimer.singleShot(20, click)
    window.explain_learning()
    assert len([s for s in observed if re.match(r'^[1-4]\. ',s)]) == 4
    assert window.nav.currentRow() == window.PAGE_KEYS.index('materials')
    assert window.material_tabs.currentIndex() == 0


@pytest.mark.parametrize('mode,sources,expected', [
    ('documental', [{'id':1}], 'Resposta dos seus documentos'),
    ('documental', [], 'Consulta aos seus documentos sem resposta'),
    ('diálogo neural experimental', [], 'Resposta da rede da TAC'),
    ('conversa guiada', [], 'Resposta preparada do programa'),
    ('consulta web documental', [{'id':1}], 'Resposta da Wikipédia'),
    ('consulta web documental', [], 'Consulta à Wikipédia sem resposta'),
    ('neural experimental', [], 'Continuação criada pela rede de escrita'),
])
def test_each_answer_identifies_actual_origin(window, core, qtapp, monkeypatch, mode, sources, expected):
    # Stub só para conferir os rótulos de apresentação e o caminho assíncrono.
    monkeypatch.setattr(core, 'chat', lambda *a,**k: {'mode':mode,'sources':sources,'text':'Resposta de teste da interface.'})
    window.input.setText('Uma pergunta de teste.')
    window.send_chat()
    pump(qtapp, lambda: window.job is None)
    assert 'TAC · ' + expected in window.transcript.toPlainText()
    assert answer_origin({'mode':mode,'sources':json.dumps(sources)}) == expected


def test_result_translation_preserves_decision_and_does_not_claim_unknown_comparisons(window, core):
    result = {'status':'rejeitado','steps':7,'selected_step':4,'device':'cuda',
              'baseline':{'loss':.1,'exact_accuracy':.9},'validation':{'loss':.3,'exact_accuracy':.6},
              'fluency_baseline':{'rule_accuracy':.1},'fluency_validation':{'rule_accuracy':.2},
              'promotion_checks':{'loss_basica':False,'conversas_revisadas':False}}
    original = json.dumps(result,sort_keys=True)
    window.conversation_result(result)
    text = window.train_label.text()
    assert 'Rejeitado' in text and 'O que melhorou' in text and 'O que piorou' in text
    assert 'verificação simples do assunto' in text and 'Acertos nas respostas básicas' in text
    assert 'nenhuma versão aprovada' in text
    assert all(s not in text for s in ('cuda','loss_basica','checkpoint','KL'))
    assert 'loss_basica' in window.technical_train_label.text()
    assert original == json.dumps(result,sort_keys=True)
    assert not core.db.query("SELECT * FROM meta WHERE key='active_conversation'")
    interrupted = training_summary({'status':'interrompido'},'conversation','Versão anterior')
    assert 'pendente' in interrupted and 'Versão anterior' in interrupted
    equal = training_summary({'status':'aprovado','baseline':1.,'validation':1.},'neural','Versão ativa')
    assert 'ficaram iguais ou não há comparação suficiente' in equal


def test_progress_hides_technical_values_and_gui_remains_responsive(window, core, qtapp):
    core.gate.set_enabled(True)
    heartbeats = []
    timer = QTimer()
    timer.timeout.connect(lambda: heartbeats.append(1))
    timer.start(10)
    def controlled_work(progress):
        progress(100, 'Diálogo 100/1000 • loss resposta 0.2 • cuda')
        time.sleep(.2)
        return 'Atividade de teste concluída.'
    window.current_training_kind = 'conversation'
    window.start_job('Treinamento neural',controlled_work,learning=True)
    pump(qtapp, lambda: window.job is None)
    timer.stop()
    assert len(heartbeats) >= 5
    assert 'Etapa 2 de 3' in window.train_label.text()
    assert 'cuda' not in window.train_label.text()
    assert 'cuda' in window.technical_train_label.text()
    assert not core.db.query('SELECT * FROM runs')


def test_saving_simple_preferences_does_not_change_training_settings(window, core):
    before = dict(core.config.values)
    window.theme_setting.setCurrentIndex(1)
    window.settings_widgets['font_size'].setValue(14)
    window.interactions_check.setChecked(True)
    window.save_preferences()
    assert core.config.get('theme') == 'light'
    assert core.config.get('save_interactions') is True
    assert all(core.config.get(k) == v for k,v in before.items() if k not in ('theme','font_size','save_interactions'))


def test_training_button_keeps_existing_engine_arguments(window, core, qtapp, monkeypatch):
    core.gate.set_enabled(True)
    calls = []
    def controlled_train(progress,**kwargs):
        calls.append(kwargs)
        return {'status':'interrompido','baseline':{},'validation':{}}
    monkeypatch.setattr(core.conversation,'train',controlled_train)
    prepared = {'counts':{'user_groups':2,'user_examples':4,'program_examples':100,
                         'reserved':{s:{'groups':1,'examples':2} for s in ('validation','test')}},
                'profile':'10m','curriculum':'fluency','resume':None}
    monkeypatch.setattr(core.conversation,'prepare_training',lambda *a:prepared)
    def confirm():
        box = qtapp.activeModalWidget()
        if box is None:
            QTimer.singleShot(10,confirm)
            return
        assert 'Este treino vai usar 2 conversas suas e 100 exercícios do programa.' in box.text()
        next(b for b in box.buttons() if b.text()=='Iniciar treino').click()
    QTimer.singleShot(10,confirm)
    window.train_conversation()
    pump(qtapp, lambda: bool(calls) and window.job is None)
    assert calls == [{'steps':core.config.get('conversation_steps'),'seconds':core.config.get('conversation_seconds'),
                      'profile':'10m','curriculum':'fluency','resume':None,'prepared':prepared}]
    assert not core.db.query('SELECT * FROM runs')
