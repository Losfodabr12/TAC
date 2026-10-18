"""Visão simples da TAC. Reutiliza as ações existentes sem alterar treinadores."""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget, QGridLayout,
)

from .presentation import friendly_progress, training_summary
from .import_ui import ConversationImportInterface


class SimpleInterface(ConversationImportInterface):
    PAGE_NAMES = ('Início', 'Conversar', 'Meus materiais', 'Melhorar a TAC', 'Preferências',
                  'Respostas guiadas', 'Treinamento e versões', 'Uso do computador', 'Ajustes e manutenção')
    PAGE_KEYS = ('home', 'chat', 'materials', 'improve', 'preferences', 'guided', 'technical', 'monitor', 'settings')

    def go(self, key):
        self.nav.setCurrentRow(self.PAGE_KEYS.index(key))

    def reorganize_ui(self):
        from .ui import button, label
        old = [self.pages.widget(i) for i in range(self.pages.count())]
        for widget in old:
            self.pages.removeWidget(widget)
        guided = self.learning_tabs.widget(1)
        self.learning_tabs.removeTab(1)
        self.learning_tabs.setTabText(0, 'Adicionar materiais')
        self.material_tabs.setTabText(1, 'Adicionar arquivo')
        self.learning_tabs.addTab(old[2], 'Guardados')
        self.auto_study_notice = label('O treino automático de escrita está permitido. Ele poderá começar depois que você guardar materiais. Essa permissão fica nas Ferramentas avançadas.', True)
        old[1].widget().layout().insertWidget(3, self.auto_study_notice)
        self.activity_controls.insertWidget(1, self.learning_button)
        self.technical_train_label = self.train_label
        self.settings_widgets['font_size'].setObjectName('normalFontSize')
        self.build_home()
        self.pages.addWidget(old[0])
        self.pages.addWidget(old[1])
        self.build_improve()
        self.build_preferences()
        self.pages.addWidget(guided)
        self.pages.addWidget(old[3])
        self.pages.addWidget(old[4])
        self.pages.addWidget(old[5])
        self.nav.clear()
        self.nav.addItems(self.PAGE_NAMES)
        self._normal_advanced_widgets = []
        self.chat_mode.setItemText(0, 'Consultar meus materiais')
        self.chat_mode.setItemText(1, 'Continuar um texto — experimental')
        self.chat_mode.setItemText(2, 'Conversar com a TAC — experimental')
        if (self.core.config.data/'tac_v2/training_result.json').exists():
            self.chat_mode.setItemText(2, 'Conversar com a TAC v2 — experimental')
            self.setWindowTitle('TAC v2 — conversa neural experimental')
            self.chat_mode.setCurrentIndex(2)
        if (self.core.config.data/'tac_v3/deployment.json').exists():
            self.chat_mode.setItemText(2, 'Conversar — comparar TAC v2 e v3')
            self.setWindowTitle('TAC — comparação v2 / v3 experimental')
        self.chat_mode.setToolTip('Escolha como a TAC responderá. A origem aparece em cada resposta.')
        self.transcript.setPlainText('TAC\nEscolha o modo acima. Cada resposta mostra de onde veio.\nUse “Como começo?” para conhecer os caminhos.')
        self.chat_tools.show()
        self.chat_readiness.show()
        self.learning_tabs.setMaximumHeight(16777215)
        self.queue_table.setColumnHidden(0, True)
        self.queue_table.setColumnWidth(2, 210)
        self.queue_table.setColumnWidth(3, 190)
        self.sources_table.setColumnHidden(0, True)
        self.sources_table.setColumnHidden(4, True)
        # Conserva todas as ações da Biblioteca; recolhe as menos frequentes.
        self.library_extra = QWidget()
        extra = QGridLayout(self.library_extra)
        extra.setContentsMargins(0, 0, 0, 0)
        extras = []
        for index in reversed(range(self.library_actions.count())):
            item = self.library_actions.itemAt(index)
            widget = item.widget()
            if isinstance(widget, QPushButton) and widget.text() != 'Usar no chat':
                self.library_actions.removeWidget(widget)
                if widget.text() == 'Reorganizar contexto':
                    old[3].widget().layout().addWidget(widget)
                else:
                    extras.insert(0, widget)
        for index, widget in enumerate(extras):
            extra.addWidget(widget, index // 2, index % 2)
        self.library_extra.hide()
        library_layout = old[2].widget().layout()
        library_layout.insertWidget(5, self.library_extra)
        self.library_actions.addWidget(button('Mais ações', lambda: self.library_extra.setVisible(not self.library_extra.isVisible())))
        replacements = {
            'Usar no chat':'Perguntar sobre este material',
            'Adicionar documentos':'Adicionar arquivo',
            'Reorganizar contexto':'Reorganizar índices',
            'Atualizar fonte':'Atualizar material', 'Excluir fonte':'Excluir material',
            'Ver correções':'Correções salvas',
            'Consultar último material estudado →':'Perguntar sobre o último material',
            'Interromper estudo':'Parar',
            'Tentar novamente e estudar':'Tentar novamente',
            'Entenda a diferença em um minuto':'Como começo?',
            'Como funciona?':'Como começo?',
            'Histórico autorizado':'Conversas salvas',
            'Praticar conversas agora':'Treinar escolha de respostas preparadas',
            'Voltar à conversa →':'Voltar à conversa',
            'Permitir treinamento neural automático após estudar materiais':'Treinar escrita automaticamente após guardar materiais',
            'Autorizar treino automático também ao abrir a TAC':'Autorizar treino de escrita também ao abrir a TAC',
            'Salvar configurações':'Salvar ajustes avançados',
            'Exportar backup completo':'Criar cópia de segurança',
            'Importar backup e reiniciar':'Recuperar uma cópia e reiniciar',
        }
        for widget in self.findChildren(QPushButton) + self.findChildren(QCheckBox):
            if widget.text() in replacements:
                widget.setText(replacements[widget.text()])
        for widget in old[1].findChildren(QLabel):
            text = widget.text()
            if text == 'Ensinar':
                widget.setText('Meus materiais')
            elif text.startswith('Escolha o que você quer melhorar.'):
                widget.setText('Adicione um arquivo ou site. Depois guarde o conteúdo para consultar.')
            elif text.startswith('Guardar conhecimento:'):
                widget.setText('Guardar permite consultar o texto. Isso não treina a conversa da TAC.')
            elif text == 'Praticar conversas':
                widget.setText('Escolher respostas preparadas')
        for widget in old[2].findChildren(QLabel):
            if widget.text() == 'Biblioteca de Conhecimento':
                widget.setText('Materiais guardados')
            elif widget.text().startswith('Trechos e fontes persistentes.'):
                widget.setText('Pesquise, abra um material ou faça uma pergunta sobre ele.')
            elif widget.text().startswith('Excluir fontes remove'):
                widget.setText('Excluir remove o texto guardado e suas referências. O arquivo original fica intacto. Um treino anterior pode continuar influenciado pelo texto.')
        for widget in guided.findChildren(QLabel):
            if widget.text() == 'Praticar conversas':
                widget.setText('Escolher respostas preparadas')
        self.terms.setText('Autorizo guardar este site e conferi seus termos de acesso. Os limites do site serão respeitados.')
        self.history_table.setHorizontalHeaderLabels(['Data', 'Origem', 'Resultado', 'Detalhe'])
        self.settings_widgets['font_size'].setToolTip('Tamanho das letras da interface.')
        self.library_query.setPlaceholderText('Pesquisar nos seus textos…')
        self.drop.setText('Arraste PDF, TXT, DOCX, Markdown ou HTML\nou clique em Adicionar arquivo')
        self.study_stage.setText('Os materiais ficam disponíveis depois que forem guardados.')
        self.activity_detail.hide()
        self.activity_counts.hide()
        self._duration_cache = {}
        self.training_phase = 'prepare'
        self.current_training_kind = None
        self.chat_mode_changed(self.chat_mode.currentIndex())
        self.refresh_training_help()

    def build_home(self):
        from .ui import button, label
        layout = self.page('Como você quer usar a TAC?', 'Consulte seus materiais ou experimente a conversa. São caminhos diferentes.')
        row = QHBoxLayout()
        row.addWidget(button('Consultar meus materiais', lambda: self.open_chat(0), True))
        row.addWidget(button('Conversar com a TAC', lambda: self.open_chat(2)))
        layout.addLayout(row)
        layout.addWidget(label('Onde cada coisa vai'))
        mapping = [
            ('Textos e livros', 'Meus materiais', 'Guarda para consultar. O treino de escrita é separado.'),
            ('Endereços de sites', 'Meus materiais', 'Guarda o conteúdo autorizado para consulta.'),
            ('Conversas de exemplo', 'Melhorar a TAC → Importar conversas', 'Entram no treino de conversa quando você iniciar o treino.'),
            ('Exercícios do programa', 'Melhorar a TAC', 'Treinam a geração de respostas e o contexto.'),
            ('Correções que você salvar', 'Correções salvas', 'Podem entrar no treino de escrita, se autorizadas.'),
            ('Conversas guardadas', 'Conversas salvas', 'Servem para consultar o histórico. Não treinam.'),
        ]
        grid = QGridLayout()
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 5)
        for col, text in enumerate(('Material', 'Onde colocar', 'O que acontece')):
            grid.addWidget(label(text), 0, col)
        for index, values in enumerate(mapping, 1):
            for col, text in enumerate(values):
                item = label(text, col != 0)
                item.setMinimumHeight(44)
                grid.addWidget(item, index, col)
        layout.addLayout(grid)
        note = label('Guardar material e treinar são coisas diferentes. Guardar permite consultar o texto. Treinar tenta melhorar uma habilidade. Um treino pode não ser aprovado.')
        note.setObjectName('notice')
        layout.addWidget(note)
        actions = QHBoxLayout()
        actions.addWidget(button('Como começo?', self.explain_learning, True))
        actions.addWidget(button('Ver formas de aprendizado', lambda: self.go('improve')))
        layout.addLayout(actions)
        layout.addStretch()

    def build_improve(self):
        from .ui import button, label
        layout = self.page('Melhorar a TAC', 'Escolha qual habilidade você quer tentar melhorar. Nenhum treino garante uma melhora.')
        for title, text, training_button, attr in (
            ('Escrita com seus textos — experimental',
             'Usa: textos guardados e correções autorizadas.\nMuda: a rede que continua textos. Não muda a conversa principal.',
             self.train_button, 'reading_time_label'),
            ('Conversa com seus exemplos — experimental',
             'Usa: suas conversas importadas e os exercícios do programa.\nMuda: a rede usada em “Conversar com a TAC”. Materiais da Biblioteca e histórico salvo não entram aqui.',
             self.conversation_train_button, 'conversation_time_label'),
        ):
            if attr == 'conversation_time_label' and (self.core.config.data/'tac_v2/encoded_ready.json').exists():
                title = 'TAC v2 — linguagem e conversação'
                text = 'Usa o corpus público preparado de português e conversas. Uma retomada continua a mesma execução com seus dados fixos.'
            card = QWidget()
            card.setObjectName('learningCard')
            body = QVBoxLayout(card)
            body.addWidget(label(title))
            body.addWidget(label(text, True))
            training_button.setText('Ensinar a escrever com meus textos' if attr == 'reading_time_label' else 'Ensinar a conversar com os exemplos escolhidos')
            if attr == 'conversation_time_label' and (self.core.config.data/'tac_v2/progress.json').exists():
                training_button.setText('Retomar treinamento TAC v2')
            body.addWidget(training_button)
            timing = label('', True)
            setattr(self, attr, timing)
            body.addWidget(timing)
            layout.addWidget(card)
        card = QWidget()
        card.setObjectName('learningCard')
        body = QVBoxLayout(card)
        body.addWidget(label('Conversas de exemplo'))
        body.addWidget(label('Importe conversas em TXT ou JSONL. Confira o relatório antes de guardar. Importar não inicia um treino.', True))
        self.import_conversations_button = button('Importar conversas', lambda: self.import_conversations(), True)
        body.addWidget(self.import_conversations_button)
        self.import_cancel_button = button('Cancelar conferência ou importação', self.cancel)
        self.import_cancel_button.hide()
        body.addWidget(self.import_cancel_button)
        actions = QHBoxLayout()
        actions.addWidget(button('Ver formatos e exemplos', self.show_conversation_examples))
        actions.addWidget(button('Escolher arquivos para o próximo treino', self.manage_conversation_imports))
        body.addLayout(actions)
        self.imports_status = label('', True)
        body.addWidget(self.imports_status)
        body.addWidget(label('Use o mesmo group nas variações de uma conversa, inclusive em arquivos diferentes. Só habilitar um arquivo não inicia treino.', True))
        layout.addWidget(card)
        self.training_material_label = label('Antes de treinar, você verá quantas conversas suas e quantos exercícios do programa serão usados.', True)
        if (self.core.config.data/'tac_v2/encoded_ready.json').exists():
            self.training_material_label.setText('TAC v2 usa os dados públicos preparados. Os exercícios e arquivos de conversa da geração antiga ficam fora desta execução.')
        layout.addWidget(self.training_material_label)
        self.refresh_import_card()
        self.training_progress = QProgressBar()
        self.training_progress.hide()
        layout.addWidget(self.training_progress)
        controls = QHBoxLayout()
        controls.addWidget(self.pause_button)
        controls.addWidget(self.train_stop_button)
        layout.addLayout(controls)
        self.train_label = label('Nenhum treino iniciado nesta sessão. Os materiais já guardados podem ser consultados sem treinar.')
        layout.addWidget(self.train_label)
        self.active_versions_label = label('', True)
        layout.addWidget(self.active_versions_label)
        layout.addWidget(button('Ver detalhes técnicos', lambda: self.go('technical')))

    def build_preferences(self):
        from .ui import button, label
        layout = self.page('Preferências', 'Suas escolhas ficam guardadas neste computador.')
        # Move os mesmos controles. Não mantém duas cópias independentes.
        form = QFormLayout()
        for text, widget in (('Tamanho das letras', self.settings_widgets['font_size']),
                             ('Tema', self.theme_setting), ('', self.interactions_check)):
            row = self.advanced_settings_form.takeRow(widget)
            if row.labelItem and row.labelItem.widget():
                row.labelItem.widget().deleteLater()
            form.addRow(text, widget)
        self.theme_setting.setItemData(0, 'dark')
        self.theme_setting.setItemData(1, 'light')
        self.theme_setting.setItemText(0, 'Escuro')
        self.theme_setting.setItemText(1, 'Claro')
        self.interactions_check.setText('Guardar minhas conversas neste computador')
        layout.addLayout(form)
        layout.addWidget(label('Conversas salvas servem para consultar o histórico. Não são usadas nos treinos.', True))
        self.preferences_research_auto = QCheckBox('Consultar a Wikipédia quando faltar informação')
        self.preferences_research_auto.setChecked(self.core.config.get('research_auto'))
        self.preferences_research_auto.toggled.connect(self.set_research_auto)
        layout.addWidget(self.preferences_research_auto)
        layout.addWidget(label('É a mesma opção da tela Conversar. A consulta não treina a conversa da TAC.', True))
        layout.addWidget(button('Salvar preferências', self.save_preferences, True))
        layout.addWidget(label('Cópias de segurança'))
        row = QHBoxLayout()
        row.addWidget(button('Criar cópia de segurança', self.backup))
        row.addWidget(button('Recuperar uma cópia', self.restore_backup))
        row.addWidget(button('Conversas salvas', self.manage_interactions))
        layout.addLayout(row)
        layout.addWidget(button('Abrir ajustes avançados', lambda: self.go('settings')))
        layout.addStretch()

    def save_preferences(self):
        from .ui import apply_theme
        if self.job:
            self.notify_error('Aguarde a atividade terminar para salvar as preferências.')
            return
        try:
            self.core.config.update(font_size=self.settings_widgets['font_size'].value(),
                                    theme=self.theme_setting.currentData(), save_interactions=self.interactions_check.isChecked())
            apply_theme(QApplication.instance(), self.core.config)
            self.statusBar().showMessage('Preferências salvas.')
        except Exception as exc:
            self.notify_error(str(exc))

    def set_research_auto(self, enabled):
        previous = self.core.config.get('research_auto')
        try:
            if enabled != previous:
                self.core.config.update(research_auto=enabled)
        except Exception as exc:
            enabled = previous
            self.notify_error(str(exc))
        for name in ('research_auto', 'preferences_research_auto'):
            widget = getattr(self, name, None)
            if widget:
                with QSignalBlocker(widget):
                    widget.setChecked(enabled)

    def open_chat(self, mode):
        if mode == 2 and not self.core.active_conversation_id():
            self.notice.setText('Ainda não há uma versão de conversa aprovada. Veja o treino com exemplos do programa em Melhorar a TAC.')
            self.notice.show()
            self.go('improve')
            return
        self.chat_mode.setCurrentIndex(mode)
        self.go('chat')

    def active_versions_text(self):
        rows = self.core.db.query("SELECT m.key,c.created FROM meta m JOIN checkpoints c ON c.id=m.value WHERE m.key IN ('active_conversation','active_checkpoint','active_dialogue')")
        versions = {}
        for row in rows:
            try:
                versions[row['key']] = 'versão de ' + datetime.fromisoformat(row['created']).astimezone().strftime('%d/%m/%Y às %H:%M:%S')
            except ValueError:
                versions[row['key']] = 'versão salva anteriormente'
        return ('Conversar com a TAC: ' + versions.get('active_conversation', 'nenhuma versão aprovada')
                + '\nContinuação de texto: ' + versions.get('active_checkpoint', 'nenhuma versão aprovada')
                + '\nRespostas preparadas: ' + versions.get('active_dialogue', 'nenhuma versão aprovada'))

    def refresh_training_help(self):
        if not hasattr(self, '_duration_cache'):
            return
        for kind, widget, limit in (('neural', self.reading_time_label, self.core.config.get('train_seconds')),
                                    ('conversation', self.conversation_time_label, self.core.config.get('conversation_seconds'))):
            v3_path=self.core.config.data/'tac_v3/progress.json'
            v3=getattr(self,'_conversation_v3_deployment',None)
            if kind=='conversation' and v3_path.exists():
                v3_state=json.loads(v3_path.read_text(encoding='utf-8'))
                if v3_state['phase']!='complete' or (v3 and self.core.active_conversation_id()==v3['checkpoint']):
                    widget.setText(f"TAC v3: {v3_state['global_step']} atualizações salvas. "
                                   + ('Orçamento concluído; qualidade experimental.' if v3_state['phase']=='complete' else 'Execução recuperável de linguagem e instruções.'))
                    continue
            continuation=self.core.config.data/'tac_v2_conversation_20261018/progress.json'
            if kind=='conversation' and continuation.exists():
                state=json.loads(continuation.read_text(encoding='utf-8'))
                finished=(continuation.parent/'result.json').exists()
                widget.setText(f"Conversação v2: {state['step']} atualizações; execução concluída, com limitações registradas."
                               if finished else f"Conversação v2: {state['step']}/{state['max_steps']} atualizações salvas; checkpoints recuperáveis.")
                continue
            v2_path=self.core.config.data/'tac_v2/progress.json'
            if kind=='conversation' and v2_path.exists():
                progress=json.loads(v2_path.read_text(encoding='utf-8'))
                widget.setText('TAC v2: treinamento concluído. A versão é experimental.' if progress['phase']=='complete'
                               else f"TAC v2: {progress['global_step']} atualizações salvas. A execução continua de um checkpoint novo ao retomar.")
                continue
            runs = self.core.db.query("SELECT id,started,finished,config FROM runs WHERE kind=? AND status IN ('aprovado','rejeitado') ORDER BY started DESC LIMIT 30", (kind,))
            signature = (runs[0]['id'] if runs else '', self.core.config.get('conversation_steps'), self.core.config.get('conversation_curriculum'), self.core.config.get('epochs'), self.core.config.get('steps_per_epoch'), self.core.config.get('device'))
            cached = self._duration_cache.get(kind)
            if not cached or cached[0] != signature:
                durations = []
                for run in runs:
                    params = json.loads(run['config'])
                    if kind == 'conversation' and (params.get('steps') != self.core.config.get('conversation_steps')
                         or params.get('batch_size') != min(8, self.core.config.get('batch_size'))
                         or params.get('accumulation') != self.core.config.get('accumulation')
                         or params.get('curriculum') != ('fluency' if self.core.config.get('conversation_curriculum') == 'fluency' else None)
                         or params.get('profile') != ('10m' if self.core.config.get('conversation_profile') == '10m' else None)):
                        continue
                    if kind == 'neural' and any(params.get(k) != self.core.config.get(k) for k in ('epochs','steps_per_epoch','context','width','layers','batch_size','accumulation','device')):
                        continue
                    if not run['finished']:
                        continue
                    duration = (datetime.fromisoformat(run['finished']) - datetime.fromisoformat(run['started'])).total_seconds()
                    if duration > 0:
                        durations.append(duration)
                text = (f'Tempo recente: mediana de {round(statistics.median(durations))} segundos em {len(durations)} execuções semelhantes.'
                        if len(durations) >= 2 else 'Tempo usual: ainda sem histórico suficiente de execuções semelhantes.')
                self._duration_cache[kind] = (signature, text)
            widget.setText(self._duration_cache[kind][1] + f'\nLimite da etapa de treino: {limit} segundos. Preparação e verificação podem levar mais tempo.')
        self.active_versions_label.setText(self.active_versions_text())

    def present_training_result(self, result, kind):
        from .presentation import imported_results_text
        self.current_training_kind = kind
        text = training_summary(result, kind, self.active_versions_text())
        comparison = imported_results_text(result.get('imported_evaluation'))
        self.train_label.setText(text + ('\n\n' + comparison if comparison else ''))
        self.active_versions_label.hide()
        self.training_progress.hide()
        compact = {k:v for k,v in result.items() if k in ('status','steps','selected_step','device','detail','promotion_checks')}
        for key in ('baseline','validation','context_validation','natural_baseline','natural_validation',
                    'session_validation','fluency_validation'):
            value = result.get(key)
            compact[key] = ({k:v for k,v in value.items() if k in ('loss','token_accuracy','exact_matches','exact_accuracy',
                'rule_matches','turns','rule_accuracy','slot_drift_rate','immediate_accuracy','recall_accuracy')}
                if isinstance(value, dict) else value)
        self.technical_train_label.setText(json.dumps(compact, ensure_ascii=False, indent=2))
        if result.get('imported_evaluation'):
            compact['imported_evaluation'] = result['imported_evaluation']
            compact['imported_counts'] = result.get('imported_counts')
            self.technical_train_label.setText(json.dumps(compact, ensure_ascii=False, indent=2))
        self.refresh_training_help()

    def explain_learning(self):
        from .ui import button, label
        dialog = QDialog(self)
        dialog.setWindowTitle('Como começo?')
        dialog.resize(650, 560)
        layout = QVBoxLayout(dialog)
        for title, text, actions in (
            ('1. Escolha seu objetivo', 'Você quer consultar textos ou experimentar a conversa?',
             [('Consultar meus textos', lambda: self.open_chat(0)), ('Experimentar a conversa', lambda: self.open_chat(2))]),
            ('2. Coloque cada material no lugar certo', 'Textos e sites vão para Meus materiais. Exemplos de conversa vão para Importar conversas.',
             [('Adicionar arquivo', self.select_files), ('Adicionar site', self.open_site), ('Importar conversas', self.import_conversations)]),
            ('3. Veja se precisa treinar', 'Consultar documentos não exige treino. Cada treino usa materiais diferentes.',
             [('Ver formas de aprendizado', lambda: self.go('improve'))]),
            ('4. Confira a origem da resposta', 'A identificação mostra se veio da rede, dos seus documentos ou da Wikipédia.',
             [('Abrir conversa', lambda: self.go('chat'))]),
        ):
            layout.addWidget(label(title))
            layout.addWidget(label(text, True))
            row = QHBoxLayout()
            for caption, action in actions:
                def run(checked=False, action=action):
                    dialog.accept()
                    action()
                row.addWidget(button(caption, run))
            layout.addLayout(row)
        layout.addWidget(button('Fechar', dialog.accept))
        dialog.exec()

    def open_site(self):
        self.go('materials')
        self.learning_tabs.setCurrentIndex(0)
        self.material_tabs.setCurrentIndex(0)
        self.url.setFocus()

    def looks_like_conversation(self, path):
        if path.suffix.lower() not in ('.txt','.md','.markdown'):
            return False
        try:
            with path.open('rb') as stream:
                raw = stream.read(65536)
            from .documents import decode_text
            text = decode_text(raw)
            return len(re.findall(r'^\s*(?:usu[aá]rio|user|assistant|assistente|tac|humano|bot|voc[eê])\s*:', text, re.I|re.M)) >= 2
        except OSError:
            return False

    def confirm_material_file(self, path):
        if not self.looks_like_conversation(path):
            return True
        box = QMessageBox(self)
        box.setWindowTitle('Este arquivo parece conter conversas')
        box.setText(f'{path.name}\n\nGuardá-lo em Meus materiais permite consultar as falas e usá-las no treino de escrita.\nEle não entrará no treino de conversa.')
        keep = box.addButton('Guardar como texto', QMessageBox.ButtonRole.AcceptRole)
        help_button = box.addButton('Usar para ensinar a conversar', QMessageBox.ButtonRole.HelpRole)
        cancel = box.addButton('Cancelar', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() == help_button:
            self.import_conversations(str(path))
        return box.clickedButton() == keep
