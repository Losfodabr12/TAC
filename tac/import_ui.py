"""Interface da importação e da conferência de dados antes do treino."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDialog,QFileDialog,QHBoxLayout,QMessageBox,QPlainTextEdit,
                              QTabWidget,QVBoxLayout,QWidget)


def report_text(preview):
    r = preview['report']
    lines = [preview['filename'], '', f"Registros analisados: {r['records']}",
             f"Conversas aceitas: {r['groups']}", f"Exemplos de resposta aceitos: {r['accepted_examples']}"]
    for split,title in (('train','Treino'),('validation','Validação'),('test','Teste')):
        data = r['by_split'][split]
        lines.append(f"{title}: {data['groups']} conversas; {data['examples']} respostas.")
    lines += ['',f"Registros com rejeições: {r['rejected_records']}",f"Exemplos rejeitados com contagem conhecida: {r['rejected_examples']}",
              f"Registros ilegíveis, sem contagem de exemplos: {r['unknown_example_count']}",f"Duplicatas ignoradas: {r['duplicates']}"]
    for issue in r['rejected']:
        lines.append(f"Linha {issue['line']}, grupo {issue.get('group') or 'não identificado'}: {issue['reason']}")
    if r['warnings']:
        lines += ['', 'Avisos'] + list(dict.fromkeys(r['warnings']))
    return '\n'.join(lines)


class ConversationImportInterface:
    def import_conversations(self, path=None):
        if self.job:
            self.notify_error('Aguarde a atividade atual terminar.')
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self,'Importar conversas',str(self.core.config.root/'examples'),
                                                'Conversas (*.jsonl *.txt)')
        if not path:
            return
        self.import_cancel = threading.Event()
        self.start_job('Conferir conversas',lambda progress:self.core.conversation.preview_import(path,progress,self.import_cancel),
                       self.show_import_preview)

    def show_import_preview(self, preview):
        from .ui import button,label
        self.last_import_preview = preview
        dialog = QDialog(self)
        dialog.setWindowTitle('Confira suas conversas antes de importar')
        dialog.resize(780,600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('Importar guarda exemplos para o treino de conversa. Não inicia treino nem muda a versão em uso.'))
        text = QPlainTextEdit(report_text(preview))
        text.setReadOnly(True)
        layout.addWidget(text)
        actions = QHBoxLayout()
        confirm = button('Importar registros válidos' if self.core.gate.enabled else 'Permitir e importar registros válidos',dialog.accept,True)
        confirm.setEnabled(preview['report']['accepted_examples'] > 0)
        actions.addWidget(confirm)
        actions.addWidget(button('Salvar relatório',lambda:self.save_import_report(preview)))
        actions.addWidget(button('Cancelar',dialog.reject))
        layout.addLayout(actions)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if not self.core.gate.enabled:
                self.core.gate.set_enabled(True)
            self.after_current_job(lambda:self.start_job('Importar conversas',lambda progress:self.core.conversation.commit_import(preview),
                                                       self.import_finished,learning=True))

    def after_current_job(self, callback):
        if self.job is not None:
            self._after_current_job = callback
        else:
            callback()

    def import_finished(self, result):
        self.notice.setText(f"Importadas {result['groups']} conversas, com {result['accepted_examples']} exemplos de resposta. Nenhum treino foi iniciado. Confira o material antes de treinar em Melhorar a TAC.")
        self.notice.show()
        self.go('improve')
        self.refresh_import_card()

    def save_import_report(self, preview):
        path,_ = QFileDialog.getSaveFileName(self,'Salvar relatório de importação',str(self.core.config.root/'reports'/'conversas_importadas.txt'),'Texto (*.txt)')
        if path:
            target = Path(path).resolve()
            if not target.is_relative_to(self.core.config.root):
                self.notify_error('Salve o relatório dentro da pasta TAC.')
                return
            target.write_text(report_text(preview),encoding='utf-8')

    def show_conversation_examples(self):
        from .ui import button,label
        dialog = QDialog(self)
        dialog.setWindowTitle('Formatos para importar conversas')
        dialog.resize(800,650)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('Exemplos completos de formato. Cada resposta da TAC gera um exemplo; group mantém a conversa inteira no mesmo destino.'))
        tabs = QTabWidget()
        for name in ('conversas_exemplo.jsonl','conversas_exemplo.txt'):
            widget = QPlainTextEdit((Path(__file__).resolve().parents[1]/'examples'/name).read_text(encoding='utf-8'))
            widget.setReadOnly(True)
            tabs.addTab(widget,name)
        layout.addWidget(tabs)
        layout.addWidget(label('A conferência usa o limite da sua versão. Com contexto de 512 unidades de texto, a resposta cabe em até 507, incluindo espaço e encerramento. O histórico pode ser recortado. O chat gera até 128 unidades por resposta.'))
        layout.addWidget(button('Fechar',dialog.accept))
        dialog.exec()

    def refresh_import_card(self):
        if not hasattr(self,'imports_status'):
            return
        row = self.core.db.query('SELECT count(*) total,coalesce(sum(enabled),0) enabled FROM dialogue_imports')[0]
        self.imports_status.setText(f"{row['total']} arquivo(s) importado(s); {row['enabled']} habilitado(s). Antes de treinar, você verá quantas conversas e exercícios serão usados.")
        self.import_conversations_button.setEnabled(self.job is None)
        self.import_cancel_button.setVisible(self.job is not None and self.job_name in ('Conferir conversas','Importar conversas'))

    def manage_conversation_imports(self):
        from .ui import button,label,table,fill_table
        dialog = QDialog(self)
        dialog.setWindowTitle('Seus arquivos de conversa')
        dialog.resize(850,520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('Habilitar escolhe o material das próximas etapas de treino. Uma retomada usa os dados salvos daquela execução.'))
        grid = table(['Arquivo','Conversas','Respostas','Próximo treino'])
        layout.addWidget(grid)
        rows = []
        def refresh():
            rows[:] = self.core.conversation.import_store().list()
            display = [{**r,'name':Path(r['origin']).name,'state':'Usar' if r['enabled'] else 'Não usar'} for r in rows]
            fill_table(grid,display,['name','groups','examples','state'])
        def toggle():
            index = grid.currentRow()
            if 0 <= index < len(rows):
                try:
                    row = rows[index]
                    self.core.conversation.import_store().set_enabled(row['id'],not row['enabled'])
                    refresh()
                    self.refresh_import_card()
                except Exception as exc:
                    self.notify_error(str(exc))
        def report():
            index = grid.currentRow()
            if 0 <= index < len(rows):
                row = rows[index]
                box = QDialog(dialog)
                box.setWindowTitle('Relatório da importação')
                body = QVBoxLayout(box)
                text = QPlainTextEdit(report_text({'filename':Path(row['origin']).name,'report':json.loads(row['report'])}))
                text.setReadOnly(True)
                body.addWidget(text)
                body.addWidget(button('Fechar',box.accept))
                box.resize(760,560)
                box.exec()
        actions = QHBoxLayout()
        actions.addWidget(button('Usar / não usar no próximo treino',toggle))
        actions.addWidget(button('Ver relatório',report))
        actions.addWidget(button('Fechar',dialog.accept))
        layout.addLayout(actions)
        refresh()
        dialog.exec()

    def request_dialogue_training(self, resume=None):
        if self.job:
            self.notify_error('Aguarde a atividade atual terminar.')
            return
        profile = self.core.config.get('conversation_profile')
        profile = profile if profile != 'current' else None
        curriculum = 'fluency' if self.core.config.get('conversation_curriculum') == 'fluency' else None
        self.start_job('Conferir material do treino',lambda progress:self.core.conversation.prepare_training(profile,curriculum,resume),
                       self.confirm_dialogue_training)

    def confirm_dialogue_training(self, prepared):
        counts = prepared['counts']
        self.last_training_preview = prepared
        message = (f"Este treino vai usar {counts['user_groups']} conversas suas e {counts['program_examples']} exercícios do programa.\n\n"
                   f"Suas conversas de treino geram {counts['user_examples']} exemplos de resposta.\n")
        for split,title in (('validation','Validação'),('test','Teste')):
            value = counts['reserved'][split]
            message += f"{title}: {value['groups']} conversas; {value['examples']} respostas.\n"
        message += '\nValidação ajuda a escolher versões que preservam as habilidades antigas. Teste fica fora da seleção. Os critérios anteriores de aprovação continuam obrigatórios.'
        if prepared.get('skipped'):
            message += f"\n{len(prepared['skipped'])} exemplo(s) ficaram fora por sobreposição com os exercícios atuais."
        if prepared.get('resume'):
            message += '\nEsta retomada usa os mesmos dados salvos na execução anterior.'
        elif prepared.get('document_dataset'):
            message += '\nTambém usa os textos opcionais que você preparou, em lotes separados de escrita. Isso não transforma documentos em conversas.'
        self.training_material_label.setText(message.split('\n\n',1)[0])
        box = QMessageBox(self)
        box.setWindowTitle('Confira o material do treino de conversa')
        box.setText(message)
        go = box.addButton('Iniciar treino',QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton('Cancelar',QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() != go:
            return
        def start():
            self.current_training_kind = 'conversation'
            self.start_job('Treinamento neural',lambda progress:self.core.conversation.train(progress,
                steps=self.core.config.get('conversation_steps'),seconds=self.core.config.get('conversation_seconds'),
                profile=prepared['profile'],curriculum=prepared['curriculum'],resume=prepared['resume'],prepared=prepared),
                self.conversation_result,learning=True)
        self.after_current_job(start)
