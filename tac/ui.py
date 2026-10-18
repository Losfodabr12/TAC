from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QPointF, QSignalBlocker
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from . import __version__
from .backups import export_backup, import_backup
from .db import now
from .presentation import QUEUE_LABELS, RESULT_LABELS, explain_error, snapshot, answer_origin, friendly_progress, user_error
from .simple_ui import SimpleInterface


DARK = """
QWidget { background:#111827; color:#e5eaf4; font-family:'Segoe UI'; font-size:13px; }
QMainWindow { background:#0b1020; }
QLabel#title { font-size:26px; font-weight:700; color:#f3f7ff; }
QLabel#muted { color:#a8b6d0; }
QWidget#activityCard { background:#17283e; border:1px solid #3b5877; border-radius:9px; }
QWidget#learningCard { background:#172438; border:1px solid #344b65; border-radius:9px; }
QWidget#futureCard { background:#161e2d; border:1px solid #303e53; border-radius:9px; }
QWidget#learningCard QLabel,QWidget#futureCard QLabel { background:transparent; }
QLabel#activityTitle { font-size:14px; font-weight:600; color:#8de7cd; border:0; background:transparent; }
QLabel#notice { background:#192e43; border:1px solid #3b5877; border-radius:6px; padding:10px; }
QTabWidget::pane { border:1px solid #344560; }
QTabBar::tab { background:#20324d; color:#b9c9df; padding:9px 18px; }
QTabBar::tab:selected { background:#235454; color:#a1f5df; }
QListWidget#nav { background:#0b1020; border:0; font-size:14px; padding:12px 4px; }
QListWidget#nav::item { padding:16px 12px; margin:3px; border-radius:8px; }
QListWidget#nav::item:selected { background:#1e3a5f; color:#76e5d2; }
QPushButton { background:#253653; border:1px solid #3a5070; border-radius:7px; padding:9px 13px; }
QPushButton:hover { background:#314768; }
QPushButton:disabled { color:#7a879c; background:#1a2539; }
QPushButton#primary { background:#166d68; border-color:#24998f; font-weight:600; }
QPushButton#learningOn { background:#164c42; font-size:13px; padding:8px; }
QPushButton#learningOff { background:#493b38; font-size:13px; padding:8px; }
QLineEdit,QPlainTextEdit,QTextBrowser,QSpinBox,QDoubleSpinBox,QComboBox { background:#0c1526; border:1px solid #344560; border-radius:5px; padding:7px; selection-background-color:#287f82; }
QTableWidget { background:#0c1526; alternate-background-color:#142039; border:1px solid #344560; gridline-color:#273852; }
QHeaderView::section { background:#20324d; color:#c8d7ed; padding:8px; border:0; }
QProgressBar { border:1px solid #344560; border-radius:5px; text-align:center; min-height:19px; }
QProgressBar::chunk { background:#238b81; }
QScrollArea { border:0; }
QCheckBox { spacing:8px; padding:4px 0; }
QToolTip { background:#20324d; color:white; border:1px solid #49988c; }
"""


def apply_theme(app, config):
    style = DARK.replace("font-size:13px", f"font-size:{config.get('font_size')}px")
    if config.get("theme") == "light":
        replacements = {"#111827": "#f0f4fa", "#e5eaf4": "#182638", "#0b1020": "#e1e8f2",
                        "#f3f7ff": "#14243c", "#a8b6d0": "#51627a", "#0c1526": "#ffffff",
                        "#253653": "#d7e3f2", "#c8d7ed": "#182638", "#20324d": "#c5d5e9",
                        "#314768": "#b9d0e6", "#142039": "#eaf0f7", '#172438':'#e8eef6',
                        '#161e2d':'#e8edf4', '#17283e':'#e0edf6', '#192e43':'#e5edf8'}
        for old, new in replacements.items():
            style = style.replace(old, new)
    app.setStyleSheet(style)


class Job(QThread):
    result = Signal(object)
    failed = Signal(str)
    update = Signal(int, str)

    def __init__(self, core, name, callback, learning=False):
        super().__init__()
        self.core, self.name, self.callback, self.learning = core, name, callback, learning

    def run(self):
        try:
            self.result.emit(self.core.perform(self.name, lambda: self.callback(self.update.emit), self.learning))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DropArea(QLabel):
    files = Signal(list)

    def __init__(self):
        super().__init__("Arraste documentos aqui\nPDF · TXT · DOCX · Markdown · HTML\nOu use Selecionar documentos")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(95)
        self.setStyleSheet("border:2px dashed #3b637b; border-radius:12px; color:#afc5dd; padding:12px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.files.emit([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()


class MonitorJob(QThread):
    measured = Signal(object)

    def __init__(self, core):
        super().__init__()
        self.core = core

    def run(self):
        try:
            info = self.core.stats()
            torch = sys.modules.get("torch")
            info["cuda"] = torch.cuda.is_available() if torch else None
            if torch and info["cuda"]:
                info["tac_vram_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            if os.name == "nt":
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode == 0:
                    parts = result.stdout.strip().splitlines()[0].split(",")
                    info["gpu"] = {"nome": parts[0].strip(), "uso_percent": parts[1].strip(), "vram_used_mb": parts[2].strip(), "vram_total_mb": parts[3].strip()}
            self.measured.emit(info)
        except Exception as exc:
            self.measured.emit({"error": str(exc)})


class LossPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.setMinimumHeight(170)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0c1526"))
        painter.setPen(QColor("#9eb0ca"))
        painter.drawText(12, 22, "Loss real por avaliação • verde: treino • azul: validação")
        if not self.rows:
            painter.drawText(12, 65, "As métricas aparecerão após um treinamento.")
            return
        all_values = [r[k] for r in self.rows for k in ("train_loss", "val_loss") if r[k] is not None]
        if not all_values:
            return
        low, high = min(all_values), max(all_values)
        high = max(high, low + 0.01)
        for key, color in (("train_loss", "#67ddba"), ("val_loss", "#6d9fff")):
            points = []
            for i, row in enumerate(self.rows):
                value = row[key]
                if value is not None:
                    points.append(QPointF(30 + i * (self.width() - 55) / max(1, len(self.rows) - 1), self.height() - 22 - (value - low) / (high - low) * (self.height() - 65)))
            painter.setPen(QPen(QColor(color), 2))
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            for point in points:
                painter.drawEllipse(point, 3, 3)
        painter.setPen(QColor("#9eb0ca"))
        painter.drawText(5, 44, f"{high:.2f}")
        painter.drawText(5, self.height() - 6, f"{low:.2f}")


def label(text, muted=False):
    item = QLabel(text)
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setWordWrap(True)
    if muted:
        item.setObjectName("muted")
    return item


def button(text, callback, primary=False):
    item = QPushButton(text)
    item.clicked.connect(callback)
    if primary:
        item.setObjectName("primary")
    return item


def table(headers):
    item = QTableWidget(0, len(headers))
    item.setHorizontalHeaderLabels(headers)
    item.setAlternatingRowColors(True)
    item.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    item.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    item.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    item.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    item.horizontalHeader().setStretchLastSection(True)
    item.verticalHeader().setVisible(False)
    return item


def fill_table(widget, rows, keys):
    selected = widget.item(widget.currentRow(), 0)
    selected_text = selected.text() if selected else None
    scroll = widget.verticalScrollBar().value()
    with QSignalBlocker(widget):
        widget.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, key in enumerate(keys):
                value = row.get(key, "")
                if isinstance(value, float):
                    value = f"{value:.5f}"
                cell = QTableWidgetItem(str(value) if value is not None else "—")
                cell.setToolTip(cell.text())
                widget.setItem(i, j, cell)
            if widget.item(i, 0).text() == selected_text:
                widget.selectRow(i)
        widget.verticalScrollBar().setValue(scroll)


class Window(SimpleInterface, QMainWindow):
    def __init__(self, core):
        super().__init__()
        self.core, self.job = core, None
        self.monitor_job = None
        self.success_callback = None
        self.current_source = None
        self.auto_armed = core.config.get("auto_on_start")
        self.pending_close = False
        self.restarting = False
        self.job_name = ""
        self.job_failed_flag = False
        self.last_study_result = None
        self.stage_text = ""
        self.last_failure = ""
        self._last_dialogue_report = None
        self.source_rows, self.checkpoint_rows, self.queue_rows = [], [], []
        self.setWindowTitle(f"TAC — Inteligência Artificial com Aprendizado Contínuo | {__version__}")
        self.resize(1240, 840)
        self.setMinimumSize(880, 640)
        base = QWidget()
        self.setCentralWidget(base)
        row = QHBoxLayout(base)
        sidebar = QWidget()
        sidebar.setFixedWidth(205)
        sl = QVBoxLayout(sidebar)
        brand = label("TAC")
        brand.setObjectName("title")
        sl.addWidget(brand)
        sl.addWidget(label("Seu assistente local", True))
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setWordWrap(True)
        self.nav.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.nav.addItems(["Conversar", "Ensinar", "Biblioteca", "Treino avançado", "Monitoramento", "Configurações"])
        sl.addWidget(self.nav, 1)
        self.advanced_check = QCheckBox("Ferramentas avançadas")
        self.advanced_check.toggled.connect(self.set_advanced)
        sl.addWidget(self.advanced_check)
        self.state_label = label("")
        sl.addWidget(self.state_label)
        sl.addWidget(label(f"v{__version__} · Dados locais", True))
        row.addWidget(sidebar)
        self.pages = QStackedWidget()
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        activity = QWidget()
        activity.setObjectName("activityCard")
        activity_layout = QVBoxLayout(activity)
        self.activity_title = label("")
        self.activity_title.setObjectName("activityTitle")
        self.activity_detail = label("")
        self.activity_counts = label("", True)
        activity_top = QHBoxLayout()
        self.activity_controls = activity_top
        activity_top.addWidget(self.activity_title, 1)
        self.activity_details_button = button("Detalhes", lambda: None)
        self.activity_details_button.setCheckable(True)
        self.activity_details_button.toggled.connect(self.show_activity_details)
        activity_top.addWidget(self.activity_details_button)
        activity_layout.addLayout(activity_top)
        activity_layout.addWidget(self.activity_detail)
        activity_layout.addWidget(self.activity_counts)
        self.activity_detail.hide()
        self.activity_counts.hide()
        content_layout.addWidget(activity)
        self.notice = label("")
        self.notice.setObjectName("notice")
        self.notice.hide()
        content_layout.addWidget(self.notice)
        content_layout.addWidget(self.pages, 1)
        row.addWidget(content, 1)
        self.build_chat()
        self.build_learning()
        self.build_library()
        self.build_training()
        self.build_monitoring()
        self.build_settings()
        self.reorganize_ui()
        self.nav.currentRowChanged.connect(self.navigate)
        self.set_advanced(False)
        self.go('home')
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(190)
        self.progress.setValue(0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Pronta. Escolha uma opção no Início ou abra Como começo?.")
        self.refresh()
        if self.core.active_conversation_id():
            self.chat_mode.setCurrentIndex(2)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(2000)
        self.slow_ticks = 0
        self.last_auto_attempt = 0
        self.last_backup_attempt = time.monotonic()  # não faz backup imediatamente na abertura
        self.progress.hide()

    def page(self, title, subtitle):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        title_label = label(title)
        title_label.setObjectName("title")
        layout.addWidget(title_label)
        layout.addWidget(label(subtitle, True))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        self.pages.addWidget(scroll)
        return layout

    def set_advanced(self, visible):
        for index in range(5, 9):
            self.nav.item(index).setHidden(not visible)
        self.chat_mode.view().setRowHidden(1, not visible)
        self.activity_details_button.setVisible(visible)
        if not visible:
            self.activity_details_button.setChecked(False)
            if self.chat_mode.currentIndex() == 1:
                self.chat_mode.setCurrentIndex(0)
        if not visible and self.nav.currentRow() >= 5:
            self.nav.setCurrentRow(0)

    def navigate(self, index):
        if index >= 5:
            self.advanced_check.setChecked(True)
        self.pages.setCurrentIndex(index)

    def show_activity_details(self, visible):
        self.activity_detail.setVisible(visible)
        self.activity_counts.setVisible(visible)

    def build_chat(self):
        layout = self.page("Conversar", "Pergunte sobre seus materiais ou converse sobre o uso da TAC.")
        self.chat_tools = QWidget()
        controls = QHBoxLayout(self.chat_tools)
        controls.setContentsMargins(0,0,0,0)
        controls.addWidget(label('Modo:'))
        self.chat_mode = QComboBox()
        self.chat_mode.addItems(["Consultar biblioteca / conversa guiada", "Continuação de texto (experimental)", "Conversar com a rede neural (experimental)"])
        self.chat_mode.currentIndexChanged.connect(self.chat_mode_changed)
        self.chat_mode.setToolTip("Consulta documental usa as fontes estudadas. Geração neural exige um checkpoint treinado.")
        controls.addWidget(self.chat_mode)
        controls.addWidget(button('Nova conversa', self.clear_conversation))
        self.conversation_version = QComboBox()
        self.conversation_version_notice = label('', True)
        self._conversation_deployment = None
        self._conversation_v3_deployment = None
        deployment_path = self.core.config.data/'tac_v2_conversation_20261018/deployment.json'
        if deployment_path.exists():
            deployment = json.loads(deployment_path.read_text(encoding='utf-8'))
            self._conversation_deployment = deployment
            self.conversation_version.addItem('Experimento — não aprovado' if not deployment.get('passed_retention_gate',False) else 'Experimento de conversação', deployment['checkpoint'])
            self.conversation_version.addItem('Referência v2 anterior', deployment['reference'])
        v3_path = self.core.config.data/'tac_v3/deployment.json'
        if v3_path.exists():
            self._conversation_v3_deployment = json.loads(v3_path.read_text(encoding='utf-8'))
            if self._conversation_deployment:
                self.conversation_version.setItemText(0, 'TAC v2 — conversação, não aprovada' if not self._conversation_deployment.get('passed_retention_gate') else 'TAC v2 — conversação')
            self.conversation_version.addItem('TAC v3 — experimental, 29,4 milhões', self._conversation_v3_deployment['checkpoint'])
        if self.conversation_version.count():
            self.conversation_version.setCurrentIndex(max(0,self.conversation_version.findData(self.core.active_conversation_id())))
            self.conversation_version.setToolTip('Trocar a versão inicia uma conversa limpa para comparação.')
            self.conversation_version.currentIndexChanged.connect(self.change_conversation_version)
        else:
            self.conversation_version.hide()
        controls.addWidget(self.conversation_version)
        controls.addStretch()
        self.chat_tools.hide()
        layout.addWidget(self.chat_tools)
        layout.addWidget(self.conversation_version_notice)
        self.conversation_version_notice.hide()
        self.context_bar = QWidget()
        context_layout = QHBoxLayout(self.context_bar)
        context_layout.setContentsMargins(0,0,0,0)
        self.context_label = label("Contexto: toda a biblioteca")
        context_layout.addWidget(self.context_label, 1)
        context_layout.addWidget(button("Usar toda a biblioteca", self.clear_context))
        self.context_bar.hide()
        layout.addWidget(self.context_bar)
        self.chat_readiness = label("")
        self.chat_readiness.hide()
        layout.addWidget(self.chat_readiness)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlainText("TAC\nOlá! Podemos começar com um oi ou uma pergunta sobre seus materiais.\n\nEm Ensinar, você pode guardar conteúdo ou praticar conversas. Use Como funciona? se quiser entender a diferença.")
        self.transcript.setMinimumHeight(180)
        layout.addWidget(self.transcript, 1)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Escreva sua mensagem…")
        self.input.setMaxLength(5000)
        self.input.returnPressed.connect(self.send_chat)
        web_controls = QHBoxLayout()
        self.research_auto = QCheckBox('Consultar a Wikipédia quando faltar informação')
        self.research_auto.setChecked(self.core.config.get('research_auto'))
        self.research_auto.setToolTip('Consulta artigos pelo assunto de perguntas públicas curtas. Não envia histórico ou documentos. Só guarda textos com novos estudos permitidos; não treina a rede.')
        self.research_auto.toggled.connect(self.set_research_auto)
        web_controls.addWidget(self.research_auto, 1)
        self.research_button = button('Consultar Wikipédia', self.ask_wikipedia)
        web_controls.addWidget(self.research_button)
        self.research_stop = button('Cancelar pesquisa', self.cancel)
        self.research_stop.hide()
        web_controls.addWidget(self.research_stop)
        layout.addLayout(web_controls)
        layout.addWidget(label('Quando a consulta for usada, a resposta virá de uma fonte. Isso não treina a conversa da TAC.', True))
        send = QHBoxLayout()
        send.addWidget(self.input, 1)
        self.send_button = button("Enviar", self.send_chat, True)
        send.addWidget(self.send_button)
        layout.addLayout(send)
        actions = QHBoxLayout()
        self.sources_button = button("Ver fontes", self.show_answer_sources)
        self.sources_button.hide()
        actions.addWidget(self.sources_button)
        actions.addWidget(button("Como funciona?", self.explain_learning))
        actions.addStretch()
        more = button("Mais opções", lambda: self.chat_extras.setVisible(not self.chat_extras.isVisible()))
        actions.addWidget(more)
        layout.addLayout(actions)
        self.chat_extras = QWidget()
        extra_layout = QHBoxLayout(self.chat_extras)
        extra_layout.setContentsMargins(0,0,0,0)
        extra_layout.addWidget(button("Registrar correção", self.correction_dialog))
        extra_layout.addWidget(button("Histórico autorizado", self.show_interactions))
        extra_layout.addWidget(button('Consultas à web', self.show_web_history))
        extra_layout.addWidget(button("Limpar conversa", self.clear_conversation))
        self.chat_extras.hide()
        layout.addWidget(self.chat_extras)

    def explain_learning(self):
        SimpleInterface.explain_learning(self)

    def show_answer_sources(self):
        if not self.core.last_answer:
            return
        answer = self.core.last_answer[1]
        dialog = QDialog(self)
        dialog.setWindowTitle("Fontes desta resposta")
        dialog.resize(760, 500)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        lines = []
        for i, source in enumerate(answer.get("sources", []), 1):
            lines.append(f"[{i}] {source['title']}\n{source['origin']}\nSeção: {source['section']}\nFonte #{source['source_id']} · trecho #{source['id']}")
        for proof in answer.get("evidence", []):
            lines.append("Trecho utilizado:\n" + proof)
        if answer.get('research'):
            research = answer['research']
            lines.append('Consulta: ' + research.get('checked_at', '') + '\n' + research.get('note', research.get('status', '')))
            lines.extend('Página não utilizada: ' + error for error in research.get('errors', []))
        if answer.get('web_verification'):
            for meta in answer['web_verification']:
                lines.append('Consulta original: ' + meta.get('checked_at', '') + '\n' + meta.get('verification_note', '')
                             + '\nAssunto verificado: ' + meta.get('verification_topic', 'não registrado'))
        if answer.get("method") and self.advanced_check.isChecked():
            lines.append("Método: " + answer["method"])
        text.setPlainText("\n\n".join(lines))
        layout.addWidget(text)
        layout.addWidget(button("Fechar", dialog.accept))
        dialog.exec()

    def practice_dialogue(self):
        if self.job:
            self.notify_error("Aguarde a atividade atual terminar.")
            return
        if not self.core.gate.enabled:
            self.core.gate.set_enabled(True)
        self.practice_progress.setRange(0, 100)
        self.practice_progress.setValue(0)
        self.start_job("Praticar conversas", lambda progress: self.core.dialogue.train(progress), self.dialogue_result, learning=True)

    def dialogue_result(self, result):
        self.practice_progress.setValue(100)
        self.practice_status.setText(f"Treino {result['status']}.\n{result['groups']} grupos de exercícios · {result['train_examples']} exemplos praticados.\nAcerto no teste separado: {result['test_correct']}/{result['test_examples']} ({result['test_accuracy']:.0%}).\nEsse resultado mede reconhecimento das intenções dos exercícios, não conversa livre.")

    def show_dialogue_report(self):
        path = self.core.config.data / "dialogue_report.json"
        if not path.exists():
            self.practice_status.setText("Nenhum exercício foi treinado ainda. Clique em Praticar conversas agora.")
            return
        result = json.loads(path.read_text(encoding="utf-8"))
        self.dialogue_result(result)

    def build_learning(self):
        outer = self.page("Ensinar", "Escolha o que você quer melhorar. São duas formas diferentes de aprender.")
        outer.addWidget(button("Entenda a diferença em um minuto", self.explain_learning))
        self.learning_tabs = QTabWidget()
        materials = QWidget()
        layout = QVBoxLayout(materials)
        layout.addWidget(label("Guardar conhecimento: ler sites e arquivos para consultar depois. Isso não muda os pesos da rede.", True))
        self.learning_tabs.addTab(materials, "Ler sites e arquivos")
        outer.addWidget(self.learning_tabs, 1)
        self.learning_button = button("", self.toggle_learning)
        self.learning_button.setMinimumHeight(32)
        layout.addWidget(self.learning_button)
        self.queue_hint = label("")
        layout.addWidget(self.queue_hint)
        self.material_tabs = QTabWidget()
        self.material_tabs.setMaximumHeight(155)
        document_page = QWidget()
        document_layout = QVBoxLayout(document_page)
        self.drop = DropArea()
        self.drop.setMinimumHeight(45)
        self.drop.setText("Arraste PDF, TXT, DOCX, Markdown ou HTML aqui · ou selecione abaixo")
        self.drop.files.connect(self.add_files)
        document_layout.addWidget(self.drop)
        documents = QHBoxLayout()
        documents.addWidget(button("Adicionar documentos", self.select_files))
        documents.addWidget(button("Testar com exemplo local", self.add_demo))
        document_layout.addLayout(documents)
        site_page = QWidget()
        site_layout = QVBoxLayout(site_page)
        site = QHBoxLayout()
        self.url = QLineEdit()
        self.url.setPlaceholderText("Endereço do site — cole https://… aqui")
        self.web_mode = QComboBox()
        self.web_mode.addItems(["Só esta página", "Documentação: páginas neste caminho"])
        self.web_mode.setToolTip("Para um artigo da Wikipédia, use Só esta página. Documentação percorre links dentro do caminho informado.")
        site.addWidget(self.url, 1)
        site.addWidget(self.web_mode)
        self.add_url_button = button("Adicionar site", self.add_url)
        site.addWidget(self.add_url_button)
        site_layout.addLayout(site)
        self.terms = QCheckBox("Autorizo estudar este site e verifiquei seus termos. A TAC respeitará robots.txt e os limites.")
        site_layout.addWidget(self.terms)
        site_layout.addStretch()
        self.material_tabs.addTab(site_page, "Adicionar site")
        self.material_tabs.addTab(document_page, "Adicionar documento ou exemplo")
        layout.addWidget(self.material_tabs)
        self.subject = QLineEdit()
        self.subject.setPlaceholderText("Assunto opcional: por exemplo, Brasil ou programação (não cole a URL aqui)")
        self.subject.hide()
        subject_check = QCheckBox("Definir assunto para os próximos materiais (opcional)")
        subject_check.toggled.connect(self.subject.setVisible)
        layout.addWidget(subject_check)
        layout.addWidget(self.subject)
        actions = QHBoxLayout()
        self.study_button = button("ESTUDAR FILA", self.process_queue, True)
        self.study_button.setMinimumHeight(42)
        self.stop_button = button("Interromper estudo", self.cancel)
        actions.addWidget(self.study_button, 2)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)
        self.study_progress = QProgressBar()
        self.study_progress.setRange(0, 100)
        self.study_progress.setValue(0)
        self.study_progress.setFormat("Nenhum estudo em andamento")
        layout.addWidget(self.study_progress)
        self.study_stage = label("Os materiais só entram na Biblioteca depois que o estudo é concluído.")
        layout.addWidget(self.study_stage)
        self.study_results = label("")
        self.study_results.setObjectName("notice")
        self.study_results.hide()
        layout.addWidget(self.study_results)
        self.study_query_button = button("Consultar último material estudado →", self.query_last_studied, True)
        self.study_query_button.hide()
        layout.addWidget(self.study_query_button)
        self.queue_details_button = button("Ver fila e histórico", lambda: None)
        self.queue_details_button.setCheckable(True)
        layout.addWidget(self.queue_details_button)
        self.queue_section = QWidget()
        queue_layout = QVBoxLayout(self.queue_section)
        queue_layout.setContentsMargins(0,0,0,0)
        self.queue_details_button.toggled.connect(self.queue_section.setVisible)
        self.queue_tabs = QTabWidget()
        self.queue_table = table(["ID", "Tipo", "Material", "Estado", "Progresso / erro"])
        self.queue_table.setMinimumHeight(120)
        self.queue_table.setColumnWidth(0, 45)
        self.queue_table.setColumnWidth(1, 75)
        self.queue_table.setColumnWidth(2, 230)
        self.queue_table.setColumnWidth(3, 180)
        self.queue_table.itemSelectionChanged.connect(self.show_queue_detail)
        self.queue_table.cellDoubleClicked.connect(lambda *_: self.show_queue_detail())
        self.queue_tabs.addTab(self.queue_table, "Fila de materiais")
        self.history_table = table(["Data UTC", "Origem", "Resultado", "Detalhe"])
        self.history_table.setMinimumHeight(120)
        self.queue_tabs.addTab(self.history_table, "Histórico de estudos")
        queue_layout.addWidget(self.queue_tabs, 1)
        selected_actions = QHBoxLayout()
        self.retry_button = button("Tentar novamente e estudar", self.retry_queue)
        selected_actions.addWidget(self.retry_button)
        selected_actions.addWidget(button("Remover item selecionado", self.remove_queue))
        queue_layout.addLayout(selected_actions)
        self.queue_detail = QPlainTextEdit()
        self.queue_detail.setReadOnly(True)
        self.queue_detail.setMinimumHeight(60)
        self.queue_detail.setMaximumHeight(75)
        self.queue_detail.setPlaceholderText("Selecione um material para ver o resultado completo ou a explicação do erro.")
        queue_layout.addWidget(self.queue_detail)
        layout.addWidget(self.queue_section)
        urgent = bool(self.core.db.query("SELECT 1 FROM queue WHERE status IN ('pendente','erro','processando') LIMIT 1"))
        self.queue_details_button.setChecked(urgent)
        self.queue_section.setVisible(urgent)
        layout.addStretch()
        practice = QWidget()
        practice_layout = QVBoxLayout(practice)
        practice_layout.addWidget(label("Praticar conversas", False))
        practice_layout.addWidget(label("Aqui a TAC faz exercícios que alteram uma rede neural. Ela aprende a reconhecer intenções de conversa e escolher respostas preparadas para elas. Os seus documentos continuam guardados separadamente."))
        practice_layout.addWidget(label("Exercícios disponíveis: linguagem formal, informal e abreviada; cumprimentos, ajuda, correções e expressões simples de alegria, tristeza ou frustração. São respostas guiadas, sem compreensão geral de emoções."))
        self.practice_button = button("Praticar conversas agora", self.practice_dialogue, True)
        practice_layout.addWidget(self.practice_button)
        self.practice_progress = QProgressBar()
        self.practice_progress.setValue(0)
        practice_layout.addWidget(self.practice_progress)
        self.practice_status = label("Ainda não há resultado de exercícios nesta sessão.")
        practice_layout.addWidget(self.practice_status)
        controls = QHBoxLayout()
        self.practice_pause_button = button("Pausar", self.pause)
        self.practice_stop_button = button("Interromper treino", self.cancel)
        controls.addWidget(self.practice_pause_button)
        controls.addWidget(self.practice_stop_button)
        practice_layout.addLayout(controls)
        practice_layout.addWidget(button("Ver resultado dos exercícios", self.show_dialogue_report))
        practice_layout.addWidget(label("Esta é uma conversa guiada, com respostas escritas para os exercícios. Não equivale a conversar livremente sobre qualquer assunto.", True))
        practice_layout.addWidget(button("Voltar à conversa →", lambda: self.go('chat')))
        practice_layout.addStretch()
        self.learning_tabs.addTab(practice, "Praticar conversas")

    def build_library(self):
        layout = self.page("Biblioteca de Conhecimento", "Trechos e fontes persistentes. Conteúdos importados são dados; instruções neles não são executadas.")
        filters = QHBoxLayout()
        self.library_query = QLineEdit()
        self.library_query.setPlaceholderText("Pesquisar trechos por assunto ou palavras…")
        self.library_query.returnPressed.connect(self.search_library)
        self.subject_filter = QLineEdit()
        self.subject_filter.setPlaceholderText("Filtrar documentos por assunto")
        self.subject_filter.textChanged.connect(self.refresh_sources)
        filters.addWidget(self.library_query, 2)
        filters.addWidget(self.subject_filter)
        filters.addWidget(button("Pesquisar", self.search_library))
        layout.addLayout(filters)
        self.sources_table = table(["ID", "Título", "Assunto", "Tipo", "Trechos", "Origem"])
        for column, width in enumerate((45, 245, 150, 65, 65)):
            self.sources_table.setColumnWidth(column, width)
        self.sources_table.setMinimumHeight(160)
        self.sources_table.itemSelectionChanged.connect(self.show_source)
        layout.addWidget(self.sources_table, 1)
        actions = QHBoxLayout()
        self.library_actions = actions
        actions.addWidget(button("Usar no chat", self.use_source))
        actions.addWidget(button("Editar assunto", self.edit_subject))
        actions.addWidget(button("Reorganizar contexto", self.reindex_selected))
        actions.addWidget(button("Atualizar fonte", self.update_source))
        actions.addWidget(button("Excluir fonte", self.delete_source))
        actions.addWidget(button("Ver correções", self.manage_corrections))
        layout.addLayout(actions)
        self.source_preview = QPlainTextEdit()
        self.source_preview.setReadOnly(True)
        self.source_preview.setMinimumHeight(220)
        layout.addWidget(self.source_preview, 1)
        layout.addWidget(label("Excluir fontes remove texto, índices e snapshots derivados. A influência em pesos já treinados pode permanecer; não existe esquecimento neural automático.", True))

    def build_training(self):
        layout = self.page("Treinamento e versões", "Redes próprias: leitura de textos e geração de diálogo. Consulte as avaliações de cada versão.")
        self.model_label = label("Nenhum checkpoint ativo")
        layout.addWidget(self.model_label)
        self.export_diagnostic_button = button('Exportar diagnóstico completo', self.export_diagnostic)
        layout.addWidget(self.export_diagnostic_button)
        layout.addWidget(label('Gera um ZIP técnico para análise. Não treina, não altera pesos e não inclui livros ou conversas privadas. Textos não verificáveis são anonimizados.', True))
        layout.addWidget(button('Ver resultados dos experimentos de conversa',self.show_conversation_research))
        layout.addWidget(button('Escolher textos opcionais para treinar escrita na conversa',self.prepare_conversation_documents))
        layout.addWidget(button('Usar somente a prática de escrita do programa',self.clear_conversation_documents))
        self.transfer_practice_check=QCheckBox('Incluir prática experimental de nomes, correções e localização no próximo treino')
        self.transfer_practice_check.setChecked(self.core.config.get('conversation_transfer_practice'))
        self.transfer_practice_check.toggled.connect(lambda value:self.core.config.update(conversation_transfer_practice=value))
        layout.addWidget(self.transfer_practice_check)
        layout.addWidget(label('As opções experimentais começam desligadas. Nenhum piloto desta fase substituiu a versão ativa. As taxas e estratégias ficam em Ajustes e manutenção.', True))
        self.train_label = label("Adicione documentos com texto suficiente e ative o aprendizado para treinar.")
        layout.addWidget(self.train_label)
        self.training_readiness = label("")
        layout.addWidget(self.training_readiness)
        layout.addWidget(button("Primeiro guardar documentos ou sites →", lambda: self.go('materials')))
        actions = QHBoxLayout()
        self.train_button = button("Treinar leitura dos documentos", self.train)
        actions.addWidget(self.train_button)
        self.pause_button = button("Pausar / continuar", self.pause)
        actions.addWidget(self.pause_button)
        self.train_stop_button = button("Interromper e salvar", self.cancel)
        actions.addWidget(self.train_stop_button)
        layout.addLayout(actions)
        self.conversation_train_button = button("Treinar diálogo e contexto · 10 milhões", self.train_conversation, True)
        layout.addWidget(self.conversation_train_button)
        layout.addWidget(label("Diálogo treina a geração das respostas dos exercícios. Mantém uma variante separada; o modelo de leitura e a biblioteca são preservados.", True))
        self.auto_check = QCheckBox("Permitir treinamento neural automático após estudar materiais")
        self.auto_check.setChecked(self.core.config.get("auto_train"))
        self.auto_check.toggled.connect(self.toggle_auto)
        layout.addWidget(self.auto_check)
        layout.addWidget(label("O estudo já permite consultas no Chat. Treinar é uma etapa experimental separada que modifica os pesos da rede.", True))
        self.plot = LossPlot()
        layout.addWidget(self.plot)
        layout.addWidget(label("Checkpoints — selecione para retomar ou restaurar; as colunas permitem comparar as perdas."))
        self.checkpoints_table = table(["ID", "Data UTC", "Passo", "Loss validação", "Estado", "Treino"])
        self.checkpoints_table.setMinimumHeight(170)
        layout.addWidget(self.checkpoints_table, 1)
        restore = QHBoxLayout()
        restore.addWidget(button("Retomar selecionado", self.resume))
        restore.addWidget(button("Restaurar modelo selecionado", self.restore_checkpoint))
        restore.addWidget(button("Ver avaliações e parâmetros", self.show_runs))
        layout.addLayout(restore)
        layout.addWidget(label("Experimento de reforço: REINFORCE em 16 somas (0..3). Recompensa = 1 se correta. Avalia as mesmas tarefas treinadas, sem alegar generalização.", True))
        layout.addWidget(button("Executar experimento de reforço", self.reinforce))

    def build_monitoring(self):
        layout = self.page("Monitoramento", "Limites de RAM são verificados entre etapas; a reserva de VRAM é configurada no PyTorch. Não são cotas rígidas do Windows.")
        self.monitor = QPlainTextEdit()
        self.monitor.setReadOnly(True)
        self.monitor.setMinimumHeight(340)
        layout.addWidget(self.monitor)
        layout.addWidget(button("Atualizar GPU / CUDA", self.monitor_gpu))
        self.gpu_label = label("GPU / CUDA: consultar pelo botão acima (nenhum acesso à internet).")
        layout.addWidget(self.gpu_label)
        layout.addWidget(button("Abrir pasta de logs", lambda: os.startfile(str(self.core.config.data / "logs"))))
        layout.addStretch()

    def build_settings(self):
        layout = self.page("Configurações", "As preferências são locais. Nenhum treinamento ou download de modelo é necessário para consultar documentos.")
        form = QFormLayout()
        self.advanced_settings_form = form
        self.settings_widgets = {}
        numeric = [
            ("font_size", "Tamanho da fonte", 11, 18),
            ("ram_mb", "Limite de RAM da TAC (MB)", 512, 8192),
            ("reserve_ram_mb", "RAM mínima livre para o sistema (MB)", 256, 8192),
            ("vram_mb", "Limite de VRAM (MB)", 256, 4608),
            ("cpu_threads", "Threads de CPU", 1, 8),
            ("batch_size", "Tamanho do lote", 1, 32), ("accumulation", "Acumulação de gradientes", 1, 16),
            ("context", "Contexto (bytes)", 16, 256), ("width", "Dimensão do modelo novo", 32, 256),
            ("layers", "Camadas do modelo novo", 1, 4), ("heads", "Cabeças de atenção", 1, 8),
            ("ff_width", "Neurônios internos do modelo novo (0 = automático)", 0, 2048),
            ("epochs", "Épocas por execução", 1, 100), ("steps_per_epoch", "Passos por época", 1, 1000),
            ("train_seconds", "Duração máxima de treino (segundos)", 1, 3600),
            ("conversation_steps", "Passos de treino de diálogo", 100, 60000),
            ("conversation_seconds", "Duração máxima de treino de diálogo (s)", 1, 3600),
            ("auto_interval_minutes", "Intervalo mínimo entre treinos automáticos (min)", 1, 1440),
            ("max_file_mb", "Limite de documento (MB)", 1, 100),
            ("web_pages", "Máximo de páginas por site", 1, 100), ("web_depth", "Profundidade de navegação", 0, 4),
            ("web_page_mb", "Limite de página (MB)", 1, 8),
            ('research_pages', 'Páginas úteis por consulta à Wikipédia', 1, 5),
            ('research_seconds', 'Limite de duração da consulta web (s)', 10, 180),
            ('research_chars', 'Caracteres guardados por página pesquisada', 2000, 100000),
            ("backup_hours", "Frequência de backup automático (horas)", 1, 720),
            ("backup_keep", "Backups automáticos mantidos", 1, 20),
        ]
        for key, title, low, high in numeric:
            widget = QSpinBox()
            widget.setRange(low, high)
            widget.setValue(self.core.config.get(key))
            self.settings_widgets[key] = widget
            form.addRow(title, widget)
        for key, title, low, high, decimals in (("learning_rate", "Taxa de aprendizado da escrita", .00001, .02, 5),
            ('conversation_learning_rate','Taxa de aprendizado da conversa',.00001,.003,5),
            ("web_interval", "Intervalo mínimo entre requisições (s)", 1, 60, 1)):
            widget = QDoubleSpinBox()
            widget.setRange(low, high)
            widget.setDecimals(decimals)
            widget.setValue(self.core.config.get(key))
            self.settings_widgets[key] = widget
            form.addRow(title, widget)
        for key,title,options in (
            ('conversation_stability','Preservação experimental',[('off','Retenção anterior'),('anchor_v1','Repetir habilidades antigas em cada passo')]),
            ('conversation_sampler','Amostragem experimental',[('replacement','Sorteio anterior'),('deck_v1','Embaralhar sem repetir no mesmo ciclo')]),
            ('conversation_generation','Geração experimental',[('greedy','Escolher a opção mais provável'),('controlled','Sortear com limites e reduzir repetição')]),
        ):
            widget=QComboBox()
            for value,text in options:widget.addItem(text,value)
            widget.setCurrentIndex(widget.findData(self.core.config.get(key)))
            setattr(self,key+'_setting',widget)
            form.addRow(title,widget)
        self.device_setting = QComboBox()
        self.device_setting.addItems(["auto", "cpu", "cuda"])
        self.device_setting.setCurrentText(self.core.config.get("device"))
        form.addRow("Dispositivo de treinamento", self.device_setting)
        self.theme_setting = QComboBox()
        self.theme_setting.addItems(["dark", "light"])
        self.theme_setting.setCurrentText(self.core.config.get("theme"))
        form.addRow("Tema da interface", self.theme_setting)
        self.data_setting = QLineEdit(self.core.config.get("data_dir"))
        self.data_setting.setToolTip("Subpasta dentro de TAC. Migração preserva a pasta anterior e reinicia a aplicação.")
        form.addRow("Pasta dos dados (dentro da TAC)", self.data_setting)
        self.startup_check = QCheckBox("Autorizar treino automático também ao abrir a TAC")
        self.startup_check.setChecked(self.core.config.get("auto_on_start"))
        form.addRow(self.startup_check)
        self.interactions_check = QCheckBox("Autorizar memória de conversa enquanto o aprendizado estiver ativado")
        self.interactions_check.setChecked(self.core.config.get("save_interactions"))
        form.addRow(self.interactions_check)
        layout.addLayout(form)
        layout.addWidget(button("Salvar configurações", self.save_settings, True))
        backups = QHBoxLayout()
        backups.addWidget(button("Exportar backup completo", self.backup))
        backups.addWidget(button("Importar backup e reiniciar", self.restore_backup))
        backups.addWidget(button("Gerenciar memória de interação", self.manage_interactions))
        layout.addLayout(backups)
        layout.addWidget(label("A arquitetura existente é preservada nos treinos incrementais. Dimensão, contexto e camadas se aplicam à criação do primeiro modelo. Para uma arquitetura diferente, use uma nova pasta de dados.", True))

    def notify_error(self, message):
        self.raw_failure = str(message)
        self.last_failure = user_error(message)
        self.core.logger.error('Erro apresentado na interface: %s', message)
        self.notice.setText("Operação não concluída\n" + self.last_failure[:1000])
        self.notice.show()
        self.statusBar().showMessage(self.last_failure[:250])
        self.refresh_activity()

    def start_job(self, name, callback, success=None, learning=False):
        if self.job is not None:
            self.notify_error("Uma operação já está em andamento. Aguarde ou cancele.")
            return False
        self.success_callback = success
        self.job_name = name
        self.job_failed_flag = False
        self.last_failure = ""
        self.raw_stage_text = name
        self.training_phase = 'prepare'
        self.stage_text = 'Preparando a atividade…'
        self.job_started = time.monotonic()
        if name != "Backup automático local":
            self.notice.hide()
        self.job = Job(self.core, name, callback, learning)
        self.job.result.connect(self.job_result)
        self.job.failed.connect(self.job_error)
        self.job.update.connect(self.job_progress)
        self.job.finished.connect(self.job_finished)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.statusBar().showMessage('Atividade em andamento. Você pode acompanhar o progresso na tela.')
        if name == "Estudar fila":
            self.study_progress.setRange(0, 0)
            self.study_progress.setFormat("Guardando materiais")
            self.study_stage.setText("Iniciando: verificando permissão e materiais da fila…")
            self.study_results.hide()
            self.study_stage.show()
        self.job.start()
        self.refresh_activity()
        return True

    def job_error(self, message):
        self.job_failed_flag = True
        self.notify_error(message)
        if self.job_name == "Estudar fila":
            self.study_results.setText("Não foi possível guardar os materiais\n" + user_error(message))
            self.study_results.show()

    def job_result(self, value):
        if self.success_callback:
            try:
                self.success_callback(value)
            except Exception as exc:
                self.core.logger.exception("Falha ao apresentar resultado")
                self.notify_error(str(exc))
        if self.job_name != "Estudar fila" and not self.job_failed_flag:
            message = value if isinstance(value, str) else ('Resposta exibida. A origem aparece junto da resposta.' if isinstance(value,dict) and 'text' in value else 'Atividade concluída. Confira o resultado na tela.')
            self.statusBar().showMessage(message, 15000)

    def job_progress(self, step, message):
        self.raw_stage_text = message
        if self.job_name in ('Treinamento neural', 'Retomar treinamento', 'Experimento de reforço', 'Praticar conversas'):
            self.stage_text, self.training_phase = friendly_progress(message, self.job_name, self.training_phase)
        else:
            self.stage_text = message.replace('Indexando e salvando:', 'Guardando:')
        self.statusBar().showMessage(self.stage_text)
        if self.job_name == "Estudar fila":
            self.study_stage.setText(self.stage_text)
            self.refresh_queue()
        elif self.job_name == "Praticar conversas":
            self.practice_progress.setValue(step)
            self.practice_status.setText(message)
        elif self.job_name in ("Treinamento neural", "Retomar treinamento", "Experimento de reforço"):
            self.technical_train_label.setText(message)
            self.train_label.setText(self.stage_text)
            self.training_progress.show()
            if self.training_phase == 'train':
                rows = self.core.db.query("SELECT kind,config FROM runs WHERE status='treinando' ORDER BY started DESC LIMIT 1")
                params = json.loads(rows[0]['config']) if rows else {}
                target = params.get('steps', params.get('epochs', 1) * params.get('steps_per_epoch', max(1, step)))
                self.training_progress.setRange(0, max(step, target))
                self.training_progress.setValue(step)
                self.training_progress.setFormat('Pratiquei %p% desta etapa')
            else:
                self.training_progress.setRange(0, 0)
        self.refresh_activity()

    def job_finished(self):
        self.job.deleteLater()
        self.job = None
        self.success_callback = None
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.training_progress.hide()
        if self.job_name == "Estudar fila":
            self.study_progress.setRange(0, 100)
            if self.last_study_result and not self.job_failed_flag:
                result = self.last_study_result
                total = max(1, result["total"])
                self.study_progress.setValue(round((result["processed"] + result["errors"]) / total * 100))
                self.study_progress.setFormat("Interrompido" if result["interrupted"] else ("Finalizado com erro(s)" if result["errors"] else "Material guardado"))
            else:
                self.study_progress.setValue(0)
                self.study_progress.setFormat("Estudo não concluído")
        self.refresh()
        if self.pending_close:
            QTimer.singleShot(0, self.close)
        if self.restarting:
            QTimer.singleShot(0, self.restart)
        callback = getattr(self, '_after_current_job', None)
        self._after_current_job = None
        if callback and not self.pending_close and not self.restarting and not self.job_failed_flag:
            QTimer.singleShot(0, callback)
        self.refresh_import_card()

    def toggle_learning(self):
        self.core.gate.set_enabled(not self.core.gate.enabled)
        self.last_failure = ""
        self.notice.hide()
        self.refresh_activity()

    def toggle_auto(self, value):
        self.core.config.update(auto_train=value)
        self.auto_armed = bool(value)
        self.last_auto_attempt = 0
        self.refresh_activity()

    def refresh_gate(self):
        enabled = self.core.gate.enabled
        self.learning_button.setText("Novos estudos: permitidos · Pausar" if enabled else "Novos estudos: pausados · Permitir")
        self.learning_button.setObjectName("learningOn" if enabled else "learningOff")
        self.learning_button.style().unpolish(self.learning_button)
        self.learning_button.style().polish(self.learning_button)
        self.state_label.setText("● Novos estudos permitidos" if enabled else "● Novos estudos pausados")

    def select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Adicionar arquivo", str(self.core.config.root), "Documentos aceitos (*.pdf *.txt *.docx *.md *.markdown *.html *.htm);;Todos os arquivos (*)")
        self.add_files(paths)

    def add_files(self, paths):
        if not paths:
            return
        if self.job:
            self.notify_error('Aguarde a atividade atual terminar para adicionar arquivos.')
            return
        from .documents import SUPPORTED
        accepted, errors = [], []
        for name in paths:
            path = Path(name)
            if path.suffix.lower() not in SUPPORTED:
                errors.append(f'{path.name}: Meus materiais aceita PDF, TXT, DOCX, Markdown e HTML. Para conversas em JSONL ou TXT, vá a Melhorar a TAC → Importar conversas. JSON não é aceito; use um objeto por linha em JSONL. Arquivos JSON/JSONL não foram adicionados à Biblioteca.')
            elif self.confirm_material_file(path):
                accepted.append(str(path))
        if errors:
            QMessageBox.information(self, 'Formato de arquivo não aceito', '\n\n'.join(errors))
        paths = accepted
        if not paths:
            return
        subject = self.subject.text().strip()
        def add(progress):
            for i, path in enumerate(paths):
                progress(i, f"Copiando {Path(path).name} para a fila")
                self.core.enqueue_file(path, subject)
            return f"{len(paths)} arquivo(s) na fila. Clique em Guardar materiais agora para disponibilizá-los nas consultas. Nenhum treino foi iniciado."
        def added(message):
            self.notice.setText(message)
            self.notice.show()
            self.go('materials')
            self.learning_tabs.setCurrentIndex(0)
            self.queue_details_button.setChecked(True)
        self.start_job("Adicionar documentos", add, added)

    def add_demo(self):
        self.add_files([str(self.core.config.root / "examples" / "primeiro_estudo.txt")])

    def add_url(self):
        try:
            if self.job:
                raise RuntimeError("Aguarde a operação atual terminar para adicionar um site.")
            if self.subject.text().strip().startswith(("http://", "https://")):
                raise ValueError("O campo Assunto é opcional e recebe palavras, como Brasil. Cole o endereço no campo Endereço do site.")
            self.core.enqueue_url(self.url.text(), self.web_mode.currentIndex() == 1, self.subject.text(), self.terms.isChecked())
            self.url.clear()
            self.refresh_queue()
            self.queue_details_button.setChecked(True)
            self.last_failure = ""
            self.notice.setText("Site na fila. A leitura ainda não começou. Clique em Guardar materiais agora para iniciar. Isso não treina a conversa da TAC.")
            self.notice.show()
            self.refresh_activity()
        except Exception as exc:
            self.notify_error(str(exc))

    def process_queue(self):
        if self.job:
            self.notify_error("Uma operação está em andamento. Aguarde ou clique em Interromper estudo.")
            return
        if not self.core.db.query("SELECT 1 FROM queue WHERE status='pendente' LIMIT 1"):
            self.notify_error("Não há materiais aguardando estudo. Adicione um site/documento ou selecione um item com erro e clique em Tentar novamente e estudar.")
            return
        if not self.core.gate.enabled:
            # O botão diz explicitamente "Ativar e estudar": o clique concede a permissão.
            self.core.gate.set_enabled(True)
        self.auto_armed = True  # sessão de estudo explicitamente autorizada
        self.last_study_result = None
        self.start_job("Estudar fila", self.core.run_queue, self.study_result, learning=True)

    def study_result(self, result):
        self.last_study_result = result
        done, failed = result["processed"], result["errors"]
        if result["interrupted"]:
            message = f"Estudo interrompido. {done} material(is) concluído(s); os restantes continuam na fila."
        elif failed and not done:
            message = "Nenhum material foi estudado. Selecione o item na fila para ver o motivo e como tentar novamente."
        elif failed:
            message = f"{done} material(is) concluído(s) e {failed} com erro. As fontes concluídas já podem ser consultadas."
        else:
            message = f"Material guardado: {done} material(is) disponível(is) para consulta. Nenhum treino foi executado nesta ação."
        self.study_results.setText(message)
        self.study_results.show()
        self.study_stage.hide()
        self.statusBar().showMessage(message)
        self.study_query_button.setVisible(bool(result["source_ids"]))
        self.refresh_queue()
        if failed:
            self.queue_details_button.setChecked(True)
            for index, row in enumerate(self.queue_rows):
                if row["status"] == "erro":
                    self.queue_table.selectRow(index)
                    self.show_queue_detail()
                    break

    def query_last_studied(self):
        ids = self.last_study_result["source_ids"] if self.last_study_result else []
        if not ids:
            recent = self.core.db.query("SELECT id FROM sources ORDER BY updated DESC,id DESC LIMIT 1")
            ids = [recent[0]["id"]] if recent else []
        if not ids:
            return
        rows = self.core.db.query("SELECT id,title FROM sources WHERE id=?", (ids[-1],))
        if rows:
            self.current_source = rows[0]["id"]
            self.context_label.setText("Contexto: " + rows[0]["title"])
            self.context_bar.show()
            self.chat_mode.setCurrentIndex(0)
            self.go('chat')
            self.input.setText("O que você aprendeu com esse documento?")
            self.send_chat()

    def queue_selected(self):
        index = self.queue_table.currentRow()
        return self.queue_rows[index] if 0 <= index < len(self.queue_rows) else None

    def retry_queue(self):
        item = self.queue_selected()
        if self.job:
            return
        if item and item["status"] in ("erro", "parcial", "pendente"):
            self.core.db.execute("UPDATE queue SET status='pendente',progress='' WHERE id=?", (item["id"],))
            self.refresh_queue()
            self.process_queue()
        elif not item:
            self.notify_error("Selecione primeiro um material na fila.")

    def remove_queue(self):
        if self.job:
            self.notify_error("Aguarde o término da operação para remover materiais da fila.")
            return
        item = self.queue_selected()
        if item and item["status"] != "processando":
            self.core.db.execute("DELETE FROM queue WHERE id=?", (item["id"],))
            # Elimina apenas a cópia gerenciada na inbox, nunca o original.
            if item["kind"] == "file":
                path = (self.core.config.data / "inbox" / item["target"]).resolve()
                if path.is_relative_to((self.core.config.data / "inbox").resolve()):
                    path.unlink(missing_ok=True)
            self.refresh_queue()
            self.refresh_activity()

    def cancel(self):
        if getattr(self, 'import_cancel', None) is not None:
            self.import_cancel.set()
        self.core.gate.cancel.set()
        self.core.gate.paused.clear()
        self.statusBar().showMessage("Parada solicitada. Aguarde o fim da etapa atual e o salvamento do progresso, quando possível.")
        self.stage_text = "Interrompendo com segurança. Aguardando terminar a etapa atual…"
        self.study_stage.setText(self.stage_text)
        self.refresh_activity()

    def pause(self):
        if self.core.operation not in ("Treinamento neural", "Retomar treinamento", "Experimento de reforço", "Praticar conversas"):
            self.notify_error("Não há treinamento em execução.")
            return
        if self.core.gate.paused.is_set():
            self.core.gate.paused.clear()
            self.pause_button.setText("Pausar / continuar")
        else:
            self.core.gate.paused.set()
            self.pause_button.setText("Continuar treinamento")

    def show_web_history(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Histórico de consultas à web')
        dialog.resize(760, 480)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        rows = self.core.db.query("SELECT created,detail FROM events WHERE kind='pesquisa_web' ORDER BY id DESC LIMIT 30")
        text.setPlainText('\n\n'.join(r['created'] + '\n' + r['detail'] for r in rows) or 'Nenhuma consulta à web registrada.')
        layout.addWidget(text)
        layout.addWidget(button('Fechar', dialog.accept))
        dialog.exec()

    def ask_wikipedia(self):
        if self.job:
            return
        from .research import public_topic
        suggested = public_topic(self.input.text()) or ''
        topic, ok = QInputDialog.getText(self, 'Consultar um artigo na Wikipédia',
                                       'Assunto público ou título do artigo (sem dados pessoais):', text=suggested)
        if not ok or not topic.strip():
            return
        if not public_topic(topic, manual=True):
            self.notify_error('Use um assunto público curto, sem URLs, dados pessoais ou conteúdo de arquivos.')
            return
        if not self.core.gate.enabled:
            self.notify_error('Novos estudos estão pausados. Permita novos estudos em Ensinar para pesquisar e guardar os textos.')
            return
        if not self.input.text().strip():
            self.input.setText('O que é ' + topic.strip() + '?')
        self.send_chat(research=True, research_topic=topic.strip())

    def send_chat(self, research=False, research_topic=None):
        question = self.input.text().strip()
        if not question or self.job:
            return
        neural = self.chat_mode.currentIndex() == 1
        conversation = self.chat_mode.currentIndex() == 2
        source = self.current_source
        def success(answer):
            start = self.transcript.document().characterCount()
            visible = answer.get("display_text", answer["text"])
            speaker = 'TAC · ' + answer_origin(answer)
            self.transcript.appendPlainText(f"Você\n{question}\n\n{speaker}\n{visible}\n\n")
            decoding = answer.get('decoding') or {}
            reason = decoding.get('stop_reason')
            if reason == 'repetition':
                self.transcript.appendPlainText('[Geração interrompida por repetição. Resposta parcial; a rede ainda apresenta falha.]\n')
            elif reason == 'length':
                self.transcript.appendPlainText('[Resposta parcial: limite de geração atingido.]\n')
            elif reason == 'role_boundary':
                self.transcript.appendPlainText('[Geração encerrada ao iniciar outro papel de conversa.]\n')
            if decoding.get('truncated_tokens', 0):
                self.transcript.appendPlainText('[Parte do histórico excedeu o contexto da rede e foi cortada.]\n')
            self.sources_button.setText(f"Ver fontes ({len(answer.get('sources', []))})")
            self.sources_button.setVisible(bool(answer.get("sources")))
            cursor = self.transcript.textCursor()
            cursor.setPosition(min(start, self.transcript.document().characterCount() - 1))
            self.transcript.setTextCursor(cursor)
            self.transcript.ensureCursorVisible()
            self.input.clear()
        online = bool(research or (self.core.config.get('research_auto') and self.core.gate.enabled))
        name = 'Conversa com consulta web' if online else ('Consulta neural' if neural or conversation else 'Consulta documental')
        self.start_job(name, lambda progress: self.core.chat(question, source, neural, conversation, progress,
                                                           research=research, research_topic=research_topic), success,
                       learning=online)

    def chat_mode_changed(self, index):
        if hasattr(self,'conversation_version'):
            self.conversation_version.setVisible(index == 2 and self.conversation_version.count() > 0)
            self.update_conversation_version_notice()
        if index == 2:
            self.clear_context()
            self.chat_readiness.setText('A TAC cria respostas neste computador usando a conversa recente. Ainda pode se confundir. Seus arquivos não são consultados neste modo.')
        elif index == 0:
            self.chat_readiness.setText('Aqui a TAC busca nos seus textos. Isso não ensina nada a ela.')
        else:
            self.chat_readiness.setText('Continua o texto com a rede de escrita. É uma experiência; não é uma resposta verificada.')
        self.chat_readiness.show()

    def clear_conversation(self):
        self.transcript.clear()
        self.core.neural_history.clear()

    def change_conversation_version(self, index):
        checkpoint = self.conversation_version.itemData(index)
        if not checkpoint or checkpoint == self.core.active_conversation_id():
            return
        if self.job:
            with QSignalBlocker(self.conversation_version):
                self.conversation_version.setCurrentIndex(self.conversation_version.findData(self.core.active_conversation_id()))
            self.notify_error('Aguarde a resposta atual antes de trocar a versão.')
            return
        def ready(_):
            self.clear_conversation()
            self.update_conversation_version_notice()
            self.statusBar().showMessage('Versão alterada. Nova conversa iniciada.')
        def failed(message):
            with QSignalBlocker(self.conversation_version):
                self.conversation_version.setCurrentIndex(self.conversation_version.findData(self.core.active_conversation_id()))
            self.update_conversation_version_notice()
        if self.start_job('Trocar versão da conversa', lambda progress:self.core.conversation.activate(checkpoint), ready, learning=True):
            self.job.failed.connect(failed)

    def update_conversation_version_notice(self):
        deployment=getattr(self,'_conversation_deployment',None)
        v3=getattr(self,'_conversation_v3_deployment',None)
        if v3 and self.chat_mode.currentIndex()==2 and self.core.active_conversation_id()==v3['checkpoint']:
            self.conversation_version_notice.setText('TAC v3 experimental. Pesos novos e treino próprio; qualidade ainda não aprovada. Compare as respostas com a v2.')
            self.conversation_version_notice.show()
            return
        if not deployment or self.chat_mode.currentIndex()!=2:
            self.conversation_version_notice.hide()
            return
        if self.core.active_conversation_id()==deployment['reference']:
            text='Referência v2 anterior. A troca de versão inicia uma conversa limpa.'
        elif not deployment.get('passed_retention_gate',False):
            text='Este experimento perdeu desempenho em tarefas anteriores. Compare com a referência; a conversação ainda falha.'
        else:
            text='Versão experimental de conversação. As respostas ainda podem estar incorretas.'
        self.conversation_version_notice.setText(text)
        self.conversation_version_notice.show()

    def clear_context(self):
        self.current_source = None
        self.context_label.setText("Contexto: toda a biblioteca")
        self.context_bar.hide()

    def selected_source(self):
        index = self.sources_table.currentRow()
        return self.source_rows[index] if 0 <= index < len(self.source_rows) else None

    def show_source(self):
        source = self.selected_source()
        if source:
            chunks = self.core.memory.chunks(source["id"])
            self.source_preview.setPlainText(f"{source['title']}\nOrigem: {source['origin']}\nAtualização UTC: {source['updated']}\n\n" + "\n\n".join(f"[{c['id']}] {c['section']}\n{c['text']}" for c in chunks[:60]))

    def use_source(self):
        source = self.selected_source()
        if source:
            self.chat_mode.setCurrentIndex(0)
            self.current_source = source["id"]
            self.context_label.setText("Contexto: " + source["title"])
            self.context_bar.show()
            self.go('chat')

    def search_library(self):
        query = self.library_query.text()
        self.start_job("Pesquisar biblioteca", lambda progress: self.core.memory.search(query, limit=30), lambda hits: self.source_preview.setPlainText("\n\n".join(f"{h['title']} — {h['section']}\n{h['text']}\nOrigem: {h['origin']}" for h in hits) or "Nenhum trecho encontrado."))

    def edit_subject(self):
        source = self.selected_source()
        if source:
            value, ok = QInputDialog.getText(self, "Assunto", "Relacionar documentos pelo mesmo assunto:", text=source["subject"])
            if ok:
                try:
                    self.core.memory.set_subject(source["id"], value)
                    self.refresh_sources()
                except Exception as exc:
                    self.notify_error(str(exc))

    def reindex_selected(self):
        source = self.selected_source()
        if source:
            def result(value):
                self.notice.setText(f"Contexto reorganizado: {value['before']} → {value['after']} trechos. O texto foi preservado; isso não treina pesos.")
                self.notice.show()
                self.refresh_sources()
            self.start_job("Reorganizar contexto", lambda progress: self.core.memory.reindex_source(source["id"]), result, learning=True)

    def update_source(self):
        source = self.selected_source()
        if not source:
            return
        if source["kind"] == "web":
            self.url.setText(source["origin"])
            self.subject.setText(source["subject"])
            self.web_mode.setCurrentIndex(0)
            self.terms.setChecked(False)
            self.material_tabs.setCurrentIndex(0)
            self.go('materials')
            self.learning_tabs.setCurrentIndex(0)
        else:
            self.add_files([source["origin"]])

    def delete_source(self):
        source = self.selected_source()
        if not source:
            return
        answer = QMessageBox.question(self, "Excluir conhecimento", f"Excluir “{source['title']}”, trechos e snapshots derivados? O arquivo original será preservado. Pesos já treinados podem manter influência do texto.")
        if answer == QMessageBox.StandardButton.Yes:
            self.start_job("Excluir fonte", lambda progress: self.core.delete_source(source["id"]))
            if self.current_source == source["id"]:
                self.clear_context()
            self.source_preview.clear()

    def correction_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Registrar correção autorizada")
        dialog.resize(650, 520)
        layout = QFormLayout(dialog)
        layout.addRow(label('Salvar não muda a resposta atual. Se você autorizar e houver texto suficiente, a correção poderá entrar no treino de escrita. Ela não entra no treino de conversa.'))
        original = QPlainTextEdit(self.core.correction_target or (self.core.last_answer[1].get("display_text", self.core.last_answer[1]["text"]) if self.core.last_answer else ""))
        correction = QPlainTextEdit()
        subject, source = QLineEdit(), QLineEdit()
        validated = QCheckBox("Conferi a correção e autorizo seu uso no treino de escrita")
        layout.addRow("Resposta original", original)
        layout.addRow("Correção", correction)
        layout.addRow("Assunto", subject)
        layout.addRow("Fonte da correção", source)
        layout.addRow(validated)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.core.memory.correct(original.toPlainText(), correction.toPlainText(), subject.text(), source.text(), validated.isChecked())
                self.statusBar().showMessage("Correção registrada. Nenhum peso foi alterado.")
            except Exception as exc:
                self.notify_error(str(exc))

    def manage_records(self, kind):
        corrections = kind == "corrections"
        dialog = QDialog(self)
        dialog.setWindowTitle("Correções salvas" if corrections else "Conversas salvas")
        dialog.resize(1000, 550)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('Correções autorizadas podem entrar no treino de escrita. Não entram no treino de conversa.' if corrections else 'Este histórico não é usado nos treinos.'))
        keys = ["id", "created", "subject", "correction", "source", "validated"] if corrections else ["id", "created", "question", "answer", "mode"]
        names = {'id':'Número', 'created':'Data', 'subject':'Assunto', 'correction':'Correção', 'source':'Fonte',
                 'validated':'Uso autorizado', 'question':'Sua pergunta', 'answer':'Resposta', 'mode':'Tipo de resposta'}
        grid = table([names[k] for k in keys])
        rows = self.core.db.query(f"SELECT * FROM {kind} ORDER BY id DESC LIMIT 500")
        fill_table(grid, rows, keys)
        layout.addWidget(grid)
        controls = QHBoxLayout()
        def selected():
            i = grid.currentRow()
            return rows[i] if 0 <= i < len(rows) else None
        def refresh():
            rows[:] = self.core.db.query(f"SELECT * FROM {kind} ORDER BY id DESC LIMIT 500")
            fill_table(grid, rows, keys)
        def remove():
            row = selected()
            if row and QMessageBox.question(dialog, "Excluir registro", "Excluir este registro permanentemente?") == QMessageBox.StandardButton.Yes:
                self.core.db.execute(f"DELETE FROM {kind} WHERE id=?", (row["id"],))
                if corrections:
                    for path in (self.core.config.data / "datasets").glob("*.json"):
                        value = json.loads(path.read_text(encoding="utf-8"))
                        if any(p.get("correction_id") == row["id"] for p in value.get("provenance", [])):
                            path.unlink()
                refresh()
        def validate():
            row = selected()
            if row:
                try:
                    with self.core.gate.writing():
                        self.core.db.execute("UPDATE corrections SET validated=? WHERE id=?", (0 if row["validated"] else 1, row["id"]))
                    refresh()
                except Exception as exc:
                    self.notify_error(str(exc))
        controls.addWidget(button("Excluir registro", remove))
        if corrections:
            controls.addWidget(button("Validar / retirar validação", validate))
        controls.addWidget(button("Fechar", dialog.accept))
        layout.addLayout(controls)
        dialog.exec()

    def manage_corrections(self):
        if not self.job:
            self.manage_records("corrections")

    def manage_interactions(self):
        if not self.job:
            self.manage_records("interactions")

    def show_interactions(self):
        rows = self.core.db.query("SELECT * FROM interactions ORDER BY id DESC LIMIT 40")
        import re
        def readable(text):
            return "\n".join(line for line in text.splitlines() if not re.match(r"^(Origem:|Fonte \[)", line))
        self.transcript.setPlainText("Conversas salvas — este histórico não é usado nos treinos.\n\n" + ("\n\n".join(f"Você\n{r['question']}\n\nTAC · {answer_origin(r)}\n{readable(r['answer'])}" for r in reversed(rows)) or "Nenhuma conversa autorizada foi armazenada."))

    def train(self):
        state = snapshot(self.core)
        if not state["text_ready"]:
            self.notify_error("Ainda não há material estudado suficiente para treinar. Adicione um documento/site na Central e conclua o estudo primeiro.")
            self.go('materials')
            return
        if not state["enabled"]:
            self.notify_error("Ative a permissão de aprendizado na Central antes de iniciar o treinamento neural.")
            return
        self.current_training_kind = 'neural'
        self.start_job("Treinamento neural", lambda progress: self.core.trainer.train(progress), self.training_result, learning=True)

    def train_conversation(self):
        v3_path=self.core.config.data/'tac_v3/progress.json'
        v3=getattr(self,'_conversation_v3_deployment',None)
        if v3_path.exists() and (json.loads(v3_path.read_text(encoding='utf-8'))['phase']!='complete'
                               or (v3 and self.core.active_conversation_id()==v3['checkpoint'])):
            self.train_v3()
            return
        if (self.core.config.data/'tac_v2/progress.json').exists():
            self.train_v2()
            return
        self.request_dialogue_training()

    def train_v2(self, checkpoint=None):
        if (self.core.config.data/'tac_v2_conversation_20261018/progress.json').exists() and checkpoint is None:
            self.train_conversation_continuation()
            return
        def run(progress):
            from .neural.v2_training import V2Trainer
            trainer = V2Trainer(self.core.config,self.core.db,self.core.gate,device='cuda')
            if checkpoint:trainer.resume(checkpoint)
            def update(message):
                total=trainer.pretrain_steps+trainer.sft_target_steps
                progress(min(99,int(100*trainer.global_step/max(1,total))),message)
            return trainer.run(update)
        self.start_job('Treinamento TAC v2',run,self.v2_result,learning=True)

    def train_conversation_continuation(self, checkpoint=None):
        def run(progress):
            from .neural.conversation_training import ConversationContinuation
            trainer=ConversationContinuation(self.core.config,self.core.db,self.core.gate)
            if checkpoint:trainer.resume(checkpoint)
            return trainer.run(lambda message:progress(min(99,int(100*trainer.step/trainer.max_steps)),message))
        self.start_job('Continuação de conversação TAC v2',run,self.continuation_result,learning=True)

    def train_v3(self, checkpoint=None):
        def run(progress):
            from .neural.v3_training import V3Trainer
            trainer=V3Trainer(self.core.config,self.core.db,self.core.gate)
            if checkpoint and not (trainer.base/'training_result.json').exists():trainer.resume(checkpoint)
            return trainer.run(lambda message:progress(0,message))
        self.start_job('Retomar treinamento',run,self.v3_result,learning=True)

    def v3_result(self, result):
        self.train_label.setText('TAC v3 interrompida e salva para retomada.' if result.get('status')=='interrompido'
                                else 'Orçamento da TAC v3 concluído. Compare as gerações; não há promoção automática nem uma nova rodada ao clicar em retomar.')
        self.training_progress.hide()
        self.refresh_training()

    def continuation_result(self, result):
        text=('Continuação interrompida e salva. Retome o mesmo treinamento.' if result.get('status')=='interrompido'
              else 'Continuação concluída. Compare as respostas livres e as limitações antes de avaliar a qualidade.')
        self.train_label.setText(text)
        self.training_progress.hide()
        self.refresh_training()

    def v2_result(self, result):
        if result.get('status')=='interrompido':
            text='TAC v2 interrompida. A sessão foi salva; use Retomar treinamento TAC v2 para continuar.'
        else:
            text='TAC v2 experimental disponível em Conversar. O treino de linguagem e conversação foi concluído; a rede ainda pode produzir respostas incorretas.'
        self.train_label.setText(text)
        self.training_progress.hide()
        self.refresh_training()

    def conversation_result(self, result):
        self.present_training_result(result, 'conversation')

    def training_result(self, result):
        if "validation" in result:
            self.present_training_result(result, 'neural')
        else:
            self.technical_train_label.setText(f"Reforço {result['status']} • acerto: {result['baseline']:.1%} → {result['accuracy']:.1%}\nAvaliado nas mesmas 16 somas treinadas; não mede generalização.")
            self.train_label.setText('Experimento separado concluído. Ele não treina a conversa da TAC. Veja o resultado nas ferramentas avançadas.')

    def selected_checkpoint(self):
        index = self.checkpoints_table.currentRow()
        return self.checkpoint_rows[index]["id"] if 0 <= index < len(self.checkpoint_rows) else None

    def resume(self):
        checkpoint = self.selected_checkpoint()
        if checkpoint:
            row = next(row for row in self.checkpoint_rows if row["id"] == checkpoint)
            if row.get('status')=='descartado_v2':
                self.notify_error('Esta versão antiga foi descartada. Selecione um checkpoint da TAC v2.')
                return
            run = self.core.db.query('SELECT config,detail FROM runs WHERE id=?',(row['run_id'],))
            if run and json.loads(run[0]['config']).get('generation')=='TAC-v3':
                self.train_v3(checkpoint)
                return
            if run:
                try:continuation=json.loads(run[0]['detail']).get('continuation')
                except (ValueError,TypeError,AttributeError):continuation=False
                if continuation or json.loads(run[0]['config']).get('continuation'):
                    self.train_conversation_continuation(checkpoint)
                    return
            if run and json.loads(run[0]['config']).get('generation')=='TAC-v2':
                self.train_v2(checkpoint)
                return
            chat = row.get("kind") == "conversation"
            if chat:
                self.request_dialogue_training(resume=checkpoint)
                return
            self.current_training_kind = 'conversation' if chat else 'neural'
            engine = self.core.conversation if chat else self.core.trainer
            self.start_job("Retomar treinamento", lambda progress: engine.train(progress, resume=checkpoint), self.conversation_result if chat else self.training_result, learning=True)

    def restore_checkpoint(self):
        checkpoint = self.selected_checkpoint()
        if checkpoint:
            row = next(row for row in self.checkpoint_rows if row["id"] == checkpoint)
            engine = self.core.conversation if row.get("kind") == "conversation" else self.core.trainer
            self.start_job("Restaurar checkpoint", lambda progress: engine.activate(checkpoint), learning=True)

    def reinforce(self):
        def run(progress):
            from .neural.reinforcement import run_experiment
            return run_experiment(self.core.config, self.core.db, self.core.gate, progress)
        self.start_job("Experimento de reforço", run, self.training_result, learning=True)

    def show_conversation_research(self):
        controlled=self.core.config.root/'reports/experimentos_0130'
        if (controlled/'summary.json').is_file():
            from .experiment_view import ExperimentDialog
            ExperimentDialog(controlled,self).exec();return
        path=self.core.config.root/'reports/research_0120/summary.json'
        if not path.is_file():
            QMessageBox.information(self,'Experimentos de conversa','Ainda não há uma rodada de experimentos concluída nesta instalação.');return
        report=json.loads(path.read_text(encoding='utf-8'))
        dialog=QDialog(self);dialog.setWindowTitle('Experimentos de conversa — resultados reais');dialog.resize(850,640)
        layout=QVBoxLayout(dialog);text=QPlainTextEdit();text.setReadOnly(True);layout.addWidget(text)
        lines=['Nenhum destes modelos substituiu a versão ativa.','120 passos por piloto; mesmo ponto de partida. Teste final reservado permanece intacto.','']
        for row in report['candidates']:
            lines.append(f"{row['name']}: taxa {row['learning_rate']:.5f}; respostas básicas {row['basic_exact']}/{row['basic_examples']}; respostas importadas de validação {row['imported_exact']}/{row['imported_examples']}.")
        reasons={'loss_basica':'a rede ficou menos segura nos exercícios básicos',
                 'conversas_revisadas':'as conversas revisadas ainda não atingiram a qualidade exigida'}
        lines+=['','Candidato comparado para diagnóstico: '+report['chosen'],
            'Por que não foi aprovado: '+ '; '.join(reasons.get(k,k) for k,v in report['checks'].items() if not v)+'.',
            'Equivalência semântica geral permanece pendente de revisão humana. As regras automáticas de fatos são limitadas.',
            'Os arquivos completos ficam em reports/research_0120.']
        development=path.parent/'novel_development.results.json'
        if development.is_file():
            lines+=['','Conversas novas de desenvolvimento (uma única avaliação, sem treino posterior):']
            observed=json.loads(development.read_text(encoding='utf-8'))
            for model,modes in observed['models'].items():
                for mode,result in modes.items():
                    scores=[v['factual_correct'] for v in result['by_skill'].values()]
                    lines.append(f"{model} / {mode}: {sum(v['correct'] for v in scores)}/{sum(v['total'] for v in scores)} fatos reconhecidos por regras limitadas.")
        text.setPlainText('\n'.join(lines));layout.addWidget(button('Fechar',dialog.accept));dialog.exec()

    def clear_conversation_documents(self):
        if self.job:
            self.notify_error('Aguarde a atividade atual terminar.'); return
        self.core.config.update(conversation_document_dataset='')
        self.statusBar().showMessage('Próxima etapa usará a prática de escrita do programa. Dados e modelos anteriores foram preservados.')

    def prepare_conversation_documents(self):
        if self.job:
            self.notify_error('Aguarde a atividade atual terminar.'); return
        if not self.core.gate.enabled:
            self.notify_error('Permita novos estudos antes de preparar os textos.'); return
        from PySide6.QtWidgets import QListWidgetItem
        dialog=QDialog(self);dialog.setWindowTitle('Textos opcionais para a próxima etapa');layout=QVBoxLayout(dialog)
        layout.addWidget(label('Escolha materiais que você autoriza usar nos pesos. Eles entram como prática de escrita, não como respostas prontas. Preparar não inicia treino.'))
        choices=QListWidget();layout.addWidget(choices)
        for source in self.core.memory.sources():
            item=QListWidgetItem(source['title']);item.setData(Qt.ItemDataRole.UserRole,source['id']);item.setCheckState(Qt.CheckState.Unchecked);choices.addItem(item)
        layout.addWidget(button('Preparar os selecionados',dialog.accept,True));layout.addWidget(button('Cancelar',dialog.reject))
        dialog.resize(650,500)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        ids=[choices.item(i).data(Qt.ItemDataRole.UserRole) for i in range(choices.count()) if choices.item(i).checkState()==Qt.CheckState.Checked]
        if not ids:
            self.notify_error('Selecione pelo menos um material.');return
        from .neural.document_replay import prepare_document_replay
        def done(result):
            self.core.config.update(conversation_document_dataset=result['dataset_hash'])
            self.notice.setText(f"Textos preparados: {result['document_samples']} trechos adicionais, até 10% dos trechos dos lotes de escrita. Nenhum treino foi iniciado.")
            self.notice.show()
        self.start_job('Preparar textos opcionais',lambda progress:prepare_document_replay(self.core,ids),done,learning=True)

    def export_diagnostic(self):
        if self.job is not None:
            self.notify_error('Aguarde a atividade atual terminar antes de exportar o diagnóstico.')
            return
        path,_ = QFileDialog.getSaveFileName(self,'Exportar diagnóstico completo',
            str(self.core.config.root/'TAC_Diagnostico.zip'),'Diagnóstico ZIP (*.zip)')
        if not path:
            return
        from .diagnostic_export import export_diagnostic
        self.start_job('Exportar diagnóstico completo',
            lambda progress:export_diagnostic(self.core,path,progress,overwrite=True),self.diagnostic_exported)

    def diagnostic_exported(self,result):
        self.notice.setText(f"Diagnóstico exportado: {result['path']}\n{result['files']} relatórios/manifesto; {result['runs']} execuções. Nenhum treino foi iniciado. Revise o resumo antes de compartilhar.")
        self.notice.show()

    def show_runs(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Avaliações reais — histórico de treinamentos")
        dialog.resize(1050, 600)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps({"runs": self.core.db.query("SELECT * FROM runs ORDER BY started DESC LIMIT 30"), "metrics": self.core.db.query("SELECT * FROM metrics ORDER BY id DESC LIMIT 200")}, ensure_ascii=False, indent=2))
        layout.addWidget(text)
        layout.addWidget(button("Fechar", dialog.accept))
        dialog.exec()

    def monitor_gpu(self):
        def read(progress):
            import torch
            result = {"torch": torch.__version__, "CUDA disponível": torch.cuda.is_available(), "CUDA runtime": torch.version.cuda}
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                result.update({"GPU": torch.cuda.get_device_name(0), "VRAM total MB": round(total / 2**20), "VRAM livre MB": round(free / 2**20), "VRAM alocada pela TAC MB": round(torch.cuda.memory_allocated() / 2**20, 1)})
            return result
        self.start_job("Diagnóstico CUDA", read, lambda value: self.gpu_label.setText(json.dumps(value, ensure_ascii=False, indent=2)))

    def save_settings(self):
        if self.job:
            self.notify_error("Aguarde a operação atual terminar para alterar configurações.")
            return
        values = {key: widget.value() for key, widget in self.settings_widgets.items()}
        values['conversation_transfer_practice']=self.transfer_practice_check.isChecked()
        for key in ('conversation_stability','conversation_sampler','conversation_generation'):
            values[key]=getattr(self,key+'_setting').currentData()
        values.update(device=self.device_setting.currentText(), theme=self.theme_setting.currentData(), auto_on_start=self.startup_check.isChecked(), save_interactions=self.interactions_check.isChecked())
        new_dir = self.data_setting.text().strip()
        try:
            self.core.config.validate({**self.core.config.values, **values, "data_dir": new_dir})
            if new_dir != self.core.config.get("data_dir"):
                from .backups import relocate_data
                def move(progress):
                    self.core.config.update(**values)
                    return relocate_data(self.core, new_dir)
                self.start_job("Migrar dados", move, self.mark_restart)
            else:
                self.core.config.update(**values)
                apply_theme(QApplication.instance(), self.core.config)
                self.statusBar().showMessage("Configurações salvas.")
            self.auto_armed = self.auto_armed or values["auto_on_start"]
        except Exception as exc:
            self.notify_error(str(exc))

    def backup(self):
        path, _ = QFileDialog.getSaveFileName(self, "Criar cópia de segurança (dentro da TAC)", str(self.core.config.data / "backups" / ("TAC_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")), "Cópia da TAC (*.zip)")
        if path:
            self.start_job("Exportar backup", lambda progress: export_backup(self.core, path))

    def restore_backup(self):
        path, _ = QFileDialog.getOpenFileName(self, "Recuperar uma cópia de segurança", str(self.core.config.data / "backups"), "Cópia da TAC (*.zip)")
        if path:
            self.start_job("Importar backup", lambda progress: import_backup(self.core, path), self.mark_restart)

    def mark_restart(self, value):
        self.restarting = True

    def restart(self):
        subprocess.Popen([sys.executable, "-m", "tac"], cwd=self.core.config.root,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.close()

    def refresh_sources(self):
        self.source_rows = self.core.memory.sources(self.subject_filter.text())
        display = [{**r,'kind': 'Site' if r['kind'] == 'web' else 'Arquivo'} for r in self.source_rows]
        fill_table(self.sources_table, display, ["id", "title", "subject", "kind", "chunks", "origin"])

    def refresh_queue(self):
        previous = self.queue_selected()
        selected_id = previous["id"] if previous else None
        scroll = self.queue_table.verticalScrollBar().value()
        self.queue_rows = self.core.db.query("SELECT * FROM queue ORDER BY id DESC LIMIT 100")
        display = []
        for row in self.queue_rows:
            options = json.loads(row["options"])
            detail = user_error(row["progress"]) if row["status"] == "erro" else row["progress"]
            if not detail and row["status"] == "pendente":
                detail = "Falta clicar em Guardar materiais agora." if self.core.gate.enabled else "Falta permitir e iniciar pelo botão verde."
            display.append({**row, "kind": "Site" if row["kind"] == "web" else "Documento", "target": options.get("origin", row["target"]), "status": QUEUE_LABELS.get(row["status"], row["status"]), "progress": detail})
        with QSignalBlocker(self.queue_table):
            fill_table(self.queue_table, display, ["id", "kind", "target", "status", "progress"])
            for i, item in enumerate(self.queue_rows):
                if item["id"] == selected_id:
                    self.queue_table.selectRow(i)
                    break
            self.queue_table.verticalScrollBar().setValue(scroll)
        history = self.core.db.query("SELECT * FROM study_history ORDER BY id DESC LIMIT 100")
        for item in history:
            item["status"] = RESULT_LABELS.get(item["status"], item["status"])
        fill_table(self.history_table, history, ["created", "origin", "status", "detail"])
        pending = sum(r["status"] == "pendente" for r in self.queue_rows)
        self.queue_tabs.setTabText(0, f"Fila de materiais · {pending} aguardando")
        if selected_id:
            self.show_queue_detail()

    def show_queue_detail(self):
        row = self.queue_selected()
        if not row:
            self.queue_detail.clear()
            return
        options = json.loads(row["options"])
        origin = options.get("origin", row["target"])
        status = row["status"]
        if status == "erro":
            detail = user_error(row["progress"])
        elif status == "pendente":
            detail = "A leitura ainda não começou. Clique em Guardar materiais agora. Se estiver pausado, Permitir e guardar materiais autoriza e inicia."
        else:
            detail = row["progress"] or "Acompanhe o progresso acima."
        self.queue_detail.setPlainText(f"{origin}\n{QUEUE_LABELS.get(status, status)}\n{detail}")
        if hasattr(self, "retry_button"):
            self.retry_button.setEnabled(self.job is None and status in ("erro", "parcial", "pendente"))

    def refresh_activity(self):
        state = snapshot(self.core)
        busy = self.job is not None
        title, detail = state["title"], state["detail"]
        if busy:
            title = "ESTUDANDO AGORA" if self.job_name == "Estudar fila" else ("TREINANDO A REDE NEURAL" if self.job_name in ("Treinamento neural", "Retomar treinamento", "Experimento de reforço", "Praticar conversas") else "EM ANDAMENTO")
            detail = self.stage_text or self.job_name
            if self.core.gate.paused.is_set():
                title, detail = "TREINAMENTO PAUSADO", "Clique em Continuar treinamento ou Interromper e salvar."
            if self.job.learning and self.core.gate.cancel.is_set():
                title = "INTERROMPENDO COM SEGURANÇA"
        elif self.last_failure:
            title, detail = "OPERAÇÃO NÃO CONCLUÍDA", self.last_failure[:300]
        readable_title = {"PRONTA PARA CONSULTAR": "Pronta para conversar", "NENHUM MATERIAL ESTUDADO AINDA": "Pronta · você pode adicionar materiais em Ensinar", "ESTUDANDO AGORA": "Lendo e guardando materiais…", "TREINANDO A REDE NEURAL": "Praticando exercícios…", "EM ANDAMENTO": "Trabalhando…", "TREINAMENTO PAUSADO": "Exercícios pausados", "OPERAÇÃO NÃO CONCLUÍDA": "Não foi possível concluir a atividade", "INTERROMPENDO COM SEGURANÇA": "Interrompendo com segurança…"}.get(title, title.capitalize())
        if busy:
            readable_title = ('Interrompendo com segurança…' if self.job.learning and self.core.gate.cancel.is_set()
                              else 'Atividade pausada. Clique em Continuar.' if self.core.gate.paused.is_set() else self.stage_text)
        elif self.last_failure:
            readable_title = 'Não foi possível concluir a atividade'
        elif state['pending']:
            readable_title = f"{state['pending']} material(is) aguardando você iniciar"
        elif state['sources']:
            readable_title = 'Pronta para consultar'
        elif state['state'] == 'empty':
            readable_title = 'Pronta. Escolha como quer usar a TAC.'
        self.activity_title.setText(readable_title)
        self.activity_detail.setText(getattr(self, 'raw_stage_text', detail)[:450] if busy else getattr(self,'raw_failure',detail)[:450] if self.last_failure else detail[:450])
        neural = "modelo treinado experimental" if state["active_checkpoint"] else "sem modelo treinado"
        self.activity_counts.setText(f"Biblioteca: {state['sources']} fonte(s) · {state['chunks']} trechos  |  Fila: {state['pending']} aguardando, {state['failed']} com erro  |  Rede: {neural}")
        self.refresh_gate()
        self.auto_study_notice.setVisible(self.core.config.get('auto_train'))
        pending = state["pending"]
        self.study_button.setText(f"Guardar materiais agora ({pending})" if state["enabled"] else f"Permitir e guardar materiais ({pending})")
        self.study_button.setEnabled(not busy and pending > 0)
        self.study_button.setToolTip("Permite novos estudos e guarda os materiais da fila para consulta." if not state["enabled"] else "Guarda os materiais da fila para consulta. Não treina a conversa.")
        self.stop_button.setEnabled(busy and self.job_name == "Estudar fila" and not self.core.gate.cancel.is_set())
        if pending:
            hint = "Os materiais aguardam você iniciar. Use o botão Guardar materiais agora."
        elif state["failed"]:
            hint = "Selecione o material com erro para ver o motivo. Use Tentar novamente e estudar após corrigir o problema."
        else:
            hint = "Adicione um material e depois clique em Guardar materiais agora."
        self.queue_hint.setText(hint)
        self.train_button.setEnabled(not busy and state["can_train"])
        v2_ready = (self.core.config.data/'tac_v2/progress.json').exists()
        self.conversation_train_button.setEnabled(not busy and state["enabled"] and (v2_ready or bool(state["active_checkpoint"])))
        self.pause_button.setEnabled(busy and self.job_name in ("Treinamento neural", "Retomar treinamento", "Experimento de reforço", "Praticar conversas"))
        self.pause_button.setVisible(busy and self.job_name in ('Treinamento neural', 'Retomar treinamento'))
        self.pause_button.setText("Continuar treinamento" if self.core.gate.paused.is_set() else "Pausar treinamento")
        self.train_stop_button.setEnabled(self.pause_button.isEnabled())
        self.train_stop_button.setVisible(not self.pause_button.isHidden())
        if not state["text_ready"]:
            self.training_readiness.setText("Treinamento indisponível: primeiro conclua o estudo de um documento ou site com texto suficiente.")
        elif not state["enabled"]:
            self.training_readiness.setText("Há material de treino, mas a permissão de aprendizado está desligada. Consultas documentais continuam disponíveis.")
        else:
            self.training_readiness.setText("Material disponível para treinar. Iniciar este treino modificará os pesos; o Chat documental já funciona sem essa etapa.")
        self.chat_mode.model().item(1).setEnabled(bool(state["active_checkpoint"]))
        has_conversation = bool(self.core.active_conversation_id())
        self.chat_mode.model().item(2).setEnabled(has_conversation)
        if has_conversation:
            self.chat_tools.show()
        if not state["active_checkpoint"] and self.chat_mode.currentIndex() == 1:
            self.chat_mode.setCurrentIndex(0)
        if self.chat_mode.currentIndex() == 0:
            self.chat_readiness.setText('Aqui a TAC busca nos seus textos. Isso não ensina nada a ela.')
        self.send_button.setEnabled(not busy)
        self.research_button.setEnabled(not busy)
        self.research_auto.setEnabled(not busy)
        self.preferences_research_auto.setEnabled(not busy)
        enabled_research = self.core.config.get('research_auto')
        for checkbox in (self.research_auto, self.preferences_research_auto):
            with QSignalBlocker(checkbox):
                checkbox.setChecked(enabled_research)
        self.research_stop.setVisible(busy and self.job_name == 'Conversa com consulta web')
        self.add_url_button.setEnabled(not busy)
        self.practice_button.setEnabled(not busy)
        self.export_diagnostic_button.setEnabled(not busy)
        self.transfer_practice_check.setEnabled(not busy)
        self.practice_button.setText("Treinar escolha de respostas preparadas" if state["enabled"] else "Permitir e treinar respostas preparadas")
        self.practice_pause_button.setEnabled(busy and self.job_name == "Praticar conversas")
        self.practice_pause_button.setText("Continuar" if self.core.gate.paused.is_set() else "Pausar")
        self.practice_stop_button.setEnabled(busy and self.job_name == "Praticar conversas")
        selected = self.queue_selected()
        self.retry_button.setEnabled(not busy and selected is not None and selected["status"] in ("erro", "pendente", "parcial"))
        self.retry_button.setText("Tentar novamente" if state["enabled"] else "Permitir e tentar novamente")
        self.refresh_import_card()
        if not busy:
            self.study_query_button.setVisible(bool(state["sources"]))

    def refresh_training(self):
        self.refresh_training_help()
        active = self.core.db.query("SELECT m.value,r.detail FROM meta m JOIN checkpoints c ON c.id=m.value JOIN runs r ON r.id=c.run_id WHERE m.key='active_checkpoint'")
        active_id = active[0]["value"] if active else ""
        chat_id = self.core.active_conversation_id() or ""
        chat_parameters = None
        self.checkpoint_rows = self.core.db.query("SELECT c.*,r.kind FROM checkpoints c JOIN runs r ON r.id=c.run_id WHERE r.kind IN ('neural','conversation') ORDER BY (c.id IN (?,?)) DESC,c.created DESC,c.rowid DESC LIMIT 100", (active_id, chat_id))
        display = [{**row, "status": ("ATIVO · " if row["id"] in (active_id, chat_id) else "") + ("diálogo · " if row["kind"] == "conversation" else "leitor · ") + row["status"]} for row in self.checkpoint_rows]
        fill_table(self.checkpoints_table, display, ["id", "created", "step", "val_loss", "status", "run_id"])
        if active:
            try:
                parameters = json.loads(active[0]["detail"]).get("parameters")
            except (ValueError, TypeError):
                parameters = None
            count = f"{parameters:,}".replace(",", ".") if parameters else "contagem disponível no Monitoramento"
            self.model_label.setText(f"Transformer ativo · {count}" + (" parâmetros" if parameters else ""))
            self.model_label.setToolTip("Checkpoint: " + active[0]["value"])
            if chat_id:
                chat_rows = self.core.db.query("SELECT r.detail FROM checkpoints c JOIN runs r ON r.id=c.run_id WHERE c.id=?", (chat_id,))
                try:
                    chat_parameters = json.loads(chat_rows[0]["detail"])["parameters"]
                    self.model_label.setText(f"Leitura: {count} parâmetros · Diálogo neural: {chat_parameters:,} parâmetros".replace(",", "."))
                except (ValueError, TypeError, KeyError, IndexError):
                    pass
        else:
            self.model_label.setText("Nenhum modelo treinado aprovado • consulta documental disponível")
            if chat_id and (self.core.config.data/'tac_v2/training_result.json').exists():
                self.model_label.setText('TAC v2 experimental · 10.004.544 parâmetros · conversa neural')
        v3=getattr(self,'_conversation_v3_deployment',None)
        if v3 and chat_id==v3['checkpoint']:
            chat_parameters=v3['parameters']
            self.model_label.setText(f"TAC v3 experimental · {chat_parameters:,} parâmetros · qualidade não aprovada".replace(',', '.'))
        caption_key = (chat_id, chat_parameters)
        if self.chat_mode.currentIndex() == 2 and getattr(self, "_chat_caption_key", None) != caption_key:
            self._chat_caption_key = caption_key
            self.chat_mode_changed(2)
        runs = self.core.db.query("SELECT id,kind,status,detail FROM runs WHERE kind IN ('neural','conversation') ORDER BY started DESC LIMIT 1")
        metrics = self.core.db.query("SELECT * FROM metrics WHERE run_id=? ORDER BY step,id", (runs[0]["id"],)) if runs else []
        by_step = {}
        for row in metrics:
            previous = by_step.get(row["step"])
            if previous and row["train_loss"] is None:
                row["train_loss"] = previous["train_loss"]
            by_step[row["step"]] = row
        self.plot.rows = list(by_step.values())
        self.plot.update()
        if self.job is None and runs and runs[0]["status"] != "treinando":
            try:
                result = json.loads(runs[0]["detail"])
                if result.get('generation')=='TAC-v3':
                    self.train_label.setText('Orçamento da TAC v3 concluído. Modelo experimental disponível para comparação; qualidade ainda não aprovada.')
                elif "validation" in result:
                    if runs[0]["kind"] == "conversation":
                        self.conversation_result(result)
                    else:
                        self.training_result(result)
            except (ValueError, TypeError):
                pass
        report_path = self.core.config.data / "dialogue_report.json"
        if self.job is None and report_path.exists():
            stamp = report_path.stat().st_mtime_ns
            if stamp != self._last_dialogue_report:
                self.dialogue_result(json.loads(report_path.read_text(encoding="utf-8")))
                self._last_dialogue_report = stamp

    def refresh(self):
        self.refresh_gate()
        self.refresh_sources()
        self.refresh_queue()
        self.refresh_training()
        self.refresh_activity()

    def tick(self):
        self.slow_ticks += 1
        self.refresh_activity()
        if (self.nav.currentRow() == 7 or self.job) and self.slow_ticks % 2 == 0 and self.monitor_job is None:
            self.monitor_job = MonitorJob(self.core)
            self.monitor_job.measured.connect(self.present_monitor)
            self.monitor_job.finished.connect(self.monitor_finished)
            self.monitor_job.start()
        if self.job:
            if self.slow_ticks % 3 == 0:
                self.refresh_queue()
                self.refresh_training()
            return
        if self.auto_armed and self.core.auto_due() and time.monotonic() - self.last_auto_attempt > self.core.config.get("auto_interval_minutes") * 60:
            if self.core.memory.sources():
                self.last_auto_attempt = time.monotonic()
                self.train()
                return
        if time.monotonic() - self.last_backup_attempt > 60 and self.core.backup_due():
            self.last_backup_attempt = time.monotonic()
            target = self.core.config.data / "backups" / ("auto_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")
            def backup(progress):
                result = export_backup(self.core, target)
                previous = sorted(target.parent.glob("auto_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                for path in previous[self.core.config.get("backup_keep"):]:
                    path.unlink()
                return result
            self.start_job("Backup automático local", backup)
            return

    def present_monitor(self, info):
        if "error" in info:
            self.monitor.setPlainText("Não foi possível atualizar o diagnóstico: " + info["error"])
            return
        self.monitor.setPlainText(f"CPU do sistema: {info['cpu_percent']:.1f}%\nRAM do sistema: {info['ram_percent']:.1f}%\nRAM livre: {info['ram_available_mb']:.0f} MB\nRAM da TAC: {info['process_mb']:.0f} MB\n\nDocumentos: {info['sources']}\nTrechos indexados: {info['chunks']}\nDados no disco: {info['disk_mb']:.2f} MB\n\nOperação: {info['operation']}\nRAM máxima configurada: {self.core.config.get('ram_mb')} MB\nVRAM máxima configurada: {self.core.config.get('vram_mb')} MB")
        models = info.get("models", [])
        if models:
            self.monitor.appendPlainText("\nModelos ativos (pesos locais):\n" + "\n".join(f"{m['name']}: {m['parameters']:,} parâmetros".replace(",", ".") for m in models))
            self.monitor.appendPlainText("Mais livros alteram os pesos durante o treino; não aumentam a quantidade de parâmetros.")
        gpu = info.get("gpu")
        cuda = "disponível" if info.get("cuda") else ("indisponível" if info.get("cuda") is False else "runtime ainda não carregado; use Atualizar GPU / CUDA")
        if gpu:
            self.gpu_label.setText(f"{gpu['nome']} • uso: {gpu['uso_percent']}%\nVRAM do sistema: {gpu['vram_used_mb']} / {gpu['vram_total_mb']} MB\nVRAM dos tensores da TAC: {info.get('tac_vram_mb', 0)} MB\nCUDA: {cuda}")
        else:
            self.gpu_label.setText("GPU não detectada pelo monitor. CUDA: " + cuda)

    def monitor_finished(self):
        self.monitor_job.deleteLater()
        self.monitor_job = None
        if self.pending_close and self.job is None:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        self.timer.stop()
        if self.job is not None or self.monitor_job is not None:
            self.pending_close = True
            self.cancel()
            event.ignore()
            self.statusBar().showMessage("Finalizando com segurança. A janela fechará após salvar o progresso.")
        else:
            self.timer.stop()
            event.accept()


def launch(core, smoke=False, screenshot=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("TAC")
    app.setOrganizationName("TAC Local")
    app.setStyle("Fusion")
    apply_theme(app, core.config)
    window = Window(core)
    window.show()
    if smoke:
        def finish():
            if screenshot:
                window.grab().save(str(screenshot))
            window.close()
        QTimer.singleShot(1200, finish)
    return app.exec()
