"""Visualização local e somente leitura dos pilotos controlados."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

LABELS={'control':'Controle', 'A_response':'Só respostas da TAC', 'B_balanced':'Conversas balanceadas',
    'C_coverage':'Revisão com mais cobertura','C_representative':'Revisão por tipos de exemplo'}
METRICS={'basic_accuracy':'Respostas básicas exatas','context_accuracy':'Contexto e lógica exatos',
    'natural_accuracy':'Diálogos e correções exatos','imported_accuracy':'Conversas importadas exatas (desenvolvimento)',
    'novel_factual_proxy':'Fatos em situações novas (regra literal limitada)',
    'closed_factual_verified_fraction':'Fatos verificados em respostas restritas (cobertura limitada)',
    'eos_rate':'Respostas com encerramento','role_rate':'Geração indevida de Usuário: (menor é melhor)'}


class ResultBars(QWidget):
    def __init__(self,summary,parent=None):
        super().__init__(parent);self.summary=summary;self.selected_metric='basic_accuracy';self.setMinimumHeight(270)

    def set_metric(self,metric):self.selected_metric=metric;self.update()

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        items=[('Ativa',self.summary['active'].get(self.selected_metric))]
        items += [(LABELS.get(a['arm'],a['arm']),a['metrics'].get(self.selected_metric,{}).get('mean')) for a in self.summary['aggregates']]
        left=260;width=max(20,self.width()-left-95);height=max(30,self.height()//max(1,len(items)))
        for index,(name,value) in enumerate(items):
            y=index*height+10;p.setPen(self.palette().text().color());p.drawText(8,y+18,name)
            p.fillRect(left,y,width,24,QColor('#253653'))
            if value is not None:
                p.fillRect(left,y,int(width*min(1,max(0,value))),24,QColor('#24998f' if index else '#578de0'))
                p.drawText(left+width+8,y+18,f'{value:.1%}')
            else:p.drawText(left+width+8,y+18,'sem dado')


class ExperimentDialog(QDialog):
    def __init__(self,folder,parent=None):
        super().__init__(parent);self.folder=Path(folder)
        self.summary=json.loads((self.folder/'summary.json').read_text(encoding='utf-8'))
        self.setWindowTitle('Experimentos de conversa — TAC 0.13');self.resize(1120,800)
        layout=QVBoxLayout(self)
        count=len(self.summary['rows'])
        title=QLabel(f'Pilotos de aprendizado • {count}/15 resultados • 120 passos');title.setObjectName('title');layout.addWidget(title)
        notice=QLabel('Nenhum piloto substituiu a versão ativa. Teste final reservado não foi usado.\n'
            'Estes resultados de desenvolvimento não demonstram conversa livre. Equivalência semântica aguarda revisão humana.')
        notice.setWordWrap(True);notice.setObjectName('notice');layout.addWidget(notice)
        self.tabs=QTabWidget();layout.addWidget(self.tabs)
        page=QWidget();box=QVBoxLayout(page);self.selector=QComboBox()
        for key,label in METRICS.items():self.selector.addItem(label,key)
        box.addWidget(self.selector);self.chart=ResultBars(self.summary);box.addWidget(self.chart)
        self.selector.currentIndexChanged.connect(lambda:self.chart.set_metric(self.selector.currentData()))
        explanation=QLabel('Barras: média das execuções concluídas de cada variante. A tabela mostra cada resultado. '
            'Os testes de fatos usam regras limitadas; frases não reconhecidas precisam de revisão humana.')
        explanation.setWordWrap(True);box.addWidget(explanation);self.tabs.addTab(page,'Comparação')
        self.table=QTableWidget(0,8);self.table.setHorizontalHeaderLabels(
            ['Variante','Semente','Básicas','Contexto','Diálogos','Importadas','Encerradas','Proteções atingidas'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row in self.summary['rows']:
            at=self.table.rowCount();self.table.insertRow(at)
            values=[LABELS.get(row['arm'],row['arm']),str(row['seed']),row['basic_count'],row['context_count'],
                row['natural_count'],row['imported_count'],f"{row['eos_rate']:.1%}",row['checks_count']]
            for col,value in enumerate(values):self.table.setItem(at,col,QTableWidgetItem(value))
        self.table.resizeColumnsToContents();self.tabs.addTab(self.table,'Todos os pilotos')
        self.details=QPlainTextEdit();self.details.setReadOnly(True)
        self.details.setPlainText('Selecione uma linha em “Todos os pilotos” para ver exposição aos dados, motivos e recursos.')
        self.table.cellClicked.connect(self.show_run);self.tabs.addTab(self.details,'Detalhes do piloto')
        self.report=QPlainTextEdit();self.report.setReadOnly(True)
        path=self.folder/'RELATORIO.md'
        self.report.setPlainText(path.read_text(encoding='utf-8') if path.exists() else 'Relatório em preparação.')
        self.tabs.addTab(self.report,'Conclusões')
        controls=QHBoxLayout();open_reports=QPushButton('Abrir pasta dos relatórios')
        open_reports.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder))))
        controls.addWidget(open_reports)
        older=QPushButton('Abrir relatórios da etapa anterior')
        older.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder.parent/'research_0120'))))
        controls.addWidget(older);close=QPushButton('Fechar');close.clicked.connect(self.accept);controls.addWidget(close)
        layout.addLayout(controls)

    def show_run(self,index,column=0):
        row=self.summary['rows'][index]
        path=(self.folder/row['result_file']).resolve()
        if not path.is_relative_to(self.folder.resolve()):raise ValueError('Resultado fora da pasta de experimentos.')
        result=json.loads(path.read_text(encoding='utf-8'))
        reasons={'loss_basica':'Ficou menos segura nos exercícios básicos.',
            'respostas_basicas':'Perdeu respostas básicas demais.', 'contexto_anterior':'Contexto ou lógica ainda não atingiram o exigido.',
            'dialogos_e_correcoes':'Diálogos ou correções ainda não atingiram o exigido.',
            'conversas_revisadas':'As conversas revisadas ainda não atingiram o exigido.',
            'sessoes_com_respostas_proprias':'Esqueceu informações em conversas com as próprias respostas.',
            'desconhecimento':'Falhou em reconhecer o que não sabe.', 'treino_concluido':'O piloto não terminou.'}
        failed=[reasons.get(k,k) for k,v in result['protection_checks'].items() if not v]
        lines=[LABELS[row['arm']]+f" • semente {row['seed']}",
            'Versão válida no chat: a mesma versão ativa anterior.',
            'Proteções: '+('todas atingidas; ainda sem promoção automática.' if not failed else ' '.join(failed)),
            f"Tempo de treino: {result['training_seconds']:.1f}s. Medições adicionais: {result['probe_seconds']:.1f}s.",
            f"Pico de memória da placa: {result['peak_vram_mb']:.1f} MiB." if result['peak_vram_mb'] is not None else 'Execução no processador.',
            '', 'Exposição real aos dados:',json.dumps(result['exposure'],ensure_ascii=False,indent=2),
            '', 'As direções dos gradientes e atualizações reais do AdamW estão no JSON deste piloto.',
            'Não foi atribuída nota semântica automática.']
        self.details.setPlainText('\n'.join(lines));self.tabs.setCurrentWidget(self.details)
