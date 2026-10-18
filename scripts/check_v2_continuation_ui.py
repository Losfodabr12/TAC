"""Valida envio real e comparação de versões na janela nativa do Windows."""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace');os.environ['QT_QPA_PLATFORM']='windows'
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import torch
from tac.core import Core
from tac.ui import Window,apply_theme
from tac.config import atomic_json
OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01'


def forbidden(*a,**kw):raise RuntimeError('Verificação da interface não pode treinar')


def main():
    torch.save=forbidden;torch.optim.AdamW.step=forbidden
    app=QApplication([]);core=Core(ROOT)
    deployment=json.loads((core.config.data/'tac_v2_conversation_20261018/deployment.json').read_text(encoding='utf-8'))
    apply_theme(app,core.config);w=Window(core);w.timer.stop();w.open_chat(2);w.show();app.processEvents()
    assert core.active_conversation_id()==deployment['checkpoint']
    assert w.conversation_version.count()==2
    assert not w.conversation_version_notice.isHidden()
    questions=['Opa, tudo bom?','O que é um lobo?','Quanto é 2 + 2?','Crie uma história curta.','Meu nome é Carlos. Qual é meu nome?']
    outputs=[];comparison={};stage=['questions'];position=[0]
    def send(question):
        w.clear_conversation();w.input.setText(question);w.send_chat();QTimer.singleShot(100,poll)
    def poll():
        if w.job is not None:QTimer.singleShot(100,poll);return
        if stage[0]=='questions':
            q,answer=core.last_answer
            assert q==questions[position[0]] and answer['checkpoint']==deployment['checkpoint'] and answer['sources']==[]
            outputs.append({'question':q,'answer':answer});position[0]+=1
            if position[0]<len(questions):send(questions[position[0]]);return
            stage[0]='reference_selected';w.conversation_version.setCurrentIndex(1);QTimer.singleShot(100,poll);return
        if stage[0]=='reference_selected':
            assert core.active_conversation_id()==deployment['reference'] and core.neural_history==[]
            stage[0]='reference_answer';send(questions[-1]);return
        if stage[0]=='reference_answer':
            comparison['reference']=core.last_answer[1]
            assert comparison['reference']['checkpoint']==deployment['reference']
            stage[0]='experiment_selected';w.conversation_version.setCurrentIndex(0);QTimer.singleShot(100,poll);return
        if stage[0]=='experiment_selected':
            assert core.active_conversation_id()==deployment['checkpoint'] and core.neural_history==[]
            stage[0]='experiment_answer';send(questions[-1]);return
        comparison['experiment']=core.last_answer[1]
        assert comparison['experiment']['text']=='Carlos'
        assert comparison['experiment']['checkpoint']==deployment['checkpoint']
        assert 'perdeu desempenho' in w.conversation_version_notice.text()
        assert w.grab().save(str(OUT/'interface_conversacao.png'))
        ref=json.loads((OUT/'reference.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(Path(ref['file']).read_bytes()).hexdigest()==ref['file_sha256']
        value={'status':'OK_native_interface_not_semantic_approval','platform':app.platformName(),
            'active_checkpoint':core.active_conversation_id(),'reference':deployment['reference'],
            'independent_questions':outputs,'real_selector_comparison':comparison,
            'version_selector_works':True,'session_cleared_on_switch':True,
            'regression_warning_visible':True,'neural_sources_only':True,'training_updates':0}
        atomic_json(OUT/'interface_verificada.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True);w.close()
    send(questions[0]);return app.exec()


if __name__=='__main__':raise SystemExit(main())
