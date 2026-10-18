"""Native Windows UI: actual v3 inference, version switching and v2 preservation."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8',errors='replace');os.environ['QT_QPA_PLATFORM']='windows'
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import torch
from tac.core import Core
from tac.ui import Window,apply_theme
from tac.config import atomic_json
OUT=ROOT/'reports/tac_v3/rodada_20261018_01'


def forbidden(*args,**kwargs):raise RuntimeError('A validação da interface não pode treinar nem salvar pesos')


def main():
    torch.save=forbidden;torch.optim.AdamW.step=forbidden
    app=QApplication([]);core=Core(ROOT);previous=core.active_conversation_id()
    configuration_before=core.config.path.read_bytes()
    deployment=json.loads((core.config.data/'tac_v3/deployment.json').read_text(encoding='utf-8'))
    apply_theme(app,core.config);w=Window(core);w.timer.stop();w.open_chat(2);w.show();app.processEvents()
    assert w.conversation_version.count()==3
    questions=['Opa, tudo bom?','O que é um lobo?','Quanto é 2 + 2?','Crie uma história curta.','Meu nome é Carlos. Qual é meu nome?',
               'opa tudo bom','o que e um lobo','quanto e 2+2']
    result={'platform':app.platformName(),'selected_before':previous,'v3_checkpoint':deployment['checkpoint'],
            'selector_options':[w.conversation_version.itemText(i) for i in range(w.conversation_version.count())],
            'neural_outputs':[],'training_updates':0,'inference_precision':'float32 (production interface)'}
    stage=['select_v3'];position=[0]
    def send(question):
        w.clear_conversation();w.input.setText(question);w.send_chat();QTimer.singleShot(100,poll)
    def _poll():
        if w.job is not None:QTimer.singleShot(100,poll);return
        if w.job_failed_flag:raise RuntimeError(w.raw_failure)
        if stage[0]=='select_v3':
            assert core.active_conversation_id()==deployment['checkpoint'] and core.neural_history==[]
            stage[0]='v3_answers';send(questions[0]);return
        if stage[0]=='v3_answers':
            q,answer=core.last_answer
            assert q==questions[position[0]] and answer['checkpoint']==deployment['checkpoint']
            assert answer['generation']=='TAC-v3' and answer['sources']==[]
            assert core.conversation._cached_model.parameter_count==29_414_400
            result['neural_outputs'].append({'question':q,'answer':answer})
            position[0]+=1
            if position[0]<len(questions):send(questions[position[0]]);return
            assert not w.conversation_version_notice.isHidden()
            assert w.grab().save(str(OUT/'interface_v3.png'))
            stage[0]='restore_v2'
            w.conversation_version.setCurrentIndex(w.conversation_version.findData(previous));QTimer.singleShot(100,poll);return
        if stage[0]=='restore_v2':
            assert core.active_conversation_id()==previous and core.neural_history==[]
            stage[0]='v2_answer';send(questions[4]);return
        result['v2_comparison_answer']=core.last_answer[1]
        assert result['v2_comparison_answer']['checkpoint']==previous
        w.clear_conversation();result.update(status='OK_interface_only_not_quality_approval',
            selected_after=core.active_conversation_id(),session_isolation_verified=True,
            v3_actual_parameters=29_414_400,v2_restored=True,
            configuration_bytes_unchanged=core.config.path.read_bytes()==configuration_before)
        assert result['configuration_bytes_unchanged']
        atomic_json(OUT/'interface_verificada.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True);w.close()
    def poll():
        try:_poll()
        except Exception as exc:
            result.update(status='FAILED',error=repr(exc))
            atomic_json(OUT/'interface_verificada.json',result)
            print(repr(exc),flush=True);app.exit(1)
    w.conversation_version.setCurrentIndex(w.conversation_version.findData(deployment['checkpoint']))
    QTimer.singleShot(100,poll)
    # Fail loudly rather than silently claiming a UI validation on timeout.
    QTimer.singleShot(240_000,lambda:app.exit(2))
    try:return app.exec()
    finally:
        if core.active_conversation_id()!=previous:core.conversation.activate(previous)


if __name__=='__main__':raise SystemExit(main())
