"""Verifica a rede ativa pelo caminho real da interface, sem respostas substitutas."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.core import Core


def main():
    core=Core(ROOT)
    answer=core.chat('Meu nome é Iolanda. Qual é meu nome?',conversation=True)
    assert answer['generation']=='TAC-v2' and answer['checkpoint']=='64f45f52fee54053a8f0c6b16d9df172'
    assert answer['mode']=='diálogo neural experimental' and answer['sources']==[]
    from PySide6.QtWidgets import QApplication
    from tac.ui import Window,apply_theme
    app=QApplication([]);apply_theme(app,core.config);w=Window(core);w.timer.stop();w.open_chat(2);w.show();app.processEvents()
    assert 'TAC v2' in w.chat_mode.currentText()
    out=ROOT/'reports/reconstrucao_zero/execucao_v2'
    w.grab().save(str(out/'interface_v2_chat.png'))
    result={'status':'OK','platform':app.platformName(),'chat_mode':w.chat_mode.currentText(),
            'active_generation':answer['generation'],'active_checkpoint':answer['checkpoint'],
            'direct_neural_answer':answer,'history_context_tokens':core.conversation._cached_model.cfg.context,
            'window_title':w.windowTitle(),'no_external_or_prepared_source':True}
    with (out/'validacao_interface.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2));w.close()


if __name__=='__main__':main()
