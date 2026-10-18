"""Uma execução substancial de continuação, sem reinicializar os pesos aprendidos."""
import json
import signal
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.core import Core
from tac.neural.conversation_training import ConversationContinuation
import torch


def main():
    if not torch.cuda.is_available():raise RuntimeError('RTX/CUDA indisponível: não iniciar silenciosamente na CPU')
    core=Core(ROOT)
    trainer=ConversationContinuation(core.config,core.db,core.gate)
    signal.signal(signal.SIGINT,lambda *a:setattr(trainer,'stop_requested',True))
    value=core.perform('Continuação de conversação TAC v2',lambda:trainer.run(lambda x:print(x,flush=True)),learning=True)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
