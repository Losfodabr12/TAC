"""Continue the same bounded v3 run after interruption; never restarts the v2."""
import json
import argparse
import signal
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.core import Core
from tac.neural.v3_training import V3Trainer
import torch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session-updates',type=int)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA indisponível: não iniciar na CPU silenciosamente')
    core=Core(ROOT);trainer=V3Trainer(core.config,core.db,core.gate)
    if args.session_updates is not None and args.session_updates<1:raise ValueError('Sessão deve ter pelo menos uma atualização')
    trainer.session_updates=args.session_updates
    signal.signal(signal.SIGINT,lambda *args:setattr(trainer,'stop_requested',True))
    result=core.perform('Treinamento TAC v3',lambda:trainer.run(lambda t:print(t,flush=True)),learning=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
