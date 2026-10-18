"""Treinamento real da geração nova, sem herança de pesos antigos."""
import argparse
import json
import os
import signal
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from tac.core import Core
from tac.neural.v2_training import V2Trainer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initialize-only',action='store_true')
    parser.add_argument('--pretrain-steps',type=int,default=12000);parser.add_argument('--sft-epochs',type=int,default=3)
    args=parser.parse_args();core=Core(ROOT)
    if not core.gate.enabled:raise RuntimeError('Aprendizado desligado. Ative antes de iniciar.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA não disponível; não iniciar silenciosamente na CPU')
    trainer=V2Trainer(core.config,core.db,core.gate,pretrain_steps=args.pretrain_steps,sft_epochs=args.sft_epochs,device='cuda')
    signal.signal(signal.SIGINT,lambda *a:setattr(trainer,'stop_requested',True))
    if args.initialize_only:
        trainer.initialize();print('TAC v2 inicializada com pesos novos:',trainer.last_checkpoint,trainer.model.parameter_count);return
    result=core.perform('TAC v2: linguagem e instruções',lambda:trainer.run(lambda x:print(x,flush=True)),learning=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
