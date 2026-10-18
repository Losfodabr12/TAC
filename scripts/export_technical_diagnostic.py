"""Gera o diagnóstico local sem treinar. Uso opcional; a interface tem o botão."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tac.core import Core
from tac.diagnostic_export import export_diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'TAC_Diagnostico.zip')
    parser.add_argument('--overwrite',action='store_true',help='Substituir o ZIP de destino escolhido.')
    args=parser.parse_args()
    result=export_diagnostic(Core(ROOT),args.output,lambda step,message:print(message,flush=True),overwrite=args.overwrite)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
