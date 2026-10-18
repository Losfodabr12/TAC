"""Cria ou importa uma ficha de revisão semântica humana, sem treinar."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.core import Core
from tac.config import atomic_json
from tac.neural.transfer_eval import review_template,validate_review

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=ROOT/'reports/research_0120/novel_development.results.json')
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--create',action='store_true')
args=parser.parse_args()
if not args.review.resolve().is_relative_to(ROOT):parser.error('Salve a ficha dentro de TAC.')
results=json.loads(args.results.read_text(encoding='utf-8'))
if args.create:
    if args.review.exists():raise FileExistsError('A ficha já existe.')
    atomic_json(args.review,review_template(results))
else:
    reviewed=json.loads(args.review.read_text(encoding='utf-8'))
    report=validate_review(results,reviewed)
    core=Core(ROOT)
    with core.gate.writing():
        core.gate.check();atomic_json(args.review.with_suffix('.scores.json'),report)
print('Ficha criada.' if args.create else 'Revisão validada; pesos e aprovação preservados.')
