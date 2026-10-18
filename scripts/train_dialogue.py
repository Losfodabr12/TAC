"""Treino real dos exercícios autorais, sem acesso à internet."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tac.core import Core
from tac.config import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--user-data", action="store_true")
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()
root = ROOT if args.user_data else ROOT / "reports" / ("dialogue_trial_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
core = Core(root)
previous_permission = core.gate.enabled
if not previous_permission:
    core.gate.set_enabled(True)  # execução explícita deste script autoriza seus exercícios
try:
    result = core.perform("Praticar conversas", lambda: core.dialogue.train(lambda s,m: print(m, flush=True), epochs=args.epochs), learning=True)
    atomic_json(ROOT / "reports" / ("dialogue_user.json" if args.user_data else "dialogue_trial.json"), result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
finally:
    if not previous_permission:
        core.gate.set_enabled(False)
