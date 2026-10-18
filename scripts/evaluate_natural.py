"""Avalia novos registros de escrita e correções; nunca treina ou ativa pesos."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import torch
from tac.core import Core
from tac.config import atomic_json
from tac.neural.conversation import ConversationData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=('validation_natural', 'test_natural'), default='validation_natural')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--label', choices=('before', 'intermediate', 'after'), default='after')
    args = parser.parse_args()
    core = Core()
    torch.set_num_threads(core.config.get('cpu_threads'))
    checkpoint = args.checkpoint or core.active_conversation_id()
    model, cp = core.conversation.load(checkpoint, args.device)
    curriculum = json.loads((ROOT/'reports/curriculum_060.json').read_text(encoding='utf-8'))
    data = ConversationData.load(core.config.data/'datasets'/f"conversation_{curriculum['hash']}.json")
    if cp.get('tokenizer_state') != data.tokenizer.state():
        raise ValueError('A avaliação exige o mesmo vocabulário local do checkpoint.')
    metrics = core.conversation.evaluate(model, data, args.device, split=args.split, generation=True)
    sessions = core.conversation.evaluate_sessions(model, data, args.device, split=args.split)
    report = {'checkpoint':checkpoint, 'checkpoint_step':cp['step'], 'model_dataset_hash':cp['dataset_hash'],
              'parameters':model.parameter_count, 'dataset_hash':data.digest, 'device':args.device,
              'metrics':metrics, 'sessions':sessions, 'split':args.split}
    atomic_json(ROOT/'reports'/f'natural_{args.label}_{checkpoint}_{args.split}.json', report)
    print(json.dumps({'metrics':{k:v for k,v in metrics.items() if k!='outputs'},
                      'sessions':{k:v for k,v in sessions.items() if k!='sessions'}}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
