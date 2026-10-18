"""Avaliação reproduzível em cenas reservadas; não treina nem ativa pesos."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import torch
from tac.core import Core
from tac.config import atomic_json
from tac.neural.conversation import ConversationData


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--split',choices=('validation_fluency','test_fluency'),default='validation_fluency')
    p.add_argument('--label',choices=('before','after','intermediate'),default='after')
    args = p.parse_args()
    c = Core()
    torch.set_num_threads(4)
    snapshot = json.loads((ROOT/'reports/curriculum_080.json').read_text(encoding='utf-8'))
    data = ConversationData.load(c.config.data/'datasets'/f"conversation_{snapshot['conversation']}.json")
    checkpoint = args.checkpoint or c.active_conversation_id()
    model,cp = c.conversation.load(checkpoint,args.device)
    assert cp['tokenizer_state'] == data.tokenizer.state()
    metrics = c.conversation.evaluate_fluency(model,data,args.device,args.split)
    loss = c.conversation.evaluate(model,data,args.device,split=args.split)
    report = {'checkpoint':checkpoint,'parameters':model.parameter_count,'dataset':data.digest,
              'split':args.split,'metrics':metrics,'token_metrics':loss,'device':args.device}
    atomic_json(ROOT/'reports'/f'fluency_{args.label}_{checkpoint}_{args.split}.json',report)
    print(json.dumps({**{k:v for k,v in metrics.items() if k!='sessions'},'token_metrics':loss},ensure_ascii=True))


if __name__ == '__main__':main()
