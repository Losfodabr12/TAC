"""Verifica caminho da interface, logits e EOS; zero atualizações de pesos."""
from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from diagnose_v2_generation import OUT, QUESTIONS, forbidden, save as save_evidence
from tac.core import Core
from tac.neural.conversation import prompt_text
from tac.neural.v2_training import model_hash


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    def save(name, value):
        save_evidence(args.prefix + name, value)
    torch.save = forbidden
    torch.optim.AdamW.step = forbidden
    from tac.research import Researcher
    Researcher.lookup = forbidden
    core = Core(ROOT)
    core.memory.answer = forbidden
    core.dialogue.reply = forbidden
    before_config = dict(core.config.values)
    # Seleciona o perfil curto e conservador, medido nas sondas. Outros campos intactos.
    core.config.update(conversation_generation='controlled')
    save('inference_config_change.json', {'before': before_config, 'after': dict(core.config.values),
         'changed_fields': [k for k in before_config if before_config[k] != core.config.values[k]]})
    active = core.active_conversation_id()
    engine = core.conversation
    # Carrega uma única vez pelo carregador real da interface.
    model, cp = engine.load(active, engine.storage.device())
    engine._cached_id, engine._cached_model, engine._cached_generation = active, model, cp.get('generation')
    initial = model_hash(model)
    tok = model.text_tokenizer
    prefill = []
    def hook(module, args):
        if not prefill:
            prefill.append(args[0].detach().cpu()[0].tolist())
    handle = model.register_forward_pre_hook(hook)
    results = []
    for q in QUESTIONS:
        core.neural_history.clear()
        prefill.clear()
        answer = core.chat(q, conversation=True)
        expected = [1, *tok.encode(prompt_text([['user', q]]), special=False)]
        assert prefill[0] == expected
        assert answer['checkpoint'] == active and answer['generation'] == 'TAC-v2'
        assert answer['sources'] == [] and answer['decoding']['generated_tokens'] <= 128
        assert answer['decoding']['truncated_tokens'] == 0
        results.append({'question': q, 'transformer_input_ids': prefill[0],
            'transformer_input_text': tok.decode(prefill[0]), 'input_matches_training_format': True, 'answer': answer})
        print(json.dumps({'question': q, 'text': answer['text'], 'stop': answer['decoding']['stop_reason']}, ensure_ascii=False), flush=True)
    handle.remove()
    save('real_interface_path.json', results)

    sensitivity = []
    for q in QUESTIONS:
        prefix = [1, *tok.encode(prompt_text([['user', q]]), special=False)]
        phases = {}
        for label, suffix in [('prompt_end', ''), ('first_answer_space', ' '), ('shared_response', ' O bom')]:
            ids = torch.tensor([prefix + tok.encode(suffix, special=False)], device=next(model.parameters()).device)
            logits = model(ids)[0][0, -1].float()
            p = logits.softmax(-1)
            values, indices = p.topk(10)
            phases[label] = {'logits': logits.cpu().tolist(), 'probabilities': p.cpu().tolist(),
                             'top': [{'id': i, 'text': tok.decode([i]), 'p': v} for i, v in zip(indices.tolist(), values.tolist())]}
        sensitivity.append({'question': q, 'phases': phases})
    differences = []
    for i, a in enumerate(sensitivity):
        for b in sensitivity[i+1:]:
            pair = {'a': a['question'], 'b': b['question']}
            for phase in a['phases']:
                l1, l2 = (torch.tensor(r['phases'][phase]['logits']) for r in (a,b))
                p1, p2 = (torch.tensor(r['phases'][phase]['probabilities']) for r in (a,b))
                pair[phase] = {'max_abs_logit_delta': float((l1-l2).abs().max()),
                               'total_variation': float((p1-p2).abs().sum()/2)}
            differences.append(pair)
    save('next_token_logits.json', {'conditions': sensitivity, 'differences': differences})

    # Na trajetória livre gulosa problemática, quanto EOS é despriorizado?
    eos_free = []
    def trace(module, args, output):
        v = output[0][0, -1].float()
        eos_free.append({'position': len(eos_free), 'eos_p': float(v.softmax(-1)[2]),
                         'eos_rank': int((v > v[2]).sum()) + 1, 'argmax': int(v.argmax())})
    handle = model.register_forward_hook(trace)
    ids = torch.tensor([[1, *tok.encode(prompt_text([['user', QUESTIONS[0]]]), special=False)]], device=next(model.parameters()).device)
    generated, info = model.generate(ids, max_new_tokens=80, greedy=True, return_info=True)
    handle.remove()
    save('eos_in_degenerate_free_run.json', {'question': QUESTIONS[0], 'options': {'greedy': True, 'repetition_penalty': 1.},
         'max_new_tokens': 80, 'info': info, 'text': tok.decode(generated[0, ids.shape[1]:].tolist()), 'eos_trace': eos_free})
    assert initial == model_hash(model)
    save('interface_preservation.json', {'active_checkpoint': active, 'weights_hash_before': initial,
          'weights_hash_after': model_hash(model), 'training_updates': 0, 'status': 'OK'})


if __name__ == '__main__':main()
