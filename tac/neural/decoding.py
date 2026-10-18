"""Parada verificável da conversa: nenhum corretor de conteúdo ou resposta pronta."""
from __future__ import annotations

import re

import torch

from ..conversation_lessons import prompt_text
from .tokenizer import ByteTokenizer


def conversation_options(mode, seed):
    """Perfis da v2; temperatura baixa e redução moderada de repetição."""
    common = {'repetition_penalty': 1.12, 'seed': seed}
    if mode == 'controlled':
        return {**common, 'greedy': False, 'temperature': .55, 'top_k': 30, 'top_p': .9}
    return {**common, 'greedy': True}


class ConversationStop:
    """Examina somente a saída nova. Interrompe em vez de forçar outro token.

    Quatro palavras iguais seguidas ou três cópias de uma frase de 2..32 palavras
    caracterizam ciclo. Quatro ocorrências próximas de uma frase de 4..12
    palavras também interrompem redundância com trechos intercalados.
    Pontuação e caixa não escondem um ciclo; grafia do texto devolvido é preservada.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cut = None
        self.reason = None
        self.detail = None

    def inspect(self, text, *, complete=False):
        role = re.search(r'(?m)^(?:Usuário|TAC):', text)
        if role:
            self.cut, self.reason = role.start(), 'role_boundary'
            self.detail = {'delimiter': role.group()}
            return self.reason
        words = list(re.finditer(r'\w+', text, re.UNICODE))
        if words and not complete and words[-1].end() == len(text):
            words.pop()  # Não confundir um subtoken com uma palavra completa.
        values = [m.group().casefold() for m in words]
        for n in range(1, min(32, len(words) // 3) + 1):
            repeats = 4 if n == 1 else 3
            if len(words) < n * repeats:
                continue
            start = len(words) - n * repeats
            block = values[start:start + n]
            if values[start:] == block * repeats:
                # Devolve o prefixo neural até a primeira cópia; o aviso da UI
                # identifica o corte. O trecho bruto continua nos diagnósticos.
                self.cut, self.reason = words[start + n - 1].end(), 'repetition'
                self.detail = {'phrase': text[words[start].start():words[start + n - 1].end()],
                               'repetitions': repeats, 'word_count': n}
                return self.reason
        # A rede também pode intercalar palavras novas entre cópias da mesma
        # expressão, como "um lobo de areia" cinco vezes numa resposta curta.
        # Só conta cópias não sobrepostas, dentro das últimas 64 palavras.
        for n in range(4, min(12, len(words) // 4) + 1):
            end = len(words)
            block = values[end - n:end]
            starts, i = [], max(0, end - 64)
            while i <= end - n:
                if values[i:i + n] == block:
                    starts.append(i)
                    i += n
                else:
                    i += 1
            if len(starts) >= 4:
                first = starts[0]
                self.cut, self.reason = words[first + n - 1].end(), 'repetition'
                self.detail = {'phrase': text[words[first].start():words[first + n - 1].end()],
                               'repetitions': len(starts), 'word_count': n, 'pattern': 'interleaved'}
                return self.reason
        return None

    def __call__(self, token_ids):
        return self.inspect(self.tokenizer.decode(token_ids))


@torch.inference_mode()
def decode_conversation(model, messages, device, max_tokens=128, generation_options=None):
    """Mesma serialização do SFT; diagnóstico separado do texto da rede."""
    tokenizer = getattr(model, 'text_tokenizer', ByteTokenizer())
    text = prompt_text(messages)
    encoded = tokenizer.encode(text, special=False)
    kept = encoded[-model.cfg.context + 1:]
    tokens = torch.tensor([[tokenizer.bos, *kept]], device=device)
    stop = ConversationStop(tokenizer)
    options = {'greedy': True} if generation_options is None else dict(generation_options)
    generated, info = model.generate(tokens, max_new_tokens=max_tokens, stop_condition=stop,
                                     return_info=True, **options)
    ids = generated[0, tokens.shape[1]:].tolist()
    raw = tokenizer.decode(ids)
    # Também examina a última palavra caso EOS ou o limite encerre sem espaço.
    if stop.reason is None:
        stop.inspect(raw, complete=True)
    if stop.reason:
        info['stop_reason'] = stop.reason
    reply = raw[:stop.cut] if stop.cut is not None else raw
    return {**info, 'text': reply.strip(), 'raw_text': raw,
            'loop': stop.detail if stop.reason == 'repetition' else None,
            'input_tokens': tokens.shape[1], 'truncated_tokens': len(encoded) - len(kept),
            'generation_options': options,
            'input_format': 'BOS + Usuário: ...\\nTAC:', 'eos_generated': bool(ids and ids[-1] == tokenizer.eos)}
