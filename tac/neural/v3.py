"""Isolated TAC v3 architecture and tokenizer; no pretrained model assets."""
import json
from .model import ModelConfig


def v3_config():
    return ModelConfig(vocab_size=8192, context=1024, width=512, heads=8,
                       layers=8, ff_width=2048, tie_embeddings=True, rope=True)


class V3Tokenizer:
    version = 'TAC-byte-bpe-v3'
    pad, bos, eos = 0, 1, 2

    def __init__(self, backend):
        self.backend = backend
        self.vocab_size = backend.get_vocab_size()
        if [backend.token_to_id(t) for t in ('<pad>', '<bos>', '<eos>')] != [0, 1, 2]:
            raise ValueError('IDs especiais incompatíveis')

    @classmethod
    def fit(cls, texts, vocab_size=8192):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(texts, trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=3, show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=['<pad>', '<bos>', '<eos>']))
        return cls(backend)

    def encode(self, text, special=True):
        # External text cannot inject reserved IDs into training or generation.
        for marker in ('<pad>', '<bos>', '<eos>'):
            text = text.replace(marker, marker.replace('<', '＜'))
        values = self.backend.encode(text, add_special_tokens=False).ids
        return [self.bos, *values, self.eos] if special else values

    def decode(self, ids):
        return self.backend.decode([int(x) for x in ids], skip_special_tokens=True)

    def state(self):
        return {'version': self.version, 'backend_json': self.backend.to_str()}

    @classmethod
    def from_state(cls, state):
        from tokenizers import Tokenizer
        if state.get('version') != cls.version or len(state['backend_json']) > 4 * 2**20:
            raise ValueError('Tokenizer v3 inválido')
        backend_json = json.loads(state['backend_json'])
        if backend_json.get('model', {}).get('type') != 'BPE':
            raise ValueError('A v3 exige tokenizer BPE local')
        tokenizer = cls(Tokenizer.from_str(state['backend_json']))
        if tokenizer.vocab_size > 8192:
            raise ValueError('Vocabulário v3 fora do limite')
        return tokenizer
