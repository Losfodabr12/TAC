"""BPE de bytes treinado somente nos textos locais fornecidos. Sem downloads."""
from collections import Counter
from functools import lru_cache
import re


class LocalBPETokenizer:
    version = "TAC-local-byte-bpe-v1"
    pad, bos, eos = 0, 1, 2

    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab_size = 259 + len(self.merges)
        self.tokens = {i + 3: bytes([i]) for i in range(256)}
        self.ranks = {}
        for index, (a, b) in enumerate(self.merges):
            token = 259 + index
            if a not in self.tokens or b not in self.tokens:
                raise ValueError("Vocabulário BPE inválido")
            self.tokens[token] = self.tokens[a] + self.tokens[b]
            self.ranks[(a, b)] = (index, token)

    @staticmethod
    def pieces(text):
        return re.findall(r"\w+|[^\w\s]+|\s+", text, re.UNICODE)

    @staticmethod
    def merge(sequence, pair, token):
        output, index = [], 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                output.append(token)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return tuple(output)

    @classmethod
    def fit(cls, texts, vocab_size=1024):
        if not 259 <= vocab_size <= 2048:
            raise ValueError("Vocabulário BPE fora dos limites")
        words = Counter(piece.encode("utf-8") for text in texts for piece in cls.pieces(text))
        sequences = {tuple(byte + 3 for byte in word): count for word, count in words.items()}
        merges = []
        for token in range(259, vocab_size):
            counts = Counter()
            for sequence, count in sequences.items():
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] += count
            if not counts:
                break
            pair = max(counts, key=lambda p: (counts[p], p))
            merges.append(pair)
            updated = Counter()
            for sequence, count in sequences.items():
                updated[cls.merge(sequence, pair, token)] += count
            sequences = dict(updated)
        return cls(merges)

    @lru_cache(maxsize=20000)
    def encode_piece(self, piece):
        sequence = tuple(byte + 3 for byte in piece.encode("utf-8"))
        while len(sequence) > 1:
            options = [(self.ranks[pair][0], pair, self.ranks[pair][1]) for pair in zip(sequence, sequence[1:]) if pair in self.ranks]
            if not options:
                break
            _, pair, token = min(options)
            sequence = self.merge(sequence, pair, token)
        return sequence

    def encode(self, text, special=True):
        tokens = [token for piece in self.pieces(text) for token in self.encode_piece(piece)]
        return [self.bos, *tokens, self.eos] if special else tokens

    def decode(self, tokens):
        return b"".join(self.tokens[token] for token in tokens if token in self.tokens).decode("utf-8", errors="replace")

    def state(self):
        return {"version": self.version, "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_state(cls, state):
        if state.get("version") != cls.version or not 0 <= len(state.get("merges", [])) <= 1789:
            raise ValueError("Tokenizador BPE incompatível")
        return cls(state["merges"])
