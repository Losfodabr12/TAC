class ByteTokenizer:
    """Vocabulário fixo UTF-8: PAD=0, BOS=1, EOS=2, bytes=3..258."""
    vocab_size = 259
    pad, bos, eos = 0, 1, 2
    version = "utf8-bytes-v1"

    def encode(self, text, special=True):
        tokens = [b + 3 for b in text.encode("utf-8")]
        return [self.bos, *tokens, self.eos] if special else tokens

    def decode(self, tokens):
        return bytes(t - 3 for t in tokens if 3 <= t < self.vocab_size).decode("utf-8", errors="replace")
