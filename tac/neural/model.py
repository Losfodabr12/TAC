from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 259
    context: int = 128
    width: int = 96
    heads: int = 4
    layers: int = 2
    ff_width: int = 0  # 0 mantém a largura original de quatro vezes o embedding.
    tie_embeddings: bool = False
    rope: bool = False


def count_parameters(cfg):
    """Contagem estrutural de parâmetros únicos, sem alocar uma rede."""
    d, f = cfg.width, cfg.ff_width or 4 * cfg.width
    return (cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2) + (0 if cfg.rope else cfg.context * d)
            + cfg.layers * (4 * d * d + 2 * d * f + 9 * d + f) + 2 * d)


class CausalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.rope = cfg.rope
        if self.rope:
            head_width = cfg.width // cfg.heads
            if head_width % 2:
                raise ValueError("RoPE exige dimensão par por cabeça")
            self.register_buffer("inv_frequency", 1 / (10000 ** (torch.arange(0, head_width, 2).float() / head_width)), persistent=False)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        hidden = cfg.ff_width or 4 * cfg.width
        self.ff = nn.Sequential(nn.Linear(cfg.width, hidden), nn.GELU(), nn.Linear(hidden, cfg.width))

    def forward(self, x, position_offset=0):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, d // self.heads).transpose(1, 2) for a in (q, k, v)]
        if self.rope:
            angles = (torch.arange(t, device=x.device).float() + position_offset)[:, None] * self.inv_frequency[None, :]
            cos, sin = angles.cos()[None, None], angles.sin()[None, None]
            def rotate(value):
                left, right = value.float().chunk(2, dim=-1)
                return torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1).to(value.dtype)
            q, k = rotate(q), rotate(k)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.proj(attention.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.ff(self.norm2(x))


class TACTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = None if cfg.rope else nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*[CausalBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.output = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None, position_offset=0):
        if position_offset < 0 or tokens.shape[1] + position_offset > self.cfg.context:
            raise ValueError("Sequência excede o contexto do modelo.")
        positions = torch.arange(tokens.shape[1], device=tokens.device) + position_offset
        x = self.tokens(tokens)
        if self.position is not None:
            x = x + self.position(positions)
        for block in self.blocks:
            x = block(x, position_offset=position_offset)
        logits = self.output(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=0)
        return logits, loss

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, seed=42, greedy=False,
                 top_k=0, top_p=1., repetition_penalty=1., stop_condition=None, return_info=False):
        if not (0 < temperature <= 5 and isinstance(top_k,int) and top_k>=0 and 0 < top_p <= 1
                and 1 <= repetition_penalty <= 3 and isinstance(max_new_tokens, int) and max_new_tokens >= 0):
            raise ValueError('Configuração de geração fora dos limites.')
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError('A geração requer uma única sequência não vazia.')
        self.eval()
        rng = torch.Generator(device=prompt.device).manual_seed(seed)
        initial_length=prompt.shape[1]
        stop_reason = 'length'
        for _ in range(min(max_new_tokens, 256)):
            logits, _ = self(prompt[:, -self.cfg.context:])
            logits = logits[:, -1, :].float() / max(0.1, temperature)
            logits[:, 0:2] = -torch.inf
            if repetition_penalty!=1. and prompt.shape[1]>initial_length:
                ids=prompt[:,initial_length:].unique()
                values=logits[:,ids]
                logits[:,ids]=torch.where(values>0,values/repetition_penalty,values*repetition_penalty)
            if 0 < top_k < logits.shape[-1]:
                threshold=logits.topk(top_k).values[:,-1:]
                logits=logits.masked_fill(logits<threshold,-torch.inf)
            if top_p < 1.:
                ordered,indices=logits.sort(dim=-1,descending=True)
                remove=ordered.softmax(-1).cumsum(-1)>top_p
                remove[:,1:]=remove[:,:-1].clone();remove[:,0]=False
                logits=logits.scatter(-1,indices,ordered.masked_fill(remove,-torch.inf))
            next_token = logits.argmax(-1, keepdim=True) if greedy else torch.multinomial(logits.softmax(-1), 1, generator=rng)
            prompt = torch.cat((prompt, next_token), dim=1)
            if next_token.item() == 2:
                stop_reason = 'eos'
                break
            if stop_condition is not None:
                reason = stop_condition(prompt[0, initial_length:].tolist())
                if reason:
                    stop_reason = reason
                    break
        if return_info:
            return prompt, {'stop_reason': stop_reason, 'generated_tokens': prompt.shape[1] - initial_length,
                            'max_new_tokens': min(max_new_tokens, 256)}
        return prompt

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def expand_model(original: TACTransformer, cfg: ModelConfig, max_parameters=500_000):
    """Amplia contexto/MLPs preservando as previsões no contexto antigo.

    Os neurônios adicionais recebem inicialização própria; suas conexões de saída
    começam em zero. Assim a ampliação não descarta os pesos já aprendidos.
    A equivalência vale antes do treino e até o limite de contexto anterior.
    """
    old = original.cfg
    if any(getattr(cfg, key) != getattr(old, key) for key in ("width", "heads", "layers", "vocab_size", "tie_embeddings", "rope")):
        raise ValueError("A expansão preserva dimensão, atenção, camadas e vocabulário.")
    if cfg.context < old.context or (cfg.ff_width or 4 * cfg.width) < (old.ff_width or 4 * old.width):
        raise ValueError("A expansão não pode reduzir contexto ou neurônios.")
    expanded = TACTransformer(cfg)
    if expanded.parameter_count > max_parameters:
        raise ValueError(f"A expansão excede o limite de {max_parameters:,} parâmetros.")
    source, destination = original.state_dict(), expanded.state_dict()
    with torch.no_grad():
        for name, target in destination.items():
            value = source[name].to(target.device)
            if target.shape == value.shape:
                target.copy_(value)
            elif name == "position.weight":
                target[:old.context].copy_(value)
                # Reutiliza posições próprias; as novas serão ajustadas no treino.
                for start in range(old.context, cfg.context, old.context):
                    target[start:start + old.context].copy_(value[:min(old.context, cfg.context - start)])
            elif name.endswith("ff.0.weight") or name.endswith("ff.0.bias"):
                target[:value.shape[0]].copy_(value)
            elif name.endswith("ff.2.weight"):
                target.zero_()
                target[:, :value.shape[1]].copy_(value)
            else:
                raise ValueError("Expansão incompatível: " + name)
    expanded.load_state_dict(destination)
    return expanded
