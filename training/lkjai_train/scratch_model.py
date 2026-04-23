import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    sequence_len: int
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    ffn_size: int
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down = nn.Linear(config.ffn_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.hidden_size // config.heads
        kv_dim = self.kv_heads * self.head_dim
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        batch, seq, _ = x.shape
        q = self.q(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        mask = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o(y)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class ScratchLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.apply(init_weights)

    def forward(self, idx, labels=None):
        seq = idx.size(1)
        x = self.tok(idx)
        cos, sin = rope_cache(seq, self.config.hidden_size // self.config.heads, idx.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return logits, loss


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def rope_cache(seq: int, dim: int, device):
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq, device=device).float()
    angles = torch.outer(pos, freq)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def save_config(config: ModelConfig, path: Path) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())
