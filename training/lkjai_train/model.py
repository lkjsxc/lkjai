from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 259
    context: int = 64
    layers: int = 2
    hidden: int = 128
    heads: int = 4
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner = int(dim * 8 / 3)
        self.gate = nn.Linear(dim, inner, bias=False)
        self.up = nn.Linear(dim, inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden)
        self.ffn_norm = RMSNorm(cfg.hidden)
        self.attn = nn.MultiheadAttention(
            cfg.hidden,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
            bias=False,
        )
        self.ffn = SwiGLU(cfg.hidden)

    def forward(self, x, mask):
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn
        return x + self.ffn(self.ffn_norm(x))


class LkjModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos = nn.Embedding(cfg.context, cfg.hidden)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        batch, seq = idx.shape
        pos = torch.arange(seq, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(seq, seq, device=idx.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def tiny_config() -> ModelConfig:
    return ModelConfig(context=32, layers=1, hidden=64, heads=4)
