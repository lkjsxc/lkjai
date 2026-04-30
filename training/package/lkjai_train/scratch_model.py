import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scratch_attention import Attention
from .scratch_layers import RMSNorm, SwiGLU, init_weights


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


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ffn = SwiGLU(config)

    def forward(self, x, cache=None, use_cache: bool = False):
        attn_out, next_cache = self.attn(self.attn_norm(x), cache, use_cache)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), next_cache


class ScratchLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.activation_checkpoint = "off"
        self.activation_checkpoint_every_n = 2
        self.checkpoint_preserve_rng = False
        self.apply(init_weights)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.activation_checkpoint = "all" if enabled else "off"

    def configure_runtime(self, settings) -> None:
        self.activation_checkpoint = settings.activation_checkpoint
        self.activation_checkpoint_every_n = settings.activation_checkpoint_every_n
        self.checkpoint_preserve_rng = settings.checkpoint_preserve_rng
        for block in self.blocks:
            block.attn.backend = settings.attention_backend

    def forward(self, idx, labels=None, cache=None, use_cache: bool = False):
        x = self.tok(idx)
        next_cache = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[index]
            if self.should_checkpoint(index, layer_cache, use_cache):
                x = checkpoint(
                    lambda value: block(value, None, False)[0],
                    x,
                    use_reentrant=False,
                    preserve_rng_state=self.checkpoint_preserve_rng,
                )
                cached = None
            else:
                x, cached = block(x, layer_cache, use_cache)
            if use_cache:
                next_cache.append(cached)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss, next_cache

    def should_checkpoint(self, index: int, layer_cache, use_cache: bool) -> bool:
        if not self.training or layer_cache is not None or use_cache:
            return False
        if self.activation_checkpoint == "all":
            return True
        if self.activation_checkpoint == "every_n":
            return index % self.activation_checkpoint_every_n == 0
        return False

def save_config(config: ModelConfig, path: Path) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())
