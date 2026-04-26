import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.hidden_size // config.heads
        self.max_len = config.sequence_len
        kv_dim = self.kv_heads * self.head_dim
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout
        self.backend = "auto"

    def forward(self, x, layer_cache=None, use_cache: bool = False):
        batch, seq, _ = x.shape
        offset = 0 if layer_cache is None else layer_cache["k"].size(2)
        q = self.q(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k_base = self.k(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v_base = self.v(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(seq, self.head_dim, x.device, offset)
        q, k_base = apply_rope(q, cos, sin), apply_rope(k_base, cos, sin)
        if layer_cache is not None:
            k_base = torch.cat([layer_cache["k"], k_base], dim=2)[:, :, -self.max_len :]
            v_base = torch.cat([layer_cache["v"], v_base], dim=2)[:, :, -self.max_len :]
        dropout_p = self.dropout if self.training else 0.0
        attn = flash_attention(q, k_base, v_base, dropout_p, layer_cache is None, self.backend)
        if attn is not None:
            pass
        elif self.heads != self.kv_heads:
            try:
                attn = F.scaled_dot_product_attention(q, k_base, v_base, dropout_p=dropout_p, is_causal=layer_cache is None, enable_gqa=True)
            except TypeError:
                repeats = self.heads // self.kv_heads
                k = k_base.repeat_interleave(repeats, dim=1)
                v = v_base.repeat_interleave(repeats, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=layer_cache is None)
        else:
            attn = F.scaled_dot_product_attention(q, k_base, v_base, dropout_p=dropout_p, is_causal=layer_cache is None)
        output = self.o(attn.transpose(1, 2).contiguous().view(batch, seq, -1))
        next_cache = {"k": k_base.detach(), "v": v_base.detach()} if use_cache else None
        return output, next_cache


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


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def rope_cache(seq: int, dim: int, device, offset: int = 0):
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(offset, offset + seq, device=device).float()
    angles = torch.outer(pos, freq)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def flash_attention(q, k, v, dropout_p: float, is_causal: bool, backend: str):
    if backend == "sdpa" or (backend == "auto" and not torch.cuda.is_available()):
        return None
    try:
        from flash_attn import flash_attn_func
    except Exception:
        if backend == "flash2":
            raise RuntimeError("TRAIN_ATTENTION_BACKEND=flash2 requires flash-attn")
        return None
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_t, k_t, v_t, dropout_p=dropout_p, causal=is_causal)
    return out.transpose(1, 2).contiguous()


def save_config(config: ModelConfig, path: Path) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters())
