from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.hidden_size // config.heads
        self.max_len = config.sequence_len
        kv_dim = self.kv_heads * self.head_dim
        self.q_size = config.hidden_size
        self.kv_size = kv_dim
        self.qkv = nn.Linear(config.hidden_size, self.q_size + 2 * kv_dim, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout
        self.backend = "auto"
        cos, sin = rope_tables(self.max_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, layer_cache=None, use_cache: bool = False):
        batch, seq, _ = x.shape
        offset = 0 if layer_cache is None else int(layer_cache.get("len", layer_cache["k"].size(2)))
        q, k_base, v_base = self.project_qkv(x, batch, seq)
        cos, sin = self.rope_slice(offset, seq, x.dtype)
        q, k_base = apply_rope(q, cos, sin), apply_rope(k_base, cos, sin)
        if layer_cache is not None:
            k_base, v_base, cache_len = append_cache(layer_cache, k_base, v_base, self.max_len)
            next_state = {"k": layer_cache["k"], "v": layer_cache["v"], "len": cache_len}
        else:
            cache_len = min(self.max_len, k_base.size(2))
            next_state = cache_state(k_base, v_base, cache_len, self.max_len)
        dropout_p = self.dropout if self.training else 0.0
        attn = attention(q, k_base, v_base, dropout_p, layer_cache is None, self.backend)
        output = self.o(attn.transpose(1, 2).contiguous().view(batch, seq, -1))
        next_cache = next_state if use_cache else None
        return output, next_cache

    def project_qkv(self, x, batch: int, seq: int):
        q_raw, k_raw, v_raw = self.qkv(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_raw.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def rope_slice(self, offset: int, seq: int, dtype):
        cos = self.rope_cos[offset : offset + seq].to(dtype=dtype)[None, None, :, :]
        sin = self.rope_sin[offset : offset + seq].to(dtype=dtype)[None, None, :, :]
        return cos, sin


def rope_tables(seq: int, dim: int):
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq).float()
    angles = torch.outer(pos, freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def append_cache(layer_cache: dict, k_new, v_new, max_len: int):
    k_cache, v_cache = layer_cache["k"], layer_cache["v"]
    current = int(layer_cache.get("len", k_cache.size(2)))
    incoming = k_new.size(2)
    total = current + incoming
    if total > max_len:
        drop = total - max_len
        keep = max(0, current - drop)
        if keep:
            k_cache[:, :, :keep].copy_(k_cache[:, :, drop:current])
            v_cache[:, :, :keep].copy_(v_cache[:, :, drop:current])
        current = keep
    k_cache[:, :, current : current + incoming].copy_(k_new.detach())
    v_cache[:, :, current : current + incoming].copy_(v_new.detach())
    length = min(max_len, current + incoming)
    return k_cache[:, :, :length], v_cache[:, :, :length], length


def cache_state(k, v, length: int, max_len: int):
    if k.size(2) == max_len:
        return {"k": k.detach(), "v": v.detach(), "len": length}
    k_store = k.new_zeros((*k.shape[:2], max_len, k.shape[-1]))
    v_store = v.new_zeros((*v.shape[:2], max_len, v.shape[-1]))
    kept = min(max_len, k.size(2))
    k_store[:, :, :kept].copy_(k[:, :, -kept:].detach())
    v_store[:, :, :kept].copy_(v[:, :, -kept:].detach())
    return {"k": k_store, "v": v_store, "len": kept}


def attention(q, k, v, dropout_p: float, is_causal: bool, backend: str):
    flash = flash_attention(q, k, v, dropout_p, is_causal, backend)
    if flash is not None:
        return flash
    return sdpa_attention(q, k, v, dropout_p, is_causal, backend)


def flash_attention(q, k, v, dropout_p: float, is_causal: bool, backend: str):
    if backend != "flash2":
        return None
    try:
        from flash_attn import flash_attn_func
    except Exception as error:
        raise RuntimeError("TRAIN_ATTENTION_BACKEND=flash2 requires flash-attn") from error
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_t, k_t, v_t, dropout_p=dropout_p, causal=is_causal)
    return out.transpose(1, 2).contiguous()


def sdpa_attention(q, k, v, dropout_p: float, is_causal: bool, backend: str):
    with sdpa_backend(backend):
        try:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=q.size(1) != k.size(1))
        except TypeError:
            if q.size(1) == k.size(1):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
            repeats = q.size(1) // k.size(1)
            return F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeats, dim=1),
                v.repeat_interleave(repeats, dim=1),
                dropout_p=dropout_p,
                is_causal=is_causal,
            )


def sdpa_backend(backend: str):
    if backend not in {"sdpa_flash", "sdpa_math"}:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return nullcontext()
    choice = SDPBackend.FLASH_ATTENTION if backend == "sdpa_flash" else SDPBackend.MATH
    return sdpa_kernel(choice)
