pub const TransformerConfig = struct {
    layers: u32 = 24,
    hidden_size: u32 = 1536,
    heads: u32 = 12,
    vocab_size: u32 = 64000,
    ff_mult: u32 = 4,
};

pub fn estimateParameterCount(cfg: TransformerConfig) u64 {
    const embed = @as(u64, cfg.vocab_size) * cfg.hidden_size;
    const attn = @as(u64, cfg.layers) * 4 * cfg.hidden_size * cfg.hidden_size;
    const ffn = @as(u64, cfg.layers) * 3 * cfg.hidden_size * (cfg.ff_mult * cfg.hidden_size);
    const norms = @as(u64, cfg.layers) * cfg.hidden_size * 2;
    return embed + attn + ffn + norms;
}

pub fn approxModelMiB(params: u64, bits_per_weight: u8) u64 {
    const bits = params * bits_per_weight;
    const bytes = bits / 8;
    const mib = bytes / (1024 * 1024);
    return mib;
}
