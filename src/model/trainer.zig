const tokenizer = @import("tokenizer.zig");
const transformer = @import("transformer.zig");
const lightweighting = @import("lightweighting.zig");

pub const TrainingSummary = struct {
    token_count: usize,
    estimated_params: u64,
    deploy_size_mib: u64,
};

pub fn planTraining(input_text: []const u8, cfg: transformer.TransformerConfig) !TrainingSummary {
    const tokens = tokenizer.tokenCount(input_text);
    const params = transformer.estimateParameterCount(cfg);
    const deploy_size = lightweighting.estimateDeploySizeMiB(params, 4);
    try lightweighting.checkDeployLimit(params, 4);
    return .{
        .token_count = tokens,
        .estimated_params = params,
        .deploy_size_mib = deploy_size,
    };
}
