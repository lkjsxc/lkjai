pub const ArtifactLimitMiB: u64 = 512;

pub fn estimateDeploySizeMiB(parameter_count: u64, quant_bits: u8) u64 {
    if (quant_bits == 0) return 0;
    const total_bits = parameter_count * quant_bits;
    const total_bytes = total_bits / 8;
    return total_bytes / (1024 * 1024);
}

pub fn checkDeployLimit(parameter_count: u64, quant_bits: u8) !void {
    const size_mib = estimateDeploySizeMiB(parameter_count, quant_bits);
    if (size_mib > ArtifactLimitMiB) return error.ArtifactTooLarge;
}
