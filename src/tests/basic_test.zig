const std = @import("std");
const tokenizer = @import("../model/tokenizer.zig");
const transformer = @import("../model/transformer.zig");
const lightweighting = @import("../model/lightweighting.zig");
const memory = @import("../storage/memory.zig");

test "tokenizer counts whitespace tokens" {
    try std.testing.expectEqual(@as(usize, 3), tokenizer.tokenCount("one two   three"));
}

test "transformer estimate is non-zero" {
    const cfg = transformer.TransformerConfig{};
    try std.testing.expect(transformer.estimateParameterCount(cfg) > 1000);
}

test "deploy size guard enforces 512 MiB cap" {
    const huge_params: u64 = 2_000_000_000;
    try std.testing.expectError(error.ArtifactTooLarge, lightweighting.checkDeployLimit(huge_params, 4));
}

test "memory store upsert delete lifecycle" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var store = memory.MemoryStore.init(gpa.allocator());
    defer store.deinit();

    try store.upsert("a", "title", "body");
    try store.upsert("a", "title2", "body2");
    try std.testing.expectEqual(@as(usize, 1), store.countMatches("title2"));
    try std.testing.expect(store.delete("a"));
}
