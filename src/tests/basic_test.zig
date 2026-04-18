const std = @import("std");
const tokenizer = @import("../model/tokenizer.zig");
const transformer = @import("../model/transformer.zig");
const lightweighting = @import("../model/lightweighting.zig");
const memory = @import("../storage/memory.zig");
const runtime_store = @import("../storage/runtime.zig");

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

test "runtime store memory adapter lifecycle" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var backing = memory.MemoryStore.init(gpa.allocator());
    defer backing.deinit();
    var store = runtime_store.RuntimeStore.fromMemory(&backing);

    try store.upsert("a", "title", "body");
    try std.testing.expectEqual(@as(usize, 1), try store.countMatches("title"));

    const listed = try store.list(gpa.allocator(), "");
    defer {
        for (listed) |item| {
            gpa.allocator().free(item.id);
            gpa.allocator().free(item.title);
            gpa.allocator().free(item.body);
        }
        gpa.allocator().free(listed);
    }

    try std.testing.expectEqual(@as(usize, 1), listed.len);
    try std.testing.expect(try store.delete("a"));
}
