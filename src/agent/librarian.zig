const std = @import("std");
const memory = @import("../storage/memory.zig");
const tokenizer = @import("../model/tokenizer.zig");

pub const Librarian = struct {
    store: *memory.MemoryStore,

    pub fn init(store: *memory.MemoryStore) Librarian {
        return .{ .store = store };
    }

    pub fn upsert(self: *Librarian, id: []const u8, title: []const u8, body: []const u8) !void {
        try self.store.upsert(id, title, body);
    }

    pub fn delete(self: *Librarian, id: []const u8) bool {
        return self.store.delete(id);
    }

    pub fn countMatches(self: *Librarian, query: []const u8) usize {
        return self.store.countMatches(query);
    }

    pub fn listJson(self: *Librarian, allocator: std.mem.Allocator, query: []const u8) ![]u8 {
        const items = try self.store.list(allocator, query);
        defer {
            for (items) |item| {
                allocator.free(item.id);
                allocator.free(item.title);
                allocator.free(item.body);
            }
            allocator.free(items);
        }

        var out = std.ArrayList(u8).init(allocator);
        try out.appendSlice("{\"status\":\"ok\",\"records\":[");
        for (items, 0..) |item, i| {
            if (i != 0) try out.appendSlice(",");
            const part = try std.fmt.allocPrint(allocator, "{{\"id\":\"{s}\",\"title\":\"{s}\"}}", .{ item.id, item.title });
            defer allocator.free(part);
            try out.appendSlice(part);
        }
        try out.appendSlice("]}");
        return out.toOwnedSlice();
    }

    pub fn buildChatReply(
        self: *Librarian,
        allocator: std.mem.Allocator,
        message: []const u8,
        match_count: usize,
        token_count: usize,
    ) ![]u8 {
        _ = self;
        const normalized = try tokenizer.normalize(allocator, message);
        defer allocator.free(normalized);
        return std.fmt.allocPrint(
            allocator,
            "Librarian summary: query=\"{s}\", tokens={d}, matching_records={d}.",
            .{ normalized, token_count, match_count },
        );
    }
};
