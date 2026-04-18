const std = @import("std");
const tokenizer = @import("../model/tokenizer.zig");
const runtime_store = @import("../storage/runtime.zig");

pub const Librarian = struct {
    store: runtime_store.RuntimeStore,

    pub fn init(store: runtime_store.RuntimeStore) Librarian {
        return .{ .store = store };
    }

    pub fn upsert(self: *Librarian, id: []const u8, title: []const u8, body: []const u8) !void {
        try self.store.upsert(id, title, body);
    }

    pub fn delete(self: *Librarian, id: []const u8) !bool {
        return try self.store.delete(id);
    }

    pub fn countMatches(self: *Librarian, query: []const u8) !usize {
        return try self.store.countMatches(query);
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
        const writer = out.writer();
        try writer.writeAll("{\"status\":\"ok\",\"records\":[");
        for (items, 0..) |item, i| {
            if (i != 0) try writer.writeByte(',');
            try writer.writeAll("{\"id\":");
            try std.json.stringify(item.id, .{}, writer);
            try writer.writeAll(",\"title\":");
            try std.json.stringify(item.title, .{}, writer);
            try writer.writeByte('}');
        }
        try writer.writeAll("]}");
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
