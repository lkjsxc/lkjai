const std = @import("std");

pub const Request = struct {
    method: []const u8,
    target: []const u8,
    headers: []const u8,
    body: []const u8,
};

pub const JsonFieldError = error{
    InvalidJson,
    OutOfMemory,
};

pub fn parse(raw: []const u8) !Request {
    const split_at = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.InvalidRequest;
    const head = raw[0..split_at];
    const body = if (split_at + 4 <= raw.len) raw[split_at + 4 ..] else "";
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return error.InvalidRequest;
    const request_line = head[0..line_end];
    const headers = if (line_end + 2 <= head.len) head[line_end + 2 ..] else "";

    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return error.InvalidRequest;
    const target = parts.next() orelse return error.InvalidRequest;
    return .{ .method = method, .target = target, .headers = headers, .body = body };
}

pub fn headerValue(headers: []const u8, name: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, ':')) |idx| {
            const key = std.mem.trim(u8, line[0..idx], " \t");
            const value = std.mem.trim(u8, line[idx + 1 ..], " \t");
            if (std.ascii.eqlIgnoreCase(key, name)) return value;
        }
    }
    return null;
}

pub fn queryValue(target: []const u8, key: []const u8) ?[]const u8 {
    const qmark = std.mem.indexOfScalar(u8, target, '?') orelse return null;
    const query = target[qmark + 1 ..];
    var pairs = std.mem.splitScalar(u8, query, '&');
    while (pairs.next()) |pair| {
        if (std.mem.indexOfScalar(u8, pair, '=')) |idx| {
            const k = pair[0..idx];
            if (std.mem.eql(u8, k, key)) return pair[idx + 1 ..];
        }
    }
    return null;
}

pub fn jsonField(allocator: std.mem.Allocator, body: []const u8, key: []const u8) JsonFieldError!?[]const u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidJson,
    };
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidJson;
    const value = parsed.value.object.get(key) orelse return null;
    if (value != .string) return error.InvalidJson;

    const duped = try allocator.dupe(u8, value.string);
    return duped;
}
