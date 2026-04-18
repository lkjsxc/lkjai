const std = @import("std");

pub const Request = struct {
    method: []const u8,
    target: []const u8,
    headers: []const u8,
    body: []const u8,
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

pub fn jsonField(body: []const u8, key: []const u8) ?[]const u8 {
    var needle_buf: [128]u8 = undefined;
    const needle = std.fmt.bufPrint(&needle_buf, "\"{s}\"", .{key}) catch return null;
    const key_pos = std.mem.indexOf(u8, body, needle) orelse return null;
    const after_key = body[key_pos + needle.len ..];
    const colon = std.mem.indexOfScalar(u8, after_key, ':') orelse return null;
    const after_colon = std.mem.trimLeft(u8, after_key[colon + 1 ..], " \t\r\n");
    if (after_colon.len == 0 or after_colon[0] != '"') return null;
    const content = after_colon[1..];
    const end_quote = std.mem.indexOfScalar(u8, content, '"') orelse return null;
    return content[0..end_quote];
}
