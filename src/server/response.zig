const std = @import("std");

pub const Response = struct {
    code: u16,
    content_type: []const u8,
    body: []const u8,
};

pub fn write(conn: std.net.Server.Connection, res: Response) !void {
    const status = switch (res.code) {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        503 => "Service Unavailable",
        else => "Internal Server Error",
    };
    var writer = conn.stream.writer();
    try writer.print(
        "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
        .{ res.code, status, res.content_type, res.body.len },
    );
    try writer.writeAll(res.body);
}
