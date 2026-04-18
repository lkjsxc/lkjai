const std = @import("std");
const config_mod = @import("config.zig");
const server = @import("server/http.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = try config_mod.Config.load(allocator);
    var app = try server.init(allocator, cfg);
    defer server.deinit(&app);

    std.log.info("lkjai listening on :{d}", .{cfg.port});
    try server.run(&app);
}

test {
    _ = @import("tests/basic_test.zig");
    _ = @import("tests/json_safety_test.zig");
}
