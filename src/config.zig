const std = @import("std");

pub const Config = struct {
    port: u16,
    admin_token: []u8,
    database_url: []u8,

    pub fn load(allocator: std.mem.Allocator) !Config {
        const port_raw = std.process.getEnvVarOwned(allocator, "PORT") catch try allocator.dupe(u8, "8080");
        const token = std.process.getEnvVarOwned(allocator, "ADMIN_TOKEN") catch try allocator.dupe(u8, "change-me");
        const db_url = std.process.getEnvVarOwned(allocator, "DATABASE_URL") catch try allocator.dupe(u8, "postgres://lkjai:lkjai@localhost:5432/lkjai");

        const parsed = std.fmt.parseInt(u16, std.mem.trim(u8, port_raw, " \n\r\t"), 10) catch 8080;
        return .{ .port = parsed, .admin_token = token, .database_url = db_url };
    }
};
