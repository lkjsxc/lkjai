const std = @import("std");
const record_mod = @import("record.zig");

pub const Record = record_mod.Record;

pub const PostgresAdapter = struct {
    allocator: std.mem.Allocator,
    database_url: []const u8,
    mutex: std.Thread.Mutex = .{},
    ready: bool = false,

    pub fn init(allocator: std.mem.Allocator, database_url: []const u8) PostgresAdapter {
        _ = allocator;
        return .{
            .allocator = std.heap.page_allocator,
            .database_url = database_url,
        };
    }

    pub fn deinit(self: *PostgresAdapter) void {
        _ = self;
    }

    pub fn upsert(self: *PostgresAdapter, id: []const u8, title: []const u8, body: []const u8) !void {
        try self.ensureSchema();
        const id_lit = try sqlLiteral(self.allocator, id);
        defer self.allocator.free(id_lit);
        const title_lit = try sqlLiteral(self.allocator, title);
        defer self.allocator.free(title_lit);
        const body_lit = try sqlLiteral(self.allocator, body);
        defer self.allocator.free(body_lit);
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "INSERT INTO runtime_records (id, title, body, updated_at) VALUES ({s}, {s}, {s}, NOW()) " ++
                "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body, updated_at = NOW();",
            .{ id_lit, title_lit, body_lit },
        );
        defer self.allocator.free(sql);

        const stdout = try self.exec(sql, 128);
        self.allocator.free(stdout);
    }

    pub fn delete(self: *PostgresAdapter, id: []const u8) !bool {
        try self.ensureSchema();
        const id_lit = try sqlLiteral(self.allocator, id);
        defer self.allocator.free(id_lit);

        const sql = try std.fmt.allocPrint(self.allocator, "DELETE FROM runtime_records WHERE id = {s} RETURNING 1;", .{id_lit});
        defer self.allocator.free(sql);
        const stdout = try self.exec(sql, 128);
        defer self.allocator.free(stdout);
        return std.mem.trim(u8, stdout, " \n\r\t").len != 0;
    }

    pub fn countMatches(self: *PostgresAdapter, query: []const u8) !usize {
        try self.ensureSchema();
        const query_lit = try sqlLiteral(self.allocator, query);
        defer self.allocator.free(query_lit);

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT COUNT(*) FROM runtime_records " ++
                "WHERE ({s} = '' OR position({s} in title) > 0 OR position({s} in body) > 0);",
            .{ query_lit, query_lit, query_lit },
        );
        defer self.allocator.free(sql);

        const stdout = try self.exec(sql, 128);
        defer self.allocator.free(stdout);
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        if (trimmed.len == 0) return error.InvalidResponse;
        return std.fmt.parseInt(usize, trimmed, 10) catch error.InvalidResponse;
    }

    pub fn list(self: *PostgresAdapter, allocator: std.mem.Allocator, query: []const u8) ![]Record {
        try self.ensureSchema();
        const query_lit = try sqlLiteral(self.allocator, query);
        defer self.allocator.free(query_lit);

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT encode(convert_to(id, 'UTF8'), 'base64') || ',' || " ++
                "encode(convert_to(title, 'UTF8'), 'base64') || ',' || " ++
                "encode(convert_to(body, 'UTF8'), 'base64') FROM runtime_records " ++
                "WHERE ({s} = '' OR position({s} in title) > 0 OR position({s} in body) > 0) " ++
                "ORDER BY updated_at DESC, id ASC;",
            .{ query_lit, query_lit, query_lit },
        );
        defer self.allocator.free(sql);

        const stdout = try self.exec(sql, 1024 * 1024);
        defer self.allocator.free(stdout);

        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        if (trimmed.len == 0) return allocator.alloc(Record, 0);

        var items = std.ArrayList(Record).init(allocator);
        errdefer {
            for (items.items) |item| {
                allocator.free(item.id);
                allocator.free(item.title);
                allocator.free(item.body);
            }
            items.deinit();
        }

        var lines = std.mem.splitScalar(u8, trimmed, '\n');
        while (lines.next()) |line_raw| {
            const line = std.mem.trimRight(u8, line_raw, "\r");
            if (line.len == 0) continue;

            var cols = std.mem.splitScalar(u8, line, ',');
            const id_b64 = cols.next() orelse return error.InvalidResponse;
            const title_b64 = cols.next() orelse return error.InvalidResponse;
            const body_b64 = cols.next() orelse return error.InvalidResponse;
            if (cols.next() != null) return error.InvalidResponse;

            const id_out = try decodeBase64(allocator, id_b64);
            errdefer allocator.free(id_out);
            const title_out = try decodeBase64(allocator, title_b64);
            errdefer allocator.free(title_out);
            const body_out = try decodeBase64(allocator, body_b64);
            errdefer allocator.free(body_out);
            try items.append(.{ .id = id_out, .title = title_out, .body = body_out });
        }
        return items.toOwnedSlice();
    }

    pub fn health(self: *PostgresAdapter) bool {
        self.ensureSchema() catch return false;
        const stdout = self.exec("SELECT 1;", 64) catch return false;
        defer self.allocator.free(stdout);
        return std.mem.eql(u8, std.mem.trim(u8, stdout, " \n\r\t"), "1");
    }

    fn ensureSchema(self: *PostgresAdapter) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.ready) return;

        const sql =
            "CREATE TABLE IF NOT EXISTS runtime_records (" ++
            "id TEXT PRIMARY KEY, " ++
            "title TEXT NOT NULL, " ++
            "body TEXT NOT NULL, " ++
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());";
        const stdout = try self.exec(sql, 128);
        self.allocator.free(stdout);
        self.ready = true;
    }

    fn exec(self: *PostgresAdapter, sql: []const u8, max_output_bytes: usize) ![]u8 {
        const argv = [_][]const u8{ "psql", self.database_url, "-X", "-A", "-t", "-q", "--set", "ON_ERROR_STOP=1", "-c", sql };
        const result = try std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = &argv,
            .max_output_bytes = max_output_bytes,
        });
        defer self.allocator.free(result.stderr);

        switch (result.term) {
            .Exited => |code| {
                if (code == 0) return result.stdout;
                self.allocator.free(result.stdout);
                std.log.err("psql exited with code {d}: {s}", .{ code, result.stderr });
                return error.StorageUnavailable;
            },
            else => {
                self.allocator.free(result.stdout);
                std.log.err("psql terminated unexpectedly: {s}", .{result.stderr});
                return error.StorageUnavailable;
            },
        }
    }
};

fn sqlLiteral(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    try out.append('\'');
    for (value) |ch| {
        if (ch == '\'') {
            try out.append('\'');
            try out.append('\'');
        } else {
            try out.append(ch);
        }
    }
    try out.append('\'');
    return out.toOwnedSlice();
}

fn decodeBase64(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
    const out_len = try std.base64.standard.Decoder.calcSizeForSlice(encoded);
    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);
    try std.base64.standard.Decoder.decode(out, encoded);
    return out;
}
