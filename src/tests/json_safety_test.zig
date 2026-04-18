const std = @import("std");
const config_mod = @import("../config.zig");
const librarian_mod = @import("../agent/librarian.zig");
const memory = @import("../storage/memory.zig");
const runtime_store = @import("../storage/runtime.zig");
const http = @import("../server/http.zig");
const req = @import("../server/request.zig");

fn testApp(allocator: std.mem.Allocator) !*http.App {
    const token = try allocator.dupe(u8, "secret");
    const db_url = try allocator.dupe(u8, "memory://example");
    const app = try allocator.create(http.App);
    app.* = try http.init(allocator, config_mod.Config{
        .port = 8080,
        .admin_token = token,
        .database_url = db_url,
    });
    return app;
}

test "jsonField decodes escaped quotes newlines and backslashes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const body = "{\"message\":\"line1\\nquote: \\\"q\\\" slash: \\\\x\"}";
    const message = (try req.jsonField(arena.allocator(), body, "message")) orelse unreachable;
    try std.testing.expectEqualStrings("line1\nquote: \"q\" slash: \\x", message);
}

test "jsonField rejects malformed and non-string JSON payloads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    try std.testing.expectError(error.InvalidJson, req.jsonField(arena.allocator(), "{\"message\":\"oops\"", "message"));
    try std.testing.expectError(error.InvalidJson, req.jsonField(arena.allocator(), "{\"message\":123}", "message"));
}

test "listJson escapes dynamic record strings" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = memory.MemoryStore.init(allocator);
    defer store.deinit();
    const store_adapter = runtime_store.RuntimeStore.fromMemory(&store);
    var librarian = librarian_mod.Librarian.init(store_adapter);

    try librarian.upsert("id\\slash", "title\n\"quoted\"", "ignored");
    const body = try librarian.listJson(allocator, "");
    defer allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\\\\") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\\\"") != null);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const records = parsed.value.object.get("records").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("id\\slash", records[0].object.get("id").?.string);
    try std.testing.expectEqualStrings("title\n\"quoted\"", records[0].object.get("title").?.string);
}

test "route returns bad_request for malformed chat body" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const req_allocator = arena.allocator();

    const app = try testApp(allocator);
    defer {
        http.deinit(app);
        allocator.destroy(app);
    }

    const response = http.routeForTest(app, req_allocator, .{
        .method = "POST",
        .target = "/api/chat",
        .headers = "x-admin-token: secret",
        .body = "{\"message\":\"unterminated\"",
    });

    try std.testing.expectEqual(@as(u16, 400), response.code);
    try std.testing.expectEqualStrings("{\"status\":\"error\",\"code\":\"bad_request\"}", response.body);
}

test "route returns bad_request for non-string chat message" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const req_allocator = arena.allocator();

    const app = try testApp(allocator);
    defer {
        http.deinit(app);
        allocator.destroy(app);
    }

    const response = http.routeForTest(app, req_allocator, .{
        .method = "POST",
        .target = "/api/chat",
        .headers = "x-admin-token: secret",
        .body = "{\"message\":42}",
    });

    try std.testing.expectEqual(@as(u16, 400), response.code);
    try std.testing.expectEqualStrings("{\"status\":\"error\",\"code\":\"bad_request\"}", response.body);
}

test "route JSON response escapes chat reply" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const req_allocator = arena.allocator();

    const app = try testApp(allocator);
    defer {
        http.deinit(app);
        allocator.destroy(app);
    }

    const response = http.routeForTest(app, req_allocator, .{
        .method = "POST",
        .target = "/api/chat",
        .headers = "x-admin-token: secret",
        .body = "{\"message\":\"Hi \\\"Team\\\" \\\\path\"}",
    });

    try std.testing.expectEqual(@as(u16, 200), response.code);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "\\\\") != null);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, response.body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("ok", parsed.value.object.get("status").?.string);
}
