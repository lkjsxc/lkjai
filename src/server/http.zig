const std = @import("std");
const config_mod = @import("../config.zig");
const memory = @import("../storage/memory.zig");
const librarian_mod = @import("../agent/librarian.zig");
const orchestrator_mod = @import("../agent/orchestrator.zig");
const req = @import("request.zig");
const resp = @import("response.zig");

pub const App = struct {
    allocator: std.mem.Allocator,
    config: config_mod.Config,
    store: memory.MemoryStore,
    librarian: librarian_mod.Librarian,
    orchestrator: orchestrator_mod.Orchestrator,
};

pub fn init(allocator: std.mem.Allocator, config: config_mod.Config) App {
    var store = memory.MemoryStore.init(allocator);
    const librarian = librarian_mod.Librarian.init(&store);
    return .{
        .allocator = allocator,
        .config = config,
        .store = store,
        .librarian = librarian,
        .orchestrator = orchestrator_mod.Orchestrator.init(16),
    };
}

pub fn deinit(self: *App) void {
    self.store.deinit();
    self.allocator.free(self.config.admin_token);
    self.allocator.free(self.config.database_url);
}

pub fn run(self: *App) !void {
    const address = try std.net.Address.parseIp("0.0.0.0", self.config.port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    while (true) {
        const conn = try server.accept();
        var thread = try std.Thread.spawn(.{}, handleConn, .{ self, conn });
        thread.detach();
    }
}

fn handleConn(app: *App, conn: std.net.Server.Connection) void {
    defer conn.stream.close();
    var arena = std.heap.ArenaAllocator.init(app.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var buf: [16384]u8 = undefined;
    const n = conn.stream.read(&buf) catch return;
    if (n == 0) return;

    const parsed = req.parse(buf[0..n]) catch {
        resp.write(conn, .{ .code = 400, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"bad_request\"}" }) catch {};
        return;
    };

    const r = route(app, allocator, parsed);
    resp.write(conn, r) catch {};
}

fn route(app: *App, allocator: std.mem.Allocator, parsed: req.Request) resp.Response {
    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.eql(u8, parsed.target, "/")) {
        return .{ .code = 200, .content_type = "text/html; charset=utf-8", .body = @embedFile("index.html") };
    }
    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.eql(u8, parsed.target, "/healthz")) {
        return .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\"}" };
    }
    if (!authorized(app, parsed.headers)) {
        return .{ .code = 401, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"unauthorized\",\"message\":\"invalid admin token\"}" };
    }

    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.startsWith(u8, parsed.target, "/api/records/list")) {
        const q = req.queryValue(parsed.target, "q") orelse "";
        const body = app.librarian.listJson(allocator, q) catch "{\"status\":\"error\",\"code\":\"list_failed\"}";
        return .{ .code = 200, .content_type = "application/json", .body = body };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/records/upsert")) {
        const id = req.jsonField(parsed.body, "id") orelse return badRequest();
        const title = req.jsonField(parsed.body, "title") orelse return badRequest();
        const body = req.jsonField(parsed.body, "body") orelse return badRequest();
        app.librarian.upsert(id, title, body) catch return serverError();
        return .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\"}" };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/records/delete")) {
        const id = req.jsonField(parsed.body, "id") orelse return badRequest();
        const found = app.librarian.delete(id);
        return if (found)
            .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\"}" }
        else
            .{ .code = 404, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"not_found\"}" };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/chat")) {
        const message = req.jsonField(parsed.body, "message") orelse return badRequest();
        if (!app.orchestrator.tryBegin()) {
            return .{ .code = 503, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"queue_full\"}" };
        }
        defer app.orchestrator.end();
        const result = app.orchestrator.runChat(allocator, &app.librarian, message) catch return serverError();
        const body = std.fmt.allocPrint(
            allocator,
            "{{\"status\":\"ok\",\"reply\":\"{s}\",\"trace\":{{\"parallel_steps\":{d},\"queue_depth\":{d}}}}}",
            .{ result.reply, result.parallel_steps, result.queue_depth },
        ) catch return serverError();
        return .{ .code = 200, .content_type = "application/json", .body = body };
    }
    return .{ .code = 404, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"not_found\"}" };
}

fn authorized(app: *App, headers: []const u8) bool {
    const token = req.headerValue(headers, "x-admin-token") orelse return false;
    return std.mem.eql(u8, token, app.config.admin_token);
}

fn badRequest() resp.Response {
    return .{ .code = 400, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"bad_request\"}" };
}

fn serverError() resp.Response {
    return .{ .code = 500, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"internal\"}" };
}
