const std = @import("std");
const config_mod = @import("../config.zig");
const memory = @import("../storage/memory.zig");
const postgres = @import("../storage/postgres.zig");
const runtime_store = @import("../storage/runtime.zig");
const librarian_mod = @import("../agent/librarian.zig");
const orchestrator_mod = @import("../agent/orchestrator.zig");
const req = @import("request.zig");
const resp = @import("response.zig");

const JsonInputError = error{
    BadRequest,
    OutOfMemory,
};

pub const App = struct {
    allocator: std.mem.Allocator,
    config: config_mod.Config,
    memory_store: ?*memory.MemoryStore,
    postgres_store: ?*postgres.PostgresAdapter,
    store: runtime_store.RuntimeStore,
    librarian: librarian_mod.Librarian,
    orchestrator: orchestrator_mod.Orchestrator,
};

pub fn init(allocator: std.mem.Allocator, config: config_mod.Config) !App {
    var app = App{
        .allocator = allocator,
        .config = config,
        .memory_store = null,
        .postgres_store = null,
        .store = undefined,
        .librarian = undefined,
        .orchestrator = orchestrator_mod.Orchestrator.init(16),
    };

    if (shouldUsePostgres(config.database_url)) {
        const store = try allocator.create(postgres.PostgresAdapter);
        store.* = postgres.PostgresAdapter.init(allocator, config.database_url);
        app.postgres_store = store;
        app.store = runtime_store.RuntimeStore.fromPostgres(store);
    } else {
        const store = try allocator.create(memory.MemoryStore);
        store.* = memory.MemoryStore.init(allocator);
        app.memory_store = store;
        app.store = runtime_store.RuntimeStore.fromMemory(store);
    }
    app.librarian = librarian_mod.Librarian.init(app.store);
    return app;
}

pub fn deinit(self: *App) void {
    self.store.deinit();
    if (self.memory_store) |store| self.allocator.destroy(store);
    if (self.postgres_store) |store| self.allocator.destroy(store);
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

pub fn routeForTest(app: *App, allocator: std.mem.Allocator, parsed: req.Request) resp.Response {
    return route(app, allocator, parsed);
}

fn route(app: *App, allocator: std.mem.Allocator, parsed: req.Request) resp.Response {
    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.eql(u8, parsed.target, "/")) {
        return .{ .code = 200, .content_type = "text/html; charset=utf-8", .body = @embedFile("index.html") };
    }
    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.eql(u8, parsed.target, "/healthz")) {
        if (app.store.health()) {
            return .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\",\"app\":\"ready\",\"storage\":\"ready\"}" };
        }
        return .{ .code = 503, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"storage_unavailable\",\"app\":\"ready\",\"storage\":\"not_ready\"}" };
    }
    if (!authorized(app, parsed.headers)) {
        return .{ .code = 401, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"unauthorized\",\"message\":\"invalid admin token\"}" };
    }

    if (std.mem.eql(u8, parsed.method, "GET") and std.mem.startsWith(u8, parsed.target, "/api/records/list")) {
        const q = req.queryValue(parsed.target, "q") orelse "";
        const body = app.librarian.listJson(allocator, q) catch return serverError();
        return .{ .code = 200, .content_type = "application/json", .body = body };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/records/upsert")) {
        const id = jsonStringField(allocator, parsed.body, "id") catch |err| return switch (err) {
            error.BadRequest => badRequest(),
            error.OutOfMemory => serverError(),
        };
        const title = jsonStringField(allocator, parsed.body, "title") catch |err| return switch (err) {
            error.BadRequest => badRequest(),
            error.OutOfMemory => serverError(),
        };
        const body = jsonStringField(allocator, parsed.body, "body") catch |err| return switch (err) {
            error.BadRequest => badRequest(),
            error.OutOfMemory => serverError(),
        };
        app.librarian.upsert(id, title, body) catch return serverError();
        return .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\"}" };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/records/delete")) {
        const id = jsonStringField(allocator, parsed.body, "id") catch |err| return switch (err) {
            error.BadRequest => badRequest(),
            error.OutOfMemory => serverError(),
        };
        const found = app.librarian.delete(id) catch return serverError();
        return if (found)
            .{ .code = 200, .content_type = "application/json", .body = "{\"status\":\"ok\"}" }
        else
            .{ .code = 404, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"not_found\"}" };
    }
    if (std.mem.eql(u8, parsed.method, "POST") and std.mem.eql(u8, parsed.target, "/api/chat")) {
        const message = jsonStringField(allocator, parsed.body, "message") catch |err| return switch (err) {
            error.BadRequest => badRequest(),
            error.OutOfMemory => serverError(),
        };
        if (!app.orchestrator.tryBegin()) {
            return .{ .code = 503, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"queue_full\"}" };
        }
        defer app.orchestrator.end();
        const result = app.orchestrator.runChat(allocator, &app.librarian, message) catch return serverError();
        const body = std.json.stringifyAlloc(
            allocator,
            .{
                .status = "ok",
                .reply = result.reply,
                .trace = .{
                    .parallel_steps = result.parallel_steps,
                    .queue_depth = result.queue_depth,
                },
            },
            .{},
        ) catch return serverError();
        return .{ .code = 200, .content_type = "application/json", .body = body };
    }
    return .{ .code = 404, .content_type = "application/json", .body = "{\"status\":\"error\",\"code\":\"not_found\"}" };
}

fn jsonStringField(allocator: std.mem.Allocator, body: []const u8, key: []const u8) JsonInputError![]const u8 {
    const value = req.jsonField(allocator, body, key) catch |err| switch (err) {
        error.InvalidJson => return error.BadRequest,
        error.OutOfMemory => return error.OutOfMemory,
    };
    return value orelse error.BadRequest;
}

fn shouldUsePostgres(database_url: []const u8) bool {
    return std.mem.startsWith(u8, database_url, "postgres://") or std.mem.startsWith(u8, database_url, "postgresql://");
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
