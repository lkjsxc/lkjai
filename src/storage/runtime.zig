const std = @import("std");
const record_mod = @import("record.zig");
const memory = @import("memory.zig");
const postgres = @import("postgres.zig");

pub const Record = record_mod.Record;

pub const RuntimeStore = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        deinit: *const fn (ptr: *anyopaque) void,
        upsert: *const fn (ptr: *anyopaque, id: []const u8, title: []const u8, body: []const u8) anyerror!void,
        delete: *const fn (ptr: *anyopaque, id: []const u8) anyerror!bool,
        count_matches: *const fn (ptr: *anyopaque, query: []const u8) anyerror!usize,
        list: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator, query: []const u8) anyerror![]Record,
        health: *const fn (ptr: *anyopaque) bool,
    };

    pub fn fromMemory(store: *memory.MemoryStore) RuntimeStore {
        return .{ .ptr = store, .vtable = &memory_vtable };
    }

    pub fn fromPostgres(adapter: *postgres.PostgresAdapter) RuntimeStore {
        return .{ .ptr = adapter, .vtable = &postgres_vtable };
    }

    pub fn deinit(self: *RuntimeStore) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn upsert(self: *RuntimeStore, id: []const u8, title: []const u8, body: []const u8) !void {
        try self.vtable.upsert(self.ptr, id, title, body);
    }

    pub fn delete(self: *RuntimeStore, id: []const u8) !bool {
        return try self.vtable.delete(self.ptr, id);
    }

    pub fn countMatches(self: *RuntimeStore, query: []const u8) !usize {
        return try self.vtable.count_matches(self.ptr, query);
    }

    pub fn list(self: *RuntimeStore, allocator: std.mem.Allocator, query: []const u8) ![]Record {
        return try self.vtable.list(self.ptr, allocator, query);
    }

    pub fn health(self: *RuntimeStore) bool {
        return self.vtable.health(self.ptr);
    }
};

fn asMemory(ptr: *anyopaque) *memory.MemoryStore {
    return @ptrCast(@alignCast(ptr));
}

fn asPostgres(ptr: *anyopaque) *postgres.PostgresAdapter {
    return @ptrCast(@alignCast(ptr));
}

fn memoryDeinit(ptr: *anyopaque) void {
    asMemory(ptr).deinit();
}
fn memoryUpsert(ptr: *anyopaque, id: []const u8, title: []const u8, body: []const u8) !void {
    try asMemory(ptr).upsert(id, title, body);
}
fn memoryDelete(ptr: *anyopaque, id: []const u8) !bool {
    return asMemory(ptr).delete(id);
}
fn memoryCount(ptr: *anyopaque, query: []const u8) !usize {
    return asMemory(ptr).countMatches(query);
}
fn memoryList(ptr: *anyopaque, allocator: std.mem.Allocator, query: []const u8) ![]Record {
    return asMemory(ptr).list(allocator, query);
}
fn memoryHealth(ptr: *anyopaque) bool {
    _ = ptr;
    return true;
}

fn postgresDeinit(ptr: *anyopaque) void {
    asPostgres(ptr).deinit();
}
fn postgresUpsert(ptr: *anyopaque, id: []const u8, title: []const u8, body: []const u8) !void {
    try asPostgres(ptr).upsert(id, title, body);
}
fn postgresDelete(ptr: *anyopaque, id: []const u8) !bool {
    return try asPostgres(ptr).delete(id);
}
fn postgresCount(ptr: *anyopaque, query: []const u8) !usize {
    return try asPostgres(ptr).countMatches(query);
}
fn postgresList(ptr: *anyopaque, allocator: std.mem.Allocator, query: []const u8) ![]Record {
    return try asPostgres(ptr).list(allocator, query);
}
fn postgresHealth(ptr: *anyopaque) bool {
    return asPostgres(ptr).health();
}

const memory_vtable = RuntimeStore.VTable{
    .deinit = memoryDeinit,
    .upsert = memoryUpsert,
    .delete = memoryDelete,
    .count_matches = memoryCount,
    .list = memoryList,
    .health = memoryHealth,
};

const postgres_vtable = RuntimeStore.VTable{
    .deinit = postgresDeinit,
    .upsert = postgresUpsert,
    .delete = postgresDelete,
    .count_matches = postgresCount,
    .list = postgresList,
    .health = postgresHealth,
};
