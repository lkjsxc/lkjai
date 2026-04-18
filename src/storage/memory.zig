const std = @import("std");
const record_mod = @import("record.zig");

pub const Record = record_mod.Record;

pub const MemoryStore = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    records: std.ArrayList(Record),

    pub fn init(allocator: std.mem.Allocator) MemoryStore {
        return .{
            .allocator = allocator,
            .records = std.ArrayList(Record).init(allocator),
        };
    }

    pub fn deinit(self: *MemoryStore) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.records.items) |record| {
            self.allocator.free(record.id);
            self.allocator.free(record.title);
            self.allocator.free(record.body);
        }
        self.records.deinit();
    }

    pub fn upsert(self: *MemoryStore, id: []const u8, title: []const u8, body: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.records.items) |*record| {
            if (std.mem.eql(u8, record.id, id)) {
                self.allocator.free(record.title);
                self.allocator.free(record.body);
                record.title = try self.allocator.dupe(u8, title);
                record.body = try self.allocator.dupe(u8, body);
                return;
            }
        }
        try self.records.append(.{
            .id = try self.allocator.dupe(u8, id),
            .title = try self.allocator.dupe(u8, title),
            .body = try self.allocator.dupe(u8, body),
        });
    }

    pub fn delete(self: *MemoryStore, id: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        var i: usize = 0;
        while (i < self.records.items.len) : (i += 1) {
            if (std.mem.eql(u8, self.records.items[i].id, id)) {
                const old = self.records.swapRemove(i);
                self.allocator.free(old.id);
                self.allocator.free(old.title);
                self.allocator.free(old.body);
                return true;
            }
        }
        return false;
    }

    pub fn countMatches(self: *MemoryStore, query: []const u8) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        var count: usize = 0;
        for (self.records.items) |record| {
            if (std.mem.indexOf(u8, record.title, query) != null or std.mem.indexOf(u8, record.body, query) != null) {
                count += 1;
            }
        }
        return count;
    }

    pub fn list(self: *MemoryStore, allocator: std.mem.Allocator, query: []const u8) ![]Record {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(Record).init(allocator);
        for (self.records.items) |record| {
            if (query.len == 0 or std.mem.indexOf(u8, record.title, query) != null or std.mem.indexOf(u8, record.body, query) != null) {
                try out.append(.{
                    .id = try allocator.dupe(u8, record.id),
                    .title = try allocator.dupe(u8, record.title),
                    .body = try allocator.dupe(u8, record.body),
                });
            }
        }
        return out.toOwnedSlice();
    }
};
