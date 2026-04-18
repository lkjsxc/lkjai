const std = @import("std");
const librarian_mod = @import("librarian.zig");
const tokenizer = @import("../model/tokenizer.zig");

pub const ChatResult = struct {
    reply: []u8,
    parallel_steps: usize,
    queue_depth: usize,
};

pub const Orchestrator = struct {
    max_in_flight: usize,
    active: std.atomic.Value(usize),

    pub fn init(max_in_flight: usize) Orchestrator {
        return .{
            .max_in_flight = max_in_flight,
            .active = std.atomic.Value(usize).init(0),
        };
    }

    pub fn tryBegin(self: *Orchestrator) bool {
        while (true) {
            const current = self.active.load(.seq_cst);
            if (current >= self.max_in_flight) return false;
            if (self.active.cmpxchgWeak(current, current + 1, .seq_cst, .seq_cst) == null) return true;
        }
    }

    pub fn end(self: *Orchestrator) void {
        _ = self.active.fetchSub(1, .seq_cst);
    }

    pub fn queueDepth(self: *Orchestrator) usize {
        const current = self.active.load(.seq_cst);
        return if (current > self.max_in_flight) current - self.max_in_flight else 0;
    }

    const MatchCtx = struct {
        librarian: *librarian_mod.Librarian,
        message: []const u8,
        out: usize = 0,
        err: ?anyerror = null,
    };

    const TokenCtx = struct {
        message: []const u8,
        out: usize = 0,
    };

    fn runMatch(ctx: *MatchCtx) void {
        ctx.out = ctx.librarian.countMatches(ctx.message) catch |err| {
            ctx.err = err;
            return;
        };
    }

    fn runTokens(ctx: *TokenCtx) void {
        ctx.out = tokenizer.tokenCount(ctx.message);
    }

    pub fn runChat(
        self: *Orchestrator,
        allocator: std.mem.Allocator,
        librarian: *librarian_mod.Librarian,
        message: []const u8,
    ) !ChatResult {
        _ = self;
        var match_ctx = MatchCtx{ .librarian = librarian, .message = message };
        var token_ctx = TokenCtx{ .message = message };

        var t1 = try std.Thread.spawn(.{}, runMatch, .{&match_ctx});
        var t2 = try std.Thread.spawn(.{}, runTokens, .{&token_ctx});
        t1.join();
        t2.join();
        if (match_ctx.err) |err| return err;

        const reply = try librarian.buildChatReply(allocator, message, match_ctx.out, token_ctx.out);
        return .{
            .reply = reply,
            .parallel_steps = 2,
            .queue_depth = 0,
        };
    }
};
