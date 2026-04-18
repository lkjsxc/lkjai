const std = @import("std");

pub fn tokenCount(input: []const u8) usize {
    var count: usize = 0;
    var it = std.mem.tokenizeAny(u8, input, " \n\r\t");
    while (it.next()) |_| count += 1;
    return count;
}

pub fn normalize(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    var it = std.mem.tokenizeAny(u8, input, " \n\r\t");
    var first = true;
    while (it.next()) |tok| {
        if (!first) try list.append(' ');
        first = false;
        for (tok) |c| try list.append(std.ascii.toLower(c));
    }
    return list.toOwnedSlice();
}
