pub const PostgresAdapter = struct {
    database_url: []const u8,

    pub fn init(database_url: []const u8) PostgresAdapter {
        return .{ .database_url = database_url };
    }

    pub fn health(self: *const PostgresAdapter) bool {
        _ = self;
        return false;
    }
};
