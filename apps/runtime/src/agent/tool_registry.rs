pub const DISABLED_DEFAULT: &[&str] = &["shell.exec", "web.fetch", "fs.write", "memory.write"];

pub fn enabled_by_default(tool: &str) -> bool {
    !DISABLED_DEFAULT.contains(&tool)
}

pub fn require_enabled(tool: &str) -> Result<(), String> {
    if enabled_by_default(tool) {
        Ok(())
    } else {
        Err(format!("tool disabled in default profile: {tool}"))
    }
}
