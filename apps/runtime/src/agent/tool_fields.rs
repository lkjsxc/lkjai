use super::action::Action;

pub fn required(action: &Action, key: &str) -> Result<String, String> {
    optional(action, key).ok_or_else(|| format!("missing string arg {key}"))
}

pub fn required_any(action: &Action, keys: &[&str]) -> Result<String, String> {
    keys.iter()
        .find_map(|key| optional(action, key))
        .ok_or_else(|| format!("missing one of {}", keys.join(", ")))
}

pub fn optional(action: &Action, key: &str) -> Option<String> {
    action.field(key)
}
