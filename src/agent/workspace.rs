use std::path::{Component, Path, PathBuf};

pub fn workspace_path(root: &Path, requested: String) -> Result<PathBuf, String> {
    let path = Path::new(&requested);
    if path.is_absolute() {
        return Err("absolute paths are outside the tool workspace".into());
    }
    let mut safe = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => safe.push(part),
            Component::CurDir => {}
            _ => return Err("path escapes the tool workspace".into()),
        }
    }
    Ok(root.join(safe))
}
