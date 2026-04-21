use serde::Serialize;
use std::{fs, path::Path};
use walkdir::WalkDir;

const DOC_LIMIT: usize = 300;
const SRC_LIMIT: usize = 200;

#[derive(Serialize)]
struct QualityResult {
    command: &'static str,
    status: &'static str,
    violations: Vec<String>,
}

pub fn check_lines() -> Result<(), Box<dyn std::error::Error>> {
    let mut violations = Vec::new();
    check_path(Path::new("docs"), DOC_LIMIT, &mut violations)?;
    check_path(Path::new("src"), SRC_LIMIT, &mut violations)?;
    check_path(Path::new("training"), SRC_LIMIT, &mut violations)?;
    check_path(Path::new("configs"), SRC_LIMIT, &mut violations)?;
    finish("check-lines", violations)
}

pub fn no_node() -> Result<(), Box<dyn std::error::Error>> {
    let mut violations = Vec::new();
    for entry in WalkDir::new(".").into_iter().filter_map(Result::ok) {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy();
        if path.starts_with("./tmp")
            || path.starts_with("./target")
            || path.starts_with("./.git")
            || path.starts_with("./docs")
        {
            continue;
        }
        if name == "package.json" || name == "package-lock.json" {
            violations.push(path.display().to_string());
        }
        if path.is_file() && node_mentioned(path)? {
            violations.push(format!("{} mentions Node", path.display()));
        }
    }
    finish("no-node", violations)
}

fn check_path(
    root: &Path,
    limit: usize,
    violations: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !root.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() || !checked_extension(path) {
            continue;
        }
        let lines = fs::read_to_string(path)?.lines().count();
        if lines > limit {
            violations.push(format!("{}: {} lines > {}", path.display(), lines, limit));
        }
    }
    Ok(())
}

fn checked_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("md" | "rs" | "py" | "sh" | "toml" | "css" | "js")
    )
}

fn node_mentioned(path: &Path) -> Result<bool, std::io::Error> {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return Ok(false);
    };
    if !matches!(ext, "yml" | "yaml" | "sh") {
        return Ok(false);
    }
    let text = fs::read_to_string(path)?;
    Ok(text.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("npm ") || lower.contains("node ")
    }))
}

fn finish(
    command: &'static str,
    violations: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = if violations.is_empty() {
        "pass"
    } else {
        "fail"
    };
    println!(
        "{}",
        serde_json::to_string(&QualityResult {
            command,
            status,
            violations: violations.clone()
        })?
    );
    if violations.is_empty() {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
