use std::{collections::BTreeMap, fs, path::Path, process::Command};

pub fn visit_tracked_dirs(errors: &mut Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .args(["-c", "safe.directory=*", "ls-files"])
        .output()?;
    if !output.status.success() {
        errors.push("git ls-files failed".into());
        return Ok(());
    }
    let mut tree: BTreeMap<String, Vec<(String, bool)>> = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let path = line.trim();
        if !path.is_empty() && !ignored_path(path) {
            collect(path, &mut tree);
        }
    }
    for (dir, children) in tree {
        if dir != "." && !ignored_path(&dir) {
            validate_dir(&dir, &children, errors);
        }
    }
    Ok(())
}

fn collect(path: &str, tree: &mut BTreeMap<String, Vec<(String, bool)>>) {
    let parts: Vec<&str> = path.split('/').collect();
    for depth in 0..parts.len() {
        let dir = if depth == 0 {
            ".".to_string()
        } else {
            parts[..depth].join("/")
        };
        let child = parts[depth].to_string();
        if hidden(&child) {
            continue;
        }
        let entry = (child, depth + 1 < parts.len());
        let children = tree.entry(dir).or_default();
        if !children.contains(&entry) {
            children.push(entry);
        }
    }
}

fn validate_dir(dir: &str, children: &[(String, bool)], errors: &mut Vec<String>) {
    if !children.iter().any(|(name, _)| name == "README.md") {
        errors.push(format!("{dir} expected README.md"));
        return;
    }
    let readme = Path::new(dir).join("README.md");
    let Ok(content) = fs::read_to_string(&readme) else {
        errors.push(format!("{} unreadable", readme.display()));
        return;
    };
    let links = links(&content);
    for (name, is_dir) in children {
        if name != "README.md"
            && !hidden(name)
            && !links.iter().any(|link| matched(link, name, *is_dir))
        {
            errors.push(format!("{dir} missing TOC link to {name}"));
        }
    }
}

fn links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(close) = rest.find("](") {
        let after = &rest[close + 2..];
        let Some(end) = after.find(')') else { break };
        links.push(after[..end].split('#').next().unwrap_or("").to_string());
        rest = &after[end + 1..];
    }
    links
}

fn matched(link: &str, target: &str, is_dir: bool) -> bool {
    link == target
        || (is_dir && (link == format!("{target}/") || link == format!("{target}/README.md")))
}

fn ignored_path(path: &str) -> bool {
    path.split('/').any(|part| {
        matches!(
            part,
            "target" | "runs" | "__pycache__" | ".pytest_cache" | "node_modules"
        )
    }) || path.contains("corpus/generated/kimi-sft-60m-v2/train/")
        || path.contains("corpus/generated/kimi-sft-60m-v2/val/")
        || path.contains("corpus/generated/kimi-sft-60m-v2/holdout/")
        || path.contains("corpus/generated/pref-v1/pairs/")
}

fn hidden(name: &str) -> bool {
    matches!(name, ".gitignore" | ".gitkeep" | "CACHEDIR.TAG")
}
