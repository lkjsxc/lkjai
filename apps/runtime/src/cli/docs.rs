use serde::Serialize;
use std::{fs, path::Path};

#[derive(Serialize)]
struct GateResult<'a> {
    command: &'a str,
    status: &'a str,
    errors: Vec<String>,
}

pub fn validate_topology() -> Result<(), Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    visit_docs_dir(Path::new("docs"), &mut errors)?;
    finish("validate-topology", errors)
}

pub fn validate_links() -> Result<(), Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    collect_link_errors(Path::new("docs"), &mut errors)?;
    finish("validate-links", errors)
}

fn visit_docs_dir(dir: &Path, errors: &mut Vec<String>) -> std::io::Result<()> {
    let entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    let visible: Vec<_> = entries
        .iter()
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .collect();
    let readme = dir.join("README.md");
    let readmes = visible
        .iter()
        .filter(|entry| entry.file_name() == "README.md")
        .count();
    if readmes != 1 {
        errors.push(format!("{} expected exactly one README.md", dir.display()));
    } else {
        require_child_links(dir, &readme, &visible, errors)?;
    }
    let children = visible
        .iter()
        .filter(|entry| entry.file_name() != "README.md")
        .count();
    if children < 2 {
        errors.push(format!("{} expected at least two children", dir.display()));
    }
    for entry in visible {
        if entry.path().is_dir() {
            visit_docs_dir(&entry.path(), errors)?;
        }
    }
    Ok(())
}

fn require_child_links(
    dir: &Path,
    readme: &Path,
    entries: &[&fs::DirEntry],
    errors: &mut Vec<String>,
) -> std::io::Result<()> {
    let content = fs::read_to_string(readme)?;
    for entry in entries {
        let name = entry.file_name();
        if name == "README.md" {
            continue;
        }
        let name = name.to_string_lossy();
        let target = if entry.path().is_dir() {
            format!("{name}/README.md")
        } else {
            name.to_string()
        };
        if !markdown_links(&content).iter().any(|link| link == &target) {
            errors.push(format!("{} missing TOC link to {}", dir.display(), target));
        }
    }
    Ok(())
}

fn collect_link_errors(dir: &Path, errors: &mut Vec<String>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_link_errors(&path, errors)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        let content = fs::read_to_string(&path)?;
        for link in markdown_links(&content) {
            let clean = link.split('#').next().unwrap_or(&link);
            if skip_link(clean) || path.parent().unwrap_or(Path::new(".")).join(clean).exists() {
                continue;
            }
            errors.push(format!("{} missing {}", path.display(), link));
        }
    }
    Ok(())
}

fn markdown_links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(close) = rest.find("](") {
        let after = &rest[close + 2..];
        if let Some(end) = after.find(')') {
            links.push(after[..end].to_string());
            rest = &after[end + 1..];
        } else {
            break;
        }
    }
    links
}

fn skip_link(link: &str) -> bool {
    link.is_empty()
        || link.starts_with("http://")
        || link.starts_with("https://")
        || link.starts_with('#')
        || link.starts_with("mailto:")
}

fn finish(command: &str, errors: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let status = if errors.is_empty() { "pass" } else { "fail" };
    println!(
        "{}",
        serde_json::to_string(&GateResult {
            command,
            status,
            errors: errors.clone()
        })?
    );
    if errors.is_empty() {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
