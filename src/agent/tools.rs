use crate::config::Config;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{fs, process::Command, time};

#[derive(Clone, Debug)]
pub enum ToolCall {
    Shell(String),
    Fetch(String),
    Read(String),
    Write { path: String, content: String },
    List(String),
}

impl ToolCall {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Shell(_) => "shell.exec",
            Self::Fetch(_) => "web.fetch",
            Self::Read(_) => "file.read",
            Self::Write { .. } => "file.write",
            Self::List(_) => "file.list",
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Shell(cmd) => cmd.clone(),
            Self::Fetch(url) => url.clone(),
            Self::Read(path) | Self::List(path) => path.clone(),
            Self::Write { path, content } => format!("{path} ({} bytes)", content.len()),
        }
    }
}

pub fn parse_tool(input: &str) -> Option<ToolCall> {
    let trimmed = input.trim();
    let (head, rest) = trimmed.split_once(' ').unwrap_or((trimmed, ""));
    match head {
        "/sh" | "/shell" => Some(ToolCall::Shell(rest.into())),
        "/fetch" => Some(ToolCall::Fetch(rest.into())),
        "/read" => Some(ToolCall::Read(rest.into())),
        "/ls" | "/list" => Some(ToolCall::List(rest.into())),
        "/write" => parse_write(rest),
        _ => parse_natural(trimmed),
    }
}

pub async fn execute(call: ToolCall, config: &Config) -> Result<String, String> {
    let timeout = Duration::from_secs(config.tool_timeout_secs);
    let output = match call {
        ToolCall::Shell(command) => time::timeout(timeout, run_shell(command)).await,
        ToolCall::Fetch(url) => time::timeout(timeout, fetch(url)).await,
        ToolCall::Read(path) => time::timeout(timeout, read_file(host_path(path))).await,
        ToolCall::Write { path, content } => {
            time::timeout(timeout, write_file(host_path(path), content)).await
        }
        ToolCall::List(path) => time::timeout(timeout, list_dir(host_path(path))).await,
    };
    match output {
        Ok(Ok(value)) => Ok(truncate(value, config.tool_output_limit)),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("tool timed out".into()),
    }
}

fn parse_write(rest: &str) -> Option<ToolCall> {
    let (path, content) = rest.split_once('\n')?;
    Some(ToolCall::Write {
        path: path.trim().into(),
        content: content.into(),
    })
}

fn parse_natural(input: &str) -> Option<ToolCall> {
    let lower = input.to_ascii_lowercase();
    if let Some(command) = prefixed(input, &lower, &["run:", "run command:", "execute:"]) {
        return Some(ToolCall::Shell(command.into()));
    }
    if let Some(url) = first_url(input) {
        if lower.starts_with("fetch ") || lower.starts_with("open ") || lower.contains(" website") {
            return Some(ToolCall::Fetch(url.into()));
        }
    }
    if let Some(path) = after_prefix(input, &lower, &["read file ", "read "]) {
        return Some(ToolCall::Read(path.into()));
    }
    if let Some(path) = after_prefix(input, &lower, &["list directory ", "list files ", "list "]) {
        return Some(ToolCall::List(path.into()));
    }
    if let Some(rest) = after_prefix(input, &lower, &["write file ", "write "]) {
        return parse_write(rest);
    }
    None
}

fn prefixed<'a>(input: &'a str, lower: &str, prefixes: &[&str]) -> Option<&'a str> {
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            let value = input[prefix.len()..].trim();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn after_prefix<'a>(input: &'a str, lower: &str, prefixes: &[&str]) -> Option<&'a str> {
    prefixed(input, lower, prefixes).filter(|value| !value.is_empty())
}

fn first_url(input: &str) -> Option<&str> {
    input
        .split_whitespace()
        .find(|word| word.starts_with("http://") || word.starts_with("https://"))
}

async fn run_shell(command: String) -> Result<String, String> {
    let mut cmd = Command::new("sh");
    cmd.arg("-lc").arg(command);
    if Path::new("/host").exists() {
        cmd.current_dir("/host");
    }
    let output = cmd.output().await.map_err(|error| error.to_string())?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(format!("exit={} \n{}", output.status, text))
}

async fn fetch(url: String) -> Result<String, String> {
    let response = reqwest::get(url).await.map_err(|error| error.to_string())?;
    response.text().await.map_err(|error| error.to_string())
}

async fn read_file(path: PathBuf) -> Result<String, String> {
    fs::read_to_string(path)
        .await
        .map_err(|error| error.to_string())
}

async fn write_file(path: PathBuf, content: String) -> Result<String, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|error| error.to_string())?;
    }
    fs::write(&path, content)
        .await
        .map_err(|error| error.to_string())?;
    Ok(format!("wrote {}", path.display()))
}

async fn list_dir(path: PathBuf) -> Result<String, String> {
    let mut entries = fs::read_dir(path)
        .await
        .map_err(|error| error.to_string())?;
    let mut names = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| error.to_string())?
    {
        names.push(entry.path().display().to_string());
    }
    names.sort();
    Ok(names.join("\n"))
}

fn host_path(path: String) -> PathBuf {
    if path.starts_with("/host/") || path == "/host" || !path.starts_with('/') {
        return PathBuf::from(path);
    }
    PathBuf::from("/host").join(path.trim_start_matches('/'))
}

fn truncate(mut value: String, limit: usize) -> String {
    if value.len() > limit {
        value.truncate(limit);
        value.push_str("\n[truncated]");
    }
    value
}
