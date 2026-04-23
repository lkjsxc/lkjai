use crate::config::Config;
use serde_json::Value;
use std::{path::PathBuf, time::Duration};
use tokio::{fs, process::Command, time};

use super::{memory::MemoryStore, workspace::workspace_path};

#[derive(Clone, Debug)]
pub enum ToolCall {
    Shell { command: String },
    Fetch { url: String },
    Read { path: String },
    Write { path: String, content: String },
    List { path: String },
    MemorySearch { query: String },
    MemoryWrite { content: String },
}

impl ToolCall {
    pub fn from_json(tool: &str, args: &Value) -> Result<Self, String> {
        match tool {
            "shell.exec" => Ok(Self::Shell {
                command: required(args, "command")?,
            }),
            "web.fetch" => Ok(Self::Fetch {
                url: required(args, "url")?,
            }),
            "fs.read" => Ok(Self::Read {
                path: required(args, "path")?,
            }),
            "fs.write" => Ok(Self::Write {
                path: required(args, "path")?,
                content: required(args, "content")?,
            }),
            "fs.list" => Ok(Self::List {
                path: required(args, "path")?,
            }),
            "memory.search" => Ok(Self::MemorySearch {
                query: required(args, "query")?,
            }),
            "memory.write" => Ok(Self::MemoryWrite {
                content: required(args, "content")?,
            }),
            other => Err(format!("unknown tool {other}")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Shell { .. } => "shell.exec",
            Self::Fetch { .. } => "web.fetch",
            Self::Read { .. } => "fs.read",
            Self::Write { .. } => "fs.write",
            Self::List { .. } => "fs.list",
            Self::MemorySearch { .. } => "memory.search",
            Self::MemoryWrite { .. } => "memory.write",
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Shell { command } => command.clone(),
            Self::Fetch { url } => url.clone(),
            Self::Read { path } | Self::List { path } => path.clone(),
            Self::Write { path, content } => format!("{path} ({} bytes)", content.len()),
            Self::MemorySearch { query } => query.clone(),
            Self::MemoryWrite { content } => content.clone(),
        }
    }
}

pub async fn execute(
    call: ToolCall,
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
) -> Result<String, String> {
    let timeout = Duration::from_secs(config.tool_timeout_secs);
    let output = match call {
        ToolCall::Shell { command } => {
            time::timeout(
                timeout,
                run_shell(command, config.tool_workspace_dir.clone()),
            )
            .await
        }
        ToolCall::Fetch { url } => time::timeout(timeout, fetch(url)).await,
        ToolCall::Read { path } => {
            time::timeout(
                timeout,
                read_file(workspace_path(&config.tool_workspace_dir, path)?),
            )
            .await
        }
        ToolCall::Write { path, content } => {
            let path = workspace_path(&config.tool_workspace_dir, path)?;
            time::timeout(timeout, write_file(path, content)).await
        }
        ToolCall::List { path } => {
            time::timeout(
                timeout,
                list_dir(workspace_path(&config.tool_workspace_dir, path)?),
            )
            .await
        }
        ToolCall::MemorySearch { query } => return memory.search(&query, 5).map(|v| v.join("\n")),
        ToolCall::MemoryWrite { content } => {
            return memory.write("run", Some(run_id), &content).map(|_| content);
        }
    };
    match output {
        Ok(Ok(value)) => Ok(truncate(value, config.tool_output_limit)),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("tool timed out".into()),
    }
}

fn required(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("missing string arg {key}"))
}

async fn run_shell(command: String, workspace: PathBuf) -> Result<String, String> {
    fs::create_dir_all(&workspace)
        .await
        .map_err(|error| error.to_string())?;
    let mut cmd = Command::new("sh");
    cmd.arg("-lc").arg(command);
    cmd.current_dir(workspace);
    let output = cmd.output().await.map_err(|error| error.to_string())?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(format!("exit={} \n{}", output.status, text))
}

async fn fetch(url: String) -> Result<String, String> {
    reqwest::get(url)
        .await
        .map_err(|error| error.to_string())?
        .text()
        .await
        .map_err(|error| error.to_string())
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

fn truncate(mut value: String, limit: usize) -> String {
    if value.len() > limit {
        value.truncate(limit);
        value.push_str("\n[truncated]");
    }
    value
}
