use std::{path::PathBuf, time::Duration};

use tokio::{fs, process::Command, time};

use crate::config::Config;

use super::{memory::MemoryStore, tools::ToolCall, workspace::workspace_path};

pub async fn execute(
    call: &ToolCall,
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
) -> Result<Option<String>, String> {
    let timeout = Duration::from_secs(config.tool_timeout_secs);
    let output = match call {
        ToolCall::Shell { command } => {
            time::timeout(
                timeout,
                run_shell(command.clone(), config.tool_workspace_dir.clone()),
            )
            .await
        }
        ToolCall::Fetch { url } => time::timeout(timeout, fetch(url.clone())).await,
        ToolCall::Read { path } => {
            time::timeout(
                timeout,
                read_file(workspace_path(&config.tool_workspace_dir, path.clone())?),
            )
            .await
        }
        ToolCall::Write { path, content } => {
            time::timeout(
                timeout,
                write_file(
                    workspace_path(&config.tool_workspace_dir, path.clone())?,
                    content.clone(),
                ),
            )
            .await
        }
        ToolCall::List { path } => {
            time::timeout(
                timeout,
                list_dir(workspace_path(&config.tool_workspace_dir, path.clone())?),
            )
            .await
        }
        ToolCall::MemorySearch { query } => {
            return memory.search(query, 5).map(|value| Some(value.join("\n")))
        }
        ToolCall::MemoryWrite { content } => {
            return memory
                .write("run", Some(run_id), content)
                .map(|_| Some(content.clone()))
        }
        _ => return Ok(None),
    };
    match output {
        Ok(Ok(value)) => Ok(Some(truncate(value, config.tool_output_limit))),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("tool timed out".into()),
    }
}

async fn run_shell(command: String, workspace: PathBuf) -> Result<String, String> {
    fs::create_dir_all(&workspace)
        .await
        .map_err(|error| error.to_string())?;
    let output = Command::new("sh")
        .arg("-lc")
        .arg(command)
        .current_dir(workspace)
        .output()
        .await
        .map_err(|error| error.to_string())?;
    Ok(format!(
        "exit={} \n{}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
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
