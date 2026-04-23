use serde_json::Value;

use crate::config::Config;

use super::{memory::MemoryStore, tool_local, tool_remote};

#[derive(Clone, Debug)]
pub enum ToolCall {
    Shell {
        command: String,
    },
    Fetch {
        url: String,
    },
    Read {
        path: String,
    },
    Write {
        path: String,
        content: String,
    },
    List {
        path: String,
    },
    MemorySearch {
        query: String,
    },
    MemoryWrite {
        content: String,
    },
    ResourceSearch {
        query: String,
        kind: String,
    },
    ResourceFetch {
        reference: String,
    },
    ResourceHistory {
        reference: String,
    },
    ResourcePreview {
        body: String,
        current_resource_id: Option<String>,
    },
    ResourceCreateNote {
        body: String,
        alias: Option<String>,
        is_private: bool,
    },
    ResourceUpdate {
        reference: String,
        body: String,
        alias: Option<String>,
        is_favorite: bool,
        is_private: bool,
    },
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
            "resource.search" => Ok(Self::ResourceSearch {
                query: required(args, "query")?,
                kind: optional(args, "kind").unwrap_or_else(|| "all".into()),
            }),
            "resource.fetch" => Ok(Self::ResourceFetch {
                reference: required_any(args, &["ref", "id"])?,
            }),
            "resource.history" => Ok(Self::ResourceHistory {
                reference: required_any(args, &["ref", "id"])?,
            }),
            "resource.preview_markdown" => Ok(Self::ResourcePreview {
                body: required(args, "body")?,
                current_resource_id: optional(args, "current_resource_id"),
            }),
            "resource.create_note" => Ok(Self::ResourceCreateNote {
                body: required(args, "body")?,
                alias: optional(args, "alias"),
                is_private: optional_bool(args, "is_private", false),
            }),
            "resource.update_resource" => Ok(Self::ResourceUpdate {
                reference: required_any(args, &["ref", "id"])?,
                body: required(args, "body")?,
                alias: optional(args, "alias"),
                is_favorite: optional_bool(args, "is_favorite", false),
                is_private: optional_bool(args, "is_private", false),
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
            Self::ResourceSearch { .. } => "resource.search",
            Self::ResourceFetch { .. } => "resource.fetch",
            Self::ResourceHistory { .. } => "resource.history",
            Self::ResourcePreview { .. } => "resource.preview_markdown",
            Self::ResourceCreateNote { .. } => "resource.create_note",
            Self::ResourceUpdate { .. } => "resource.update_resource",
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
            Self::ResourceSearch { query, kind } => format!("{query} [{kind}]"),
            Self::ResourceFetch { reference } | Self::ResourceHistory { reference } => {
                reference.clone()
            }
            Self::ResourcePreview { body, .. } | Self::ResourceCreateNote { body, .. } => {
                body.chars().take(80).collect()
            }
            Self::ResourceUpdate {
                reference, body, ..
            } => format!("{reference}: {}", body.chars().take(60).collect::<String>()),
        }
    }
}

pub async fn execute(
    call: ToolCall,
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
) -> Result<String, String> {
    if let Some(value) = tool_local::execute(&call, config, memory, run_id).await? {
        return Ok(value);
    }
    if let Some(value) = tool_remote::execute(&call, config).await? {
        return Ok(value);
    }
    Err(format!("tool not implemented: {}", call.name()))
}

fn required(args: &Value, key: &str) -> Result<String, String> {
    optional(args, key).ok_or_else(|| format!("missing string arg {key}"))
}

fn required_any(args: &Value, keys: &[&str]) -> Result<String, String> {
    keys.iter()
        .find_map(|key| optional(args, key))
        .ok_or_else(|| format!("missing one of {}", keys.join(", ")))
}

fn optional(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn optional_bool(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}
