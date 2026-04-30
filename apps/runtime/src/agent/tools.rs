use super::{memory::MemoryStore, tool_local, tool_remote};
use crate::config::Config;

#[derive(Clone, Debug)]
pub enum ToolCall {
    AgentFinish { content: String },
    AgentThink { content: String },
    Shell { command: String },
    Fetch { url: String },
    Read { path: String },
    Write {
        path: String,
        content: String,
    },
    List { path: String },
    MemorySearch { query: String },
    MemoryWrite { content: String },
    ResourceSearch {
        query: String,
        kind: String,
        sort: Option<String>,
        cursor: Option<String>,
        limit: Option<String>,
        direction: Option<String>,
        scope: Option<String>,
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
        is_favorite: bool,
        is_private: bool,
    },
    ResourceCreateMedia {
        path: String,
        alias: Option<String>,
        is_favorite: bool,
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
    pub fn from_fields(action: &super::action::Action) -> Result<Self, String> {
        let tool = action.tool.as_str();
        match tool {
            "agent.finish" => Ok(Self::AgentFinish {
                content: required(action, "content")?,
            }),
            "agent.think" => Ok(Self::AgentThink {
                content: required(action, "content")?,
            }),
            "shell.exec" => Ok(Self::Shell {
                command: required(action, "command")?,
            }),
            "web.fetch" => Ok(Self::Fetch {
                url: required(action, "url")?,
            }),
            "fs.read" => Ok(Self::Read {
                path: required(action, "path")?,
            }),
            "fs.write" => Ok(Self::Write {
                path: required(action, "path")?,
                content: required(action, "content")?,
            }),
            "fs.list" => Ok(Self::List {
                path: required(action, "path")?,
            }),
            "memory.search" => Ok(Self::MemorySearch {
                query: required(action, "query")?,
            }),
            "memory.write" => Ok(Self::MemoryWrite {
                content: required(action, "content")?,
            }),
            "resource.search" => Ok(Self::ResourceSearch {
                query: required_any(action, &["query", "q"])?,
                kind: optional(action, "kind").unwrap_or_else(|| "all".into()),
                sort: optional(action, "sort"),
                cursor: optional(action, "cursor"),
                limit: optional(action, "limit"),
                direction: optional(action, "direction"),
                scope: optional(action, "scope"),
            }),
            "resource.fetch" => Ok(Self::ResourceFetch {
                reference: required_any(action, &["ref", "id"])?,
            }),
            "resource.history" => Ok(Self::ResourceHistory {
                reference: required_any(action, &["ref", "id"])?,
            }),
            "resource.preview_markdown" => Ok(Self::ResourcePreview {
                body: required(action, "body")?,
                current_resource_id: optional(action, "current_resource_id"),
            }),
            "resource.create_note" => Ok(Self::ResourceCreateNote {
                body: required(action, "body")?,
                alias: optional(action, "alias"),
                is_favorite: action.bool_field("is_favorite", false),
                is_private: action.bool_field("is_private", false),
            }),
            "resource.create_media" => Ok(Self::ResourceCreateMedia {
                path: required(action, "path")?,
                alias: optional(action, "alias"),
                is_favorite: action.bool_field("is_favorite", false),
                is_private: action.bool_field("is_private", false),
            }),
            "resource.update_resource" => Ok(Self::ResourceUpdate {
                reference: required_any(action, &["ref", "id"])?,
                body: required(action, "body")?,
                alias: optional(action, "alias"),
                is_favorite: action.bool_field("is_favorite", false),
                is_private: action.bool_field("is_private", false),
            }),
            other => Err(format!("unknown tool {other}")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AgentFinish { .. } => "agent.finish",
            Self::AgentThink { .. } => "agent.think",
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
            Self::ResourceCreateMedia { .. } => "resource.create_media",
            Self::ResourceUpdate { .. } => "resource.update_resource",
        }
    }

    pub fn summary(&self) -> String {
        super::tool_summary::summary(self)
    }
}
pub async fn execute(
    call: ToolCall,
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
) -> Result<String, String> {
    if let ToolCall::AgentFinish { content } = call {
        return Ok(content);
    }
    if let ToolCall::AgentThink { content } = call {
        return Ok(content);
    }
    if let Some(value) = tool_local::execute(&call, config, memory, run_id).await? {
        return Ok(value);
    }
    if let Some(value) = tool_remote::execute(&call, config).await? {
        return Ok(value);
    }
    Err(format!("tool not implemented: {}", call.name()))
}

fn required(action: &super::action::Action, key: &str) -> Result<String, String> {
    optional(action, key).ok_or_else(|| format!("missing string arg {key}"))
}

fn required_any(action: &super::action::Action, keys: &[&str]) -> Result<String, String> {
    keys.iter()
        .find_map(|key| optional(action, key))
        .ok_or_else(|| format!("missing one of {}", keys.join(", ")))
}

fn optional(action: &super::action::Action, key: &str) -> Option<String> {
    action.field(key)
}
