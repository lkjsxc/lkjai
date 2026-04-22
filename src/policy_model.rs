use std::{path::PathBuf, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::model_client::{ModelMessage, ModelStatus};

#[derive(Clone)]
pub struct PolicyModel {
    path: PathBuf,
    policy: Arc<PolicyFile>,
}

#[derive(Clone, Deserialize)]
struct PolicyFile {
    rules: Vec<PolicyRule>,
    fallback: ActionTemplate,
}

#[derive(Clone, Deserialize)]
struct PolicyRule {
    any: Vec<String>,
    action: ActionTemplate,
}

#[derive(Clone, Deserialize, Serialize)]
struct ActionTemplate {
    kind: String,
    thought: String,
    tool: Option<String>,
    args: Option<serde_json::Value>,
    content: Option<String>,
}

impl PolicyModel {
    pub fn load(path: PathBuf) -> Self {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| default_policy());
        let policy = serde_json::from_str(&text).unwrap_or_else(|_| {
            serde_json::from_str(&default_policy()).expect("default policy must parse")
        });
        Self {
            path,
            policy: Arc::new(policy),
        }
    }

    pub fn status(&self) -> ModelStatus {
        ModelStatus {
            model: "trained-policy".into(),
            api_url: format!("policy://{}", self.path.display()),
            loaded: true,
            message: format!("{} rules loaded", self.policy.rules.len()),
        }
    }

    pub fn chat(&self, messages: &[ModelMessage]) -> Result<String, String> {
        let prompt = messages
            .last()
            .map(|message| message.content.as_str())
            .unwrap_or("");
        if let Some(observation) = last_event(prompt, "observation:") {
            let clean = observation.replace("\\n", "\n");
            return serialize_action(final_action(format!("Observed:\n{clean}")));
        }
        let text = last_event(prompt, "user:").unwrap_or_else(|| prompt.to_string());
        let lower = text.to_ascii_lowercase();
        let action = self
            .policy
            .rules
            .iter()
            .find(|rule| rule.any.iter().any(|term| lower.contains(term)))
            .map(|rule| fill(rule.action.clone(), &text))
            .unwrap_or_else(|| fill(self.policy.fallback.clone(), &text));
        serialize_action(action)
    }
}

fn fill(mut action: ActionTemplate, input: &str) -> ActionTemplate {
    if let Some(args) = action.args.as_mut() {
        replace_value(args, "{{memory}}", &memory_content(input));
        replace_value(args, "{{path}}", &path_hint(input));
        replace_value(args, "{{url}}", &url_hint(input));
    }
    if let Some(content) = action.content.as_mut() {
        *content = content.replace("{{input}}", input);
    }
    action
}

fn replace_value(value: &mut serde_json::Value, needle: &str, replacement: &str) {
    match value {
        serde_json::Value::String(text) => *text = text.replace(needle, replacement),
        serde_json::Value::Object(map) => {
            for item in map.values_mut() {
                replace_value(item, needle, replacement);
            }
        }
        _ => {}
    }
}

fn path_hint(input: &str) -> String {
    input
        .split_whitespace()
        .find(|part| part.starts_with('/') || *part == ".")
        .unwrap_or(".")
        .trim_matches('"')
        .to_string()
}

fn url_hint(input: &str) -> String {
    input
        .split_whitespace()
        .find(|part| part.starts_with("http://") || part.starts_with("https://"))
        .unwrap_or("")
        .trim_matches('"')
        .to_string()
}

fn memory_content(input: &str) -> String {
    input
        .replace("remember that", "")
        .replace("Remember that", "")
        .trim()
        .trim_matches('.')
        .to_string()
}

fn last_event(prompt: &str, prefix: &str) -> Option<String> {
    prompt.lines().rev().find_map(|line| {
        line.strip_prefix(prefix)
            .map(|value| value.trim().to_string())
    })
}

fn final_action(content: String) -> ActionTemplate {
    ActionTemplate {
        kind: "final".into(),
        thought: "respond after observation".into(),
        tool: None,
        args: None,
        content: Some(content),
    }
}

fn serialize_action(action: ActionTemplate) -> Result<String, String> {
    serde_json::to_string(&action).map_err(|error| error.to_string())
}

fn default_policy() -> String {
    r#"{"rules":[],"fallback":{"kind":"final","thought":"fallback","content":"I am running from a trained local policy."}}"#.into()
}
