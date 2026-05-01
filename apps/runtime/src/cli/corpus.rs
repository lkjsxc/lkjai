use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use crate::agent::validate_action_text;

#[derive(Deserialize)]
struct Row {
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: String,
}

pub fn validate_actions(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    for path in paths {
        for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Row>(&line) {
                Ok(row) => validate_row(path, index + 1, row, &mut errors),
                Err(error) => errors.push(format!(
                    "{}:{} bad json: {}",
                    path.display(),
                    index + 1,
                    error
                )),
            }
        }
    }
    let status = if errors.is_empty() { "pass" } else { "fail" };
    println!(
        "{}",
        serde_json::json!({
            "command": "corpus validate-actions",
            "status": status,
            "errors": errors,
        })
    );
    if status == "pass" {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

fn validate_row(path: &PathBuf, line: usize, row: Row, errors: &mut Vec<String>) {
    let mut tools = Vec::new();
    for message in row
        .messages
        .iter()
        .filter(|message| message.role == "assistant")
    {
        match validate_action_text(&message.content) {
            Ok(tool) => tools.push(tool),
            Err(error) => errors.push(format!("{}:{} {}", path.display(), line, error)),
        }
    }
    if tools.is_empty() {
        errors.push(format!(
            "{}:{} missing assistant action",
            path.display(),
            line
        ));
        return;
    }
    validate_sequence(path, line, &tools, errors);
}

fn validate_sequence(path: &PathBuf, line: usize, tools: &[String], errors: &mut Vec<String>) {
    let last = tools.last().map(String::as_str).unwrap_or("");
    if last != "agent.finish" && last != "agent.request_confirmation" {
        errors.push(format!(
            "{}:{} last assistant action is {}",
            path.display(),
            line,
            last
        ));
    }
    for tool in tools {
        if matches!(
            tool.as_str(),
            "resource.create_note" | "resource.create_media" | "resource.update_resource"
        ) && !tools
            .iter()
            .any(|name| name == "agent.request_confirmation")
        {
            errors.push(format!(
                "{}:{} mutation without confirmation",
                path.display(),
                line
            ));
        }
    }
}
