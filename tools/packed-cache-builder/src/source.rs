use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use walkdir::WalkDir;

use crate::error::{Error, Result};

pub(crate) fn encode_source_tokens(
    source: &Path,
    tokenizer: &Tokenizer,
    tokenizer_path: &Path,
    needed: usize,
) -> Result<(Vec<u32>, u64)> {
    let mut ids = Vec::with_capacity(needed);
    let mut example_count = 0u64;
    for path in source_jsonl_paths(source)? {
        let file = File::open(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        for (line_index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&line).map_err(|source| Error::Json {
                path: path.with_file_name(format!(
                    "{}:{}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("jsonl"),
                    line_index + 1
                )),
                source,
            })?;
            let mut texts = Vec::new();
            collect_text_fields(&value, &mut texts);
            for text in texts {
                example_count += 1;
                let encoding =
                    tokenizer
                        .encode(text.as_str(), false)
                        .map_err(|message| Error::Tokenizer {
                            path: tokenizer_path.to_path_buf(),
                            message: message.to_string(),
                        })?;
                for id in encoding.get_ids() {
                    if ids.len() == needed {
                        return Ok((ids, example_count));
                    }
                    ids.push(*id);
                }
            }
        }
    }
    Ok((ids, example_count))
}

fn collect_text_fields(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if (key == "text" || key == "content") && child.is_string() {
                    output.push(child.as_str().unwrap_or_default().to_string());
                } else {
                    collect_text_fields(child, output);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_text_fields(item, output);
            }
        }
        _ => {}
    }
}

pub(crate) fn source_jsonl_paths(source: &Path) -> Result<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if !source.is_dir() {
        return Err(Error::Message(format!(
            "source is not a file or directory: {}",
            source.display()
        )));
    }
    let mut paths = Vec::new();
    for entry in WalkDir::new(source).sort_by_file_name() {
        let entry = entry.map_err(|source| Error::Message(source.to_string()))?;
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|ext| ext.to_str()) == Some("jsonl")
        {
            paths.push(entry.path().to_path_buf());
        }
    }
    Ok(paths)
}
