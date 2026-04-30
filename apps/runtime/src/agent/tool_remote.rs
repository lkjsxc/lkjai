use reqwest::{header, multipart, Method};
use serde_json::json;
use tokio::fs;

use crate::config::Config;

use super::{tools::ToolCall, workspace::workspace_path};

pub async fn execute(call: &ToolCall, config: &Config) -> Result<Option<String>, String> {
    let (method, path, body, query, form) = match call {
        ToolCall::ResourceSearch {
            query,
            kind,
            sort,
            cursor,
            limit,
            direction,
            scope,
        } => (
            Method::GET,
            "/api/resources/search".to_string(),
            None,
            resource_query(query, kind, sort, cursor, limit, direction, scope),
            None,
        ),
        ToolCall::ResourceFetch { reference } => (
            Method::GET,
            format!("/api/resources/{reference}"),
            None,
            Vec::new(),
            None,
        ),
        ToolCall::ResourceHistory { reference } => (
            Method::GET,
            format!("/api/resources/{reference}/history"),
            None,
            Vec::new(),
            None,
        ),
        ToolCall::ResourcePreview {
            body,
            current_resource_id,
        } => (
            Method::POST,
            "/admin/markdown-preview".to_string(),
            Some(json!({"body": body, "current_resource_id": current_resource_id})),
            Vec::new(),
            None,
        ),
        ToolCall::ResourceCreateNote {
            body,
            alias,
            is_favorite,
            is_private,
        } => (
            Method::POST,
            "/api/resources/notes".to_string(),
            Some(
                json!({"body": body, "alias": alias, "is_favorite": is_favorite, "is_private": is_private}),
            ),
            Vec::new(),
            None,
        ),
        ToolCall::ResourceCreateMedia {
            path,
            alias,
            is_favorite,
            is_private,
        } => (
            Method::POST,
            "/api/resources/media".to_string(),
            None,
            Vec::new(),
            Some(media_form(config, path, alias, *is_favorite, *is_private).await?),
        ),
        ToolCall::ResourceUpdate {
            reference,
            body,
            alias,
            is_favorite,
            is_private,
        } => (
            Method::PUT,
            format!("/api/resources/{reference}"),
            Some(
                json!({"body": body, "alias": alias, "is_favorite": is_favorite, "is_private": is_private}),
            ),
            Vec::new(),
            None,
        ),
        _ => return Ok(None),
    };
    let url = format!("{}{}", config.kjxlkj_api_url.trim_end_matches('/'), path);
    let client = reqwest::Client::new();
    let mut request = client.request(method, &url);
    if !query.is_empty() {
        request = request.query(&query);
    }
    if !config.kjxlkj_session_cookie.is_empty() {
        request = request.header(header::COOKIE, &config.kjxlkj_session_cookie);
    }
    if let Some(body) = body {
        request = request.json(&body);
    }
    if let Some(form) = form {
        request = request.multipart(form);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("kjxlkj request failed: {error}"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| format!("kjxlkj response failed: {error}"))?;
    if status.is_success() {
        Ok(Some(text))
    } else {
        Err(format!("kjxlkj returned {status}: {text}"))
    }
}

fn resource_query(
    query: &str,
    kind: &str,
    sort: &Option<String>,
    cursor: &Option<String>,
    limit: &Option<String>,
    direction: &Option<String>,
    scope: &Option<String>,
) -> Vec<(&'static str, String)> {
    let mut values = vec![("q", query.to_string()), ("kind", kind.to_string())];
    for (key, value) in [
        ("sort", sort),
        ("cursor", cursor),
        ("limit", limit),
        ("direction", direction),
        ("scope", scope),
    ] {
        if let Some(value) = value {
            values.push((key, value.clone()));
        }
    }
    values
}

async fn media_form(
    config: &Config,
    path: &str,
    alias: &Option<String>,
    is_favorite: bool,
    is_private: bool,
) -> Result<multipart::Form, String> {
    let safe_path = workspace_path(&config.tool_workspace_dir, path.to_string())?;
    let bytes = fs::read(&safe_path)
        .await
        .map_err(|error| error.to_string())?;
    let filename = safe_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("upload.bin")
        .to_string();
    let part = multipart::Part::bytes(bytes).file_name(filename);
    let mut form = multipart::Form::new()
        .part("file", part)
        .text("is_favorite", is_favorite.to_string())
        .text("is_private", is_private.to_string());
    if let Some(alias) = alias {
        form = form.text("alias", alias.clone());
    }
    Ok(form)
}
