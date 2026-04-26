use reqwest::{header, Method};
use serde_json::json;

use crate::config::Config;

use super::tools::ToolCall;

pub async fn execute(call: &ToolCall, config: &Config) -> Result<Option<String>, String> {
    let (method, path, body, query) = match call {
        ToolCall::ResourceSearch { query, kind } => (
            Method::GET,
            "/api/resources/search".to_string(),
            None,
            vec![("q", query.clone()), ("kind", kind.clone())],
        ),
        ToolCall::ResourceFetch { reference } => (
            Method::GET,
            format!("/api/resources/{reference}"),
            None,
            Vec::new(),
        ),
        ToolCall::ResourceHistory { reference } => (
            Method::GET,
            format!("/api/resources/{reference}/history"),
            None,
            Vec::new(),
        ),
        ToolCall::ResourcePreview {
            body,
            current_resource_id,
        } => (
            Method::POST,
            "/api/resources/preview-markdown".to_string(),
            Some(json!({"body": body, "current_resource_id": current_resource_id})),
            Vec::new(),
        ),
        ToolCall::ResourceCreateNote {
            body,
            alias,
            is_private,
        } => (
            Method::POST,
            "/api/resources/notes".to_string(),
            Some(json!({"body": body, "alias": alias, "is_private": is_private})),
            Vec::new(),
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
