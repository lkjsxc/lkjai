use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Action {
    pub tool: String,
    pub reasoning: Option<String>,
    fields: BTreeMap<String, String>,
}

impl Action {
    pub fn new(tool: String, fields: BTreeMap<String, String>) -> Self {
        Self {
            tool,
            reasoning: None,
            fields,
        }
    }

    pub fn fields(&self) -> BTreeMap<String, String> {
        self.fields.clone()
    }

    pub fn field(&self, key: &str) -> Option<String> {
        self.fields.get(key).filter(|v| !v.is_empty()).cloned()
    }

    pub fn bool_field(&self, key: &str, default: bool) -> bool {
        self.field(key)
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(default)
    }

    pub fn signature(&self) -> String {
        format!("{}:{:?}", self.tool, self.fields)
    }
}

pub fn parse(text: &str) -> Result<Action, String> {
    let body = action_body(text)?;
    let fields = child_fields(body)?;
    let tool = fields
        .get("tool")
        .filter(|v| !v.is_empty())
        .cloned()
        .ok_or_else(|| "XML action missing tool".to_string())?;
    Ok(Action {
        tool,
        reasoning: fields.get("reasoning").filter(|v| !v.is_empty()).cloned(),
        fields,
    })
}

fn action_body(text: &str) -> Result<&str, String> {
    let trimmed = text.trim();
    if trimmed.contains("<action ") {
        return Err("XML action tags must not use attributes".into());
    }
    let start = trimmed
        .find("<action>")
        .ok_or_else(|| "missing <action>".to_string())?;
    let content_start = start + "<action>".len();
    let end = trimmed[content_start..]
        .find("</action>")
        .map(|idx| content_start + idx)
        .ok_or_else(|| "missing </action>".to_string())?;
    if !trimmed[end + "</action>".len()..].trim().is_empty() {
        return Err("prose after </action>".into());
    }
    Ok(&trimmed[content_start..end])
}

fn child_fields(mut body: &str) -> Result<BTreeMap<String, String>, String> {
    let mut fields = BTreeMap::new();
    loop {
        body = body.trim_start();
        if body.is_empty() {
            return Ok(fields);
        }
        let close = body
            .find('>')
            .ok_or_else(|| "malformed child tag".to_string())?;
        let open = &body[..=close];
        if !open.starts_with('<') || open.starts_with("</") || open.contains(' ') {
            return Err("child tags must be simple and attribute-free".into());
        }
        let key = &open[1..open.len() - 1];
        if key.is_empty() || !key.chars().all(valid_tag_char) {
            return Err(format!("invalid child tag {key}"));
        }
        let end_tag = format!("</{key}>");
        let value_start = close + 1;
        let value_end = body[value_start..]
            .find(&end_tag)
            .map(|idx| value_start + idx)
            .ok_or_else(|| format!("missing {end_tag}"))?;
        if fields
            .insert(key.to_string(), unescape(&body[value_start..value_end]))
            .is_some()
        {
            return Err(format!("duplicate child tag {key}"));
        }
        body = &body[value_end + end_tag.len()..];
    }
}

fn valid_tag_char(char: char) -> bool {
    char.is_ascii_lowercase() || char.is_ascii_digit() || char == '_' || char == '.'
}

fn unescape(text: &str) -> String {
    text.trim()
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}
