# Agent Schema

## Action JSON

The model must return one JSON object.

```json
{
  "kind": "tool_call",
  "thought": "brief private planning note",
  "tool": "fs.read",
  "args": {"path": "README.md"}
}
```

```json
{
  "kind": "final",
  "thought": "brief private planning note",
  "content": "assistant response"
}
```

```json
{
  "kind": "request_confirmation",
  "summary": "Create a public research note",
  "operation": "resource.create_note",
  "pending_tool_call": {
    "tool": "resource.create_note",
    "args": {"body": "# Research", "is_private": false}
  }
}
```

## Event Kinds

- `user`
- `assistant`
- `plan`
- `tool_call`
- `tool_result`
- `observation`
- `memory_write`
- `confirmation_request`
- `error`
