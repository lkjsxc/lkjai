# Agent Schema

## Action JSON

The model must return one JSON object.

```json
{
  "kind": "tool_call",
  "thought": "brief private planning note",
  "tool": "fs.read",
  "args": {"path": "/home/user/file.txt"}
}
```

```json
{
  "kind": "final",
  "thought": "brief private planning note",
  "content": "assistant response"
}
```

## Event Kinds

- `user`: user input.
- `assistant`: final assistant output.
- `plan`: model action thought or repair note.
- `tool_call`: validated tool request.
- `tool_result`: raw tool result or error.
- `observation`: compact observation passed back to the model.
- `memory_write`: durable memory write.
- `error`: model, validation, or runtime error.

## Required Event Fields

- `kind`: event kind.
- `content`: human-readable content.
- `tool`: optional tool name.
- `timestamp`: RFC 3339 timestamp.
- `step`: optional loop step number.

## Tool Names

- `shell.exec`.
- `web.fetch`.
- `fs.read`.
- `fs.write`.
- `fs.list`.
- `memory.search`.
- `memory.write`.
