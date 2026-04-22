# Agent Failure Handling

## Invalid Model Output

- Invalid JSON gets one repair attempt by default.
- The repair prompt includes the validation error and required schema.
- If repair fails, the turn stops with `invalid_action`.

## Tool Errors

- Tool calls are logged before execution.
- Tool results are logged after execution.
- Failed tool calls become `tool_result` and `observation` events.
- Tool errors do not panic the web process.

## Limits

- Every tool has a timeout.
- Textual tool output is truncated to `TOOL_OUTPUT_LIMIT`.
- The agent loop stops at `AGENT_MAX_STEPS`.
- The model prompt must favor recent context, summaries, and retrieved memory
  over full transcript replay.
