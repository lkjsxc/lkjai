# Agent Loop

## Goal

Execute one user turn as a bounded multi-step loop.

## Default Flow

1. Append the user message to the run transcript.
2. Load recent transcript events.
3. Load the rolling summary for older events.
4. Retrieve relevant durable memories.
5. Verify the model server is reachable.
6. Build the model prompt with system policy, tools, memory, and recent context.
7. Ask the model for one XML action.
8. Validate the XML action and tool fields.
9. Append a visible `reasoning` event when `<reasoning>` is present.
10. Stop as `repeat_action` if the same non-terminal action repeats.
11. Execute the requested tool.
12. Append the tool result as an observation when the tool is external.
13. Repeat until the executed tool is `agent.finish` or a stop rule fires.

## Model Unreachable Guard

- Before step 6, if the model server health probe fails, the loop stops
  immediately with `model_error`.
- No prompt is built and no request is sent when the server is down.

## Defaults

- `AGENT_MAX_STEPS=6`.
- `AGENT_REPAIR_ATTEMPTS=1`.
- `MODEL_MAX_NEW_TOKENS=512`.
- `MODEL_TEMPERATURE=0.2`.
- The agent must never execute an unvalidated tool call.
- The agent must not synthesize fake tool results.
- `<reasoning>` is visible and brief.
- `agent.think` emits a `plan` event and does not emit duplicate tool events.
- Prompt context is compact and should avoid replaying duplicate tool output.
- Simple everyday chat should use `agent.finish` directly.

## Stop Reasons

- `finish`: `agent.finish` produced the final answer.
- `max_steps`: loop reached `AGENT_MAX_STEPS`.
- `invalid_action`: XML action could not be repaired.
- `tool_error`: configured fatal tool failure.
- `model_error`: model server call failed or model server is unreachable.
- `repeat_action`: repeated identical non-terminal action was blocked.

## Verification

```bash
curl -sf -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"list files"}' | jq '.stop_reason'
```

Expected: one of the stop reasons above; never an empty string.
