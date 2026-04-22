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
7. Ask the model for one strict JSON action.
8. Validate the JSON action.
9. Execute a tool when the action is `tool_call`.
10. Append the tool result as an observation.
11. Repeat until the action is `final` or a stop rule fires.

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

## Stop Reasons

- `final`: model produced a valid final answer.
- `max_steps`: loop reached `AGENT_MAX_STEPS`.
- `invalid_action`: model output could not be repaired.
- `tool_error`: configured fatal tool failure.
- `model_error`: model server call failed or model server is unreachable.

## Verification

```bash
curl -sf -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"list files"}' | jq '.stop_reason'
```

Expected: one of the stop reasons above; never an empty string.
