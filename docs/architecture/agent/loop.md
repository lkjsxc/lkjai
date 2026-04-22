# Agent Loop

## Goal

- Execute one user turn as a bounded multi-step loop.
- Preserve every meaningful step as structured transcript events.
- Let the model plan, call tools, observe results, revise, and answer.

## Default Flow

1. Append the user message to the run transcript.
2. Load recent transcript events.
3. Load the rolling summary for older events.
4. Retrieve relevant durable memories.
5. Build the model prompt with system policy, tools, memory, and recent context.
6. Ask the model for one strict JSON action.
7. Validate the JSON action.
8. Execute a tool when the action is `tool_call`.
9. Append the tool result as an observation.
10. Repeat until the action is `final` or a stop rule fires.

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
- `model_error`: model server call failed.
