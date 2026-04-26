# Agent Assessment

## Current Artifact State

- Latest behavioral report: `data/train/runs/behavioral-eval.json`.
- Current artifact pass rate: `0.0` from `0/200` cases.
- Current artifact XML validity: `0.02` from `4/200` cases.
- Current fixed eval: `12/13` artifact checks passed.
- Missing fixed eval target: tokenizer train tokens `>= 450000000`.

## Observed Chat Behavior

Recent transcripts under `data/agent/runs/` show:

- simple greetings can trigger repeated unrelated `fs.read` calls,
- failed reads target repository docs that are not mounted in the tool
  workspace,
- the model often repeats the same failed action,
- some final answers are generic task-completion text,
- older runs include non-XML or JSON-shaped assistant output.

## Improvement Priorities

1. Teach everyday conversation to finish directly with `agent.finish`.
2. Remove generic final answers such as `Completed task for ...` from accepted
   conversational behavior.
3. Make brief `<reasoning>` visible so operators can inspect why a tool was
   chosen.
4. Align runtime tool workspace contents with corpus paths.
5. Expand valid XML action rows before increasing advanced tool complexity.
6. Report everyday-chat pass rate separately from repository tool tasks.
7. Block repeated identical non-terminal actions during runtime probes.

## Manual Probe Set

- `Hello`
- `What can you help me with?`
- `Please keep answers concise.`
- `Explain lkjai in one sentence.`
- `Thanks`
- `Remember that I prefer short answers.`
- `What do I prefer?`
- `List files in the workspace.`
- `Read docs/README.md and summarize the canon.`
