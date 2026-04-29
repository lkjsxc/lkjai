# Training Operations

Use this subtree for scratch tokenizer/model training behavior, competency
gating, and operator-facing acceptance flow.

## Read This Section When

- You need to run the scratch training Compose profile.
- You need fixed-eval competency criteria and artifact schema.
- You need start/continue/stop rules for long GPU runs.

## Child Index

- [long-run.md](long-run.md): real training runtime contract and environment knobs
- [competency-gate.md](competency-gate.md): fixed-eval threshold and acceptance policy
- [iteration.md](iteration.md): baseline, pass-rate ladder, and accepted-run log
- [kimi-corpus-smoke.md](kimi-corpus-smoke.md): latest Kimi smoke report
- [kimi-corpus/README.md](kimi-corpus/README.md): Kimi synthetic corpus workflow,
  schema, quality gates, and long-run commands
- [kimi-corpus-generation.md](kimi-corpus-generation.md): KimiCode full-corpus
  generation prompt background
- [agent-assessment.md](agent-assessment.md): current observed agent behavior and
  improvement priorities
- [full-corpus-status.md](full-corpus-status.md): generated chunk status and
  remaining token gap
- [model-pure-recovery.md](model-pure-recovery.md): no-fallback recovery path
  for broken XML-action chat artifacts
- [xml-action-repair.md](xml-action-repair.md): focused repair corpus and SFT
  pass for broken XML action generation
