# Documentation Canon

`docs/` is the active canon for `lkjai`.

The project is a local agentic multi-turn AI system for an RTX 3070 8GB
workstation. It is not primarily a from-scratch small LLM platform.

## Global Rules

1. Keep one canonical owner for each rule.
2. Keep every docs directory to one `README.md` plus multiple children.
3. Keep every docs file at 300 lines or fewer.
4. Keep every authored source file at 200 lines or fewer.
5. Prefer short declarative bullets over narrative prose.
6. Remove stale contracts instead of preserving conflicting versions.
7. Document exact commands, paths, defaults, and payload shapes when they matter.
8. Optimize for LLM retrieval before human ornament.

## Top-Level Sections

- [vision/README.md](vision/README.md): purpose, principles, and LLM readability
- [getting-started/README.md](getting-started/README.md): quickstart and verification
- [product/README.md](product/README.md): chat, tools, memory, and HTTP API
- [architecture/README.md](architecture/README.md): agent, model, memory, runtime, and training design
- [operations/README.md](operations/README.md): Compose, deployment, training runs, and gates
- [repository/README.md](repository/README.md): repository layout, workflow, and authoring rules
- [research/README.md](research/README.md): model, tuning, and serving references
- [decisions/README.md](decisions/README.md): keep/remove list and rejected alternatives

## Recommended Reading Order

1. [vision/purpose.md](vision/purpose.md)
2. [decisions/keep-remove.md](decisions/keep-remove.md)
3. [architecture/agent/loop.md](architecture/agent/loop.md)
4. [architecture/agent/schema.md](architecture/agent/schema.md)
5. [architecture/memory/store.md](architecture/memory/store.md)
6. [architecture/model/config.md](architecture/model/config.md)
7. [architecture/training/pipeline.md](architecture/training/pipeline.md)
8. [operations/compose.md](operations/compose.md)
9. [operations/quality.md](operations/quality.md)
10. [product/api.md](product/api.md)
