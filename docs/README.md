# Documentation Canon

`docs/` is the only active canon for project intent, product behavior, model
design, training operations, runtime shape, verification, and repository rules.

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

- [vision/README.md](vision/README.md): purpose, principles, and LLM-readable topology rules
- [getting-started/README.md](getting-started/README.md): quickstart and verification entrypoints
- [product/README.md](product/README.md): chat UI, agent tools, and API contracts
- [architecture/README.md](architecture/README.md): model, training, and runtime architecture
- [operations/README.md](operations/README.md): Compose profiles, long-run training ops, deployment, and quality gates
- [repository/README.md](repository/README.md): repository layout, workflow, and authoring rules
- [research/README.md](research/README.md): external model, corpus, and lightweighting references

## Recommended Reading Order

1. [vision/purpose.md](vision/purpose.md)
2. [repository/layout.md](repository/layout.md)
3. [architecture/model/config.md](architecture/model/config.md)
4. [architecture/training/corpus.md](architecture/training/corpus.md)
5. [architecture/training/pipeline.md](architecture/training/pipeline.md)
6. [operations/compose.md](operations/compose.md)
7. [operations/training/long-run.md](operations/training/long-run.md)
8. [operations/training/competency-gate.md](operations/training/competency-gate.md)
9. [operations/quality.md](operations/quality.md)
10. [product/agent-tools.md](product/agent-tools.md)
11. [architecture/runtime/web.md](architecture/runtime/web.md)
