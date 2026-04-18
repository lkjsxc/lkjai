# Purpose Contract

## Goal

`lkjai` is a Zig-first platform for building and operating an end-to-end LLM stack, from strict from-scratch training to a production-facing librarian AI agent.

## Product Intent

- Train a dense transformer from random initialization.
- Keep a deployment-ready inference artifact at `<= 512 MiB`.
- Operate a private single-operator chat console.
- Provide an orchestration-style agent that can process many requests in parallel.
- Manage librarian records with deterministic, auditable behavior.

## Non-Goals

- Backward compatibility with previous project structures.
- Multi-user public SaaS scope in v1.
- Weakly defined behavior not anchored in docs contracts.

