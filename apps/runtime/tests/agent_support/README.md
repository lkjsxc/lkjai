# Agent Test Support

## Purpose

Shared fixtures for runtime integration tests live here.

## Contents

- [mod.rs](mod.rs): fake model server, fake resource server, config builder,
  and XML action helpers.

## Rules

- Keep helpers deterministic.
- Do not call external services from tests.
