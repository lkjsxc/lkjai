# Runtime Tests

## Purpose

Integration tests verify the runtime agent loop, configuration, and resource
tool behavior.

## Contents

- [agent_support/README.md](agent_support/README.md): shared test helpers.
- [agent_tests.rs](agent_tests.rs): agent loop and local-tool behavior.
- [config_tests.rs](config_tests.rs): environment default behavior.
- [resource_tests.rs](resource_tests.rs): `kjxlkj` resource tool behavior.

## Rules

- Tests use fake HTTP servers instead of real model or `kjxlkj` services.
- Mutation tests must prove confirmation is required before execution.
