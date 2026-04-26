# Configs

## Purpose

Project configuration files that are intended to be committed live here.

## Contents

- [corpus/](corpus/): generation and corpus-size configuration.
- [curriculum/](curriculum/): training curriculum configuration.
- [training/](training/): scratch training run configuration.

## Rules

- Prefer explicit config files over large Compose environment blocks.
- Keep local secrets and host-specific overrides in ignored files.
