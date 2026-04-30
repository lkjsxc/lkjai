# Training

## Purpose

Training is a retired product-code location kept only to make the native
migration explicit.

## Contents

- Product train and serve code lives in [../native/](../native/).
- Historical Python package and tests were removed.
- Training runbooks live in [../docs/operations/training/](../docs/operations/training/).

## Rules

- Long jobs run through Docker Compose.
- Generated artifacts belong under `data/`, not in this directory.
