# Storage Contract

## Goal

Persist librarian records and runtime metadata with PostgreSQL.

## Canonical Backend

- PostgreSQL is the source of truth backend.
- Connection string comes from `DATABASE_URL` (compose path: `postgres://lkjai:lkjai@postgres:5432/lkjai`).

## Record Shape

- `id` (text primary key)
- `title` (text)
- `body` (text)
- `updated_at` (timestamp)

## Adapter Rules

- Runtime uses a storage interface to keep domain logic decoupled from DB driver details.
- Runtime selects PostgreSQL storage when `DATABASE_URL` starts with `postgres://` or `postgresql://`.
- In-memory fallback only applies when a non-PostgreSQL URL is configured.
- Write and delete operations must return explicit status values.
