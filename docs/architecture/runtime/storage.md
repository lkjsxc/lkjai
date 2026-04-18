# Storage Contract

## Goal

Persist librarian records and runtime metadata with PostgreSQL.

## Canonical Backend

- PostgreSQL is the source of truth backend.
- Connection string comes from `DATABASE_URL`.

## Record Shape

- `id` (text primary key)
- `title` (text)
- `body` (text)
- `updated_at` (timestamp)

## Adapter Rules

- Runtime uses a storage interface to keep domain logic decoupled from DB driver details.
- Bootstrap implementation may use in-memory fallback for local development, but PostgreSQL contract remains canonical.
- Write and delete operations must return explicit status values.

