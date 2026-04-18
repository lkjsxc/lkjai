# Quickstart

## Goal

Run `lkjai` locally with PostgreSQL and open the private operator console.

## Contract

1. Copy environment file if needed:
   ```bash
   cp .env.example .env
   ```
2. Build and start runtime services:
   ```bash
   docker compose up -d --build postgres app
   ```
3. Verify service health:
   ```bash
   curl -sS http://127.0.0.1:8080/healthz
   ```
4. Open:
   - `http://127.0.0.1:8080/` for the private console.

## Required Environment

- `ADMIN_TOKEN`: required for all write and chat actions.
- `DATABASE_URL`: PostgreSQL connection string.
- `PORT`: app listen port, default `8080`.

## Notes

- Current bootstrap implementation includes a mock storage adapter while preserving PostgreSQL contracts and compose service topology.
- Operator API calls must include the admin token.

