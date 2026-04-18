# Verification

## Goal

Run the canonical quality pipeline using Docker Compose.

## Command Bundle

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.verify.yml build app verify
docker compose -f docker-compose.yml -f docker-compose.verify.yml up -d postgres app
docker compose -f docker-compose.yml -f docker-compose.verify.yml run --rm verify
docker compose -f docker-compose.yml -f docker-compose.verify.yml down -v
```

## Pass Conditions

- Docs topology and line-limit checks pass.
- Zig build and tests pass.
- App starts and `/healthz` returns success.
- Artifact-size gate evaluates configured deploy model metadata and enforces `<= 512 MiB`.

## Stop Rule

- Any non-zero command exit blocks acceptance.

