# Compose Contract

## Services

- `postgres`: canonical persistence service.
- `app`: Zig runtime server.
- `verify`: docs + Zig quality gate runner.

## Runtime Commands

```bash
docker compose up -d --build postgres app
docker compose ps
docker compose down -v
```

## Verification Commands

```bash
docker compose -f docker-compose.yml -f docker-compose.verify.yml build app verify
docker compose -f docker-compose.yml -f docker-compose.verify.yml up -d postgres app
docker compose -f docker-compose.yml -f docker-compose.verify.yml run --rm verify
docker compose -f docker-compose.yml -f docker-compose.verify.yml down -v
```

