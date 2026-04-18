# API Contract

## Authentication

- Header: `x-admin-token: <token>`
- Missing or invalid token:
  ```json
  { "status": "error", "code": "unauthorized", "message": "invalid admin token" }
  ```

## Chat

- `POST /api/chat`
- Request:
  ```json
  { "message": "find records about distributed indexing" }
  ```
- Response:
  ```json
  {
    "status": "ok",
    "reply": "...",
    "trace": { "parallel_steps": 3, "queue_depth": 0 }
  }
  ```

## Upsert Record

- `POST /api/records/upsert`
- Request:
  ```json
  { "id": "rec-001", "title": "Indexing", "body": "..." }
  ```

## Delete Record

- `POST /api/records/delete`
- Request:
  ```json
  { "id": "rec-001" }
  ```

## List Records

- `GET /api/records/list?q=<query>`
- Response:
  ```json
  {
    "status": "ok",
    "records": [{ "id": "rec-001", "title": "Indexing" }]
  }
  ```

