FROM rust:1.93-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin lkjai

FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/lkjai /app/lkjai

ENV APP_HOST=127.0.0.1
ENV APP_PORT=8080
ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/data/models/lkj-150m

EXPOSE 8080
CMD ["/app/lkjai"]
