FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}
ARG CUDA_COMPUTE_CAP=86
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.93.0

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --features cuda --bin lkjai

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/lkjai /app/lkjai

ENV APP_HOST=127.0.0.1
ENV APP_PORT=8080
ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/data/train/models/lkj-150m
ENV INFERENCE_DEVICE=cuda

EXPOSE 8080
CMD ["/app/lkjai"]
