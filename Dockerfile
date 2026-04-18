FROM alpine:3.20 AS builder
WORKDIR /workspace
RUN apk add --no-cache zig build-base
COPY . .
RUN zig build -Doptimize=ReleaseSafe

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=builder /workspace/zig-out/bin/lkjai /usr/local/bin/lkjai
ENV PORT=8080
EXPOSE 8080
CMD ["lkjai"]
