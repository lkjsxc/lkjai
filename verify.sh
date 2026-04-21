#!/bin/sh
set -eu

echo "== rust fmt =="
cargo fmt -- --check

echo "== rust tests =="
cargo test

echo "== python tests =="
python3 -m pytest -o cache_dir=/tmp/pytest-cache training/tests

echo "== docs topology =="
cargo run --bin lkjai -- docs validate-topology

echo "== docs links =="
cargo run --bin lkjai -- docs validate-links

echo "== line limits =="
cargo run --bin lkjai -- quality check-lines

echo "== forbidden js runtime check =="
cargo run --bin lkjai -- quality no-node

echo "== training smoke =="
python3 -m lkjai_train.cli --data-dir "$DATA_DIR" smoke

echo "== gates passed =="
