# Kimi Authentication

## Purpose

Verify Kimi credentials without printing secrets before launching a pilot or
long corpus run.

## Key Discovery

The generator loads keys from:

- `MOONSHOT_API_KEY`
- `KIMI_API_KEY`
- `MOONSHOT_API_KEYS`
- an explicit `--api-key-file`
- `/home/lkjsxc/private/archived/security/password.md`

Only tokens matching `sk-...` are treated as API keys. Keys starting with
`sk-kimi-` are routed to the Kimi Code OpenAI-compatible endpoint
`https://api.kimi.com/coding/v1` with model `kimi-for-coding`. Other keys use
the Moonshot Open Platform endpoint unless overridden. Logs and reports may
print SHA-256 fingerprints, never raw keys.

## Fingerprint Check

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, 'tools/kimi-corpus')
from kimi_lib.kimi_keys import load_api_keys, fingerprint
keys = load_api_keys('')
print({'key_count': len(keys), 'fingerprints': [fingerprint(k) for k in keys]})
PY
```

## API Smoke

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --api-provider kimi-api \
  --target-tokens 1000 \
  --mode sft \
  --parallelism 1 \
  --batch-documents 1 \
  --max-calls 1 \
  --output-dir data/kimi_synthetic/auth-smoke \
  --run-dir runs/kimi_auth_smoke
```

If every candidate returns `invalid_authentication_error`, stop before the
pilot run. Fix or replace the key source, rerun the smoke, then delete the
ignored failed staging directory.
