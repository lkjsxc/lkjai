# Packed Cache Builder Source

## Contents

- [build.rs](build.rs): deterministic cache construction.
- [digest.rs](digest.rs): SHA-256 file and payload checksums.
- [error.rs](error.rs): shared error type.
- [main.rs](main.rs): CLI entrypoint.
- [metadata.rs](metadata.rs): metadata and config types.
- [source.rs](source.rs): JSONL text extraction and tokenizer encoding.
- [tests.rs](tests.rs): deterministic and rejection tests.
- [validate.rs](validate.rs): strict cache validation.
