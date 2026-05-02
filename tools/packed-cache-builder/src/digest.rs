use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::source::source_jsonl_paths;

pub(crate) fn digest_source(source: &Path) -> Result<String> {
    if source.is_file() {
        return digest_path(source);
    }
    let mut hasher = Sha256::new();
    for path in source_jsonl_paths(source)? {
        let relative = path.strip_prefix(source).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(read_bytes(&path)?);
        hasher.update([0]);
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

pub(crate) fn digest_path(path: &Path) -> Result<String> {
    let bytes = read_bytes(path)?;
    Ok(digest_bytes(&bytes))
}

pub(crate) fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize().as_slice())
}

pub(crate) fn digest_packed_data(tokens: &[u8], loss_mask: &[u8], starts: &[u8]) -> String {
    let mut hasher = Sha256::new();
    for (name, bytes) in [
        ("tokens.bin", tokens),
        ("loss_mask.bin", loss_mask),
        ("starts.bin", starts),
    ] {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(bytes);
        hasher.update([0]);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(crate) fn read_to_string(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bytes)
}

pub(crate) fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_err(|source| Error::Io {
        path: PathBuf::from(path),
        source,
    })?;
    file.write_all(bytes).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}
