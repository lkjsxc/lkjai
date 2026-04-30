use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct PackedCache {
    tokens: Vec<u8>,
    mask: Vec<u8>,
    starts: Vec<u8>,
}

impl PackedCache {
    fn open(cache_dir: &Path) -> std::io::Result<Self> {
        Ok(Self {
            tokens: fs::read(cache_dir.join("tokens.bin"))?,
            mask: fs::read(cache_dir.join("loss_mask.bin"))?,
            starts: fs::read(cache_dir.join("starts.bin"))?,
        })
    }

    fn start_count(&self) -> usize {
        self.starts.len() / 8
    }

    fn start_at(&self, index: usize) -> usize {
        let offset = index * 8;
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&self.starts[offset..offset + 8]);
        u64::from_le_bytes(bytes) as usize
    }

    fn token_at(&self, index: usize) -> u32 {
        let offset = index * 2;
        if offset + 2 > self.tokens.len() {
            return 0;
        }
        let mut bytes = [0_u8; 2];
        bytes.copy_from_slice(&self.tokens[offset..offset + 2]);
        u16::from_le_bytes(bytes) as u32
    }

    fn mask_at(&self, index: usize) -> u8 {
        self.mask.get(index).copied().unwrap_or(0)
    }

    fn window_checksum(&self, index: usize, sequence_len: usize) -> u64 {
        let start = self.start_at(index);
        let mut checksum = 0_u64;
        for pos in 0..sequence_len {
            let token = self.token_at(start + pos) as u64;
            let label_token = self.token_at(start + pos + 1) as u64;
            let train = self.mask_at(start + pos + 1) != 0;
            checksum = checksum.wrapping_add(token);
            if train {
                checksum = checksum.wrapping_add(label_token << 1);
            }
        }
        checksum
    }
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let flag = format!("--{name}");
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|items| items[0] == flag)
        .and_then(|items| items[1].parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_path() -> PathBuf {
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|items| items[0] == "--cache-dir")
        .map(|items| PathBuf::from(&items[1]))
        .unwrap_or_else(|| PathBuf::from("data/train/datasets/packed/train-causal_lm_full-seq1024"))
}

fn main() -> std::io::Result<()> {
    let cache_dir = parse_path();
    let sequence_len = parse_arg("sequence-len", 1024_usize);
    let requested = parse_arg("windows", 10_000_usize);
    let cache = PackedCache::open(&cache_dir)?;
    let windows = requested.min(cache.start_count());
    let started = Instant::now();
    let mut checksum = 0_u64;
    for index in 0..windows {
        checksum = checksum.wrapping_add(cache.window_checksum(index, sequence_len));
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    println!(
        "{{\n  \"cache_dir\": \"{}\",\n  \"sequence_len\": {},\n  \"windows_requested\": {},\n  \"windows_read\": {},\n  \"elapsed_seconds\": {:.9},\n  \"windows_per_second\": {:.3},\n  \"tokens_per_second\": {:.3},\n  \"checksum\": {}\n}}",
        cache_dir.display(),
        sequence_len,
        requested,
        windows,
        elapsed,
        windows as f64 / elapsed,
        (windows * sequence_len) as f64 / elapsed,
        checksum
    );
    Ok(())
}
