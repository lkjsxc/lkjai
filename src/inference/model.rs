use candle_core::{DType, Device, Result, Tensor};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Deserialize)]
struct ModelCfg {
    vocab_size: usize,
    context: usize,
    layers: usize,
    hidden: usize,
    heads: usize,
}

pub struct LkjModel {
    cfg: ModelCfg,
    device: Device,
    tok: Tensor,
    pos: Tensor,
    norm: Tensor,
    head: Tensor,
    blocks: Vec<Block>,
}

struct Block {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    qkv: Tensor,
    out: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl LkjModel {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: ModelCfg = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)
            .map_err(candle_core::Error::msg)?;
        let device = Device::cuda_if_available(0)?;
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };
        let mut ws = candle_core::safetensors::load(dir.join("model.safetensors"), &device)?;
        let tok = take(&mut ws, "tok.weight", dtype)?;
        let pos = take(&mut ws, "pos.weight", dtype)?;
        let norm = take(&mut ws, "norm.weight", dtype)?;
        let head = take(&mut ws, "head.weight", dtype)?;
        if cfg.hidden % cfg.heads != 0 || tok.dim(0)? != cfg.vocab_size {
            candle_core::bail!("config does not match exported tensor shapes");
        }
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block {
                attn_norm: take(&mut ws, &format!("blocks.{i}.attn_norm.weight"), dtype)?,
                ffn_norm: take(&mut ws, &format!("blocks.{i}.ffn_norm.weight"), dtype)?,
                qkv: take(&mut ws, &format!("blocks.{i}.attn.in_proj_weight"), dtype)?,
                out: take(&mut ws, &format!("blocks.{i}.attn.out_proj.weight"), dtype)?,
                gate: take(&mut ws, &format!("blocks.{i}.ffn.gate.weight"), dtype)?,
                up: take(&mut ws, &format!("blocks.{i}.ffn.up.weight"), dtype)?,
                down: take(&mut ws, &format!("blocks.{i}.ffn.down.weight"), dtype)?,
            });
        }
        Ok(Self {
            cfg,
            device,
            tok,
            pos,
            norm,
            head,
            blocks,
        })
    }

    pub fn context(&self) -> usize {
        self.cfg.context
    }

    pub fn device_name(&self) -> String {
        format!("{:?}", self.device)
    }

    pub fn next_token(&self, ids: &[u32]) -> Result<u32> {
        let logits = self.forward(ids)?;
        select_token(logits, ids, self.cfg.vocab_size)
    }

    fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let seq = ids.len();
        let ids_t = Tensor::from_slice(ids, seq, &self.device)?;
        let pos_ids: Vec<u32> = (0..seq as u32).collect();
        let pos_t = Tensor::from_slice(&pos_ids, seq, &self.device)?;
        let mut x = self
            .tok
            .index_select(&ids_t, 0)?
            .broadcast_add(&self.pos.index_select(&pos_t, 0)?)?;
        for block in &self.blocks {
            let attn = block.attention(&rms_norm(&x, &block.attn_norm)?, &self.cfg)?;
            x = x.add(&attn)?;
            let ffn = block.ffn(&rms_norm(&x, &block.ffn_norm)?)?;
            x = x.add(&ffn)?;
        }
        let h = rms_norm(&x, &self.norm)?.get(seq - 1)?;
        self.head.matmul(&h.unsqueeze(1)?)?.squeeze(1)
    }
}

impl Block {
    fn attention(&self, x: &Tensor, cfg: &ModelCfg) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let head_dim = cfg.hidden / cfg.heads;
        let qkv = x.matmul(&self.qkv.t()?)?;
        let q = heads(&qkv.narrow(1, 0, cfg.hidden)?, cfg.heads, head_dim)?;
        let k = heads(&qkv.narrow(1, cfg.hidden, cfg.hidden)?, cfg.heads, head_dim)?;
        let v = heads(
            &qkv.narrow(1, cfg.hidden * 2, cfg.hidden)?,
            cfg.heads,
            head_dim,
        )?;
        let scores = q.matmul(&k.transpose(1, 2)?)? * (1.0 / (head_dim as f64).sqrt());
        let scores = scores?.broadcast_add(&causal_mask(seq, cfg.heads, x.device())?)?;
        let probs = softmax_last(&scores)?;
        let y = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((seq, cfg.hidden))?;
        y.matmul(&self.out.t()?)
    }

    fn ffn(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.matmul(&self.gate.t()?)?.silu()?;
        let up = x.matmul(&self.up.t()?)?;
        gate.mul(&up)?.matmul(&self.down.t()?)
    }
}

fn take(ws: &mut HashMap<String, Tensor>, key: &str, dtype: DType) -> Result<Tensor> {
    ws.remove(key)
        .ok_or_else(|| candle_core::Error::msg(format!("missing tensor {key}")))?
        .to_dtype(dtype)
}

fn heads(x: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    let seq = x.dim(0)?;
    x.reshape((seq, heads, head_dim))?.transpose(0, 1)
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(x.rank() - 1)? + 1e-6)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&scale)?.broadcast_mul(weight)
}

fn causal_mask(seq: usize, heads: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0f32; seq * seq];
    for row in 0..seq {
        for col in row + 1..seq {
            values[row * seq + col] = -1.0e9;
        }
    }
    Tensor::from_vec(values, (seq, seq), device)?.broadcast_as((heads, seq, seq))
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let dim = x.rank() - 1;
    let shifted = x.broadcast_sub(&x.max_keepdim(dim)?)?;
    let exp = shifted.exp()?;
    exp.broadcast_div(&exp.sum_keepdim(dim)?)
}

fn select_token(logits: Tensor, ids: &[u32], vocab_size: usize) -> Result<u32> {
    let mut scores = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for token in [0usize, 2usize] {
        if token < scores.len() {
            scores[token] = f32::NEG_INFINITY;
        }
    }
    for id in ids.iter().rev().take(64) {
        let index = *id as usize;
        if index < scores.len() {
            scores[index] = f32::NEG_INFINITY;
        }
    }
    scores
        .into_iter()
        .take(vocab_size)
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id as u32)
        .ok_or_else(|| candle_core::Error::msg("empty logits"))
}
