use candle_core::Result;
use rand::Rng;

pub fn sample_top_k(
    scores: &[f32],
    vocab_size: usize,
    top_k: usize,
    temperature: f32,
) -> Result<u32> {
    let mut candidates: Vec<(usize, f32)> = scores
        .into_iter()
        .take(vocab_size)
        .enumerate()
        .filter_map(|(id, score)| score.is_finite().then_some((id, *score)))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(top_k.max(1));
    let Some((best, best_score)) = candidates.first().copied() else {
        candle_core::bail!("empty logits");
    };
    if temperature <= 0.0 || candidates.len() == 1 {
        return Ok(best as u32);
    }
    let mut weights = Vec::with_capacity(candidates.len());
    let mut total = 0.0f32;
    for (_, score) in &candidates {
        let weight = ((*score - best_score) / temperature).exp();
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return Ok(best as u32);
    }
    let mut threshold = rand::rng().random_range(0.0..total);
    for ((id, _), weight) in candidates.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(*id as u32);
        }
    }
    Ok(best as u32)
}
