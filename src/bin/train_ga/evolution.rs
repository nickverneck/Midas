use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub fn crossover(a: &[f32], b: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if rng.gen_bool(0.5) {
            out.push(a[i]);
        } else {
            out.push(b[i]);
        }
    }
    out
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    generation: usize,
    pop: Vec<Vec<f32>>,
}

fn parse_generation_from_path(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_string_lossy();
    let marker = "checkpoint_gen";
    let start = stem.find(marker)? + marker.len();
    let digits: String = stem[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub fn load_checkpoint(path: &Path) -> Result<(usize, Vec<Vec<f32>>)> {
    let data = std::fs::read(path)?;
    if let Ok(ckpt) = bincode::deserialize::<Checkpoint>(&data) {
        return Ok((ckpt.generation.saturating_add(1), ckpt.pop));
    }
    let pop: Vec<Vec<f32>> = bincode::deserialize(&data)?;
    let start_gen = parse_generation_from_path(path)
        .unwrap_or(0)
        .saturating_add(1);
    Ok((start_gen, pop))
}

pub fn save_checkpoint(path: &Path, generation: usize, pop: &[Vec<f32>]) -> Result<()> {
    let data = bincode::serialize(&Checkpoint {
        generation,
        pop: pop.to_vec(),
    })?;
    std::fs::write(path, data)?;
    Ok(())
}
