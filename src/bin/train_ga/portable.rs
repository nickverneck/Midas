use anyhow::{Result, bail};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct PortableLayer {
    name: String,
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Serialize)]
struct PortablePolicy {
    format_version: u32,
    architecture: &'static str,
    input_dim: usize,
    hidden_dim: usize,
    hidden_layers: usize,
    output_dim: usize,
    action_labels: [&'static str; 4],
    layers: Vec<PortableLayer>,
}

pub fn save_policy_json(
    input_dim: usize,
    hidden: usize,
    layers: usize,
    genome: &[f32],
    path: &Path,
) -> Result<()> {
    let payload = build_policy(input_dim, hidden, layers, genome)?;
    let json = serde_json::to_string_pretty(&payload)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn build_policy(
    input_dim: usize,
    hidden: usize,
    layers: usize,
    genome: &[f32],
) -> Result<PortablePolicy> {
    let mut parsed_layers = Vec::with_capacity(layers + 1);
    let mut offset = 0usize;
    let mut in_dim = input_dim;

    for idx in 0..layers {
        let out_dim = hidden;
        let weight_len = in_dim * out_dim;
        let bias_len = out_dim;
        parsed_layers.push(PortableLayer {
            name: format!("layer_{idx}"),
            in_dim,
            out_dim,
            weight: genome[offset..offset + weight_len].to_vec(),
            bias: genome[offset + weight_len..offset + weight_len + bias_len].to_vec(),
        });
        offset += weight_len + bias_len;
        in_dim = hidden;
    }

    let out_dim = 4usize;
    let weight_len = in_dim * out_dim;
    let bias_len = out_dim;
    parsed_layers.push(PortableLayer {
        name: "out".to_string(),
        in_dim,
        out_dim,
        weight: genome[offset..offset + weight_len].to_vec(),
        bias: genome[offset + weight_len..offset + weight_len + bias_len].to_vec(),
    });
    offset += weight_len + bias_len;

    if offset != genome.len() {
        bail!(
            "portable policy export expected {} parameters but received {}",
            offset,
            genome.len()
        );
    }

    Ok(PortablePolicy {
        format_version: 1,
        architecture: "mlp-tanh",
        input_dim,
        hidden_dim: hidden,
        hidden_layers: layers,
        output_dim: out_dim,
        action_labels: ["buy", "sell", "hold", "revert"],
        layers: parsed_layers,
    })
}
