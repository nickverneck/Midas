use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::actions::{POLICY_ACTION_DIM, POLICY_ACTION_LABELS};

#[derive(Serialize, Deserialize)]
struct PortableLayer {
    name: String,
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct PortablePolicy {
    format_version: u32,
    architecture: String,
    input_dim: usize,
    hidden_dim: usize,
    hidden_layers: usize,
    output_dim: usize,
    action_labels: [String; POLICY_ACTION_DIM],
    layers: Vec<PortableLayer>,
}

pub struct LoadedPolicy {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub hidden_layers: usize,
    pub genome: Vec<f32>,
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

pub fn load_policy_json(path: &Path) -> Result<LoadedPolicy> {
    let text = std::fs::read_to_string(path)?;
    let payload: PortablePolicy = serde_json::from_str(&text)?;

    if payload.architecture != "mlp-tanh" {
        bail!(
            "unsupported portable policy architecture '{}' in {}",
            payload.architecture,
            path.display()
        );
    }
    if payload.output_dim != POLICY_ACTION_DIM {
        bail!(
            "portable policy output dim {} does not match expected {} in {}",
            payload.output_dim,
            POLICY_ACTION_DIM,
            path.display()
        );
    }
    if payload.layers.len() != payload.hidden_layers + 1 {
        bail!(
            "portable policy layer count {} does not match hidden_layers {} in {}",
            payload.layers.len(),
            payload.hidden_layers,
            path.display()
        );
    }

    let mut genome = Vec::new();
    for (idx, layer) in payload.layers.iter().enumerate() {
        let expected_out = if idx == payload.hidden_layers {
            POLICY_ACTION_DIM
        } else {
            payload.hidden_dim
        };
        if layer.out_dim != expected_out {
            bail!(
                "portable policy layer '{}' out_dim {} does not match expected {} in {}",
                layer.name,
                layer.out_dim,
                expected_out,
                path.display()
            );
        }
        if layer.weight.len() != layer.in_dim * layer.out_dim {
            bail!(
                "portable policy layer '{}' weight len {} does not match {} x {} in {}",
                layer.name,
                layer.weight.len(),
                layer.in_dim,
                layer.out_dim,
                path.display()
            );
        }
        if layer.bias.len() != layer.out_dim {
            bail!(
                "portable policy layer '{}' bias len {} does not match out_dim {} in {}",
                layer.name,
                layer.bias.len(),
                layer.out_dim,
                path.display()
            );
        }
        genome.extend_from_slice(&layer.weight);
        genome.extend_from_slice(&layer.bias);
    }

    Ok(LoadedPolicy {
        input_dim: payload.input_dim,
        hidden_dim: payload.hidden_dim,
        hidden_layers: payload.hidden_layers,
        genome,
    })
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

    let out_dim = POLICY_ACTION_DIM;
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
        format_version: 2,
        architecture: "mlp-tanh".to_string(),
        input_dim,
        hidden_dim: hidden,
        hidden_layers: layers,
        output_dim: out_dim,
        action_labels: POLICY_ACTION_LABELS.map(str::to_string),
        layers: parsed_layers,
    })
}
