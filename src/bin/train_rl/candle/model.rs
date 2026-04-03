use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{Dropout, Linear, Module, ModuleT, VarBuilder, VarMap, linear};
use std::path::Path;

pub(crate) struct Mlp {
    hidden_layers: Vec<Linear>,
    out: Linear,
    dropout: Option<Dropout>,
}

impl Mlp {
    pub(crate) fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.hidden_layers {
            xs = layer.forward(&xs)?;
            xs = xs.tanh()?;
            if let Some(dropout) = &self.dropout {
                xs = dropout.forward_t(&xs, train)?;
            }
        }
        Ok(self.out.forward(&xs)?)
    }
}

pub(crate) fn build_policy(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    action_dim: usize,
    dropout: f64,
) -> Result<Mlp> {
    build_mlp(vb, input_dim, hidden, layers, action_dim, dropout)
}

pub(super) fn build_value(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    dropout: f64,
) -> Result<Mlp> {
    build_mlp(vb, input_dim, hidden, layers, 1, dropout)
}

fn build_mlp(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    output_dim: usize,
    dropout: f64,
) -> Result<Mlp> {
    let mut hidden_layers = Vec::with_capacity(layers);
    let mut in_dim = input_dim;
    for i in 0..layers {
        hidden_layers.push(linear(in_dim, hidden, vb.pp(format!("layer_{i}")))?);
        in_dim = hidden;
    }
    Ok(Mlp {
        hidden_layers,
        out: linear(in_dim, output_dim, vb.pp("out"))?,
        dropout: (dropout > 0.0).then(|| Dropout::new(dropout as f32)),
    })
}

pub(crate) fn load_checkpoint(varmap: &mut VarMap, path: &Path) -> Result<()> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        bail!(
            "candle RL checkpoints must be .safetensors files (received {})",
            path.display()
        );
    }
    varmap
        .load(path)
        .with_context(|| format!("load candle checkpoint {}", path.display()))
}
