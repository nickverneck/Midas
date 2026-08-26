use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Dropout, Init, Linear, Module, ModuleT, VarBuilder, VarMap, linear};
use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
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

/// Populate a model's variables using a host-side seeded RNG.
///
/// Candle's CPU backend does not expose a seedable device RNG, so its default
/// Kaiming initializers cannot be made reproducible through the device API.
/// Creating the tensors here keeps CPU runs reproducible while preserving the
/// same Kaiming-normal weights and uniform biases used by candle_nn::linear.
pub(crate) fn initialize_seeded_mlp(
    varmap: &mut VarMap,
    prefix: &str,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    output_dim: usize,
    seed_rng: &mut StdRng,
    device: &Device,
) -> Result<()> {
    let mut in_dim = input_dim;
    for layer_idx in 0..layers {
        initialize_seeded_linear(
            varmap,
            &format!("{prefix}.layer_{layer_idx}"),
            in_dim,
            hidden,
            seed_rng,
            device,
        )?;
        in_dim = hidden;
    }
    initialize_seeded_linear(
        varmap,
        &format!("{prefix}.out"),
        in_dim,
        output_dim,
        seed_rng,
        device,
    )?;
    Ok(())
}

fn initialize_seeded_linear(
    varmap: &mut VarMap,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    seed_rng: &mut StdRng,
    device: &Device,
) -> Result<()> {
    let weight_path = format!("{prefix}.weight");
    let bias_path = format!("{prefix}.bias");
    let weight_std = (2.0 / in_dim as f64).sqrt();
    let normal = Normal::new(0.0, weight_std).context("create seeded Kaiming initializer")?;
    let weights = (0..out_dim * in_dim)
        .map(|_| normal.sample(seed_rng) as f32)
        .collect::<Vec<_>>();
    let bias_bound = 1.0 / (in_dim as f64).sqrt();
    let biases = (0..out_dim)
        .map(|_| seed_rng.gen_range(-bias_bound..=bias_bound) as f32)
        .collect::<Vec<_>>();

    varmap.get(
        (out_dim, in_dim),
        &weight_path,
        Init::Const(0.0),
        DType::F32,
        device,
    )?;
    varmap.get(out_dim, &bias_path, Init::Const(0.0), DType::F32, device)?;
    varmap.set_one(
        &weight_path,
        Tensor::from_vec(weights, (out_dim, in_dim), device)?,
    )?;
    varmap.set_one(&bias_path, Tensor::from_vec(biases, out_dim, device)?)?;
    Ok(())
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
