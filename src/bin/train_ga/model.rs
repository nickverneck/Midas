use anyhow::Result;
use tch::nn;

pub fn param_count(input_dim: usize, hidden: usize, layers: usize) -> usize {
    let mut count = 0;
    let mut in_dim = input_dim;
    for _ in 0..layers {
        count += in_dim * hidden + hidden;
        in_dim = hidden;
    }
    count += in_dim * 4 + 4;
    count
}

pub fn build_mlp(p: &nn::Path, input_dim: i64, hidden: i64, layers: usize) -> nn::Sequential {
    let mut seq = nn::seq();
    let mut in_dim = input_dim;
    for i in 0..layers {
        let linear = nn::linear(p / format!("layer_{i}"), in_dim, hidden, Default::default());
        seq = seq.add(linear).add_fn(|xs| xs.tanh());
        in_dim = hidden;
    }
    let out = nn::linear(p / "out", in_dim, 4, Default::default());
    seq.add(out)
}

pub fn load_params_from_vec(vs: &nn::VarStore, genome: &[f32]) {
    use tch::no_grad;
    let vars = vs.trainable_variables();
    let mut offset = 0;
    for mut v in vars {
        let numel = v.numel();
        let slice = &genome[offset..offset + numel as usize];
        let t = tch::Tensor::f_from_slice(slice)
            .expect("tensor from genome")
            .reshape(&v.size())
            .to_device(v.device());
        no_grad(|| {
            v.copy_(&t);
        });
        offset += numel as usize;
    }
}

pub fn save_policy(
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    device: tch::Device,
    genome: &[f32],
    path: &std::path::Path,
) -> Result<()> {
    let vs = nn::VarStore::new(device);
    let _policy = build_mlp(&vs.root(), obs_dim as i64, hidden as i64, layers);
    load_params_from_vec(&vs, genome);
    vs.save(path)?;
    Ok(())
}
