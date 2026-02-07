use anyhow::Result;
use tch::Tensor;
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

pub struct BatchedPolicy {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
}

impl BatchedPolicy {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let last = self.weights.len().saturating_sub(1);
        let mut x = input.shallow_clone();
        for (idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let x_batched = x.unsqueeze(-1);
            let mut y = w.bmm(&x_batched).squeeze_dim(-1) + b;
            if idx != last {
                y = y.tanh();
            }
            x = y;
        }
        x
    }
}

pub fn build_batched_policy(
    genomes: &[Vec<f32>],
    input_dim: usize,
    hidden: usize,
    layers: usize,
    device: tch::Device,
) -> BatchedPolicy {
    assert!(!genomes.is_empty(), "batch policy requires genomes");
    let layer_count = layers + 1;
    let mut weight_layers: Vec<Vec<Tensor>> = (0..layer_count)
        .map(|_| Vec::with_capacity(genomes.len()))
        .collect();
    let mut bias_layers: Vec<Vec<Tensor>> = (0..layer_count)
        .map(|_| Vec::with_capacity(genomes.len()))
        .collect();

    for genome in genomes {
        let mut offset = 0usize;
        let mut in_dim = input_dim;
        for layer in 0..layers {
            let w_len = in_dim * hidden;
            let b_len = hidden;
            let w = Tensor::f_from_slice(&genome[offset..offset + w_len])
                .expect("tensor from genome")
                .reshape(&[hidden as i64, in_dim as i64])
                .to_device(device);
            offset += w_len;
            let b = Tensor::f_from_slice(&genome[offset..offset + b_len])
                .expect("tensor from genome")
                .reshape(&[hidden as i64])
                .to_device(device);
            offset += b_len;
            weight_layers[layer].push(w);
            bias_layers[layer].push(b);
            in_dim = hidden;
        }

        let w_len = in_dim * 4;
        let b_len = 4;
        let w = Tensor::f_from_slice(&genome[offset..offset + w_len])
            .expect("tensor from genome")
            .reshape(&[4, in_dim as i64])
            .to_device(device);
        offset += w_len;
        let b = Tensor::f_from_slice(&genome[offset..offset + b_len])
            .expect("tensor from genome")
            .reshape(&[4])
            .to_device(device);
        offset += b_len;
        weight_layers[layers].push(w);
        bias_layers[layers].push(b);
        debug_assert_eq!(offset, genome.len(), "genome length mismatch");
    }

    let weights = weight_layers
        .into_iter()
        .map(|ws| Tensor::stack(&ws, 0))
        .collect();
    let biases = bias_layers
        .into_iter()
        .map(|bs| Tensor::stack(&bs, 0))
        .collect();

    BatchedPolicy { weights, biases }
}
