use tch::nn;

pub fn build_mlp(
    p: &nn::Path,
    input_dim: i64,
    hidden: i64,
    layers: usize,
    output_dim: i64,
    dropout: f64,
) -> nn::SequentialT {
    let mut seq = nn::seq_t();
    let mut in_dim = input_dim;
    for i in 0..layers {
        let layer = nn::linear(p / format!("layer_{i}"), in_dim, hidden, Default::default());
        seq = seq
            .add(layer)
            .add_fn_t(move |xs, train| xs.tanh().dropout(dropout, train));
        in_dim = hidden;
    }
    let out = nn::linear(p / "out", in_dim, output_dim, Default::default());
    seq.add(out)
}

pub fn build_policy(
    p: &nn::Path,
    input_dim: i64,
    hidden: i64,
    layers: usize,
    action_dim: i64,
    dropout: f64,
) -> nn::SequentialT {
    build_mlp(p, input_dim, hidden, layers, action_dim, dropout)
}

pub fn build_value(
    p: &nn::Path,
    input_dim: i64,
    hidden: i64,
    layers: usize,
    dropout: f64,
) -> nn::SequentialT {
    build_mlp(p, input_dim, hidden, layers, 1, dropout)
}
