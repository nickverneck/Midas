use tch::Device;

fn main() {
    println!("tch crate version: {}", env!("CARGO_PKG_VERSION"));
    println!("cuda available: {}", tch::Cuda::is_available());
    println!("cudnn available: {}", tch::Cuda::cudnn_is_available());

    let device = Device::default();
    println!("default device: {:?}", device);

    // MPS support is exposed via Device::Mps in tch when libtorch has MPS.
    let mps_device = Device::Mps;
    let mps_available = std::panic::catch_unwind(|| {
        let _t = tch::Tensor::zeros(&[1], (tch::Kind::Float, mps_device));
    })
    .is_ok();
    println!("mps device usable: {}", mps_available);
}
