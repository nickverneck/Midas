use tch::Device;

fn main() {
    println!("tch crate version: {}", env!("CARGO_PKG_VERSION"));
    println!("cuda available: {}", tch::Cuda::is_available());
    println!("cudnn available: {}", tch::Cuda::cudnn_is_available());
    if tch::Cuda::is_available() {
        println!("tch cuda device count: {}", tch::Cuda::device_count());
    }
    if let Ok(cuda_ver) = std::env::var("TORCH_CUDA_VERSION") {
        println!("env TORCH_CUDA_VERSION: {}", cuda_ver);
    }
    if tch::Cuda::is_available() {
        let cuda_tensor_ok = std::panic::catch_unwind(|| {
            let _t = tch::Tensor::zeros(&[1], (tch::Kind::Float, Device::Cuda(0)));
        })
        .is_ok();
        println!("cuda tensor usable: {}", cuda_tensor_ok);
    }

    // MPS support is exposed via Device::Mps in tch when libtorch has MPS.
    let mps_device = Device::Mps;
    let mps_available = std::panic::catch_unwind(|| {
        let _t = tch::Tensor::zeros(&[1], (tch::Kind::Float, mps_device));
    })
    .is_ok();
    println!("mps device usable: {}", mps_available);

    let device = if mps_available { Device::Mps } else { Device::Cpu };
    println!("selected device: {:?}", device);
}
