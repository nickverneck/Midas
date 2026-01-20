use tch::Device;

fn main() {
    println!("tch crate version: {}", env!("CARGO_PKG_VERSION"));
    println!("cuda available: {}", tch::Cuda::is_available());
    println!("cudnn available: {}", tch::Cuda::cudnn_is_available());
    if tch::Cuda::is_available() {
        println!("tch cuda device count: {}", tch::Cuda::device_count());
        if tch::Cuda::device_count() > 0 {
            println!("tch cuda device 0 name: {}", tch::Cuda::device_name(0).unwrap_or_else(|_| "unknown".to_string()));
        }
    }
    println!("tch cuda version: {}", tch::Cuda::version());

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
