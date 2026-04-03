#[cfg(feature = "backend-candle-cuda")]
use anyhow::Context;
use anyhow::{Result, bail};
use candle_core::Device;
use midas_env::ml::ComputeRuntime;

pub(crate) fn resolve_device(requested: ComputeRuntime) -> Result<Device> {
    match requested {
        ComputeRuntime::Cpu => Ok(Device::Cpu),
        ComputeRuntime::Auto => auto_device(),
        ComputeRuntime::Cuda => explicit_cuda_device(),
        ComputeRuntime::Mps => bail!(
            "candle RL backend in this branch does not expose Apple GPU execution; use --device cpu for Candle or use Burn/libtorch for Apple GPU experiments"
        ),
    }
}

pub(crate) fn runtime_from_device(device: &Device) -> ComputeRuntime {
    match device {
        Device::Cuda(_) => ComputeRuntime::Cuda,
        Device::Metal(_) => ComputeRuntime::Mps,
        _ => ComputeRuntime::Cpu,
    }
}

pub(crate) fn print_device(device: &Device) {
    match device {
        Device::Cpu => println!("info: candle backend using cpu"),
        Device::Cuda(_) => println!("info: candle backend using cuda:0"),
        Device::Metal(_) => println!("info: candle backend using mps"),
    }
}

fn auto_device() -> Result<Device> {
    #[cfg(feature = "backend-candle-cuda")]
    {
        if Device::new_cuda(0).is_ok() {
            return Device::new_cuda(0).context("initialize candle cuda device");
        }
    }

    Ok(Device::Cpu)
}

fn explicit_cuda_device() -> Result<Device> {
    #[cfg(feature = "backend-candle-cuda")]
    {
        return Device::new_cuda(0).context("initialize candle cuda device");
    }

    #[cfg(not(feature = "backend-candle-cuda"))]
    {
        bail!(
            "candle cuda support is not compiled into this build; re-run with the 'backend-candle-cuda' Cargo feature"
        )
    }
}
