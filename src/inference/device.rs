use candle_core::{Device, Result};

pub fn select_device(request: &str) -> Result<Device> {
    match request {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            if !cfg!(feature = "cuda") {
                candle_core::bail!("cuda requested but lkjai was built without the cuda feature");
            }
            Device::new_cuda(0)
        }
        "auto" => match Device::new_cuda(0) {
            Ok(device) => Ok(device),
            Err(_) => Ok(Device::Cpu),
        },
        other => {
            candle_core::bail!("unknown INFERENCE_DEVICE={other}; expected cuda, cpu, or auto")
        }
    }
}
