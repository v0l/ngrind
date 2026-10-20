use super::Error;
use cudarc::driver::{CudaContext, CudaModule, CudaStream, sys};
use std::sync::Arc;

pub struct Device {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub name: String,
    pub sm_count: u32,
    pub ordinal: u32,
}

pub fn device_count() -> u32 {
    let present = std::panic::catch_unwind(|| unsafe { cudarc::driver::sys::is_culib_present() })
        .unwrap_or(false);
    if !present {
        return 0;
    }
    std::panic::catch_unwind(|| CudaContext::device_count().unwrap_or(0).max(0) as u32).unwrap_or(0)
}

pub fn device_name(index: u32) -> String {
    std::panic::catch_unwind(|| {
        CudaContext::new(index as usize)
            .and_then(|c| c.name())
            .unwrap_or_default()
    })
    .unwrap_or_default()
}

pub fn devices() -> Vec<String> {
    (0..device_count()).map(device_name).collect()
}

impl Device {
    pub fn open(ordinal: u32, fatbin: &'static [u8]) -> Result<Device, Error> {
        if ordinal >= device_count() {
            return Err(Error::NoDevice(ordinal));
        }
        let ctx = CudaContext::new(ordinal as usize)?;
        let stream = ctx.new_stream()?;
        let module = ctx.load_module(cudarc::nvrtc::Ptx::from_binary(fatbin.to_vec()))?;
        let name = ctx.name()?;
        let sm_count = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            as u32;
        Ok(Device {
            ctx,
            stream,
            module,
            name,
            sm_count,
            ordinal,
        })
    }
}
