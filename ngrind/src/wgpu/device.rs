use super::Error;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub software: bool,
}

impl GpuInfo {
    pub fn label(&self) -> String {
        if self.software {
            format!("{} ({}, software)", self.name, self.backend)
        } else {
            format!("{} ({})", self.name, self.backend)
        }
    }
}

pub struct Gpu {
    #[cfg(test)]
    _exclusive: std::sync::MutexGuard<'static, ()>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
}

fn describe(adapter: &wgpu::Adapter) -> GpuInfo {
    let info = adapter.get_info();
    GpuInfo {
        name: info.name,
        backend: format!("{:?}", info.backend),
        software: info.device_type == wgpu::DeviceType::Cpu,
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut all = instance.enumerate_adapters(wgpu::Backends::all()).await;
    all.sort_by_key(|a| {
        let info = a.get_info();
        let rank = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        };
        let api = match info.backend {
            wgpu::Backend::Vulkan => 0,
            wgpu::Backend::Metal => 0,
            wgpu::Backend::Dx12 => 1,
            _ => 2,
        };
        (rank, api)
    });
    all
}

#[cfg(target_arch = "wasm32")]
async fn adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut last = String::from("this browser has no WebGPU");
    for preference in [
        wgpu::PowerPreference::HighPerformance,
        wgpu::PowerPreference::None,
    ] {
        let options = wgpu::RequestAdapterOptions {
            power_preference: preference,
            ..Default::default()
        };
        match instance.request_adapter(&options).await {
            Ok(adapter) => return vec![adapter],
            Err(e) => last = e.to_string(),
        }
    }
    web_sys::console::error_1(&format!("request_adapter: {last}").into());
    Vec::new()
}

pub async fn list() -> Vec<GpuInfo> {
    adapters().await.iter().map(describe).collect()
}

impl Gpu {
    pub async fn open(index: usize) -> Result<Gpu, Error> {
        #[cfg(test)]
        let exclusive = {
            static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            LOCK.lock().unwrap_or_else(|e| e.into_inner())
        };
        let all = adapters().await;
        let adapter = all
            .into_iter()
            .nth(index)
            .ok_or_else(|| Error::NoAdapter(format!("adapter {index} does not exist")))?;
        let info = describe(&adapter);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ngrind"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Device(e.to_string()))?;
        Ok(Gpu {
            #[cfg(test)]
            _exclusive: exclusive,
            device,
            queue,
            info,
        })
    }
}
