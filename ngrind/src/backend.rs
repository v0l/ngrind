#[cfg(not(target_arch = "wasm32"))]
use crate::{
    PowFound,
    control::{Control, OnFound},
    event::Template,
    pattern::Pattern,
};
#[cfg(not(target_arch = "wasm32"))]
use secp256k1::Keypair;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cuda,
    Wgpu,
    Cpu,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Cuda => "cuda",
            Kind::Wgpu => "wgpu",
            Kind::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Backend {
    pub kind: Kind,
    pub index: usize,
    pub name: String,
    pub software: bool,
}

impl Backend {
    pub fn label(&self) -> String {
        match self.kind {
            Kind::Cpu => format!("cpu, {}", self.name),
            _ => format!("{}, {}", self.kind.label(), self.name),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn available() -> Vec<Backend> {
    let mut all = Vec::new();
    #[cfg(feature = "cuda")]
    for (i, name) in crate::cuda::devices().into_iter().enumerate() {
        all.push(Backend {
            kind: Kind::Cuda,
            index: i,
            name,
            software: false,
        });
    }
    #[cfg(feature = "wgpu")]
    for (i, info) in pollster::block_on(crate::wgpu::list())
        .into_iter()
        .enumerate()
    {
        all.push(Backend {
            kind: Kind::Wgpu,
            index: i,
            name: info.label(),
            software: info.software,
        });
    }
    all.push(Backend {
        kind: Kind::Cpu,
        index: 0,
        name: format!("{} threads", crate::cpu::cores()),
        software: false,
    });
    all
}

#[cfg(target_arch = "wasm32")]
pub async fn available_web() -> Vec<Backend> {
    let mut all = Vec::new();
    #[cfg(feature = "wgpu")]
    for (i, info) in crate::wgpu::list().await.into_iter().enumerate() {
        all.push(Backend {
            kind: Kind::Wgpu,
            index: i,
            name: info.label(),
            software: info.software,
        });
    }
    all.push(Backend {
        kind: Kind::Cpu,
        index: 0,
        name: format!("{} threads", crate::cpu::cores()),
        software: false,
    });
    all
}

pub fn best(all: &[Backend]) -> Backend {
    all.iter()
        .find(|b| b.kind == Kind::Cuda)
        .or_else(|| all.iter().find(|b| b.kind == Kind::Wgpu && !b.software))
        .or_else(|| all.iter().find(|b| b.kind == Kind::Cpu))
        .or_else(|| all.first())
        .cloned()
        .unwrap_or(Backend {
            kind: Kind::Cpu,
            index: 0,
            name: String::new(),
            software: false,
        })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn vanity(
    backend: &Backend,
    ctl: &dyn Control,
    pattern: &Pattern,
    threads: usize,
    found: OnFound<'_>,
) -> Result<(), String> {
    match backend.kind {
        Kind::Cpu => {
            crate::cpu::vanity(ctl, pattern, threads, found);
            Ok(())
        }
        #[cfg(feature = "cuda")]
        Kind::Cuda => crate::cuda::vanity(ctl, pattern, backend.index as u32, found),
        #[cfg(feature = "wgpu")]
        Kind::Wgpu => crate::wgpu::vanity_blocking(ctl, pattern, backend.index, found),
        #[allow(unreachable_patterns)]
        other => Err(format!("this build has no {} support", other.label())),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn pow(
    backend: &Backend,
    ctl: &dyn Control,
    template: &Template,
    signer: Option<&Keypair>,
    threads: usize,
    found: &(dyn Fn(PowFound) + Sync),
) -> Result<(), String> {
    match backend.kind {
        Kind::Cpu => crate::cpu::pow(ctl, template, signer, threads, found),
        #[cfg(feature = "cuda")]
        Kind::Cuda => crate::cuda::pow(ctl, template, signer, backend.index as u32, found),
        #[cfg(feature = "wgpu")]
        Kind::Wgpu => crate::wgpu::pow_blocking(ctl, template, signer, backend.index, found),
        #[allow(unreachable_patterns)]
        other => Err(format!("this build has no {} support", other.label())),
    }
}
