use crate::job::{Backend, Link, Spawn};
use ngrind::event::Template;
use ngrind::{Pattern, cpu, keys};
use secp256k1::Keypair;

#[derive(Clone)]
pub struct Cpu {
    pub threads: usize,
}

#[derive(Clone)]
pub struct Gpu {
    pub device: u32,
}

pub fn gpu_devices() -> Vec<String> {
    #[cfg(feature = "cuda")]
    {
        ngrind::cuda::devices()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

pub fn vanity(pattern: Pattern, cpu: Cpu, gpu: Gpu) -> Spawn {
    Box::new(move |backend, link| {
        let pattern = pattern.clone();
        let (cpu, gpu) = (cpu.clone(), gpu.clone());
        std::thread::spawn(move || {
            let found = |secret: [u8; 32], public: [u8; 32]| link.vanity(secret, public);
            let result = match backend {
                Backend::Cpu => {
                    cpu::vanity(&link, &pattern, cpu.threads, &found);
                    Ok(())
                }
                Backend::Gpu => gpu_vanity(&link, &pattern, gpu.device, &found),
            };
            link.finish(result);
        })
    })
}

#[cfg(feature = "cuda")]
fn gpu_vanity(
    link: &Link,
    pattern: &Pattern,
    device: u32,
    found: ngrind::OnFound,
) -> Result<(), String> {
    ngrind::cuda::vanity(link, pattern, device, found)
}

#[cfg(not(feature = "cuda"))]
fn gpu_vanity(_: &Link, _: &Pattern, _: u32, _: ngrind::OnFound) -> Result<(), String> {
    Err("this build has no GPU support".into())
}

pub fn pow(template: Template, secret: Option<[u8; 32]>, cpu: Cpu, gpu: Gpu) -> Spawn {
    Box::new(move |backend, link| {
        let template = template.clone();
        let (cpu, gpu) = (cpu.clone(), gpu.clone());
        std::thread::spawn(move || {
            let signer = secret
                .and_then(|s| secp256k1::SecretKey::from_byte_array(s).ok())
                .map(|sk| Keypair::from_secret_key(keys::secp(), &sk));
            let found = |f: ngrind::PowFound| link.pow(f);
            let result = match backend {
                Backend::Cpu => cpu::pow(&link, &template, signer.as_ref(), cpu.threads, &found),
                Backend::Gpu => gpu_pow(&link, &template, signer.as_ref(), gpu.device, &found),
            };
            link.finish(result);
        })
    })
}

#[cfg(feature = "cuda")]
fn gpu_pow(
    link: &Link,
    template: &Template,
    signer: Option<&Keypair>,
    device: u32,
    found: &(dyn Fn(ngrind::PowFound) + Sync),
) -> Result<(), String> {
    ngrind::cuda::pow(link, template, signer, device, found)
}

#[cfg(not(feature = "cuda"))]
fn gpu_pow(
    _: &Link,
    _: &Template,
    _: Option<&Keypair>,
    _: u32,
    _: &(dyn Fn(ngrind::PowFound) + Sync),
) -> Result<(), String> {
    Err("this build has no GPU support".into())
}
