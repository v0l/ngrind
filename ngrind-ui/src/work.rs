use crate::job::{Backend, Handle, Spawn};
use ngrind::Pattern;
use ngrind::backend::Backend as Device;
use ngrind::event::Template;
use secp256k1::Keypair;

fn device(all: &[Device], backend: Backend) -> Option<Device> {
    all.iter().find(|d| d.kind == backend.kind()).cloned()
}

fn signer(secret: Option<[u8; 32]>) -> Option<Keypair> {
    secret
        .and_then(|s| secp256k1::SecretKey::from_secret_bytes(s).ok())
        .map(|sk| Keypair::from_secret_key(&sk))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn vanity(pattern: Pattern, threads: usize, picks: Vec<Device>) -> Spawn {
    Box::new(move |backend, link| {
        let pattern = pattern.clone();
        let picked = device(&picks, backend);
        Handle::spawn(move || {
            let found = |secret: [u8; 32], public: [u8; 32]| link.vanity(secret, public);
            let result = match picked {
                Some(d) => ngrind::backend::vanity(&d, &link, &pattern, threads, &found),
                None => Err(format!("no {} device is available", backend.label())),
            };
            link.finish(result);
        })
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn pow(
    template: Template,
    secret: Option<[u8; 32]>,
    threads: usize,
    picks: Vec<Device>,
) -> Spawn {
    Box::new(move |backend, link| {
        let template = template.clone();
        let picked = device(&picks, backend);
        Handle::spawn(move || {
            let signer = signer(secret);
            let found = |f: ngrind::PowFound| link.pow(f);
            let result = match picked {
                Some(d) => {
                    ngrind::backend::pow(&d, &link, &template, signer.as_ref(), threads, &found)
                }
                None => Err(format!("no {} device is available", backend.label())),
            };
            link.finish(result);
        })
    })
}

#[cfg(target_arch = "wasm32")]
pub fn vanity(pattern: Pattern, _threads: usize, picks: Vec<Device>) -> Spawn {
    Box::new(move |backend, link| {
        let pattern = pattern.clone();
        let picked = device(&picks, backend);
        Handle::spawn_local(async move {
            let found = |secret: [u8; 32], public: [u8; 32]| link.vanity(secret, public);
            let result = match picked {
                Some(d) => ngrind::wgpu::vanity(&link, &pattern, d.index, &found).await,
                None => Err(format!("no {} device is available", backend.label())),
            };
            link.finish(result);
        })
    })
}

#[cfg(target_arch = "wasm32")]
pub fn pow(
    template: Template,
    secret: Option<[u8; 32]>,
    _threads: usize,
    picks: Vec<Device>,
) -> Spawn {
    Box::new(move |backend, link| {
        let template = template.clone();
        let picked = device(&picks, backend);
        Handle::spawn_local(async move {
            let signer = signer(secret);
            let found = |f: ngrind::PowFound| link.pow(f);
            let result = match picked {
                Some(d) => {
                    ngrind::wgpu::pow(&link, &template, signer.as_ref(), d.index, &found).await
                }
                None => Err(format!("no {} device is available", backend.label())),
            };
            link.finish(result);
        })
    })
}
