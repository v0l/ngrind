#![cfg(target_arch = "wasm32")]

use ngrind::control::{Control, Phase};
use ngrind::event::Template;
use ngrind::{Alphabet, Pattern, keys};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

struct Progress {
    done: AtomicU64,
    stop: AtomicBool,
    log: Mutex<Vec<String>>,
    limit: usize,
    found: AtomicUsize,
}

impl Control for Progress {
    fn add(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn phase(&self, _: Phase) {}

    fn log(&self, text: &str) {
        self.log
            .lock()
            .expect("single threaded")
            .push(text.to_string());
    }
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Job {
    progress: Arc<Progress>,
    results: Arc<Mutex<Vec<String>>>,
    error: Arc<Mutex<Option<String>>>,
    finished: Arc<AtomicBool>,
}

#[wasm_bindgen]
impl Job {
    pub fn done(&self) -> f64 {
        self.progress.done.load(Ordering::Relaxed) as f64
    }

    pub fn stop(&self) {
        self.progress.stop.store(true, Ordering::Relaxed);
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().expect("single threaded").clone()
    }

    pub fn results(&self) -> Vec<String> {
        self.results.lock().expect("single threaded").clone()
    }

    pub fn log(&self) -> Vec<String> {
        self.progress.log.lock().expect("single threaded").clone()
    }
}

fn job(limit: usize) -> (Job, Arc<Progress>) {
    let progress = Arc::new(Progress {
        done: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        log: Mutex::new(Vec::new()),
        limit,
        found: AtomicUsize::new(0),
    });
    let job = Job {
        progress: progress.clone(),
        results: Arc::new(Mutex::new(Vec::new())),
        error: Arc::new(Mutex::new(None)),
        finished: Arc::new(AtomicBool::new(false)),
    };
    (job, progress)
}

#[wasm_bindgen]
pub async fn adapters() -> Vec<String> {
    ngrind::wgpu::list()
        .await
        .into_iter()
        .map(|i| i.label())
        .collect()
}

#[wasm_bindgen]
pub fn vanity(prefix: String, suffix: String, hex: bool, limit: usize) -> Result<Job, JsError> {
    let alphabet = if hex { Alphabet::Hex } else { Alphabet::Npub };
    let pattern = Pattern::parse(alphabet, &prefix, &suffix).map_err(|e| JsError::new(&e))?;
    let (handle, progress) = job(limit.max(1));
    let (results, error, finished) = (
        handle.results.clone(),
        handle.error.clone(),
        handle.finished.clone(),
    );
    wasm_bindgen_futures::spawn_local(async move {
        let found = |secret: [u8; 32], public: [u8; 32]| {
            if progress.found.fetch_add(1, Ordering::Relaxed) >= progress.limit {
                return;
            }
            results.lock().expect("single threaded").push(format!(
                "{}\n{}",
                keys::npub(&public),
                keys::nsec(&secret)
            ));
            if progress.found.load(Ordering::Relaxed) >= progress.limit {
                progress.stop.store(true, Ordering::Relaxed);
            }
        };
        if let Err(e) = ngrind::wgpu::vanity(progress.as_ref(), &pattern, 0, &found).await {
            *error.lock().expect("single threaded") = Some(e);
        }
        finished.store(true, Ordering::Relaxed);
    });
    Ok(handle)
}

#[wasm_bindgen]
pub fn pow(
    pubkey: String,
    content: String,
    kind: u16,
    difficulty: u32,
    limit: usize,
) -> Result<Job, JsError> {
    let public = keys::parse_public(&pubkey).map_err(|e| JsError::new(&e))?;
    let template = Template {
        pubkey: keys::hex(&public),
        kind,
        tags: Vec::new(),
        content,
        difficulty,
        fast: false,
    };
    let (handle, progress) = job(limit.max(1));
    let (results, error, finished) = (
        handle.results.clone(),
        handle.error.clone(),
        handle.finished.clone(),
    );
    wasm_bindgen_futures::spawn_local(async move {
        let found = |f: ngrind::PowFound| {
            if progress.found.fetch_add(1, Ordering::Relaxed) >= progress.limit {
                return;
            }
            results.lock().expect("single threaded").push(f.event);
            if progress.found.load(Ordering::Relaxed) >= progress.limit {
                progress.stop.store(true, Ordering::Relaxed);
            }
        };
        if let Err(e) = ngrind::wgpu::pow(progress.as_ref(), &template, None, 0, &found).await {
            *error.lock().expect("single threaded") = Some(e);
        }
        finished.store(true, Ordering::Relaxed);
    });
    Ok(handle)
}
