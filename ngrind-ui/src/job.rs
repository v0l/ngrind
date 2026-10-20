use ngrind::{Control, Phase};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const WINDOW: f64 = 8.0;
const SAMPLE_EVERY: Duration = Duration::from_millis(500);
const HISTORY: usize = 180;
const LOG_LINES: usize = 200;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Backend {
    Gpu,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Gpu => "gpu",
            Backend::Cpu => "cpu",
        }
    }

    pub fn slot(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Kind {
    Npub,
    Event,
}

#[derive(Clone, Debug)]
pub struct VanityFound {
    pub secret: [u8; 32],
    pub public: [u8; 32],
    pub after: u64,
}

#[derive(Clone, Debug)]
pub struct PowHit {
    pub found: ngrind::PowFound,
    pub after: u64,
}

pub enum Msg {
    Phase(Phase),
    Log(String),
    Vanity(VanityFound),
    Pow(PowHit),
    Failed(String),
    Ended,
}

pub struct Shared {
    done: AtomicU64,
    stop: AtomicBool,
    found: AtomicUsize,
    limit: usize,
}

#[derive(Clone)]
pub struct Link {
    tx: Sender<Msg>,
    ctx: egui::Context,
    shared: Arc<Shared>,
}

impl Control for Link {
    fn add(&self, n: u64) {
        Link::add(self, n);
    }

    fn stopped(&self) -> bool {
        Link::stopped(self)
    }

    fn phase(&self, phase: Phase) {
        Link::phase(self, phase);
    }

    fn log(&self, text: &str) {
        Link::log(self, text);
    }
}

impl Link {
    pub fn pair(ctx: &egui::Context, limit: usize) -> (Link, Receiver<Msg>) {
        let (tx, rx) = channel();
        let shared = Arc::new(Shared {
            done: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            found: AtomicUsize::new(0),
            limit,
        });
        (
            Link {
                tx,
                ctx: ctx.clone(),
                shared,
            },
            rx,
        )
    }

    fn send(&self, msg: Msg) {
        let _ = self.tx.send(msg);
        self.ctx.request_repaint();
    }

    pub fn phase(&self, phase: Phase) {
        self.send(Msg::Phase(phase));
    }

    pub fn log(&self, text: impl Into<String>) {
        self.send(Msg::Log(text.into()));
    }

    pub fn add(&self, n: u64) {
        self.shared.done.fetch_add(n, Ordering::Relaxed);
    }

    pub fn done(&self) -> u64 {
        self.shared.done.load(Ordering::Relaxed)
    }

    pub fn stopped(&self) -> bool {
        self.shared.stop.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    fn counted(&self) -> bool {
        let before = self.shared.found.fetch_add(1, Ordering::Relaxed);
        if before + 1 >= self.shared.limit {
            self.stop();
        }
        if before >= self.shared.limit {
            self.shared.found.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    pub fn vanity(&self, secret: [u8; 32], public: [u8; 32]) {
        if self.counted() {
            self.send(Msg::Vanity(VanityFound {
                secret,
                public,
                after: self.done(),
            }));
        }
    }

    pub fn pow(&self, found: ngrind::PowFound) {
        if self.counted() {
            self.send(Msg::Pow(PowHit {
                found,
                after: self.done(),
            }));
        }
    }

    pub fn finish(&self, result: Result<(), String>) {
        if let Err(e) = result {
            self.send(Msg::Failed(e));
        }
        self.send(Msg::Ended);
    }
}

pub type Spawn = Box<dyn Fn(Backend, Link) -> JoinHandle<()>>;

pub struct Run {
    pub kind: Kind,
    pub backend: Backend,
    pub target: String,
    pub expected: f64,
    pub started: Instant,
    pub ended: Option<Instant>,
    pub phase: Phase,
    pub error: Option<String>,
    pub rate: f64,
    pub history: Vec<f32>,
    pub vanity: Vec<VanityFound>,
    pub pow: Vec<PowHit>,
    pub log: Vec<(f64, String)>,
    pub saved_vanity: usize,
    pub saved_pow: usize,
    link: Link,
    rx: Receiver<Msg>,
    spawn: Spawn,
    handle: Option<JoinHandle<()>>,
    next: Option<Backend>,
    banked: f64,
    since: Option<Instant>,
    window: VecDeque<(f64, u64)>,
    last_sample: Instant,
}

impl Run {
    pub fn start(
        ctx: &egui::Context,
        kind: Kind,
        backend: Backend,
        target: String,
        expected: f64,
        limit: usize,
        spawn: Spawn,
    ) -> Run {
        let (link, rx) = Link::pair(ctx, limit);
        let handle = spawn(backend, link.clone());
        let now = Instant::now();
        let mut run = Run {
            kind,
            backend,
            target,
            expected,
            started: now,
            ended: None,
            phase: Phase::Setup,
            error: None,
            rate: 0.0,
            history: Vec::new(),
            vanity: Vec::new(),
            pow: Vec::new(),
            log: Vec::new(),
            saved_vanity: 0,
            saved_pow: 0,
            link,
            rx,
            spawn,
            handle: Some(handle),
            next: None,
            banked: 0.0,
            since: None,
            window: VecDeque::new(),
            last_sample: now,
        };
        run.note(format!("started on {}", backend.label()));
        run
    }

    pub fn live(&self) -> bool {
        self.ended.is_none()
    }

    pub fn switching(&self) -> Option<Backend> {
        self.next
    }

    pub fn stop(&self) {
        self.link.stop();
    }

    pub fn shutdown(&mut self) {
        self.next = None;
        self.link.stop();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    pub fn switch(&mut self, backend: Backend) {
        if !self.live() {
            return;
        }
        if backend == self.backend || self.next.is_some() {
            return;
        }
        self.next = Some(backend);
        self.note(format!("switching to {}", backend.label()));
        self.link.stop();
    }

    pub fn done(&self) -> u64 {
        self.link.done()
    }

    pub fn elapsed(&self) -> f64 {
        self.banked + self.since.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0)
    }

    pub fn average(&self) -> f64 {
        let t = self.elapsed();
        if t > 0.0 { self.done() as f64 / t } else { 0.0 }
    }

    pub fn found(&self) -> usize {
        self.vanity.len() + self.pow.len()
    }

    fn note(&mut self, text: impl Into<String>) {
        let t = self.started.elapsed().as_secs_f64();
        self.log.push((t, text.into()));
        if self.log.len() > LOG_LINES {
            self.log.remove(0);
        }
    }

    fn bank(&mut self) {
        if let Some(s) = self.since.take() {
            self.banked += s.elapsed().as_secs_f64();
        }
    }

    fn limit_hit(&self) -> bool {
        self.link.shared.found.load(Ordering::Relaxed) >= self.link.shared.limit
    }

    pub fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Phase(p) => {
                    self.phase = p;
                    match p {
                        Phase::Search => {
                            self.since.get_or_insert_with(Instant::now);
                        }
                        Phase::Setup => self.bank(),
                    }
                }
                Msg::Log(t) => self.note(t),
                Msg::Vanity(f) => {
                    self.note("match found");
                    self.vanity.push(f);
                }
                Msg::Pow(f) => {
                    self.note(format!(
                        "event found with {} leading zero bits",
                        f.found.bits
                    ));
                    self.pow.push(f);
                }
                Msg::Failed(e) => {
                    self.note(format!("failed: {e}"));
                    self.error = Some(e);
                }
                Msg::Ended => self.worker_ended(),
            }
        }
        if self.ended.is_none() && self.handle.as_ref().is_some_and(|h| h.is_finished()) {
            let mut seen = false;
            while let Ok(msg) = self.rx.try_recv() {
                if let Msg::Ended = msg {
                    seen = true;
                }
            }
            if seen {
                self.worker_ended();
            } else {
                self.error = Some("the worker stopped unexpectedly".into());
                self.finish();
            }
        }
        self.sample();
    }

    fn worker_ended(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        match self.next.take() {
            Some(next) if self.error.is_none() && !self.limit_hit() => {
                self.bank();
                self.backend = next;
                self.phase = Phase::Setup;
                self.window.clear();
                self.rate = 0.0;
                self.link.shared.stop.store(false, Ordering::Relaxed);
                self.handle = Some((self.spawn)(next, self.link.clone()));
                self.note(format!("running on {}", next.label()));
            }
            _ => self.finish(),
        }
    }

    fn finish(&mut self) {
        if self.ended.is_some() {
            return;
        }
        self.bank();
        self.ended = Some(Instant::now());
        self.rate = 0.0;
        let how = if self.error.is_some() {
            "stopped on an error"
        } else if self.limit_hit() {
            "finished"
        } else {
            "stopped"
        };
        self.note(how);
    }

    fn sample(&mut self) {
        if self.ended.is_some() || self.since.is_none() {
            return;
        }
        if self.last_sample.elapsed() < SAMPLE_EVERY {
            return;
        }
        self.last_sample = Instant::now();
        let t = self.elapsed();
        self.window.push_back((t, self.done()));
        while self.window.len() > 2 && t - self.window[0].0 > WINDOW {
            self.window.pop_front();
        }
        if let (Some(first), Some(last)) = (self.window.front(), self.window.back()) {
            let dt = last.0 - first.0;
            if dt > 0.0 {
                self.rate = (last.1 - first.1) as f64 / dt;
            }
        }
        self.history.push(self.rate as f32);
        if self.history.len() > HISTORY {
            self.history.remove(0);
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::{self, Cpu, Gpu};
    use ngrind::{Alphabet, Pattern};

    fn wait(run: &mut Run, mut until: impl FnMut(&Run) -> bool) {
        let started = Instant::now();
        while !until(run) {
            run.poll();
            assert!(started.elapsed() < Duration::from_secs(120), "timed out");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn long_search() -> Spawn {
        let pattern = Pattern::parse(Alphabet::Hex, "", "0123456789").unwrap();
        work::vanity(pattern, Cpu { threads: 2 }, Gpu { device: 0 })
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn a_run_moves_between_devices_and_keeps_counting() {
        let ctx = egui::Context::default();
        let mut run = Run::start(
            &ctx,
            Kind::Npub,
            Backend::Cpu,
            "x".into(),
            1e12,
            1,
            long_search(),
        );
        wait(&mut run, |r| r.done() > 100_000);
        let before = run.done();
        run.switch(Backend::Gpu);
        assert_eq!(run.switching(), Some(Backend::Gpu));
        wait(&mut run, |r| {
            r.backend == Backend::Gpu && r.phase == Phase::Search && r.done() > before + 10_000_000
        });
        assert!(run.live() && run.error.is_none(), "{:?}", run.error);
        run.switch(Backend::Cpu);
        wait(&mut run, |r| {
            r.backend == Backend::Cpu && r.phase == Phase::Search
        });
        let after = run.done();
        wait(&mut run, |r| r.done() > after + 100_000);
        run.shutdown();
        run.poll();
        assert!(run.error.is_none());
    }

    #[test]
    fn stopping_ends_the_run() {
        let ctx = egui::Context::default();
        let mut run = Run::start(
            &ctx,
            Kind::Npub,
            Backend::Cpu,
            "x".into(),
            1e12,
            1,
            long_search(),
        );
        wait(&mut run, |r| r.done() > 0);
        run.stop();
        wait(&mut run, |r| !r.live());
        assert!(run.ended.is_some() && run.found() == 0);
    }
}
