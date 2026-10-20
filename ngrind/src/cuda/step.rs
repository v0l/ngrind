use super::{Device, Error};
use crate::pattern::Pattern;
use crate::stepping::{Decoded, Decoder, start_states, table, unlimbs};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use web_time::{Duration, Instant};

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/step.fatbin"));

const THREADS: u32 = 128;
const MAX_SYMBOLS: usize = 52;
const RECORD: usize = 10;
const SLOTS: usize = 2;
const TARGET_DISPATCH: Duration = Duration::from_millis(80);
const MAX_ITERS: u32 = 64;
pub const HALVES: [usize; 5] = [4, 8, 16, 32, 64];

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    threads: u32,
    iters: u32,
    bits: u32,
    count: u32,
    max_hits: u32,
    pad: u32,
    at: [u32; MAX_SYMBOLS],
    want: [u32; MAX_SYMBOLS],
}

unsafe impl DeviceRepr for Params {}

#[derive(Clone, Debug)]
pub struct StepConfig {
    pub threads: u32,
    pub half: usize,
    pub fixed_iters: Option<u32>,
    pub max_hits: usize,
}

impl Default for StepConfig {
    fn default() -> Self {
        StepConfig {
            threads: 0,
            half: 32,
            fixed_iters: None,
            max_hits: 64,
        }
    }
}

struct Slot {
    out: CudaSlice<u32>,
    host: PinnedHostSlice<u32>,
    done: CudaEvent,
    advanced: u64,
    keys: u64,
}

pub struct StepVanity {
    dev: Device,
    func: CudaFunction,
    table: CudaSlice<u64>,
    state: CudaSlice<u64>,
    slots: Vec<Slot>,
    decoder: Decoder,
    symbols: Vec<(u32, u32)>,
    config: StepConfig,
    iters: u32,
    advanced: u64,
    head: usize,
    in_flight: usize,
    last: Instant,
    keys_generated: u64,
    matches_seen: u64,
    rejected: u64,
}

fn as_u64(words: &[u32]) -> Vec<u64> {
    words
        .chunks_exact(2)
        .map(|w| w[0] as u64 | (w[1] as u64) << 32)
        .collect()
}

impl StepVanity {
    pub fn new(
        pattern: &Pattern,
        device: u32,
        mut config: StepConfig,
    ) -> Result<StepVanity, Error> {
        if !HALVES.contains(&config.half) {
            return Err(Error::Gpu(format!("half must be one of {HALVES:?}")));
        }
        let symbols = pattern.symbols();
        if symbols.is_empty() || symbols.len() > MAX_SYMBOLS {
            return Err(Error::Gpu(format!(
                "a pattern needs 1 to {MAX_SYMBOLS} characters"
            )));
        }
        let dev = Device::open(device, FATBIN)?;
        unsafe { dev.ctx.disable_event_tracking() };
        let func = dev
            .module
            .load_function(&format!("ngrind_step_{}", config.half))?;
        if config.threads == 0 {
            let blocks = func
                .occupancy_max_active_blocks_per_multiprocessor(THREADS, 0, None)?
                .max(1);
            config.threads = dev.sm_count * blocks * THREADS * 4;
        }
        let stream = dev.stream.clone();
        let origin = crate::keys::random_secret();
        let table = stream.clone_htod(&as_u64(&table(config.half)))?;
        let state = stream.clone_htod(&as_u64(&start_states(&origin, config.threads)))?;
        let words = 1 + config.max_hits * RECORD;
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let out = stream.alloc_zeros::<u32>(words)?;
            let host = unsafe { dev.ctx.alloc_pinned_with_flags::<u32>(words, 0) }?;
            let done = dev.ctx.new_event(None)?;
            slots.push(Slot {
                out,
                host,
                done,
                advanced: 0,
                keys: 0,
            });
        }
        stream.synchronize()?;
        let iters = config.fixed_iters.unwrap_or(1);
        Ok(StepVanity {
            dev,
            func,
            table,
            state,
            slots,
            decoder: Decoder {
                origin,
                half: config.half,
                pattern: pattern.clone(),
            },
            symbols,
            config,
            iters,
            advanced: 0,
            head: 0,
            in_flight: 0,
            last: Instant::now(),
            keys_generated: 0,
            matches_seen: 0,
            rejected: 0,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.dev.name
    }

    pub fn keys_generated(&self) -> u64 {
        self.keys_generated
    }

    pub fn matches_seen(&self) -> u64 {
        self.matches_seen
    }

    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    fn enqueue(&mut self, index: usize) -> Result<(), Error> {
        let mut params = Params {
            threads: self.config.threads,
            iters: self.iters,
            bits: self.decoder.pattern.symbol_bits(),
            count: self.symbols.len() as u32,
            max_hits: self.config.max_hits as u32,
            pad: 0,
            at: [0; MAX_SYMBOLS],
            want: [0; MAX_SYMBOLS],
        };
        for (k, (at, want)) in self.symbols.iter().enumerate() {
            params.at[k] = *at;
            params.want[k] = *want;
        }
        let stream = self.dev.stream.clone();
        let slot = &mut self.slots[index];
        stream.memset_zeros(&mut slot.out)?;
        let cfg = LaunchConfig {
            grid_dim: (self.config.threads.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.func);
        launch
            .arg(&params)
            .arg(&self.table)
            .arg(&mut self.state)
            .arg(&mut slot.out);
        unsafe { launch.launch(cfg) }
            .map_err(|e| Error::Gpu(format!("kernel launch failed: {e}")))?;
        stream.memcpy_dtoh(&slot.out, &mut slot.host)?;
        slot.done.record(&stream)?;
        slot.advanced = self.advanced;
        slot.keys = self.config.threads as u64 * self.iters as u64 * self.decoder.per_iteration();
        self.advanced += self.iters as u64;
        self.in_flight += 1;
        Ok(())
    }

    pub fn step(&mut self) -> Result<Vec<crate::stepping::Found>, Error> {
        while self.in_flight < SLOTS {
            let index = (self.head + self.in_flight) % SLOTS;
            self.enqueue(index)?;
        }
        let slot = &self.slots[self.head];
        slot.done.synchronize()?;
        let words: Vec<u32> =
            unsafe { std::slice::from_raw_parts(slot.host.as_ptr()?, slot.host.len()) }.to_vec();
        let (advanced, keys) = (slot.advanced, slot.keys);
        self.keys_generated += keys;
        self.head = (self.head + 1) % SLOTS;
        self.in_flight -= 1;

        let elapsed = self.last.elapsed();
        self.last = Instant::now();
        if self.config.fixed_iters.is_none() {
            if elapsed < TARGET_DISPATCH / 2 && self.iters < MAX_ITERS {
                self.iters *= 2;
            } else if elapsed > TARGET_DISPATCH * 2 && self.iters > 1 {
                self.iters /= 2;
            }
        }

        let found = words[0] as usize;
        self.matches_seen += found as u64;
        let mut hits = Vec::new();
        for k in 0..found.min(self.config.max_hits) {
            let record = &words[1 + k * RECORD..1 + (k + 1) * RECORD];
            match self.decoder.decode(
                advanced,
                record[0] as u64,
                record[1],
                unlimbs(&record[2..10]),
            ) {
                Decoded::Hit(private, x) => hits.push((private, x)),
                Decoded::Rejected => self.rejected += 1,
            }
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys;
    use crate::pattern::Alphabet;
    use crate::stepping::{offset_secret, thread_secret};

    fn have_gpu() -> bool {
        if super::super::device_count() == 0 {
            eprintln!("skipped: no CUDA device");
            return false;
        }
        true
    }

    fn replay(threads: u32, half: usize, dispatches: i128) {
        let pattern = Pattern::parse(Alphabet::Hex, "a", "").unwrap();
        let config = StepConfig {
            threads,
            half,
            fixed_iters: Some(1),
            max_hits: 4096,
        };
        let mut miner = StepVanity::new(&pattern, 0, config).unwrap();
        for _ in 0..dispatches {
            miner.step().unwrap();
        }
        let per = 2 * half as i128 + 1;
        let mut cpu = 0u64;
        for d in 0..dispatches {
            for t in 0..threads as u64 {
                let base = thread_secret(&miner.decoder.origin, t).unwrap();
                for k in -(half as i128)..=(half as i128) {
                    let key = offset_secret(&base, d * per + k).unwrap();
                    if pattern.matches(&keys::public_of(&key.to_secret_bytes()).unwrap()) {
                        cpu += 1;
                    }
                }
            }
        }
        assert_eq!(miner.matches_seen(), cpu, "{threads} threads, half {half}");
        assert_eq!(miner.rejected(), 0);
    }

    #[test]
    fn the_cpu_replays_every_dispatch() {
        if !have_gpu() {
            return;
        }
        for half in HALVES {
            replay(256, half, 3);
        }
    }

    #[test]
    fn finds_npub_and_hex_patterns() {
        if !have_gpu() {
            return;
        }
        type Accept = fn(&str, &str) -> bool;
        let cases: [(Alphabet, &str, &str, Accept); 4] = [
            (Alphabet::Hex, "abc", "", |_, h| h.starts_with("abc")),
            (Alphabet::Hex, "a", "1", |_, h| {
                h.starts_with('a') && h.ends_with('1')
            }),
            (Alphabet::Npub, "acd", "", |n, _| n.starts_with("npub1acd")),
            (Alphabet::Npub, "a", "q", |n, _| {
                n.starts_with("npub1a") && n[..n.len() - 6].ends_with('q')
            }),
        ];
        for (alphabet, prefix, suffix, accept) in cases {
            let pattern = Pattern::parse(alphabet, prefix, suffix).unwrap();
            let mut miner = StepVanity::new(&pattern, 0, StepConfig::default()).unwrap();
            let started = Instant::now();
            let mut found = Vec::new();
            while found.len() < 5 {
                found.extend(miner.step().unwrap());
                assert!(started.elapsed() < Duration::from_secs(60));
            }
            assert_eq!(miner.rejected(), 0);
            for (secret, public) in found {
                assert_eq!(keys::public_of(&secret).unwrap(), public);
                let (n, h) = (keys::npub(&public), keys::hex(&public));
                assert!(accept(&n, &h), "{n} {h}");
            }
        }
    }

    #[test]
    fn every_key_is_checked_exactly_once() {
        if !have_gpu() {
            return;
        }
        let pattern = Pattern::parse(Alphabet::Hex, "abcde", "").unwrap();
        let mut miner = StepVanity::new(&pattern, 0, StepConfig::default()).unwrap();
        while miner.keys_generated() < 16_000_000_000 {
            miner.step().unwrap();
        }
        let expected = miner.keys_generated() as f64 / pattern.expected();
        let seen = miner.matches_seen() as f64;
        assert!(
            (seen - expected).abs() < 5.0 * expected.sqrt(),
            "saw {seen}, expected {expected:.0}"
        );
    }
}
