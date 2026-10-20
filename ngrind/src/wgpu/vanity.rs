use super::vanity_shader::{self as sh, RECORD, U_WORDS, WORKGROUP, out_words};
use super::{Error, Gpu};
use crate::keys;
use crate::pattern::Pattern;
use crate::stepping::{Decoded, Decoder, start_states, table};
#[cfg(test)]
use crate::stepping::{offset_secret, thread_secret};
use futures_channel::oneshot;
use std::sync::Arc;
use web_time::{Duration, Instant};

pub use crate::stepping::Found;

const SLOTS: usize = 2;
const TARGET_DISPATCH: Duration = Duration::from_millis(80);
const MAX_ITERS: u32 = 64;

#[derive(Clone, Debug)]
pub struct VanityConfig {
    pub threads: u32,
    pub half: usize,
    pub fixed_iters: Option<u32>,
    pub max_hits: usize,
}

impl Default for VanityConfig {
    fn default() -> Self {
        VanityConfig {
            threads: 65536,
            half: 16,
            fixed_iters: None,
            max_hits: 8,
        }
    }
}

struct Slot {
    out: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
    advanced: u64,
    keys: u64,
    pending: Option<oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

pub struct WgpuVanity {
    gpu: Arc<Gpu>,
    pipeline: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
    slots: Vec<Slot>,
    decoder: Decoder,
    symbols: Vec<(u32, u32)>,
    config: VanityConfig,
    iters: u32,
    advanced: u64,
    head: usize,
    in_flight: usize,
    last: Instant,
    keys_generated: u64,
    matches_seen: u64,
    rejected: u64,
}

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn storage(read_only: bool, binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl WgpuVanity {
    pub fn new(
        gpu: Arc<Gpu>,
        pattern: &Pattern,
        config: VanityConfig,
    ) -> Result<WgpuVanity, Error> {
        let symbols = pattern.symbols();
        if symbols.is_empty() || symbols.len() > sh::MAX_SYMBOLS {
            return Err(Error::Gpu(format!(
                "a pattern needs 1 to {} characters",
                sh::MAX_SYMBOLS
            )));
        }
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vanity"),
            entries: &[
                storage(true, 0),
                storage(true, 1),
                storage(false, 2),
                storage(false, 3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vanity"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vanity"),
            source: wgpu::ShaderSource::Wgsl(sh::source(config.half, config.max_hits).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vanity"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buffer = |label: &str, size: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let origin = keys::random_secret();
        let uniforms = buffer(
            "uniforms",
            U_WORDS,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let tbl = table(config.half);
        let gt = buffer(
            "table",
            tbl.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        gpu.queue.write_buffer(&gt, 0, &bytes_of(&tbl));
        let state = buffer(
            "state",
            config.threads as usize * 16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        gpu.queue
            .write_buffer(&state, 0, &bytes_of(&start_states(&origin, config.threads)));

        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let out = buffer(
                "out",
                out_words(config.max_hits),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            let read = buffer(
                "read",
                out_words(config.max_hits),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vanity"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                ],
            });
            slots.push(Slot {
                out,
                read,
                bind,
                advanced: 0,
                keys: 0,
                pending: None,
            });
        }
        let config_iters = config.fixed_iters.unwrap_or(1);
        Ok(WgpuVanity {
            gpu,
            pipeline,
            uniforms,
            slots,
            decoder: Decoder {
                origin,
                half: config.half,
                pattern: pattern.clone(),
            },
            symbols,
            config,
            iters: config_iters,
            advanced: 0,
            head: 0,
            in_flight: 0,
            last: Instant::now(),
            keys_generated: 0,
            matches_seen: 0,
            rejected: 0,
        })
    }

    pub fn device_name(&self) -> String {
        self.gpu.info.label()
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

    fn per_iteration(&self) -> u64 {
        self.decoder.per_iteration()
    }

    fn enqueue(&mut self, index: usize) {
        let per = self.per_iteration();
        let mut u = vec![0u32; U_WORDS];
        u[sh::U_THREADS] = self.config.threads;
        u[sh::U_ITERS] = self.iters;
        u[sh::U_BITS] = self.decoder.pattern.symbol_bits();
        u[sh::U_COUNT] = self.symbols.len() as u32;
        for (k, (at, value)) in self.symbols.iter().enumerate() {
            u[sh::U_SYMBOLS + 2 * k] = *at;
            u[sh::U_SYMBOLS + 2 * k + 1] = *value;
        }
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let slot = &mut self.slots[index];
        self.gpu
            .queue
            .write_buffer(&self.uniforms, 0, &bytes_of(&u));
        encoder.clear_buffer(&slot.out, 0, None);
        {
            let groups = self.config.threads.div_ceil(WORKGROUP);
            let x = groups.min(65535);
            let y = groups.div_ceil(x);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &slot.bind, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(
            &slot.out,
            0,
            &slot.read,
            0,
            (out_words(self.config.max_hits) * 4) as u64,
        );
        self.gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = oneshot::channel();
        slot.read.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        slot.pending = Some(rx);
        slot.advanced = self.advanced;
        slot.keys = self.config.threads as u64 * self.iters as u64 * per;
        self.advanced += self.iters as u64;
        self.in_flight += 1;
    }

    pub async fn step(&mut self) -> Result<Vec<Found>, Error> {
        while self.in_flight < SLOTS {
            let index = (self.head + self.in_flight) % SLOTS;
            self.enqueue(index);
        }
        let rx = self.slots[self.head]
            .pending
            .take()
            .expect("an in-flight slot is mapped");
        #[cfg(not(target_arch = "wasm32"))]
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(e.to_string()))?;
        rx.await
            .map_err(|_| Error::Gpu("the GPU dropped a readback".into()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;

        let slot = &self.slots[self.head];
        let words: Vec<u32> = {
            let view = slot.read.slice(..).get_mapped_range();
            let words = view
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("four bytes")))
                .collect();
            drop(view);
            slot.read.unmap();
            words
        };
        let (advanced, keys) = (slot.advanced, slot.keys);
        self.keys_generated += keys;
        self.head = (self.head + 1) % SLOTS;
        self.in_flight -= 1;

        let elapsed = self.last.elapsed();
        self.last = Instant::now();
        if self.config.fixed_iters.is_some() {
        } else if elapsed < TARGET_DISPATCH / 2 && self.iters < MAX_ITERS {
            self.iters *= 2;
        } else if elapsed > TARGET_DISPATCH * 2 && self.iters > 1 {
            self.iters /= 2;
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
                crate::stepping::unlimbs(&record[2..10]),
            ) {
                Decoded::Hit(private, x) => hits.push((private, x)),
                Decoded::Rejected => self.rejected += 1,
            }
        }
        Ok(hits)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn step_blocking(&mut self) -> Result<Vec<Found>, Error> {
        pollster::block_on(self.step())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::pattern::Alphabet;

    fn collect(
        adapter: usize,
        pattern: Pattern,
        want: usize,
        config: VanityConfig,
    ) -> Option<Vec<Found>> {
        let gpu = crate::wgpu::test_gpu(adapter)?;
        let mut miner = WgpuVanity::new(gpu, &pattern, config).unwrap();
        let started = Instant::now();
        let mut found = Vec::new();
        while found.len() < want {
            found.extend(miner.step_blocking().unwrap());
            assert!(
                started.elapsed() < Duration::from_secs(120),
                "{} found of {want}",
                found.len()
            );
        }
        assert_eq!(
            miner.rejected(),
            0,
            "the shader reported keys that do not check out"
        );
        Some(found)
    }

    fn check(pattern: Pattern, want: usize, accept: impl Fn(&str, &str) -> bool) {
        let Some(found) = collect(
            0,
            pattern,
            want,
            VanityConfig {
                threads: 4096,
                half: 8,
                fixed_iters: None,
                max_hits: 8,
            },
        ) else {
            return;
        };
        for (secret, public) in found {
            assert_eq!(keys::public_of(&secret).unwrap(), public);
            let (npub, hex) = (keys::npub(&public), keys::hex(&public));
            assert!(accept(&npub, &hex), "{npub} {hex}");
        }
    }

    #[test]
    fn hex_prefix() {
        check(
            Pattern::parse(Alphabet::Hex, "ab", "").unwrap(),
            20,
            |_, h| h.starts_with("ab"),
        );
    }

    #[test]
    fn hex_prefix_and_suffix() {
        check(
            Pattern::parse(Alphabet::Hex, "a", "1").unwrap(),
            20,
            |_, h| h.starts_with('a') && h.ends_with('1'),
        );
    }

    #[test]
    fn npub_prefix() {
        check(
            Pattern::parse(Alphabet::Npub, "ac", "").unwrap(),
            20,
            |n, _| n.starts_with("npub1ac"),
        );
    }

    #[test]
    fn npub_suffix() {
        check(
            Pattern::parse(Alphabet::Npub, "", "as").unwrap(),
            10,
            |n, _| n[..n.len() - 6].ends_with("as"),
        );
    }

    #[test]
    fn npub_prefix_and_suffix() {
        check(
            Pattern::parse(Alphabet::Npub, "a", "q").unwrap(),
            10,
            |n, _| n.starts_with("npub1a") && n[..n.len() - 6].ends_with('q'),
        );
    }

    fn replay(threads: u32, half: usize, dispatches: i128) {
        let pattern = Pattern::parse(Alphabet::Hex, "a", "").unwrap();
        let Some(gpu) = crate::wgpu::test_gpu(0) else {
            return;
        };
        let mut miner = WgpuVanity::new(
            gpu,
            &pattern,
            VanityConfig {
                threads,
                half,
                fixed_iters: Some(1),
                max_hits: 4096,
            },
        )
        .unwrap();
        let per = 2 * half as i128 + 1;
        for _ in 0..dispatches {
            miner.step_blocking().unwrap();
        }
        let mut cpu_matches = 0u64;
        for d in 0..dispatches {
            for t in 0..threads as u64 {
                let base = thread_secret(&miner.decoder.origin, t).unwrap();
                for k in -(half as i128)..=(half as i128) {
                    let key = offset_secret(&base, d * per + k).unwrap();
                    if pattern.matches(&keys::public_of(&key.to_secret_bytes()).unwrap()) {
                        cpu_matches += 1;
                    }
                }
            }
        }
        assert_eq!(
            miner.matches_seen(),
            cpu_matches,
            "{threads} threads, half {half}: the shader and the CPU disagree about the same keys"
        );
    }

    #[test]
    fn the_cpu_replays_every_dispatch() {
        for half in [1, 2, 4, 8, 16, 32] {
            replay(256, half, 3);
        }
        replay(64, 64, 2);
        replay(64, 127, 1);
    }

    #[test]
    fn every_key_is_checked_exactly_once() {
        let pattern = Pattern::parse(Alphabet::Hex, "abcde", "").unwrap();
        let Some(gpu) = crate::wgpu::test_gpu(0) else {
            return;
        };
        let mut miner = WgpuVanity::new(
            gpu,
            &pattern,
            VanityConfig {
                threads: 65536,
                half: 32,
                fixed_iters: None,
                max_hits: 8,
            },
        )
        .unwrap();
        while miner.keys_generated() < 16_000_000_000 {
            miner.step_blocking().unwrap();
        }
        let expected = miner.keys_generated() as f64 / pattern.expected();
        let seen = miner.matches_seen() as f64;
        let sigma = expected.sqrt();
        assert!(
            (seen - expected).abs() < 5.0 * sigma,
            "saw {seen} matches, expected {expected:.0} +- {sigma:.0}"
        );
    }

    #[test]
    fn software_adapter_agrees() {
        let infos = pollster::block_on(super::super::list());
        if let Some(i) = infos.iter().position(|i| i.software) {
            let found = collect(
                i,
                Pattern::parse(Alphabet::Hex, "a", "").unwrap(),
                4,
                VanityConfig {
                    threads: 256,
                    half: 4,
                    fixed_iters: None,
                    max_hits: 8,
                },
            )
            .expect("the adapter was just listed");
            for (secret, public) in found {
                assert_eq!(keys::public_of(&secret).unwrap(), public);
            }
        }
    }
}
