use super::pow_shader::{self as sh, HEADER, OUT_WORDS, WORKGROUP};
use super::{Error, Gpu};
use crate::powlayout::{LOW_DIGITS, NONCES_PER_UNIT, Plan, PowHit, UNITS_PER_HIGH};
use crate::sha;
use futures_channel::oneshot;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use web_time::{Duration, Instant};

const SLOTS: usize = 2;
const FIRST_LAUNCH: u32 = 1 << 10;
const MAX_LAUNCH: u32 = 1 << 22;
const TARGET_DISPATCH: Duration = Duration::from_millis(80);
const BASE: u64 = 100_000_000;

struct Slot {
    params: wgpu::Buffer,
    out: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
    units: u64,
    mapped: Option<oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

struct State {
    plan: Plan,
    target: u32,
    cursor: u64,
    slots: Vec<Slot>,
    head: usize,
    in_flight: usize,
    launch: u32,
    last: Instant,
    pipelines: HashMap<(usize, usize, usize, usize), wgpu::ComputePipeline>,
}

pub struct WgpuPow {
    gpu: Arc<Gpu>,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    state: Option<State>,
    attempts: u64,
}

impl WgpuPow {
    pub fn new(gpu: Arc<Gpu>) -> WgpuPow {
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pow"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pow"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        WgpuPow {
            gpu,
            layout,
            pipeline_layout,
            state: None,
            attempts: 0,
        }
    }

    pub fn device_name(&self) -> String {
        self.gpu.info.label()
    }

    pub fn attempts(&self) -> u64 {
        self.attempts
    }

    pub fn init(&mut self, prefix: &str, suffix: &str, target: u32) -> Result<(), Error> {
        self.state = None;
        self.attempts = 0;
        if !(1..=256).contains(&target) {
            return Err(Error::InvalidDifficulty(target));
        }
        let plan = Plan::new(prefix.as_bytes(), suffix.as_bytes());
        let max_trail = plan.layouts.iter().map(|l| l.n_trail).max().unwrap_or(0);
        let words = (HEADER + max_trail * 64) as u64;
        let device = &self.gpu.device;
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"),
                size: words * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: (OUT_WORDS * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read"),
                size: (OUT_WORDS * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pow"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out.as_entire_binding(),
                    },
                ],
            });
            slots.push(Slot {
                params,
                out,
                read,
                bind,
                units: 0,
                mapped: None,
            });
        }
        self.state = Some(State {
            plan,
            target,
            cursor: 0,
            slots,
            head: 0,
            in_flight: 0,
            launch: FIRST_LAUNCH,
            last: Instant::now(),
            pipelines: HashMap::new(),
        });
        Ok(())
    }

    fn enqueue(&mut self, index: usize) -> Result<(), Error> {
        let st = self.state.as_mut().expect("checked by mine");
        let start = st.cursor;
        let which = st
            .plan
            .find(start)
            .ok_or_else(|| Error::Gpu("unit outside every nonce layout".into()))?;
        let l = &st.plan.layouts[which];
        let end = l.first_unit + l.units;
        let count = (end - start).min(st.launch as u64);

        let offset = start - l.first_unit;
        let high = l.high_base + offset / UNITS_PER_HIGH;
        let mut header = [0u32; HEADER];
        header[sh::P_MIDSTATE..sh::P_MIDSTATE + 8].copy_from_slice(&st.plan.midstate);
        header[sh::P_TPL..sh::P_TPL + 32].copy_from_slice(&l.tpl);
        header[sh::P_HIGH_LO] = (high % BASE) as u32;
        header[sh::P_HIGH_HI] = (high / BASE) as u32;
        header[sh::P_R0] = (offset % UNITS_PER_HIGH) as u32;
        header[sh::P_COUNT] = count as u32;
        header[sh::P_TARGET] = st.target;
        header[sh::P_FIRST_TARGET] = st.target.min(32);

        let key = (l.b0, l.wv, l.digits - LOW_DIGITS, l.n_trail);
        let pipeline = match st.pipelines.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let source = sh::source(key.0, key.1, key.2, key.3);
                let module = self
                    .gpu
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("pow"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                let pipeline =
                    self.gpu
                        .device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("pow"),
                            layout: Some(&self.pipeline_layout),
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                e.insert(pipeline)
            }
        };

        let slot = &mut st.slots[index];
        let mut bytes = Vec::with_capacity((HEADER + l.trail_kw.len()) * 4);
        for w in header.iter().chain(l.trail_kw.iter()) {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        self.gpu.queue.write_buffer(&slot.params, 0, &bytes);

        let groups = count.div_ceil(WORKGROUP as u64) as u32;
        let x = groups.min(65535);
        let y = groups.div_ceil(x);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&slot.out, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &slot.bind, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&slot.out, 0, &slot.read, 0, (OUT_WORDS * 4) as u64);
        self.gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = oneshot::channel();
        slot.read.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        slot.mapped = Some(rx);
        slot.units = count;
        st.cursor += count;
        st.in_flight += 1;
        Ok(())
    }

    pub async fn mine(&mut self) -> Result<Option<PowHit>, Error> {
        if self.state.is_none() {
            return Err(Error::Gpu("miner is not initialised".into()));
        }
        loop {
            let st = self.state.as_ref().expect("checked");
            if st.in_flight >= SLOTS || st.cursor >= st.plan.total_units {
                break;
            }
            let index = (st.head + st.in_flight) % SLOTS;
            self.enqueue(index)?;
        }
        let st = self.state.as_mut().expect("checked");
        if st.in_flight == 0 {
            return Err(Error::Exhausted);
        }
        let rx = st.slots[st.head]
            .mapped
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

        let st = self.state.as_mut().expect("checked");
        let words: Vec<u32> = {
            let slot = &st.slots[st.head];
            let view = slot.read.slice(..).get_mapped_range();
            let words = view
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("four bytes")))
                .collect();
            drop(view);
            slot.read.unmap();
            words
        };
        let units = st.slots[st.head].units;
        self.attempts += units * NONCES_PER_UNIT;
        st.head = (st.head + 1) % SLOTS;
        st.in_flight -= 1;

        let elapsed = st.last.elapsed();
        st.last = Instant::now();
        if elapsed < TARGET_DISPATCH / 2 && st.launch < MAX_LAUNCH {
            st.launch *= 2;
        } else if elapsed > TARGET_DISPATCH * 2 && st.launch > FIRST_LAUNCH {
            st.launch /= 2;
        }

        if words[0] == 0 {
            return Ok(None);
        }
        let high = words[2] as u64 * BASE + words[1] as u64;
        let nonce = high * 10_000 + words[3] as u64;
        let mut digest = Sha256::new();
        digest.update(&st.plan.prefix);
        digest.update(nonce.to_string());
        digest.update(&st.plan.suffix);
        let hash: [u8; 32] = digest.finalize().into();
        for i in 0..8 {
            if words[4 + i].to_be_bytes() != hash[i * 4..i * 4 + 4] {
                return Err(Error::Gpu(format!(
                    "GPU hash for nonce {nonce} does not match the CPU hash"
                )));
            }
        }
        let bits = sha::leading_zero_bits(&hash);
        if bits < st.target {
            return Err(Error::Gpu(format!(
                "GPU nonce {nonce} has only {bits} leading zero bits"
            )));
        }
        Ok(Some(PowHit {
            nonce,
            event_id: hash,
            difficulty: bits,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn mine_blocking(&mut self) -> Result<Option<PowHit>, Error> {
        pollster::block_on(self.mine())
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::event::{Template, leading_zero_bits};

    fn run(adapter: usize, content: &str, fast: bool, difficulty: u32) {
        let infos = pollster::block_on(super::super::list());
        let Some(gpu) = crate::wgpu::test_gpu(adapter) else {
            return;
        };
        eprintln!("adapter {adapter}: {} of {}", gpu.info.label(), infos.len());
        let template = Template {
            pubkey: "ef".repeat(32),
            kind: 1,
            tags: vec![vec!["t".into(), "x".into()]],
            content: content.into(),
            difficulty,
            fast,
        };
        let created_at = 1_700_000_000;
        let split = template.split(created_at);
        let mut miner = WgpuPow::new(gpu);
        miner
            .init(&split.prefix, &split.suffix, difficulty)
            .unwrap();
        let started = Instant::now();
        let hit = loop {
            if let Some(hit) = miner.mine_blocking().unwrap() {
                break hit;
            }
            assert!(started.elapsed() < Duration::from_secs(120), "no hit");
        };
        let id = template.id_of(created_at, &template.mined(hit.nonce));
        assert_eq!(id, hit.event_id);
        assert!(leading_zero_bits(&id) >= difficulty);
        eprintln!(
            "  nonce {} after {} hashes in {:?}",
            hit.nonce,
            miner.attempts(),
            started.elapsed()
        );
    }

    #[test]
    fn first_adapter_finds_verified_events() {
        for fast in [false, true] {
            let Some(i) = pollster::block_on(super::super::list())
                .iter()
                .position(|a| !a.software)
            else {
                return;
            };
            run(i, "hello from wgpu \"quoted\" 日本語", fast, 20);
        }
    }

    #[test]
    fn a_long_note_spills_into_a_second_block() {
        let Some(i) = pollster::block_on(super::super::list())
            .iter()
            .position(|a| !a.software)
        else {
            return;
        };
        run(i, &"long note ".repeat(30), false, 18);
    }

    #[test]
    fn software_adapter_agrees() {
        let infos = pollster::block_on(super::super::list());
        if let Some(i) = infos.iter().position(|i| i.software) {
            run(i, "software rasteriser", false, 16);
        }
    }
}
