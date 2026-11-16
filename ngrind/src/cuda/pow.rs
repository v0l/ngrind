use super::{Device, Error};
use crate::powlayout::{LOW_DIGITS, NONCES_PER_UNIT, Plan, PowHit};
use crate::sha;
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pow.fatbin"));

const SLOTS: usize = 2;
const PARAM_TRAIL: usize = 6;
const OUT_WORDS: usize = 6;

const fn parse(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let mut n = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        n = n * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    n
}

const THREADS: u32 = parse(env!("NGRIND_POW_THREADS"));
const WAVES: u32 = parse(env!("NGRIND_POW_WAVES"));

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    midstate: [u32; 8],
    tpl: [u32; 32],
    kw: [u32; PARAM_TRAIL * 64],
    trail_kw: u64,
    out: u64,
    unit_start: u64,
    high_base: u64,
    unit_count: u32,
    n_trail: i32,
    high_digits: i32,
    target: i32,
    first_target: i32,
}

unsafe impl DeviceRepr for Params {}

struct DeviceLayout {
    device_trail: Option<CudaSlice<u32>>,
    launch_units: u32,
}

struct Slot {
    device: CudaSlice<u64>,
    host: PinnedHostSlice<u64>,
    done: CudaEvent,
    units: u64,
}

struct State {
    target: u32,
    plan: Plan,
    device_layouts: Vec<DeviceLayout>,
    cursor: u64,
    limit: u64,
    head: usize,
    in_flight: usize,
}

pub struct PowMiner {
    dev: Option<Device>,
    funcs: Vec<Option<CudaFunction>>,
    state: Option<State>,
    slots: Vec<Slot>,
    trails: Vec<CudaSlice<u32>>,
    attempts: u64,
}

impl Default for PowMiner {
    fn default() -> Self {
        Self::new()
    }
}

impl PowMiner {
    pub fn new() -> PowMiner {
        PowMiner {
            dev: None,
            funcs: (0..32).map(|_| None).collect(),
            state: None,
            slots: Vec::new(),
            trails: Vec::new(),
            attempts: 0,
        }
    }

    pub fn attempts(&self) -> u64 {
        self.attempts
    }

    pub fn device_name(&self) -> String {
        self.dev
            .as_ref()
            .map(|d| d.name.clone())
            .unwrap_or_default()
    }

    pub fn init(
        &mut self,
        prefix: &str,
        suffix: &str,
        target: u32,
        device: u32,
    ) -> Result<(), Error> {
        self.release_state()?;
        self.attempts = 0;
        if !(1..=256).contains(&target) {
            return Err(Error::InvalidDifficulty(target));
        }
        if self.dev.as_ref().is_none_or(|d| d.ordinal != device) {
            self.funcs.iter_mut().for_each(|f| *f = None);
            self.slots.clear();
            self.trails.clear();
            self.dev = None;
            let dev = Device::open(device, FATBIN)?;
            unsafe { dev.ctx.disable_event_tracking() };
            self.dev = Some(dev);
        }
        let state = self.setup(prefix.as_bytes(), suffix.as_bytes(), target)?;
        self.state = Some(state);
        Ok(())
    }

    fn release_state(&mut self) -> Result<(), Error> {
        if let Some(state) = self.state.take() {
            if state.in_flight > 0 {
                if let Some(dev) = self.dev.as_ref() {
                    dev.stream.synchronize()?;
                }
            }
            for layout in state.device_layouts {
                if let Some(buf) = layout.device_trail {
                    self.trails.push(buf);
                }
            }
        }
        Ok(())
    }

    pub fn allocations(&self) -> usize {
        self.slots.len() + self.trails.len()
    }

    fn trail_buffer(&mut self, stream: &Arc<CudaStream>, words: &[u32]) -> Result<CudaSlice<u32>, Error> {
        match self.trails.iter().position(|b| b.len() >= words.len()) {
            Some(i) => {
                let mut buf = self.trails.swap_remove(i);
                stream.memcpy_htod(words, &mut buf)?;
                Ok(buf)
            }
            None => Ok(stream.clone_htod(words)?),
        }
    }

    fn function(&mut self, b0: usize, wv: usize) -> Result<&CudaFunction, Error> {
        let slot = b0 * 16 + wv;
        if self.funcs[slot].is_none() {
            let dev = self
                .dev
                .as_ref()
                .ok_or_else(|| Error::Gpu("miner is not initialised".into()))?;
            self.funcs[slot] = Some(dev.module.load_function(&format!("ngrind_pow_{b0}_{wv}"))?);
        }
        Ok(self.funcs[slot].as_ref().expect("just loaded"))
    }

    fn setup(&mut self, prefix: &[u8], suffix: &[u8], target: u32) -> Result<State, Error> {
        let plan = Plan::new(prefix, suffix);
        let (sm_count, stream) = {
            let dev = self.dev.as_ref().expect("opened by init");
            (dev.sm_count, dev.stream.clone())
        };
        let mut device_layouts = Vec::with_capacity(plan.layouts.len());
        for l in &plan.layouts {
            let device_trail = if l.n_trail > 0 {
                Some(self.trail_buffer(&stream, &l.trail_kw)?)
            } else {
                None
            };
            let func = self.function(l.b0, l.wv)?;
            let blocks = func
                .occupancy_max_active_blocks_per_multiprocessor(THREADS, 0, None)?
                .max(1);
            device_layouts.push(DeviceLayout {
                device_trail,
                launch_units: sm_count * blocks * THREADS * WAVES,
            });
        }

        let ctx = stream.context().clone();
        while self.slots.len() < SLOTS {
            let device = stream.alloc_zeros::<u64>(OUT_WORDS)?;
            let host = unsafe { ctx.alloc_pinned_with_flags::<u64>(OUT_WORDS, 0) }?;
            let done = ctx.new_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            ))?;
            self.slots.push(Slot {
                device,
                host,
                done,
                units: 0,
            });
        }
        let limit = plan.total_units;
        Ok(State {
            target,
            plan,
            device_layouts,
            cursor: 0,
            limit,
            head: 0,
            in_flight: 0,
        })
    }

    fn enqueue(&mut self, index: usize) -> Result<(), Error> {
        let (b0, wv, params, count) = {
            let st = self.state.as_mut().expect("checked by mine");
            let start = st.cursor;
            let index_of = st
                .plan
                .find(start)
                .ok_or_else(|| Error::Gpu("unit outside every nonce layout".into()))?;
            let l = &st.plan.layouts[index_of];
            let dl = &st.device_layouts[index_of];
            let end = (l.first_unit + l.units).min(st.limit);
            let count = (end - start).min(dl.launch_units as u64) as u32;
            let dev = self.dev.as_ref().expect("opened by init");
            let mut kw = [0u32; PARAM_TRAIL * 64];
            let n = l.trail_kw.len().min(PARAM_TRAIL * 64);
            kw[..n].copy_from_slice(&l.trail_kw[..n]);
            let trail_ptr = match &dl.device_trail {
                Some(buf) => device_pointer(buf, dev),
                None => 0,
            };
            let out_ptr = device_pointer(&self.slots[index].device, dev);
            let params = Params {
                midstate: st.plan.midstate,
                tpl: l.tpl,
                kw,
                trail_kw: trail_ptr,
                out: out_ptr,
                unit_start: start - l.first_unit,
                high_base: l.high_base,
                unit_count: count,
                n_trail: l.n_trail as i32,
                high_digits: (l.digits - LOW_DIGITS) as i32,
                target: st.target as i32,
                first_target: st.target.min(32) as i32,
            };
            (l.b0, l.wv, params, count)
        };
        let func = self.function(b0, wv)?.clone();
        let dev = self.dev.as_ref().expect("opened by init");
        let st = self.state.as_mut().expect("checked by mine");
        let slot = &mut self.slots[index];
        dev.stream.memset_zeros(&mut slot.device)?;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = dev.stream.launch_builder(&func);
        launch.arg(&params);
        unsafe { launch.launch(cfg) }
            .map_err(|e| Error::Gpu(format!("kernel launch failed: {e}")))?;
        dev.stream.memcpy_dtoh(&slot.device, &mut slot.host)?;
        slot.done.record(&dev.stream)?;
        slot.units = count as u64;
        st.cursor += count as u64;
        st.in_flight += 1;
        Ok(())
    }

    pub fn mine(&mut self) -> Result<Option<PowHit>, Error> {
        if self.state.is_none() {
            return Err(Error::Gpu("miner is not initialised".into()));
        }
        loop {
            let st = self.state.as_ref().expect("checked");
            if st.in_flight >= SLOTS || st.cursor >= st.limit {
                break;
            }
            let index = (st.head + st.in_flight) % SLOTS;
            self.enqueue(index)?;
        }
        let st = self.state.as_mut().expect("checked");
        if st.in_flight == 0 {
            return Err(Error::Exhausted);
        }
        let slot = &self.slots[st.head];
        slot.done.synchronize()?;
        let out: [u64; OUT_WORDS] =
            unsafe { std::slice::from_raw_parts(slot.host.as_ptr()?, OUT_WORDS) }
                .try_into()
                .expect("six words");
        self.attempts += slot.units * NONCES_PER_UNIT;
        st.head = (st.head + 1) % SLOTS;
        st.in_flight -= 1;

        let found = (out[5] & 0xffff_ffff) != 0;
        if !found {
            return Ok(None);
        }
        let nonce = out[0];
        let mut digest = Sha256::new();
        digest.update(&st.plan.prefix);
        digest.update(nonce.to_string());
        digest.update(&st.plan.suffix);
        let hash: [u8; 32] = digest.finalize().into();
        for i in 0usize..8 {
            let word = if i.is_multiple_of(2) {
                out[1 + i / 2] as u32
            } else {
                (out[1 + i / 2] >> 32) as u32
            };
            if word.to_be_bytes() != hash[i * 4..i * 4 + 4] {
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
}

fn device_pointer<T>(buf: &CudaSlice<T>, dev: &Device) -> u64 {
    use cudarc::driver::DevicePtr;
    let (ptr, _guard) = buf.device_ptr(&dev.stream);
    ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> bool {
        super::super::device_count() > 0
    }

    #[test]
    fn buffers_are_reused_across_init_calls() {
        if !gpu() {
            return;
        }
        let mut miner = PowMiner::new();
        miner.init("{\"a\":\"", "\"}", 8, 0).unwrap();
        let first = miner.allocations();
        let ptr = device_pointer(&miner.slots[0].device, miner.dev.as_ref().unwrap());
        for _ in 0..5 {
            miner.init("{\"a\":\"", "\"}", 8, 0).unwrap();
            assert!(miner.mine().is_ok());
        }
        assert_eq!(miner.allocations(), first);
        assert_eq!(
            device_pointer(&miner.slots[0].device, miner.dev.as_ref().unwrap()),
            ptr
        );
    }
}
