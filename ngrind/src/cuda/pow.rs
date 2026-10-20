use super::sha::{self, K};
use super::{Device, Error};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use sha2::{Digest, Sha256};

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pow.fatbin"));

const NONCES_PER_UNIT: u64 = 1000;
const UNITS_PER_HIGH: u64 = 10;
const LOW_DIGITS: usize = 4;
const MIN_NONCE_DIGITS: usize = 12;
const MAX_NONCE_DIGITS: usize = 20;
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

#[derive(Debug, Clone)]
pub struct PowHit {
    pub nonce: u64,
    pub event_id: [u8; 32],
    pub difficulty: u32,
}

struct Layout {
    digits: usize,
    b0: usize,
    wv: usize,
    n_trail: usize,
    cost: usize,
    high_base: u64,
    units: u64,
    first_unit: u64,
    tpl: [u32; 32],
    trail_kw: Vec<u32>,
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
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    target: u32,
    midstate: [u32; 8],
    layouts: Vec<Layout>,
    cursor: u64,
    limit: u64,
    slots: Vec<Slot>,
    head: usize,
    in_flight: usize,
}

pub struct PowMiner {
    dev: Option<Device>,
    funcs: Vec<Option<CudaFunction>>,
    state: Option<State>,
    attempts: u64,
}

fn pow10(n: usize) -> u64 {
    10u64.pow(n as u32)
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
        self.state = None;
        self.attempts = 0;
        if !(1..=256).contains(&target) {
            return Err(Error::InvalidDifficulty(target));
        }
        if self.dev.as_ref().is_none_or(|d| d.ordinal != device) {
            self.funcs.iter_mut().for_each(|f| *f = None);
            self.dev = None;
            let dev = Device::open(device, FATBIN)?;
            unsafe { dev.ctx.disable_event_tracking() };
            self.dev = Some(dev);
        }
        let state = self.setup(prefix.as_bytes(), suffix.as_bytes(), target)?;
        self.state = Some(state);
        Ok(())
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
        let (midstate, consumed) = sha::midstate(prefix);
        let rem = prefix.len() - consumed;
        let mut layouts: Vec<Layout> = (MIN_NONCE_DIGITS..=MAX_NONCE_DIGITS)
            .filter(|digits| (rem + digits - LOW_DIGITS) % 4 == 0)
            .map(|digits| build_layout(prefix, suffix, consumed, rem, digits))
            .collect();
        layouts.sort_by(|a, b| a.cost.cmp(&b.cost).then(b.digits.cmp(&a.digits)));

        let (sm_count, stream) = {
            let dev = self.dev.as_ref().expect("opened by init");
            (dev.sm_count, dev.stream.clone())
        };
        let mut total = 0u64;
        for l in &mut layouts {
            l.first_unit = total;
            total += l.units;
            if l.n_trail > 0 {
                l.device_trail = Some(stream.clone_htod(&l.trail_kw)?);
            }
            let func = self.function(l.b0, l.wv)?;
            let blocks = func
                .occupancy_max_active_blocks_per_multiprocessor(THREADS, 0, None)?
                .max(1);
            l.launch_units = sm_count * blocks * THREADS * WAVES;
        }

        let ctx = stream.context().clone();
        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let device = stream.alloc_zeros::<u64>(OUT_WORDS)?;
            let host = unsafe { ctx.alloc_pinned_with_flags::<u64>(OUT_WORDS, 0) }?;
            let done = ctx.new_event(None)?;
            slots.push(Slot {
                device,
                host,
                done,
                units: 0,
            });
        }
        Ok(State {
            prefix: prefix.to_vec(),
            suffix: suffix.to_vec(),
            target,
            midstate,
            layouts,
            cursor: 0,
            limit: total,
            slots,
            head: 0,
            in_flight: 0,
        })
    }

    fn enqueue(&mut self, index: usize) -> Result<(), Error> {
        let (b0, wv, params, count) = {
            let st = self.state.as_mut().expect("checked by mine");
            let start = st.cursor;
            let l = st
                .layouts
                .iter()
                .find(|l| start >= l.first_unit && start < l.first_unit + l.units)
                .ok_or_else(|| Error::Gpu("unit outside every nonce layout".into()))?;
            let end = (l.first_unit + l.units).min(st.limit);
            let count = (end - start).min(l.launch_units as u64) as u32;
            let dev = self.dev.as_ref().expect("opened by init");
            let mut kw = [0u32; PARAM_TRAIL * 64];
            let n = l.trail_kw.len().min(PARAM_TRAIL * 64);
            kw[..n].copy_from_slice(&l.trail_kw[..n]);
            let trail_ptr = match &l.device_trail {
                Some(buf) => device_pointer(buf, dev),
                None => 0,
            };
            let out_ptr = device_pointer(&st.slots[index].device, dev);
            let params = Params {
                midstate: st.midstate,
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
        let slot = &mut st.slots[index];
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
        let slot = &st.slots[st.head];
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
        digest.update(&st.prefix);
        digest.update(nonce.to_string());
        digest.update(&st.suffix);
        let hash: [u8; 32] = digest.finalize().into();
        for i in 0..8 {
            let word = if i % 2 == 0 {
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

fn build_layout(
    prefix: &[u8],
    suffix: &[u8],
    consumed: usize,
    rem: usize,
    digits: usize,
) -> Layout {
    let high = digits - LOW_DIGITS;
    let low_pos = rem + high;
    let b0 = low_pos / 64;
    let wv = (low_pos % 64) / 4;
    let tail_len = rem + digits + suffix.len();
    let blocks = (tail_len + 9).div_ceil(64);
    let n_trail = blocks - b0 - 1;
    let cost = (64 - wv) + 24 + 64 * n_trail;
    let high_base = pow10(high - 1);
    let high_max = (pow10(high) - 1).min((u64::MAX - 9999) / 10000);
    let units = (high_max - high_base + 1) * UNITS_PER_HIGH;

    let mut tail = vec![0u8; blocks * 64];
    tail[..rem].copy_from_slice(&prefix[consumed..]);
    tail[rem + digits..rem + digits + suffix.len()].copy_from_slice(suffix);
    tail[tail_len] = 0x80;
    let bits = ((prefix.len() + digits + suffix.len()) as u64) * 8;
    for i in 0..8 {
        tail[blocks * 64 - 1 - i] = (bits >> (8 * i)) as u8;
    }

    let mut tpl = [0u32; 32];
    for (i, word) in tpl.iter_mut().enumerate().take((b0 + 1) * 16) {
        *word = u32::from_be_bytes(tail[i * 4..i * 4 + 4].try_into().expect("four bytes"));
    }
    let mut trail_kw = Vec::with_capacity(n_trail * 64);
    for b in 0..n_trail {
        let w = sha::schedule(&tail[(b0 + 1 + b) * 64..(b0 + 2 + b) * 64]);
        trail_kw.extend((0..64).map(|i| w[i].wrapping_add(K[i])));
    }
    Layout {
        digits,
        b0,
        wv,
        n_trail,
        cost,
        high_base,
        units,
        first_unit: 0,
        tpl,
        trail_kw,
        device_trail: None,
        launch_units: 0,
    }
}
