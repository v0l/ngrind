use crate::sha::{self, K};

pub const NONCES_PER_UNIT: u64 = 1000;
pub const UNITS_PER_HIGH: u64 = 10;
pub const LOW_DIGITS: usize = 4;
pub const MIN_NONCE_DIGITS: usize = 12;
pub const MAX_NONCE_DIGITS: usize = 20;

#[derive(Debug, Clone)]
pub struct PowHit {
    pub nonce: u64,
    pub event_id: [u8; 32],
    pub difficulty: u32,
}

pub struct Layout {
    pub digits: usize,
    pub b0: usize,
    pub wv: usize,
    pub n_trail: usize,
    pub cost: usize,
    pub high_base: u64,
    pub units: u64,
    pub first_unit: u64,
    pub tpl: [u32; 32],
    pub trail_kw: Vec<u32>,
}

pub struct Plan {
    pub prefix: Vec<u8>,
    pub suffix: Vec<u8>,
    pub midstate: [u32; 8],
    pub layouts: Vec<Layout>,
    pub total_units: u64,
}

fn pow10(n: usize) -> u64 {
    10u64.pow(n as u32)
}

impl Plan {
    pub fn new(prefix: &[u8], suffix: &[u8]) -> Plan {
        let (midstate, consumed) = sha::midstate(prefix);
        let rem = prefix.len() - consumed;
        let mut layouts: Vec<Layout> = (MIN_NONCE_DIGITS..=MAX_NONCE_DIGITS)
            .filter(|digits| (rem + digits - LOW_DIGITS).is_multiple_of(4))
            .map(|digits| build_layout(prefix, suffix, consumed, rem, digits))
            .collect();
        layouts.sort_by(|a, b| a.cost.cmp(&b.cost).then(b.digits.cmp(&a.digits)));
        let mut total_units = 0;
        for l in &mut layouts {
            l.first_unit = total_units;
            total_units += l.units;
        }
        Plan {
            prefix: prefix.to_vec(),
            suffix: suffix.to_vec(),
            midstate,
            layouts,
            total_units,
        }
    }

    pub fn find(&self, unit: u64) -> Option<usize> {
        self.layouts
            .iter()
            .position(|l| unit >= l.first_unit && unit < l.first_unit + l.units)
    }
}

impl Layout {
    pub fn nonce(&self, unit: u64, low: u32) -> u64 {
        let high = self.high_base + unit / UNITS_PER_HIGH;
        high * 10_000 + (unit % UNITS_PER_HIGH) * 1000 + low as u64
    }
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
    }
}
