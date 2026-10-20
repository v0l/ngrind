use crate::keys;
use crate::pattern::Pattern;
use secp256k1::{PublicKey, Scalar, SecretKey};

pub const STRIDE_SHIFT: usize = 48;

pub type Found = ([u8; 32], [u8; 32]);

pub fn limbs(bytes: &[u8]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (k, limb) in out.iter_mut().enumerate() {
        let at = (7 - k) * 4;
        *limb = u32::from_be_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
    }
    out
}

pub fn unlimbs(words: &[u32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for k in 0..8 {
        out[(7 - k) * 4..(7 - k) * 4 + 4].copy_from_slice(&words[k].to_be_bytes());
    }
    out
}

fn scalar(value: u128) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[16..].copy_from_slice(&value.to_be_bytes());
    Scalar::from_be_bytes(bytes).expect("far below the curve order")
}

fn secret(value: u128) -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[16..].copy_from_slice(&value.to_be_bytes());
    SecretKey::from_secret_bytes(bytes).expect("a small non-zero scalar")
}

pub fn offset_secret(base: &SecretKey, delta: i128) -> Result<SecretKey, String> {
    let magnitude = scalar(delta.unsigned_abs());
    if delta >= 0 {
        base.add_tweak(&magnitude).map_err(|e| e.to_string())
    } else {
        base.negate()
            .add_tweak(&magnitude)
            .map(|k| k.negate())
            .map_err(|e| e.to_string())
    }
}

pub fn thread_secret(origin: &SecretKey, thread: u64) -> Result<SecretKey, String> {
    offset_secret(origin, (thread as i128) << STRIDE_SHIFT)
}

pub fn table(half: usize) -> Vec<u32> {
    let mut words = Vec::with_capacity((half + 1) * 16);
    let mut push = |k: u128| {
        let point = PublicKey::from_secret_key(&secret(k)).serialize_uncompressed();
        words.extend_from_slice(&limbs(&point[1..33]));
        words.extend_from_slice(&limbs(&point[33..65]));
    };
    for i in 0..half {
        push(i as u128 + 1);
    }
    push(2 * half as u128 + 1);
    words
}

pub fn start_states(origin: &SecretKey, threads: u32) -> Vec<u32> {
    let mut words = Vec::with_capacity(threads as usize * 16);
    let step = PublicKey::from_secret_key(&secret(1u128 << STRIDE_SHIFT));
    let mut point = PublicKey::from_secret_key(origin);
    for _ in 0..threads {
        let bytes = point.serialize_uncompressed();
        words.extend_from_slice(&limbs(&bytes[1..33]));
        words.extend_from_slice(&limbs(&bytes[33..65]));
        point = point
            .combine(&step)
            .expect("distinct multiples never cancel");
    }
    words
}

pub struct Decoder {
    pub origin: SecretKey,
    pub half: usize,
    pub pattern: Pattern,
}

pub enum Decoded {
    Hit([u8; 32], [u8; 32]),
    Rejected,
}

impl Decoder {
    pub fn per_iteration(&self) -> u64 {
        2 * self.half as u64 + 1
    }

    pub fn decode(&self, advanced: u64, thread: u64, code: u32, x: [u8; 32]) -> Decoded {
        let delta =
            advanced as i128 * self.per_iteration() as i128 + code as i128 - self.half as i128;
        match thread_secret(&self.origin, thread).and_then(|s| offset_secret(&s, delta)) {
            Ok(key) => {
                let private = key.to_secret_bytes();
                if keys::public_of(&private).is_ok_and(|p| p == x) && self.pattern.matches(&x) {
                    Decoded::Hit(private, x)
                } else {
                    Decoded::Rejected
                }
            }
            Err(_) => Decoded::Rejected,
        }
    }
}
