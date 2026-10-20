mod device;
mod field;
mod pow;
mod pow_shader;
mod run;
mod vanity;
mod vanity_shader;

pub use device::{Gpu, GpuInfo, list};
pub use pow::WgpuPow;
pub use run::{pow, vanity};
#[cfg(not(target_arch = "wasm32"))]
pub use run::{pow_blocking, vanity_blocking};
pub use vanity::{VanityConfig, WgpuVanity};

use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    NoAdapter(String),
    Device(String),
    Gpu(String),
    InvalidDifficulty(u32),
    Exhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter(e) => write!(f, "no usable GPU: {e}"),
            Error::Device(e) => write!(f, "GPU device: {e}"),
            Error::Gpu(e) => write!(f, "GPU mining failed: {e}"),
            Error::InvalidDifficulty(d) => write!(f, "difficulty {d} is outside 1..=256"),
            Error::Exhausted => write!(f, "nonce space exhausted for this template"),
        }
    }
}

impl std::error::Error for Error {}
