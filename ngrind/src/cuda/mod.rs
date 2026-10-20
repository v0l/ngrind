mod device;
mod pow;
mod run;
mod step;
mod vanity;

pub use crate::powlayout::PowHit;
pub use device::{Device, device_count, device_name, devices};
pub use pow::PowMiner;
pub use run::{pow, vanity, vanity_sequential};
pub use step::{StepConfig, StepVanity};
pub use vanity::{Bech32Check, Config, SearchMode, VanityHit, VanityMiner, VanityMode};

use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    InvalidDifficulty(u32),
    NoDevice(u32),
    Driver(String),
    Gpu(String),
    Exhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDifficulty(d) => write!(f, "difficulty {d} is outside 1..=256"),
            Error::NoDevice(d) => write!(f, "CUDA device {d} does not exist"),
            Error::Driver(e) => write!(f, "CUDA driver: {e}"),
            Error::Gpu(e) => write!(f, "GPU mining failed: {e}"),
            Error::Exhausted => write!(f, "nonce space exhausted for this template"),
        }
    }
}

impl std::error::Error for Error {}

impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error::Driver(e.to_string())
    }
}
