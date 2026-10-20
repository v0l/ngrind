#![cfg_attr(test, allow(clippy::await_holding_lock, clippy::arc_with_non_send_sync))]
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
#![allow(clippy::chunks_exact_to_as_chunks)]
pub mod backend;
pub mod control;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod event;
pub mod gtable;
pub mod keys;
pub mod pattern;
pub mod powlayout;
pub mod sha;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use control::{Control, OnFound, Phase};
pub use event::{PowFound, Template};
pub use pattern::{Alphabet, Pattern};
