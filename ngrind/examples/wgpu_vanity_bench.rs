use ngrind::wgpu::{Gpu, VanityConfig, WgpuVanity};
use ngrind::{Alphabet, Pattern};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let gpu = Arc::new(pollster::block_on(Gpu::open(0)).unwrap());
    let pattern = Pattern::parse(Alphabet::Hex, "", "0123456789").unwrap();
    for (threads, half) in [
        (16384u32, 8usize),
        (16384, 16),
        (65536, 16),
        (65536, 32),
        (131072, 32),
        (262144, 32),
        (65536, 64),
    ] {
        let mut m = WgpuVanity::new(
            gpu.clone(),
            &pattern,
            VanityConfig {
                threads,
                half,
                fixed_iters: None,
                max_hits: 8,
            },
        )
        .unwrap();
        for _ in 0..4 {
            m.step_blocking().unwrap();
        }
        let begin = Instant::now();
        let base = m.keys_generated();
        while begin.elapsed() < Duration::from_secs(4) {
            m.step_blocking().unwrap();
        }
        println!(
            "threads {threads:>6} half {half:>3}: {:.2} Mkeys/s",
            (m.keys_generated() - base) as f64 / begin.elapsed().as_secs_f64() / 1e6
        );
    }
}
