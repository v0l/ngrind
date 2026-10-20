use ngrind::wgpu::{Gpu, WgpuPow};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let gpu = Arc::new(pollster::block_on(Gpu::open(0)).unwrap());
    let prefix = r#"[0,"efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef",1700000000,1,[["nonce",""#;
    let suffix = r#"","60"]],"gm"]"#;
    let mut pow = WgpuPow::new(gpu);
    pow.init(prefix, suffix, 60).unwrap();
    for _ in 0..8 {
        pow.mine_blocking().unwrap();
    }
    let begin = Instant::now();
    let base = pow.attempts();
    while begin.elapsed() < Duration::from_secs(4) {
        pow.mine_blocking().unwrap();
    }
    println!(
        "{:.0}",
        (pow.attempts() - base) as f64 / begin.elapsed().as_secs_f64() / 1e6
    );
}
