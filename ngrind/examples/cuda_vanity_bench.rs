use ngrind::cuda::{StepConfig, StepVanity};
use ngrind::{Alphabet, Pattern};
use std::time::{Duration, Instant};

fn main() {
    let pattern = Pattern::parse(Alphabet::Hex, "", "0123456789").unwrap();
    for _round in 0..3 {
        for half in [16usize, 32] {
            for threads in [0u32, 131072] {
                let config = StepConfig {
                    threads,
                    half,
                    ..StepConfig::default()
                };
                let mut m = StepVanity::new(&pattern, 0, config).unwrap();
                for _ in 0..6 {
                    m.step().unwrap();
                }
                let begin = Instant::now();
                let base = m.keys_generated();
                while begin.elapsed() < Duration::from_secs(3) {
                    m.step().unwrap();
                }
                let rate = (m.keys_generated() - base) as f64 / begin.elapsed().as_secs_f64();
                println!(
                    "half {half:>2} threads {:>7}: {:.2} Gkeys/s",
                    if threads == 0 {
                        "auto".into()
                    } else {
                        threads.to_string()
                    },
                    rate / 1e9
                );
            }
        }
    }
}
