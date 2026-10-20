use super::vanity::{Bech32Check, Config};
use super::{PowMiner, SearchMode, VanityMiner, VanityMode, device_name};
use crate::PowFound;
use crate::control::{Control, OnFound, Phase};
use crate::event::{Template, leading_zero_bits, unix_now};
use crate::gtable::GTable;
use crate::keys;
use crate::pattern::{Alphabet, Pattern};
use secp256k1::Keypair;
use std::path::Path;
use std::time::{Duration, Instant};

const REFRESH: Duration = Duration::from_secs(30);

struct Plan {
    text: String,
    mode: VanityMode,
    split: usize,
    verify: Option<Bech32Check>,
}

fn plan(pattern: &Pattern) -> Plan {
    let (p, s) = (pattern.prefix.as_str(), pattern.suffix.as_str());
    let both = format!("{p}{s}");
    match (pattern.alphabet, p.is_empty(), s.is_empty()) {
        (Alphabet::Hex, _, true) => Plan {
            text: p.into(),
            mode: VanityMode::HexPrefix,
            split: p.len(),
            verify: None,
        },
        (Alphabet::Hex, true, _) => Plan {
            text: s.into(),
            mode: VanityMode::HexSuffix,
            split: 0,
            verify: None,
        },
        (Alphabet::Hex, _, _) => Plan {
            text: both,
            mode: VanityMode::HexBoth,
            split: p.len(),
            verify: None,
        },
        (Alphabet::Npub, _, true) => match pattern.prefilter_hex() {
            Some(hex) => Plan {
                text: hex,
                mode: VanityMode::HexPrefix,
                split: 0,
                verify: Some(Bech32Check {
                    prefix: p.into(),
                    suffix: String::new(),
                }),
            },
            None => Plan {
                text: p.into(),
                mode: VanityMode::Bech32Prefix,
                split: p.len(),
                verify: None,
            },
        },
        (Alphabet::Npub, true, _) => Plan {
            text: s.into(),
            mode: VanityMode::Bech32Suffix,
            split: 0,
            verify: None,
        },
        (Alphabet::Npub, _, _) => Plan {
            text: both,
            mode: VanityMode::Bech32Both,
            split: p.len(),
            verify: None,
        },
    }
}

pub fn vanity(
    ctl: &dyn Control,
    pattern: &Pattern,
    device: u32,
    found: OnFound,
) -> Result<(), String> {
    search(ctl, pattern, device, None, found)
}

pub fn vanity_sequential(
    ctl: &dyn Control,
    pattern: &Pattern,
    device: u32,
    checkpoint: &Path,
    found: OnFound,
) -> Result<(), String> {
    search(ctl, pattern, device, Some(checkpoint), found)
}

const CHECKPOINT_EVERY: Duration = Duration::from_secs(60);

fn offset_from(checkpoint: &Path) -> Option<[u8; 32]> {
    let text = std::fs::read_to_string(checkpoint).ok()?;
    let value = text.lines().find_map(|l| l.strip_prefix("startOffset="))?;
    let value = value.trim();
    if value.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn search(
    ctl: &dyn Control,
    pattern: &Pattern,
    device: u32,
    checkpoint: Option<&Path>,
    found: OnFound,
) -> Result<(), String> {
    ctl.phase(Phase::Setup);
    ctl.log("building the secp256k1 table");
    let table = GTable::shared();
    if ctl.stopped() {
        return Ok(());
    }
    ctl.log(&format!("starting on {}", device_name(device)));
    let plan = plan(pattern);
    let mut start = [0u8; 32];
    let mut resumed = false;
    match checkpoint {
        Some(path) => match offset_from(path) {
            Some(saved) => {
                start = saved;
                resumed = true;
            }
            None => start = rand::random(),
        },
        None => start[24..].copy_from_slice(&rand::random::<u64>().to_le_bytes()),
    }
    let mut miner = VanityMiner::new(
        &table,
        Config {
            pattern: &plan.text,
            mode: plan.mode,
            split: Some(plan.split),
            search: if checkpoint.is_some() {
                SearchMode::Sequential
            } else {
                SearchMode::Random
            },
            start,
            verify: plan.verify,
            device,
        },
    )
    .map_err(|e| e.to_string())?;
    if let (Some(path), true) = (checkpoint, resumed) {
        miner.load_checkpoint(path)?;
        ctl.log(&format!("resumed at {:.2} %", miner.progress() * 100.0));
    }
    ctl.phase(Phase::Search);

    let (mut counted, mut rejected) = (miner.keys_generated(), 0u32);
    let mut saved = Instant::now();
    while !ctl.stopped() {
        if let Some(path) = checkpoint {
            if saved.elapsed() >= CHECKPOINT_EVERY {
                miner.save_checkpoint(path).map_err(|e| e.to_string())?;
                saved = Instant::now();
            }
            if miner.progress() >= 1.0 {
                ctl.log("the whole search space has been covered");
                break;
            }
        }
        miner.iterate().map_err(|e| e.to_string())?;
        let hits = miner.collect().map_err(|e| e.to_string())?;
        let now = miner.keys_generated();
        ctl.add(now - counted);
        counted = now;
        for hit in hits {
            let genuine = keys::public_of(&hit.private_key).is_ok_and(|p| p == hit.public_key)
                && pattern.matches(&hit.public_key);
            if genuine {
                found(hit.private_key, hit.public_key);
            } else {
                rejected += 1;
                ctl.log(&format!(
                    "discarded a match that failed the check ({rejected})"
                ));
            }
        }
    }
    if let Some(path) = checkpoint {
        miner.save_checkpoint(path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn pow(
    ctl: &dyn Control,
    template: &Template,
    signer: Option<&Keypair>,
    device: u32,
    found: &(dyn Fn(PowFound) + Sync),
) -> Result<(), String> {
    ctl.phase(Phase::Setup);
    let mut miner = PowMiner::new();
    let mut created_at = unix_now();
    let split = template.split(created_at);
    miner
        .init(&split.prefix, &split.suffix, template.difficulty, device)
        .map_err(|e| e.to_string())?;
    ctl.log(&format!("starting on {}", miner.device_name()));
    ctl.phase(Phase::Search);

    let mut counted = 0u64;
    let mut refreshed = Instant::now();
    let mut renew = false;
    while !ctl.stopped() {
        let hit = match miner.mine() {
            Ok(hit) => hit,
            Err(super::Error::Exhausted) => {
                renew = true;
                None
            }
            Err(e) => return Err(e.to_string()),
        };
        let total = miner.attempts();
        ctl.add(total - counted);
        counted = total;

        if let Some(hit) = hit {
            let mined = template.mined(hit.nonce);
            let id = template.id_of(created_at, &mined);
            if id != hit.event_id || leading_zero_bits(&id) < template.difficulty {
                return Err("the nonce from the GPU does not reproduce the event id".into());
            }
            found(PowFound::verified(template, created_at, hit.nonce, signer)?);
            renew = true;
        }

        if renew || refreshed.elapsed() >= REFRESH {
            if ctl.stopped() {
                break;
            }
            renew = false;
            created_at = unix_now().max(created_at + 1);
            let split = template.split(created_at);
            miner
                .init(&split.prefix, &split.suffix, template.difficulty, device)
                .map_err(|e| e.to_string())?;
            counted = 0;
            refreshed = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[derive(Default)]
    struct Stop {
        done: AtomicU64,
        stop: AtomicBool,
    }

    impl Control for Stop {
        fn add(&self, n: u64) {
            self.done.fetch_add(n, Ordering::Relaxed);
        }

        fn stopped(&self) -> bool {
            self.stop.load(Ordering::Relaxed)
        }
    }

    fn hits(pattern: Pattern, want: usize) -> Vec<([u8; 32], [u8; 32])> {
        let ctl = Stop::default();
        let got = Mutex::new(Vec::new());
        vanity(&ctl, &pattern, 0, &|s, p| {
            let mut v = got.lock().unwrap();
            if v.len() < want {
                v.push((s, p));
            }
            if v.len() >= want {
                ctl.stop.store(true, Ordering::Relaxed);
            }
        })
        .expect("gpu vanity");
        got.into_inner().unwrap()
    }

    fn check(pattern: Pattern, want: usize, accept: impl Fn(&str, &str) -> bool) {
        let found = hits(pattern, want);
        assert_eq!(found.len(), want);
        for (secret, public) in found {
            assert_eq!(keys::public_of(&secret).unwrap(), public);
            let (npub, hex) = (keys::npub(&public), keys::hex(&public));
            assert!(accept(&npub, &hex), "{npub} {hex}");
        }
    }

    #[test]
    fn npub_prefix_through_the_hex_prefilter() {
        check(
            Pattern::parse(Alphabet::Npub, "ac", "").unwrap(),
            3,
            |n, _| n.starts_with("npub1ac"),
        );
    }

    #[test]
    fn short_npub_prefix_without_a_prefilter() {
        check(
            Pattern::parse(Alphabet::Npub, "a", "").unwrap(),
            2,
            |n, _| n.starts_with("npub1a"),
        );
    }

    #[test]
    fn npub_suffix() {
        check(
            Pattern::parse(Alphabet::Npub, "", "as").unwrap(),
            2,
            |n, _| n[..n.len() - 6].ends_with("as"),
        );
    }

    #[test]
    fn hex_prefix_and_suffix_with_uneven_split() {
        check(
            Pattern::parse(Alphabet::Hex, "abc", "1").unwrap(),
            2,
            |_, h| h.starts_with("abc") && h.ends_with('1'),
        );
    }

    #[test]
    fn npub_prefix_and_suffix() {
        check(
            Pattern::parse(Alphabet::Npub, "ac", "q").unwrap(),
            2,
            |n, _| n.starts_with("npub1ac") && n[..n.len() - 6].ends_with('q'),
        );
    }

    #[test]
    fn pow_events_verify() {
        for fast in [false, true] {
            let template = Template {
                pubkey: "ef".repeat(32),
                kind: 1,
                tags: vec![vec!["t".into(), "x".into()]],
                content: "hello from the gpu \"quoted\" 日本語".into(),
                difficulty: 28,
                fast,
            };
            let ctl = Stop::default();
            let got = Mutex::new(Vec::new());
            pow(&ctl, &template, None, 0, &|f| {
                let mut v = got.lock().unwrap();
                v.push(f);
                if v.len() >= 2 {
                    ctl.stop.store(true, Ordering::Relaxed);
                }
            })
            .expect("gpu pow");
            let got = got.into_inner().unwrap();
            assert!(got.len() >= 2, "fast {fast}");
            for f in got {
                let id = template.id_of(f.created_at, &template.mined(f.nonce));
                assert_eq!(keys::hex(&id), f.id);
                assert!(leading_zero_bits(&id) >= 28);
            }
        }
    }
}
