use crate::PowFound;
use crate::event::{Template, unix_now};
use crate::keys;
use crate::pattern::Pattern;
use secp256k1::Keypair;
use secp256k1::{PublicKey, Scalar, SecretKey};
use sha2::compress256;
use sha2::digest::generic_array::GenericArray;
use sha2::digest::typenum::U64;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

const VANITY_BATCH: u64 = 2048;
const POW_BATCH: u64 = 1 << 14;
const POW_CHUNK: u64 = 1 << 22;
const FIRST_NONCE: u64 = 100_000_000_000;
const REFRESH: std::time::Duration = std::time::Duration::from_secs(30);

const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

type Block = GenericArray<u8, U64>;

pub use crate::control::{Control, OnFound, Phase};

pub fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn one() -> SecretKey {
    let mut bytes = [0u8; 32];
    bytes[31] = 1;
    SecretKey::from_byte_array(bytes).expect("one is a valid secret")
}

fn offset(i: u64) -> Option<Scalar> {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&i.to_be_bytes());
    Scalar::from_be_bytes(bytes).ok()
}

pub fn vanity(ctl: &dyn Control, pattern: &Pattern, threads: usize, found: OnFound) {
    ctl.phase(Phase::Search);
    ctl.log(&format!("searching on {} threads", threads.max(1)));
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| vanity_thread(ctl, pattern, found));
        }
    });
}

fn vanity_thread(link: &dyn Control, pattern: &Pattern, found: OnFound) {
    let secp = keys::secp();
    let step = PublicKey::from_secret_key(secp, &one());
    let mut base = keys::random_secret();
    let mut point = PublicKey::from_secret_key(secp, &base);
    let mut i = 0u64;
    let mut batch = 0u64;
    loop {
        let compressed = point.serialize();
        let mut x = [0u8; 32];
        x.copy_from_slice(&compressed[1..]);
        if pattern.matches(&x)
            && !link.stopped()
            && let Some(secret) = offset(i).and_then(|o| base.add_tweak(&o).ok())
            && keys::public_of(&secret.secret_bytes()).is_ok_and(|p| p == x)
        {
            found(secret.secret_bytes(), x);
        }
        i += 1;
        batch += 1;
        point = match point.combine(&step) {
            Ok(p) => p,
            Err(_) => {
                base = keys::random_secret();
                i = 0;
                PublicKey::from_secret_key(secp, &base)
            }
        };
        if batch == VANITY_BATCH {
            link.add(batch);
            batch = 0;
            if link.stopped() {
                return;
            }
        }
    }
}

struct Tail {
    blocks: Vec<Block>,
    at: usize,
    digits: Vec<u8>,
}

impl Tail {
    fn new(prefix_rem: &[u8], first: u64, suffix: &[u8], total_prefix: usize) -> Tail {
        let digits: Vec<u8> = first.to_string().into_bytes();
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(prefix_rem);
        let at = bytes.len();
        bytes.extend_from_slice(&digits);
        bytes.extend_from_slice(suffix);
        let message = total_prefix + digits.len() + suffix.len();
        bytes.push(0x80);
        while bytes.len() % 64 != 56 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&((message as u64) * 8).to_be_bytes());
        let blocks = bytes
            .chunks_exact(64)
            .map(Block::clone_from_slice)
            .collect();
        Tail { blocks, at, digits }
    }

    fn write(&mut self, k: usize) {
        let pos = self.at + k;
        self.blocks[pos / 64][pos % 64] = self.digits[k];
    }

    fn step(&mut self) {
        for k in (0..self.digits.len()).rev() {
            if self.digits[k] < b'9' {
                self.digits[k] += 1;
                self.write(k);
                return;
            }
            self.digits[k] = b'0';
            self.write(k);
        }
    }
}

fn pow10_above(n: u64) -> u64 {
    let mut p = 10u64;
    while p <= n {
        match p.checked_mul(10) {
            Some(next) => p = next,
            None => return u64::MAX,
        }
    }
    p
}

fn clears(state: &[u32; 8], target: u32) -> bool {
    if state[0].leading_zeros() < target.min(32) {
        return false;
    }
    let mut bits = 0;
    for word in state {
        if *word == 0 {
            bits += 32;
        } else {
            bits += word.leading_zeros();
            break;
        }
    }
    bits >= target
}

pub fn pow_search(
    link: &dyn Control,
    prefix: &str,
    suffix: &str,
    target: u32,
    threads: usize,
    deadline: Instant,
) -> Option<u64> {
    let bytes = prefix.as_bytes();
    let full = bytes.len() / 64 * 64;
    let mut midstate = SHA256_IV;
    let blocks: Vec<Block> = bytes[..full]
        .chunks_exact(64)
        .map(Block::clone_from_slice)
        .collect();
    compress256(&mut midstate, &blocks);
    let rem = &bytes[full..];

    let chunk = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let found: Mutex<Option<u64>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let mut state;
                while !done.load(Ordering::Relaxed) && !link.stopped() && Instant::now() < deadline
                {
                    let start = FIRST_NONCE.saturating_add(
                        chunk
                            .fetch_add(1, Ordering::Relaxed)
                            .saturating_mul(POW_CHUNK),
                    );
                    if start == u64::MAX {
                        return;
                    }
                    let end = start.saturating_add(POW_CHUNK).min(pow10_above(start));
                    let mut tail = Tail::new(rem, start, suffix.as_bytes(), bytes.len());
                    let mut batch = 0u64;
                    for nonce in start..end {
                        state = midstate;
                        compress256(&mut state, &tail.blocks);
                        if clears(&state, target) {
                            done.store(true, Ordering::Relaxed);
                            let mut slot = found.lock().expect("no panic holds this lock");
                            slot.get_or_insert(nonce);
                            link.add(batch + 1);
                            return;
                        }
                        tail.step();
                        batch += 1;
                        if batch == POW_BATCH {
                            link.add(batch);
                            batch = 0;
                            if done.load(Ordering::Relaxed) || link.stopped() {
                                return;
                            }
                        }
                    }
                    link.add(batch);
                }
            });
        }
    });
    let nonce = *found.lock().expect("no panic holds this lock");
    nonce
}

pub fn pow(
    ctl: &dyn Control,
    template: &Template,
    signer: Option<&Keypair>,
    threads: usize,
    found: &(dyn Fn(PowFound) + Sync),
) -> Result<(), String> {
    ctl.phase(Phase::Search);
    ctl.log(&format!("searching on {} threads", threads.max(1)));
    let mut created_at = unix_now();
    while !ctl.stopped() {
        let split = template.split(created_at);
        let deadline = Instant::now() + REFRESH;
        if let Some(nonce) = pow_search(
            ctl,
            &split.prefix,
            &split.suffix,
            template.difficulty,
            threads,
            deadline,
        ) {
            found(PowFound::verified(template, created_at, nonce, signer)?);
        }
        created_at = unix_now().max(created_at + 1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Template, leading_zero_bits};
    use crate::pattern::{Alphabet, Pattern};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Duration;

    #[derive(Default)]
    struct Counter {
        done: AtomicU64,
        stop: AtomicBool,
    }

    impl Control for Counter {
        fn add(&self, n: u64) {
            self.done.fetch_add(n, Ordering::Relaxed);
        }

        fn stopped(&self) -> bool {
            self.stop.load(Ordering::Relaxed)
        }
    }

    fn first_match(pattern: &Pattern) -> ([u8; 32], [u8; 32]) {
        let ctl = Counter::default();
        let hit = Mutex::new(None);
        vanity(&ctl, pattern, 2, &|s, p| {
            hit.lock().unwrap().get_or_insert((s, p));
            ctl.stop.store(true, Ordering::Relaxed);
        });
        hit.into_inner()
            .unwrap()
            .expect("a short pattern is found in a blink")
    }

    #[test]
    fn cpu_finds_a_hex_prefix_and_the_secret_matches() {
        let (secret, public) = first_match(&Pattern::parse(Alphabet::Hex, "ab", "").unwrap());
        assert!(keys::hex(&public).starts_with("ab"));
        assert_eq!(keys::public_of(&secret).unwrap(), public);
    }

    #[test]
    fn cpu_finds_an_npub_prefix_and_suffix() {
        let (_, public) = first_match(&Pattern::parse(Alphabet::Npub, "a", "s").unwrap());
        let text = keys::npub(&public);
        assert!(text.starts_with("npub1a"), "{text}");
        assert!(text[..text.len() - 6].ends_with('s'), "{text}");
    }

    #[test]
    fn cpu_pow_hashes_match_the_event_id() {
        for (content, tags, fast) in [
            ("gm", vec![], false),
            (
                "a longer note with \"quotes\" and unicode 日本語 that spills into a second block of the hash",
                vec![],
                false,
            ),
            ("fast", vec![vec!["t".to_string(), "x".to_string()]], true),
        ] {
            let template = Template {
                pubkey: "cd".repeat(32),
                kind: 1,
                tags,
                content: content.into(),
                difficulty: 16,
                fast,
            };
            let split = template.split(1_700_000_000);
            let ctl = Counter::default();
            let nonce = pow_search(
                &ctl,
                &split.prefix,
                &split.suffix,
                16,
                4,
                Instant::now() + Duration::from_secs(60),
            )
            .expect("16 bits is quick");
            let mined = template.mined(nonce);
            let id = template.id_of(1_700_000_000, &mined);
            assert!(
                leading_zero_bits(&id) >= 16,
                "{content}: {}",
                keys::hex(&id)
            );
        }
    }

    #[test]
    fn cpu_pow_stops_at_the_deadline() {
        let template = Template {
            pubkey: "cd".repeat(32),
            kind: 1,
            tags: vec![],
            content: "x".into(),
            difficulty: 60,
            fast: false,
        };
        let split = template.split(1);
        let ctl = Counter::default();
        let started = Instant::now();
        let nonce = pow_search(
            &ctl,
            &split.prefix,
            &split.suffix,
            60,
            2,
            started + Duration::from_millis(300),
        );
        assert!(nonce.is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(ctl.done.load(Ordering::Relaxed) > 0);
    }
}
