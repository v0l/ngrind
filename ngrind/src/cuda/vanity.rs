use super::{Device, Error};
use crate::gtable::{CHUNKS, GTable, POINT_BYTES, VALUES};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PinnedHostSlice, PushKernelArg};
use std::path::Path;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vanity.fatbin"));

const fn parse(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let mut n = 0u64;
    let mut i = 0;
    while i < bytes.len() {
        n = n * 10 + (bytes[i] - b'0') as u64;
        i += 1;
    }
    n
}

const BLOCKS: u32 = parse(env!("NGRIND_BLOCKS")) as u32;
const THREADS: u32 = parse(env!("NGRIND_THREADS")) as u32;
const KEYS_PER_THREAD: u64 = parse(env!("NGRIND_KEYS_PER_THREAD"));
const TOTAL_THREADS: usize = BLOCKS as usize * THREADS as usize;
const RAND_STATE_BYTES: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VanityMode {
    HexPrefix = 0,
    HexSuffix = 1,
    HexBoth = 2,
    Bech32Prefix = 3,
    Bech32Suffix = 4,
    Bech32Both = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Random,
    Sequential,
}

#[derive(Debug, Clone)]
pub struct VanityHit {
    pub private_key: [u8; 32],
    pub public_key: [u8; 32],
}

pub struct VanityMiner {
    dev: Device,
    random: CudaFunction,
    sequential: CudaFunction,
    table_x: CudaSlice<u8>,
    table_y: CudaSlice<u8>,
    pattern: CudaSlice<u8>,
    pattern_len: u8,
    mode: VanityMode,
    search: SearchMode,
    states: Option<CudaSlice<u8>>,
    found: CudaSlice<u8>,
    keys: CudaSlice<u8>,
    publics: CudaSlice<u8>,
    host_found: PinnedHostSlice<u8>,
    verify: Option<Bech32Check>,
    label: String,
    start: [u8; 32],
    keys_generated: u64,
    matches_found: u64,
    iteration: u64,
    total_iterations: u64,
    pending: bool,
}

#[derive(Debug, Clone)]
pub struct Bech32Check {
    pub prefix: String,
    pub suffix: String,
}

impl Bech32Check {
    fn accepts(&self, public: &[u8; 32]) -> bool {
        let text = crate::keys::npub(public);
        let data = &text[5..text.len() - 6];
        data.starts_with(&self.prefix) && data.ends_with(&self.suffix)
    }
}

pub struct Config<'a> {
    pub pattern: &'a str,
    pub mode: VanityMode,
    pub split: Option<usize>,
    pub search: SearchMode,
    pub start: [u8; 32],
    pub verify: Option<Bech32Check>,
    pub device: u32,
}

impl VanityMiner {
    pub fn new(table: &GTable, config: Config<'_>) -> Result<VanityMiner, Error> {
        let dev = Device::open(config.device, FATBIN)?;
        unsafe { dev.ctx.disable_event_tracking() };
        let stream = dev.stream.clone();
        let table_bytes = CHUNKS * VALUES * POINT_BYTES;
        assert_eq!(table.x.len(), table_bytes);
        assert_eq!(table.y.len(), table_bytes);
        let table_x = stream.clone_htod(&table.x)?;
        let table_y = stream.clone_htod(&table.y)?;

        let limit = if config.mode as u32 >= VanityMode::Bech32Prefix as u32 {
            52
        } else {
            16
        };
        let bytes = config.pattern.as_bytes();
        let pattern_len = bytes.len().min(limit);
        let mut padded = vec![0u8; limit];
        padded[..pattern_len].copy_from_slice(&bytes[..pattern_len]);
        let pattern = stream.clone_htod(&padded)?;

        let split = config.split.unwrap_or(pattern_len / 2).min(pattern_len) as u8;
        write_symbol(&dev, "d_vanitySplit", &[split])?;

        let mut states = None;
        let random_init = dev.module.load_function("ngrind_rand_init")?;
        let random = dev.module.load_function("ngrind_vanity_random")?;
        let sequential = dev.module.load_function("ngrind_vanity_sequential")?;
        let cfg = LaunchConfig {
            grid_dim: (BLOCKS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        match config.search {
            SearchMode::Random => {
                let buffer = stream.alloc_zeros::<u8>(TOTAL_THREADS * RAND_STATE_BYTES)?;
                let seed =
                    u64::from_le_bytes(config.start[24..32].try_into().expect("eight bytes"));
                let mut launch = stream.launch_builder(&random_init);
                launch.arg(&buffer).arg(&seed);
                unsafe { launch.launch(cfg) }.map_err(|e| Error::Gpu(e.to_string()))?;
                states = Some(buffer);
            }
            SearchMode::Sequential => write_symbol(&dev, "d_startOffset", &config.start)?,
        }

        let found = stream.alloc_zeros::<u8>(TOTAL_THREADS)?;
        let keys = stream.alloc_zeros::<u8>(TOTAL_THREADS * 32)?;
        let publics = stream.alloc_zeros::<u8>(TOTAL_THREADS * 32)?;
        let host_found = unsafe { dev.ctx.alloc_pinned_with_flags::<u8>(TOTAL_THREADS, 0) }?;
        stream.synchronize()?;

        let total_iterations = match config.search {
            SearchMode::Random => 0,
            SearchMode::Sequential => {
                let bits = match (
                    &config.verify,
                    config.mode as u32 >= VanityMode::Bech32Prefix as u32,
                ) {
                    (Some(v), _) => 5 * (v.prefix.len() + v.suffix.len()),
                    (None, true) => 5 * pattern_len,
                    (None, false) => 4 * pattern_len,
                };
                let space = if bits >= 64 { u64::MAX } else { 1u64 << bits };
                let per_iteration = TOTAL_THREADS as u64 * KEYS_PER_THREAD;
                space.div_ceil(per_iteration)
            }
        };

        Ok(VanityMiner {
            dev,
            random,
            sequential,
            table_x,
            table_y,
            pattern,
            pattern_len: pattern_len as u8,
            mode: config.mode,
            search: config.search,
            states,
            found,
            keys,
            publics,
            host_found,
            verify: config.verify,
            label: format!(
                "{}:{}:{}",
                config.mode as u32,
                config.pattern,
                config.split.unwrap_or(pattern_len / 2)
            ),
            start: config.start,
            keys_generated: 0,
            matches_found: 0,
            iteration: 0,
            total_iterations,
            pending: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        self.dev.ctx.synchronize()?;
        self.dev.ctx.check_err()?;
        Ok(())
    }

    pub fn device_name(&self) -> &str {
        &self.dev.name
    }

    pub fn iterate(&mut self) -> Result<(), Error> {
        let stream = self.dev.stream.clone();
        stream.memset_zeros(&mut self.found)?;
        let cfg = LaunchConfig {
            grid_dim: (BLOCKS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mode = self.mode as i32;
        match self.search {
            SearchMode::Sequential => {
                let mut launch = stream.launch_builder(&self.sequential);
                launch
                    .arg(&self.iteration)
                    .arg(&self.table_x)
                    .arg(&self.table_y)
                    .arg(&self.pattern)
                    .arg(&self.pattern_len)
                    .arg(&mode)
                    .arg(&self.found)
                    .arg(&self.keys)
                    .arg(&self.publics);
                unsafe { launch.launch(cfg) }.map_err(|e| Error::Gpu(e.to_string()))?;
                self.iteration += 1;
            }
            SearchMode::Random => {
                let states = self.states.as_ref().expect("allocated for random search");
                let mut launch = stream.launch_builder(&self.random);
                launch
                    .arg(states)
                    .arg(&self.table_x)
                    .arg(&self.table_y)
                    .arg(&self.pattern)
                    .arg(&self.pattern_len)
                    .arg(&mode)
                    .arg(&self.found)
                    .arg(&self.keys)
                    .arg(&self.publics);
                unsafe { launch.launch(cfg) }.map_err(|e| Error::Gpu(e.to_string()))?;
            }
        }
        stream.memcpy_dtoh(&self.found, &mut self.host_found)?;
        self.keys_generated += TOTAL_THREADS as u64 * KEYS_PER_THREAD;
        self.pending = true;
        Ok(())
    }

    pub fn collect(&mut self) -> Result<Vec<VanityHit>, Error> {
        if !self.pending {
            return Ok(Vec::new());
        }
        self.pending = false;
        let stream = self.dev.stream.clone();
        stream.synchronize()?;
        let threads: Vec<usize> = {
            let flags = self.host_found.as_slice()?;
            flags
                .iter()
                .enumerate()
                .filter(|(_, f)| **f != 0)
                .map(|(i, _)| i)
                .collect()
        };
        if threads.is_empty() {
            return Ok(Vec::new());
        }
        let secrets = stream.clone_dtoh(&self.keys)?;
        let publics = stream.clone_dtoh(&self.publics)?;
        stream.synchronize()?;
        let mut hits = Vec::new();
        for i in threads {
            let public: [u8; 32] = publics[i * 32..i * 32 + 32].try_into().expect("32 bytes");
            if let Some(check) = &self.verify
                && !check.accepts(&public)
            {
                continue;
            }
            let mut private_key = [0u8; 32];
            for (k, b) in private_key.iter_mut().enumerate() {
                *b = secrets[i * 32 + 31 - k];
            }
            self.matches_found += 1;
            hits.push(VanityHit {
                private_key,
                public_key: public,
            });
        }
        Ok(hits)
    }

    pub fn keys_generated(&self) -> u64 {
        self.keys_generated
    }

    pub fn matches_found(&self) -> u64 {
        self.matches_found
    }

    pub fn progress(&self) -> f64 {
        if self.search != SearchMode::Sequential || self.total_iterations == 0 {
            return 0.0;
        }
        self.iteration as f64 / self.total_iterations as f64
    }

    pub fn save_checkpoint(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut text = String::new();
        text.push_str("# ngrind sequential search checkpoint\n");
        text.push_str(
            "# WARNING: this file contains your search offset, protect it like a private key\n",
        );
        text.push_str(&format!("pattern={}\n", self.label));
        text.push_str(&format!("startOffset={}\n", crate::keys::hex(&self.start)));
        text.push_str(&format!("currentIteration={}\n", self.iteration));
        text.push_str(&format!("keysGenerated={}\n", self.keys_generated));
        text.push_str(&format!("matchesFound={}\n", self.matches_found));
        text.push_str(&format!("totalIterations={}\n", self.total_iterations));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        options.open(path)?.write_all(text.as_bytes())
    }

    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut offset = None;
        let mut label = None;
        let (mut iteration, mut keys, mut matches) =
            (self.iteration, self.keys_generated, self.matches_found);
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            match key {
                "pattern" => label = Some(value.trim().to_string()),
                "startOffset" => offset = Some(value.trim().to_string()),
                "currentIteration" => {
                    iteration = value.trim().parse().map_err(|_| "bad iteration")?
                }
                "keysGenerated" => keys = value.trim().parse().map_err(|_| "bad key count")?,
                "matchesFound" => matches = value.trim().parse().map_err(|_| "bad match count")?,
                _ => {}
            }
        }
        if label.as_deref() != Some(self.label.as_str()) {
            return Err("checkpoint was made for a different pattern".into());
        }
        let offset = offset.ok_or("checkpoint has no startOffset")?;
        if offset != crate::keys::hex(&self.start) {
            return Err("checkpoint is from a different search session".into());
        }
        self.iteration = iteration;
        self.keys_generated = keys;
        self.matches_found = matches;
        Ok(())
    }
}

fn write_symbol(dev: &Device, name: &str, bytes: &[u8]) -> Result<(), Error> {
    let mut symbol = dev.module.get_global(name, &dev.stream)?;
    dev.stream.memcpy_htod(bytes, &mut symbol)?;
    dev.stream.synchronize()?;
    Ok(())
}
