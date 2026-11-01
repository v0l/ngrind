use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use ngrind::event::{PowFound, Template, parse_tags};
use ngrind::{Alphabet, Control, Pattern, Phase, cpu, keys};
use secp256k1::{Keypair, SecretKey};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "ngrind",
    version,
    about = "Mine vanity npubs and NIP-13 proof of work events"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Device {
    Auto,
    Cuda,
    Wgpu,
    Cpu,
}

#[derive(Subcommand)]
enum Command {
    Vanity(VanityArgs),
    Pow(PowArgs),
    Devices,
}

#[derive(Parser)]
struct Common {
    #[arg(long, value_enum, default_value = "auto")]
    device: Device,

    #[arg(long, default_value_t = 0)]
    gpu: u32,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long, default_value_t = 1)]
    count: usize,
}

#[derive(Parser)]
struct VanityArgs {
    #[command(flatten)]
    common: Common,

    #[arg(long)]
    prefix: Option<String>,

    #[arg(long)]
    suffix: Option<String>,

    #[arg(long)]
    hex: bool,

    #[arg(long)]
    checkpoint: Option<PathBuf>,
}

#[derive(Parser)]
struct PowArgs {
    #[command(flatten)]
    common: Common,

    #[arg(long)]
    file: Option<PathBuf>,

    #[arg(long)]
    event: Option<String>,

    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=256))]
    difficulty: u32,

    /// Secret key as nsec or 64 digit hex. Visible to every process on the
    /// machine (ps, /proc/<pid>/cmdline) and to the shell history; prefer
    /// --nsec-stdin or --nsec-file.
    #[arg(long)]
    nsec: Option<String>,

    /// Read the secret key from the first line of standard input.
    #[arg(long, conflicts_with_all = ["nsec", "nsec_file"])]
    nsec_stdin: bool,

    /// Read the secret key from the first line of a file.
    #[arg(long, conflicts_with_all = ["nsec", "nsec_stdin"])]
    nsec_file: Option<PathBuf>,

    #[arg(long)]
    fast: bool,
}

struct Progress {
    done: AtomicU64,
    stop: Arc<AtomicBool>,
    unit: &'static str,
    started: Mutex<Option<Instant>>,
}

impl Control for Progress {
    fn add(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn phase(&self, phase: Phase) {
        if phase == Phase::Search {
            self.started
                .lock()
                .expect("no panic holds this lock")
                .get_or_insert_with(Instant::now);
        }
    }

    fn log(&self, text: &str) {
        eprintln!("{text}");
    }
}

fn report(progress: &Arc<Progress>, expected: f64) -> std::thread::JoinHandle<()> {
    let progress = progress.clone();
    std::thread::spawn(move || {
        let mut last = (Instant::now(), 0u64);
        while !progress.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(250));
            if last.0.elapsed() < Duration::from_secs(5) {
                continue;
            }
            let done = progress.done.load(Ordering::Relaxed);
            let rate = (done - last.1) as f64 / last.0.elapsed().as_secs_f64();
            let chance = if expected > 0.0 {
                -(-(done as f64) / expected).exp_m1() * 100.0
            } else {
                0.0
            };
            eprintln!(
                "{} {}, {}/s, {chance:.1} % chance by now",
                count(done as f64),
                progress.unit,
                count(rate)
            );
            last = (Instant::now(), done);
        }
    })
}

fn count(v: f64) -> String {
    let units = ["", "k", "M", "G", "T", "P"];
    let (mut x, mut i) = (v, 0);
    while x >= 1000.0 && i + 1 < units.len() {
        x /= 1000.0;
        i += 1;
    }
    format!("{x:.2} {}", units[i]).trim_end().to_string()
}

fn setup(unit: &'static str) -> Result<Arc<Progress>> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))?;
    Ok(Arc::new(Progress {
        done: AtomicU64::new(0),
        stop,
        unit,
        started: Mutex::new(None),
    }))
}

fn finish(progress: &Progress, matched: usize) {
    let elapsed = progress
        .started
        .lock()
        .expect("no panic holds this lock")
        .map(|s| s.elapsed().as_secs_f64());
    let done = progress.done.load(Ordering::Relaxed);
    eprintln!(
        "{matched} found after {} {}",
        count(done as f64),
        progress.unit
    );
    if let Some(t) = elapsed.filter(|t| *t > 0.0) {
        eprintln!("{:.1} s, {}/s on average", t, count(done as f64 / t));
    }
}

fn vanity(args: VanityArgs) -> Result<()> {
    let alphabet = if args.hex {
        Alphabet::Hex
    } else {
        Alphabet::Npub
    };
    let pattern = Pattern::parse(
        alphabet,
        args.prefix.as_deref().unwrap_or_default(),
        args.suffix.as_deref().unwrap_or_default(),
    )
    .map_err(anyhow::Error::msg)?;
    eprintln!(
        "searching for {}, 1 in {} keys",
        pattern.describe(),
        count(pattern.expected())
    );
    let progress = setup("keys")?;
    let reporter = report(&progress, pattern.expected());
    let hits = Mutex::new(0usize);
    let limit = args.common.count.max(1);
    let found = |secret: [u8; 32], public: [u8; 32]| {
        let mut n = hits.lock().expect("no panic holds this lock");
        if *n >= limit {
            return;
        }
        *n += 1;
        println!("npub  {}", keys::npub(&public));
        println!("hex   {}", keys::hex(&public));
        println!("nsec  {}", keys::nsec(&secret));
        println!();
        if *n >= limit {
            progress.stop.store(true, Ordering::SeqCst);
        }
    };
    let threads = args.common.threads.unwrap_or_else(cpu::cores);
    let result = match (&args.checkpoint, args.common.device) {
        (Some(path), Device::Auto | Device::Cuda) => {
            sequential(&progress, &pattern, &args, path, &found)
        }
        (Some(_), _) => Err("--checkpoint needs the CUDA backend".to_string()),
        (None, _) => {
            let backend = choose(&args.common)?;
            eprintln!("using {}", backend.label());
            ngrind::backend::vanity(&backend, progress.as_ref(), &pattern, threads, &found)
        }
    };
    progress.stop.store(true, Ordering::SeqCst);
    let _ = reporter.join();
    finish(&progress, *hits.lock().expect("no panic holds this lock"));
    result.map_err(anyhow::Error::msg)
}

#[cfg(feature = "cuda")]
fn sequential(
    progress: &Arc<Progress>,
    pattern: &Pattern,
    args: &VanityArgs,
    path: &std::path::Path,
    found: ngrind::OnFound,
) -> Result<(), String> {
    ngrind::cuda::vanity_sequential(progress.as_ref(), pattern, args.common.gpu, path, found)
}

#[cfg(not(feature = "cuda"))]
fn sequential(
    _: &Arc<Progress>,
    _: &Pattern,
    _: &VanityArgs,
    _: &std::path::Path,
    _: ngrind::OnFound,
) -> Result<(), String> {
    Err("this build has no CUDA support".into())
}

fn choose(common: &Common) -> Result<ngrind::backend::Backend> {
    use ngrind::backend::{Kind, available, best};
    let all = available();
    let want = match common.device {
        Device::Auto => return Ok(best(&all)),
        Device::Cuda => Kind::Cuda,
        Device::Wgpu => Kind::Wgpu,
        Device::Cpu => Kind::Cpu,
    };
    all.into_iter()
        .filter(|b| b.kind == want)
        .nth(if want == Kind::Cpu {
            0
        } else {
            common.gpu as usize
        })
        .with_context(|| {
            format!(
                "no {} device {} in this build or on this machine",
                want.label(),
                common.gpu
            )
        })
}

fn read_secret(read: &mut dyn std::io::Read) -> Result<[u8; 32]> {
    let mut text = String::new();
    read.read_to_string(&mut text)
        .context("reading the secret key")?;
    let line = text.lines().next().unwrap_or_default().trim();
    keys::parse_secret_any(line).map_err(anyhow::Error::msg)
}

fn signer_of(args: &PowArgs) -> Result<Option<Keypair>> {
    let secret = if args.nsec_stdin {
        eprintln!("waiting for the nsec on standard input (first line)");
        Some(read_secret(&mut std::io::stdin().lock())?)
    } else if let Some(path) = &args.nsec_file {
        Some(read_secret(&mut std::fs::File::open(path).with_context(|| {
            format!("reading {}", path.display())
        })?)?)
    } else {
        if args.nsec.is_some() {
            eprintln!(
                "warning: --nsec puts the secret key on the command line, where ps, \
                 /proc/<pid>/cmdline and the shell history can read it; \
                 prefer --nsec-stdin or --nsec-file"
            );
        }
        match args.nsec.as_deref() {
            Some(text) => Some(keys::parse_secret_any(text).map_err(anyhow::Error::msg)?),
            None => None,
        }
    };
    secret
        .map(|secret| {
            Ok(Keypair::from_secret_key(&SecretKey::from_secret_bytes(
                secret,
            )?))
        })
        .transpose()
}

fn pow(args: PowArgs) -> Result<()> {
    let text = match (&args.event, &args.file) {
        (Some(json), _) => json.clone(),
        (None, Some(path)) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        }
        _ => bail!("give the event with --event or --file"),
    };
    let event: serde_json::Value = serde_json::from_str(&text).context("the event is not JSON")?;
    let signer = signer_of(&args)?;
    let pubkey = match &signer {
        Some(pair) => keys::hex(&pair.x_only_public_key().0.to_byte_array()),
        None => event["pubkey"]
            .as_str()
            .context("the event has no pubkey and no --nsec was given")?
            .to_string(),
    };
    let kind = event["kind"]
        .as_u64()
        .and_then(|k| u16::try_from(k).ok())
        .context("kind must be an integer between 0 and 65535")?;
    let content = event
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();
    let tags = match event.get("tags") {
        Some(tags) => parse_tags(&tags.to_string()).map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    let template = Template {
        pubkey,
        kind,
        tags,
        content,
        difficulty: args.difficulty,
        fast: args.fast,
    };
    let expected = (args.difficulty as f64).exp2();
    eprintln!(
        "difficulty {} bits, 1 in {} hashes",
        args.difficulty,
        count(expected)
    );

    let progress = setup("hashes")?;
    let reporter = report(&progress, expected);
    let limit = args.common.count.max(1);
    let hits: Mutex<Vec<PowFound>> = Mutex::new(Vec::new());
    let found = |f: PowFound| {
        let mut list = hits.lock().expect("no panic holds this lock");
        if list.len() >= limit {
            return;
        }
        println!("{}", f.event);
        if !f.signed {
            eprintln!(
                "unsigned: use created_at {} and the tags above, then sign it yourself",
                f.created_at
            );
        }
        list.push(f);
        if list.len() >= limit {
            progress.stop.store(true, Ordering::SeqCst);
        }
    };
    let threads = args.common.threads.unwrap_or_else(cpu::cores);
    let backend = choose(&args.common)?;
    eprintln!("using {}", backend.label());
    let result = ngrind::backend::pow(
        &backend,
        progress.as_ref(),
        &template,
        signer.as_ref(),
        threads,
        &found,
    );
    progress.stop.store(true, Ordering::SeqCst);
    let _ = reporter.join();
    finish(
        &progress,
        hits.lock().expect("no panic holds this lock").len(),
    );
    result.map_err(anyhow::Error::msg)
}

fn devices() {
    for b in ngrind::backend::available() {
        println!("{:<5} {}  {}", b.kind.label(), b.index, b.name);
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Vanity(args) => vanity(args),
        Command::Pow(args) => pow(args),
        Command::Devices => {
            devices();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_an_nsec_from_a_stream() {
        let nsec = keys::nsec(&[7u8; 32]);
        let mut read = std::io::Cursor::new(format!("{nsec}\nand more\n"));
        assert_eq!(read_secret(&mut read).unwrap(), [7u8; 32]);
    }

    #[test]
    fn reads_a_hex_secret_from_a_stream() {
        let mut read = std::io::Cursor::new(format!("0x{}\n", keys::hex(&[9u8; 32])));
        assert!(read_secret(&mut read).is_err());
        let mut read = std::io::Cursor::new(format!("{}\n", keys::hex(&[9u8; 32])));
        assert_eq!(read_secret(&mut read).unwrap(), [9u8; 32]);
    }

    #[test]
    fn rejects_a_stream_without_a_secret() {
        let mut npub = std::io::Cursor::new(keys::npub(&[7u8; 32]));
        assert!(read_secret(&mut npub).is_err());
        let mut empty = std::io::Cursor::new("");
        assert!(read_secret(&mut empty).is_err());
    }
}
