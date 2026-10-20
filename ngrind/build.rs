use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const KERNELS: [&str; 2] = ["pow", "vanity"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    for var in [
        "NGRIND_CUDA_ARCHS",
        "NGRIND_BLOCKS",
        "NGRIND_THREADS",
        "NGRIND_KEYS_PER_THREAD",
        "NGRIND_POW_THREADS",
        "NGRIND_POW_WAVES",
        "CUDA_PATH",
        "CUDA_HOME",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let blocks = setting("NGRIND_BLOCKS", "3072");
    let threads = setting("NGRIND_THREADS", "256");
    let keys = setting("NGRIND_KEYS_PER_THREAD", "256");
    let pow_threads = setting("NGRIND_POW_THREADS", "256");
    let pow_waves = setting("NGRIND_POW_WAVES", "1");
    println!("cargo:rustc-env=NGRIND_BLOCKS={blocks}");
    println!("cargo:rustc-env=NGRIND_THREADS={threads}");
    println!("cargo:rustc-env=NGRIND_KEYS_PER_THREAD={keys}");
    println!("cargo:rustc-env=NGRIND_POW_THREADS={pow_threads}");
    println!("cargo:rustc-env=NGRIND_POW_WAVES={pow_waves}");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = find_nvcc();
    let archs = architectures(&nvcc);
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");

    for name in KERNELS {
        let mut cmd = Command::new(&nvcc);
        cmd.arg("-allow-unsupported-compiler")
            .arg("-O2")
            .arg("--fatbin")
            .arg(format!("-I{}", kernels.display()))
            .arg(format!("-DNOSTR_BLOCKS_PER_GRID={blocks}"))
            .arg(format!("-DNOSTR_THREADS_PER_BLOCK={threads}"))
            .arg(format!("-DKEYS_PER_THREAD_BATCH={keys}"))
            .arg(format!("-DPOW_THREADS_PER_BLOCK={pow_threads}"))
            .arg(format!("-DPOW_LAUNCH_WAVES={pow_waves}"));
        for arch in &archs {
            cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
        let newest = archs
            .iter()
            .max_by_key(|a| a.parse::<u32>().unwrap_or(0))
            .expect("at least one architecture");
        cmd.arg(format!(
            "-gencode=arch=compute_{newest},code=compute_{newest}"
        ));
        cmd.arg("-o")
            .arg(out.join(format!("{name}.fatbin")))
            .arg(kernels.join(format!("{name}.cu")));
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("could not run {}: {e}", nvcc.display()));
        assert!(status.success(), "nvcc failed on {name}.cu");
    }
}

fn setting(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn find_nvcc() -> PathBuf {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let nvcc = Path::new(&root).join("bin/nvcc");
            if nvcc.exists() {
                return nvcc;
            }
        }
    }
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        let nvcc = Path::new(root).join("bin/nvcc");
        if nvcc.exists() {
            return nvcc;
        }
    }
    PathBuf::from("nvcc")
}

fn architectures(nvcc: &Path) -> Vec<String> {
    if let Ok(list) = env::var("NGRIND_CUDA_ARCHS") {
        let archs: Vec<String> = list
            .split([',', ' '])
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(str::to_string)
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        && out.status.success()
    {
        let mut found: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| l.trim().replace('.', ""))
            .filter(|c| !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit()))
            .collect();
        found.sort();
        found.dedup();
        if !found.is_empty() {
            return found;
        }
    }
    let out = Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .expect("nvcc is needed to build the CUDA kernels");
    let newest = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("compute_").map(str::to_string))
        .filter_map(|s| s.parse::<u32>().ok())
        .max()
        .expect("nvcc lists no architectures");
    vec![newest.to_string()]
}
