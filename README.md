# ngrind

Mines vanity npubs and NIP-13 proof of work events on the GPU or the CPU. Three backends, picked at run time:

| backend | for | notes |
|---|---|---|
| `cuda` | NVIDIA | kernels built by `nvcc`, loaded through `cudarc`. Fastest by a wide margin. |
| `wgpu` | any GPU with Vulkan, Metal or DirectX 12, and the browser (WebGPU) | WGSL shaders. No toolkit needed. |
| `cpu` | everything | |

`auto` (the default) picks cuda if there is an NVIDIA driver, then a hardware wgpu adapter, then the cpu. A binary does not link `libcudart` and runs on a machine with no NVIDIA driver.

| crate | what |
|---|---|
| `ngrind` | library: the miners, event templates, key and pattern handling |
| `ngrind-cli` | the `ngrind` command |
| `ngrind-ui` | window in the bench-suite style, with a backend toggle that works mid-run |
| `ngrind-web` | the same miners in a web page, on WebGPU |

On an RTX PRO 6000 (a measured example, shared with other work):

| | cuda | wgpu | cpu, 48 threads |
|---|---|---|---|
| vanity keys | 12 Gkeys/s | 1.9 to 2.6 Gkeys/s | 9 Mkeys/s |
| proof of work | 10.9 GH/s | 5.1 to 7.5 GH/s | 0.65 GH/s |

Both GPU vanity miners step along the curve and share one inversion across each batch of keys. `--checkpoint` still uses the older scalar-multiply kernel (about 240 Mkeys/s), because its saved offsets follow that kernel's search order.

## Build

```bash
cargo build --release
cargo build --release --no-default-features --features wgpu
cargo build --release --no-default-features
./ngrind-web/build.sh
./ngrind-web/deploy.sh
```

The first needs the CUDA toolkit (`nvcc`) and builds kernels for the architectures `nvidia-smi` reports. The second needs no CUDA. The third is cpu only. `ngrind-web/build.sh` builds the full egui app for the browser into `target/web`, with the wgpu and cpu backends. Like gcad it uses worker threads, so it needs nightly with `rust-src`, `clang` for the secp256k1 C code, and a page served with COOP/COEP headers (`ngrind-web/_headers`, or `python3 ngrind-web/serve.py`). `deploy.sh` publishes it to https://ngrind.pages.dev.

Build settings, all optional environment variables: `NGRIND_CUDA_ARCHS` (for example `86,89,120`), `NGRIND_BLOCKS`, `NGRIND_THREADS`, `NGRIND_KEYS_PER_THREAD`, `NGRIND_POW_THREADS`, `NGRIND_POW_WAVES`, `CUDA_PATH`.

## Use

```bash
ngrind devices
ngrind vanity --prefix alice --device wgpu
ngrind vanity --prefix alice --count 3
ngrind vanity --prefix ac --suffix q --device cpu
ngrind vanity --hex --prefix cafe
ngrind vanity --prefix satoshi --checkpoint run.ckpt
ngrind pow --file note.json --difficulty 28 --nsec nsec1...
ngrind pow --event '{"pubkey":"...","kind":1,"content":"gm"}' --difficulty 24 --fast
```

Found keys print to stdout. Progress, speed and the chance of a hit so far go to stderr every five seconds. With `--checkpoint` the search is sequential and resumable (cuda only). The checkpoint holds the search offset, so keep it private (mode 0600).

## Credits

The secp256k1 device maths in `ngrind/kernels/GPUMath.h` is from [VanitySearch](https://github.com/JeanLucPons/VanitySearch) by Jean Luc Pons (GPL-3.0). The vanity and PoW kernels, the NIP-13 nonce layout and the original GPU miner are from [rummage](https://github.com/rossbates/rummage) by rossbates (MIT, see `NOTICE`). Licensed GPL-3.0-or-later.
