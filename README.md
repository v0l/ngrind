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
| `ngrind-web` | the full `ngrind-ui` app in the browser, with the wgpu and cpu backends |

## Compared with rummage and rana

Measured on one machine: an RTX PRO 6000 Blackwell Max-Q (shared with other work, so GPU figures vary by about 20 %) and 48 CPU threads. Vanity rates are for an npub prefix, PoW for a short kind 1 note.

| | vanity keys | proof of work | GPU | in the browser |
|---|---|---|---|---|
| ngrind, cuda | 14 Gkeys/s | 7 to 11 GH/s | NVIDIA | no |
| ngrind, wgpu | 2.3 to 2.5 Gkeys/s | 5 to 7.5 GH/s | any (Vulkan, Metal, DX12, WebGPU) | yes |
| ngrind, cpu | 18 Mkeys/s | 0.65 GH/s | none | yes, with worker threads |
| rummage (C++ and CUDA) | 217 Mkeys/s | none | NVIDIA | no |
| rana 0.5.5 | 1.2 to 1.4 Mkeys/s | none | none | no |

On an Apple M1 Pro (MacBook Pro, 10 cores, 16 GB) the wgpu backend runs on Metal:

| ngrind on M1 Pro | vanity keys | proof of work |
|---|---|---|
| wgpu (Metal) | 152 Mkeys/s | 370 MH/s |
| cpu, 10 threads | 2.9 Mkeys/s | 40 MH/s |

rummage only searches for vanity keys. The vanity gap is the method: rummage and rana compute every key with a full scalar multiplication, while ngrind steps along the curve from a random point and shares one field inversion across each batch of keys, as VanitySearch does. rana uses every core, but each key goes through the generic `nostr-sdk` key path.

`--checkpoint` still uses the older scalar-multiply kernel (about 240 Mkeys/s), because its saved offsets follow that kernel's search order.

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

The secp256k1 device maths in `ngrind/kernels/GPUMath.h` is from [VanitySearch](https://github.com/JeanLucPons/VanitySearch) by Jean Luc Pons (GPL-3.0). The stepped vanity search (`ngrind/kernels/step.cu` and its WGSL twin) is ngrind's own code, but it follows VanitySearch's method: walk out from a centre point in both directions and share one field inversion across the group. The scalar-multiply vanity kernel kept for `--checkpoint` (`ngrind/kernels/vanity.cu`) is from [rummage](https://github.com/rossbates/rummage) by rossbates (MIT, see `NOTICE`). Licensed GPL-3.0-or-later.
