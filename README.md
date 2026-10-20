# ngrind

Mines vanity npubs and NIP-13 proof of work events on an NVIDIA GPU or on the CPU. Pure Rust on the host, with the CUDA kernels compiled to a fatbin by `nvcc` at build time and loaded through `cudarc` at run time. A binary does not link `libcudart`, and runs on a machine with no NVIDIA driver (CPU only).

| crate | what |
|---|---|
| `ngrind` | library: GPU and CPU miners, event templates, key and pattern handling |
| `ngrind-cli` | the `ngrind` command |
| `ngrind-ui` | window in the bench-suite style, with a GPU/CPU toggle that works mid-run |

## Build

```bash
cargo build --release
cargo build --release --no-default-features
```

The first needs the CUDA toolkit (`nvcc`). It builds kernels for the architectures `nvidia-smi` reports. The second needs no CUDA at all and only offers the CPU.

Build settings, all optional environment variables: `NGRIND_CUDA_ARCHS` (for example `86,89,120`), `NGRIND_BLOCKS`, `NGRIND_THREADS`, `NGRIND_KEYS_PER_THREAD`, `NGRIND_POW_THREADS`, `NGRIND_POW_WAVES`, `CUDA_PATH`.

## Use

```bash
ngrind devices
ngrind vanity --prefix alice --count 3
ngrind vanity --prefix ac --suffix q --device cpu
ngrind vanity --hex --prefix cafe
ngrind vanity --prefix satoshi --checkpoint run.ckpt
ngrind pow --file note.json --difficulty 28 --nsec nsec1...
ngrind pow --event '{"pubkey":"...","kind":1,"content":"gm"}' --difficulty 24 --fast
```

Found keys print to stdout. Progress, speed and the chance of a hit so far go to stderr every five seconds. With `--checkpoint` the search is sequential and resumable. The checkpoint holds the search offset, so keep it private (mode 0600).

## Credits

The secp256k1 device maths in `ngrind/kernels/GPUMath.h` is from [VanitySearch](https://github.com/JeanLucPons/VanitySearch) by Jean Luc Pons (GPL-3.0). The vanity and PoW kernels, the NIP-13 nonce layout and the original GPU miner are from [rummage](https://github.com/rossbates/rummage) by rossbates (MIT, see `NOTICE`). Licensed GPL-3.0-or-later.
