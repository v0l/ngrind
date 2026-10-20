# Changelog

All notable changes to ngrind are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-20

### Added
- Vanity npub and hex key search, and NIP-13 proof of work, on three backends: cuda, wgpu and cpu.
- A stepped CUDA vanity kernel, about 14 Gkeys/s on an RTX PRO 6000.
- WGSL shaders for any GPU with Vulkan, Metal, DirectX 12 or WebGPU.
- `ngrind` command line tool, with resumable sequential search on cuda (`--checkpoint`).
- `ngrind-ui` window in the bench-suite style, with a backend toggle that works mid-run.
- The full window app in the browser at https://ngrind.pages.dev, with the wgpu and cpu backends.
- Release archives for Linux, macOS and Windows, a macOS `.dmg` and a Windows `.msi`.

[Unreleased]: https://github.com/v0l/ngrind/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/v0l/ngrind/releases/tag/v0.1.0
