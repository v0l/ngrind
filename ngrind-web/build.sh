#!/bin/sh
set -e
cd "$(dirname "$0")/.."
version=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')
bindgen=target/tools/bin/wasm-bindgen
if ! "$bindgen" --version 2>/dev/null | grep -q " $version$"; then
    cargo install --locked --root target/tools wasm-bindgen-cli --version "$version"
fi
link="--shared-memory --import-memory --max-memory=4294967296 --export=__wasm_init_tls --export=__tls_size --export=__tls_align --export=__tls_base"
flags="-C target-feature=+atomics,+bulk-memory,+mutable-globals --cfg getrandom_backend=\"wasm_js\""
for arg in $link; do flags="$flags -C link-arg=$arg"; done
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="$flags"
cargo +nightly build --release --target wasm32-unknown-unknown -p ngrind-web -Z build-std=panic_abort,std
out=target/web
rm -rf "$out"
"$bindgen" --target web --no-typescript --out-dir "$out" target/wasm32-unknown-unknown/release/ngrind_web.wasm
sed -i 's/pkg.default(data.module, data.memory)/pkg.default({ module_or_path: data.module, memory: data.memory })/' "$out"/snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js
if command -v wasm-opt >/dev/null; then
    wasm-opt -O2 --enable-threads --enable-bulk-memory --enable-nontrapping-float-to-int -o "$out/ngrind_web_bg.wasm" "$out/ngrind_web_bg.wasm"
fi
cp ngrind-web/index.html ngrind-web/_headers "$out/"
echo "built $out; serve it with: python3 ngrind-web/serve.py 8080"
