#!/bin/sh
set -e
cd "$(dirname "$0")/.."
version=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')
bindgen=target/tools/bin/wasm-bindgen
if ! "$bindgen" --version 2>/dev/null | grep -q " $version$"; then
    cargo install --locked --root target/tools wasm-bindgen-cli --version "$version"
fi
cargo build --release --target wasm32-unknown-unknown -p ngrind-web
out=target/web
rm -rf "$out"
"$bindgen" --target web --no-typescript --out-dir "$out" target/wasm32-unknown-unknown/release/ngrind_web.wasm
if command -v wasm-opt >/dev/null; then
    wasm-opt -O2 --enable-bulk-memory --enable-nontrapping-float-to-int -o "$out/ngrind_web_bg.wasm" "$out/ngrind_web_bg.wasm"
fi
cp ngrind-web/index.html "$out/index.html"
echo "built $out"
