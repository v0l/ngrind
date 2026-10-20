use super::field;
use std::fmt::Write;

pub const WORKGROUP: u32 = 64;
pub const MAX_SYMBOLS: usize = 52;
pub const RECORD: usize = 10;

pub fn out_words(max_hits: usize) -> usize {
    1 + max_hits * RECORD
}

pub const U_THREADS: usize = 0;
pub const U_ITERS: usize = 1;
pub const U_BITS: usize = 2;
pub const U_COUNT: usize = 3;
pub const U_SYMBOLS: usize = 4;
pub const U_WORDS: usize = U_SYMBOLS + 2 * MAX_SYMBOLS;

pub fn source(half: usize, max_hits: usize) -> String {
    assert!((1..=127).contains(&half));
    let mut s = field::library();
    let _ = writeln!(s, "const HALF: u32 = {half}u;");
    let _ = writeln!(s, "const PREFW: u32 = {}u;", (half + 1) * 8);
    let _ = writeln!(s, "const MAXH: u32 = {max_hits}u;");
    s.push_str(
        r#"
@group(0) @binding(0) var<storage, read> u: array<u32>;
@group(0) @binding(1) var<storage, read> gt: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<atomic<u32>>;

fn gx(i: u32) -> F {
    var r: F;
    for (var k = 0u; k < 8u; k = k + 1u) { r[k] = gt[i * 16u + k]; }
    return r;
}

fn gy(i: u32) -> F {
    var r: F;
    for (var k = 0u; k < 8u; k = k + 1u) { r[k] = gt[i * 16u + 8u + k]; }
    return r;
}

fn stash(a: ptr<function, array<u32, PREFW>>, i: u32, v: F) {
    for (var k = 0u; k < 8u; k = k + 1u) { (*a)[i * 8u + k] = v[k]; }
}

fn fetch(a: ptr<function, array<u32, PREFW>>, i: u32) -> F {
    var r: F;
    for (var k = 0u; k < 8u; k = k + 1u) { r[k] = (*a)[i * 8u + k]; }
    return r;
}

fn symbol(x: F, i: u32, bits: u32) -> u32 {
    var xv = x;
    let pos = i * bits;
    let t = 7u - (pos >> 5u);
    let off = pos & 31u;
    let hi = xv[t];
    var lo = 0u;
    if (t > 0u) { lo = xv[t - 1u]; }
    var w = hi << off;
    if (off > 0u) { w = w | (lo >> (32u - off)); }
    return w >> (32u - bits);
}

fn check(x: F, t: u32, code: u32) {
    let n = u[3u];
    let bits = u[2u];
    for (var s = 0u; s < n; s = s + 1u) {
        if (symbol(x, u[4u + 2u * s], bits) != u[5u + 2u * s]) { return; }
    }
    let k = atomicAdd(&out[0], 1u);
    if (k < MAXH) {
        let b = 1u + k * 10u;
        atomicStore(&out[b], t);
        atomicStore(&out[b + 1u], code);
        for (var i = 0u; i < 8u; i = i + 1u) { atomicStore(&out[b + 2u + i], x[i]); }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    if (t >= u[0u]) { return; }
    var px: F;
    var py: F;
    for (var k = 0u; k < 8u; k = k + 1u) {
        px[k] = state[t * 16u + k];
        py[k] = state[t * 16u + 8u + k];
    }
    let zero = F(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let iters = u[1u];
    for (var it = 0u; it < iters; it = it + 1u) {
        let c = it * (2u * HALF + 1u) + HALF;
        check(px, t, c);

        var pref: array<u32, PREFW>;
        var acc = fsub(gx(0u), px);
        stash(&pref, 0u, acc);
        for (var i = 1u; i <= HALF; i = i + 1u) {
            acc = fmul(acc, fsub(gx(i), px));
            stash(&pref, i, acc);
        }
        var inv = finv(acc);

        let jump_dx = fsub(gx(HALF), px);
        let jump_inv = fmul(inv, fetch(&pref, HALF - 1u));
        inv = fmul(inv, jump_dx);
        let lam = fmul(fsub(gy(HALF), py), jump_inv);
        let nx = fsub(fsub(fsqr(lam), px), gx(HALF));
        let ny = fsub(fmul(lam, fsub(px, nx)), py);

        for (var ii = 0u; ii < HALF; ii = ii + 1u) {
            let i = HALF - 1u - ii;
            let dx = fsub(gx(i), px);
            var inv_i = inv;
            if (i > 0u) {
                inv_i = fmul(inv, fetch(&pref, i - 1u));
                inv = fmul(inv, dx);
            }
            let gyi = gy(i);
            let lp = fmul(fsub(gyi, py), inv_i);
            check(fsub(fsub(fsqr(lp), px), gx(i)), t, c + i + 1u);
            let lm = fmul(fsub(fsub(zero, gyi), py), inv_i);
            check(fsub(fsub(fsqr(lm), px), gx(i)), t, c - i - 1u);
        }
        px = nx;
        py = ny;
    }
    for (var k = 0u; k < 8u; k = k + 1u) {
        state[t * 16u + k] = px[k];
        state[t * 16u + 8u + k] = py[k];
    }
}
"#,
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates() {
        for half in [1, 8, 32, 64, 127] {
            let src = source(half, 8);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{half}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{half}: {e:?}"));
        }
    }
}
