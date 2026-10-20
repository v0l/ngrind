use crate::sha::K;
use std::fmt::Write;

pub const WORKGROUP: u32 = 64;
pub const OUT_WORDS: usize = 12;
pub const HEADER: usize = 64;

pub const P_MIDSTATE: usize = 0;
pub const P_TPL: usize = 8;
pub const P_HIGH_LO: usize = 40;
pub const P_HIGH_HI: usize = 41;
pub const P_R0: usize = 42;
pub const P_COUNT: usize = 43;
pub const P_TARGET: usize = 44;
pub const P_FIRST_TARGET: usize = 45;

const NAMES: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

fn var(prefix: &str, role: usize, i: usize) -> String {
    format!("{prefix}{}", NAMES[(8 + role - i % 8) % 8])
}

fn round(s: &mut String, prefix: &str, i: usize, kw: &str) {
    let (a, b, c, d) = (
        var(prefix, 0, i),
        var(prefix, 1, i),
        var(prefix, 2, i),
        var(prefix, 3, i),
    );
    let (e, f, g, h) = (
        var(prefix, 4, i),
        var(prefix, 5, i),
        var(prefix, 6, i),
        var(prefix, 7, i),
    );
    let _ = writeln!(
        s,
        "{{ let t1 = {h} + bsig1({e}) + ch({e}, {f}, {g}) + ({kw}); let t2 = bsig0({a}) + maj({a}, {b}, {c}); {d} = {d} + t1; {h} = t1 + t2; }}"
    );
}

fn schedule(s: &mut String, w: &str, i: usize) {
    let _ = writeln!(
        s,
        "{w}{} = {w}{} + ssig1({w}{}) + {w}{} + ssig0({w}{});",
        i & 15,
        i & 15,
        (i - 2) & 15,
        (i - 7) & 15,
        (i - 15) & 15
    );
}

fn declare_state(s: &mut String, prefix: &str, from: &str) {
    for (k, n) in NAMES.iter().enumerate() {
        let _ = writeln!(s, "var {prefix}{n}: u32 = {from}{k};");
    }
}

fn fold_state(s: &mut String, into: &str, prefix: &str) {
    for (k, n) in NAMES.iter().enumerate() {
        let _ = writeln!(s, "{into}{k} = {into}{k} + {prefix}{n};");
    }
}

fn k_plus(i: usize, w: &str) -> String {
    format!("0x{:08x}u + {w}", K[i])
}

fn digit(src: &str, j: usize) -> String {
    let scale = 10u32.pow((j % 8) as u32);
    let place = if scale == 1 {
        src.to_string()
    } else {
        format!("({src} / {scale}u)")
    };
    format!("(0x30u + ({place} % 10u))")
}

pub fn source(b0: usize, wv: usize, high_digits: usize, n_trail: usize) -> String {
    let mut s = String::new();
    s.push_str(
        r#"
@group(0) @binding(0) var<storage, read> p: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<atomic<u32>, 12>;

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn bsig0(x: u32) -> u32 { return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u); }
fn bsig1(x: u32) -> u32 { return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u); }
fn ssig0(x: u32) -> u32 { return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u); }
fn ssig1(x: u32) -> u32 { return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u); }
fn ch(x: u32, y: u32, z: u32) -> u32 { return z ^ (x & (y ^ z)); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) | (z & (x | y)); }

"#,
    );
    let _ = writeln!(
        s,
        "@compute @workgroup_size({WORKGROUP})\nfn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
    );
    let _ = writeln!(
        s,
        "let gid = (wid.y * nwg.x + wid.x) * {WORKGROUP}u + lid.x;"
    );
    let _ = writeln!(s, "if (gid >= p[{P_COUNT}u]) {{ return; }}");
    let _ = writeln!(s, "let r = p[{P_R0}u] + gid;");
    let _ = writeln!(s, "let stp = r / 10u;");
    let _ = writeln!(s, "let thousands = r % 10u;");
    let _ = writeln!(s, "var hl = p[{P_HIGH_LO}u] + stp;");
    let _ = writeln!(s, "var hh = p[{P_HIGH_HI}u];");
    let _ = writeln!(
        s,
        "if (hl >= 100000000u) {{ hl = hl - 100000000u; hh = hh + 1u; }}"
    );

    let words = (b0 + 1) * 16;
    for i in 0..words {
        let _ = writeln!(s, "var m{i}: u32 = p[{}u];", P_TPL + i);
    }
    for j in 0..high_digits {
        let word = (b0 * 16 + wv) as i64 - 1 - (j / 4) as i64;
        assert!(word >= 0, "nonce digits fall before the template");
        let src = if j < 8 { "hl" } else { "hh" };
        let _ = writeln!(
            s,
            "m{word} = m{word} | ({} << {}u);",
            digit(src, j),
            8 * (j & 3)
        );
    }
    for i in 0..8 {
        let _ = writeln!(s, "var st{i}: u32 = p[{}u];", P_MIDSTATE + i);
    }
    if b0 == 1 {
        for i in 0..16 {
            let _ = writeln!(s, "var f{i}: u32 = m{i};");
        }
        declare_state(&mut s, "x", "st");
        for i in 0..16 {
            round(&mut s, "x", i, &k_plus(i, &format!("f{i}")));
        }
        for i in 16..64 {
            schedule(&mut s, "f", i);
            round(&mut s, "x", i, &k_plus(i, &format!("f{}", i & 15)));
        }
        fold_state(&mut s, "st", "x");
    }
    for i in 0..16 {
        let _ = writeln!(s, "let w{i}: u32 = m{};", b0 * 16 + i);
    }
    declare_state(&mut s, "q", "st");
    for i in 0..wv {
        round(&mut s, "q", i, &k_plus(i, &format!("w{i}")));
    }

    let _ = writeln!(s, "let low_base = 0x30303030u | (thousands << 24u);");
    let _ = writeln!(s, "for (var hd = 0u; hd < 10u; hd = hd + 1u) {{");
    let _ = writeln!(s, "if (atomicLoad(&out[0]) != 0u) {{ return; }}");
    let _ = writeln!(s, "for (var td = 0u; td < 10u; td = td + 1u) {{");
    let _ = writeln!(s, "for (var od = 0u; od < 10u; od = od + 1u) {{");
    for i in 0..16 {
        if i == wv {
            let _ = writeln!(
                s,
                "var n{i}: u32 = low_base | (hd << 16u) | (td << 8u) | od;"
            );
        } else {
            let _ = writeln!(s, "var n{i}: u32 = w{i};");
        }
    }
    for n in NAMES {
        let _ = writeln!(s, "var v{n}: u32 = q{n};");
    }
    for i in wv..16 {
        round(&mut s, "v", i, &k_plus(i, &format!("n{i}")));
    }
    for i in 16..64 {
        schedule(&mut s, "n", i);
        round(&mut s, "v", i, &k_plus(i, &format!("n{}", i & 15)));
    }
    for (i, n) in NAMES.iter().enumerate() {
        let _ = writeln!(s, "var s{i}: u32 = st{i} + v{n};");
    }
    if n_trail > 0 {
        let _ = writeln!(s, "for (var b = 0u; b < {n_trail}u; b = b + 1u) {{");
        let _ = writeln!(s, "let base = {HEADER}u + b * 64u;");
        for (i, n) in NAMES.iter().enumerate() {
            let _ = writeln!(s, "var y{n}: u32 = s{i};");
        }
        for i in 0..64 {
            round(&mut s, "y", i, &format!("p[base + {i}u]"));
        }
        for (i, n) in NAMES.iter().enumerate() {
            let _ = writeln!(s, "s{i} = s{i} + y{n};");
        }
        let _ = writeln!(s, "}}");
    }
    let _ = writeln!(s, "if (countLeadingZeros(s0) >= p[{P_FIRST_TARGET}u]) {{");
    let _ = writeln!(s, "var bits = 0u;");
    let _ = writeln!(s, "var done = false;");
    for i in 0..8 {
        let _ = writeln!(
            s,
            "if (!done) {{ if (s{i} != 0u) {{ bits = bits + countLeadingZeros(s{i}); done = true; }} else {{ bits = bits + 32u; }} }}"
        );
    }
    let _ = writeln!(s, "if (bits >= p[{P_TARGET}u]) {{");
    let _ = writeln!(s, "if (atomicAdd(&out[0], 1u) == 0u) {{");
    let _ = writeln!(s, "atomicStore(&out[1], hl);");
    let _ = writeln!(s, "atomicStore(&out[2], hh);");
    let _ = writeln!(
        s,
        "atomicStore(&out[3], thousands * 1000u + hd * 100u + td * 10u + od);"
    );
    for i in 0..8 {
        let _ = writeln!(s, "atomicStore(&out[{}u], s{i});", 4 + i);
    }
    let _ = writeln!(s, "}}\nreturn;\n}}\n}}");
    let _ = writeln!(s, "}}\n}}\n}}");
    let _ = writeln!(s, "}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layout_parses_and_validates() {
        for b0 in 0..2 {
            for wv in 0..16 {
                let room = 4 * (b0 * 16 + wv);
                for digits in [room.min(8), room.min(16)] {
                    for n_trail in [0, 2] {
                        let src = source(b0, wv, digits, n_trail);
                        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                            panic!("{b0} {wv} {digits}: {}", e.emit_to_string(&src))
                        });
                        naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            naga::valid::Capabilities::all(),
                        )
                        .validate(&module)
                        .unwrap_or_else(|e| panic!("{b0} {wv} {digits}: {e:?}"));
                    }
                }
            }
        }
    }
}
