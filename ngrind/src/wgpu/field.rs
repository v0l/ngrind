use std::fmt::Write;

pub const PRELUDE: &str = r#"
alias F = array<u32, 8>;
alias W = array<u32, 16>;

struct FC { v: F, c: u32 }

const CC: F = F(977u, 1u, 0u, 0u, 0u, 0u, 0u, 0u);

fn cy(a: u32, s: u32) -> u32 { return select(0u, 1u, s < a); }
fn bw(a: u32, b: u32) -> u32 { return select(0u, 1u, a < b); }

fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
    let mid = lh + hl;
    let c1 = select(0u, 0x10000u, mid < lh);
    let lo = ll + (mid << 16u);
    let c2 = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (mid >> 16u) + c1 + c2);
}

"#;

fn add8() -> String {
    let mut s = String::from("fn add8(a: F, b: F) -> FC {\n    var r: F;\n    var c = 0u;\n");
    s.push_str("    { let x = a[0] + b[0]; c = cy(a[0], x); r[0] = x; }\n");
    for i in 1..8 {
        let _ = writeln!(
            s,
            "    {{ let x = a[{i}] + b[{i}]; let c1 = cy(a[{i}], x); let u = x + c; let c2 = cy(x, u); r[{i}] = u; c = c1 + c2; }}"
        );
    }
    s.push_str("    return FC(r, c);\n}\n\n");
    s
}

fn sub8() -> String {
    let mut s = String::from("fn sub8(a: F, b: F) -> FC {\n    var r: F;\n    var c = 0u;\n");
    s.push_str("    { let x = a[0] - b[0]; c = bw(a[0], b[0]); r[0] = x; }\n");
    for i in 1..8 {
        let _ = writeln!(
            s,
            "    {{ let x = a[{i}] - b[{i}]; let c1 = bw(a[{i}], b[{i}]); let u = x - c; let c2 = bw(x, c); r[{i}] = u; c = c1 + c2; }}"
        );
    }
    s.push_str("    return FC(r, c);\n}\n\n");
    s
}

fn accumulate(s: &mut String, product: &str) {
    let _ = writeln!(
        s,
        "    {{ let p = {product}; let s0 = c0 + p.x; let k0 = cy(c0, s0); c0 = s0; let s1 = c1 + p.y; let k1 = cy(c1, s1); let s1b = s1 + k0; let k1b = cy(s1, s1b); c1 = s1b; c2 = c2 + k1 + k1b; }}"
    );
}

fn mul_wide(square: bool) -> String {
    let name = if square { "sqr_wide" } else { "mul_wide" };
    let args = if square { "a: F" } else { "a: F, b: F" };
    let mut s = format!(
        "fn {name}({args}) -> W {{\n    var t: W;\n    var c0 = 0u; var c1 = 0u; var c2 = 0u;\n"
    );
    for k in 0..15usize {
        let lo = k.saturating_sub(7);
        let hi = k.min(7);
        for i in lo..=hi {
            let j = k - i;
            if square {
                if i > j {
                    continue;
                }
                let times = if i < j { 2 } else { 1 };
                for _ in 0..times {
                    accumulate(&mut s, &format!("mulhilo(a[{i}], a[{j}])"));
                }
            } else {
                accumulate(&mut s, &format!("mulhilo(a[{i}], b[{j}])"));
            }
        }
        let _ = writeln!(s, "    t[{k}] = c0; c0 = c1; c1 = c2; c2 = 0u;");
    }
    s.push_str("    t[15] = c0;\n    return t;\n}\n\n");
    s
}

fn reduce() -> String {
    let mut s = String::from(
        "fn reduce(t: W) -> F {\n    var r: array<u32, 10>;\n    var mc = 0u; var carry = 0u;\n",
    );
    for i in 0..9usize {
        let a = if i < 8 {
            format!("t[{i}]")
        } else {
            "0u".into()
        };
        let m = if i < 8 {
            let _ = writeln!(s, "    var m{i}: u32;");
            let _ = writeln!(
                s,
                "    {{ let p = mulhilo(t[{}], 977u); m{i} = p.x + mc; let nm = p.y + cy(p.x, m{i}); mc = nm; }}",
                8 + i
            );
            format!("m{i}")
        } else {
            "mc".into()
        };
        let d = if i >= 1 {
            format!("t[{}]", 8 + i - 1)
        } else {
            "0u".into()
        };
        let _ = writeln!(
            s,
            "    {{ let s1 = {a} + {m}; let c1 = cy({a}, s1); let s2 = s1 + {d}; let c2 = cy(s1, s2); let s3 = s2 + carry; let c3 = cy(s2, s3); r[{i}] = s3; carry = c1 + c2 + c3; }}"
        );
    }
    s.push_str("    r[9] = carry;\n");
    s.push_str("    let q = mulhilo(r[8], 977u);\n");
    s.push_str("    let r9m = r[9] * 977u;\n");
    s.push_str("    let v = r[8] + q.y + r9m;\n");
    s.push_str("    let cv = select(0u, 1u, v < r[8]);\n");
    s.push_str("    var lo: F;\n");
    for i in 0..8 {
        let _ = writeln!(s, "    lo[{i}] = r[{i}];");
    }
    s.push_str("    let ad = F(q.x, v, r[9] + cv, 0u, 0u, 0u, 0u, 0u);\n");
    s.push_str("    let f = add8(lo, ad);\n");
    s.push_str("    var x = f.v;\n");
    s.push_str("    if (f.c != 0u) { x = add8(f.v, CC).v; }\n");
    s.push_str("    let y = add8(x, CC);\n");
    s.push_str("    if (y.c != 0u) { return y.v; }\n");
    s.push_str("    return x;\n}\n\n");
    s
}

const TAIL: &str = r#"
fn fadd(a: F, b: F) -> F {
    let s = add8(a, b);
    let y = add8(s.v, CC);
    if (s.c != 0u || y.c != 0u) { return y.v; }
    return s.v;
}

fn fsub(a: F, b: F) -> F {
    let d = sub8(a, b);
    if (d.c == 0u) { return d.v; }
    return sub8(d.v, CC).v;
}

fn fmul(a: F, b: F) -> F { return reduce(mul_wide(a, b)); }
fn fsqr(a: F) -> F { return reduce(sqr_wide(a)); }

fn fsqrn(a: F, n: u32) -> F {
    var t = a;
    for (var i = 0u; i < n; i = i + 1u) { t = fsqr(t); }
    return t;
}

fn finv(a: F) -> F {
    let x2 = fmul(fsqr(a), a);
    let x3 = fmul(fsqr(x2), a);
    let x6 = fmul(fsqrn(x3, 3u), x3);
    let x9 = fmul(fsqrn(x6, 3u), x3);
    let x11 = fmul(fsqrn(x9, 2u), x2);
    let x22 = fmul(fsqrn(x11, 11u), x11);
    let x44 = fmul(fsqrn(x22, 22u), x22);
    let x88 = fmul(fsqrn(x44, 44u), x44);
    let x176 = fmul(fsqrn(x88, 88u), x88);
    let x220 = fmul(fsqrn(x176, 44u), x44);
    let x223 = fmul(fsqrn(x220, 3u), x3);
    var t = fmul(fsqrn(x223, 23u), x22);
    t = fmul(fsqrn(t, 5u), a);
    t = fmul(fsqrn(t, 3u), x2);
    t = fmul(fsqrn(t, 2u), a);
    return t;
}
"#;

pub fn library() -> String {
    let mut s = String::from(PRELUDE);
    s.push_str(&add8());
    s.push_str(&sub8());
    s.push_str(&mul_wide(false));
    s.push_str(&mul_wide(true));
    s.push_str(&reduce());
    s.push_str(TAIL);
    s
}

#[cfg(test)]
pub fn selftest_source() -> String {
    let mut s = library();
    s.push_str(
        r#"
@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> res: array<u32>;

fn load8(at: u32) -> F {
    var r: F;
    for (var i = 0u; i < 8u; i = i + 1u) { r[i] = src_words[at + i]; }
    return r;
}

fn store8(at: u32, v: F) {
    for (var i = 0u; i < 8u; i = i + 1u) { res[at + i] = v[i]; }
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let n = wid.x;
    let a = load8(n * 16u);
    let b = load8(n * 16u + 8u);
    store8(n * 40u, fadd(a, b));
    store8(n * 40u + 8u, fsub(a, b));
    store8(n * 40u + 16u, fmul(a, b));
    store8(n * 40u + 24u, fsqr(a));
    store8(n * 40u + 32u, finv(a));
}
"#,
    );
    s
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn prime() -> BigUint {
        (BigUint::from(1u8) << 256u32) - BigUint::from(0x1_0000_03D1u64)
    }

    fn to_words(v: &BigUint) -> [u32; 8] {
        let mut out = [0u32; 8];
        for (i, d) in v.iter_u32_digits().enumerate() {
            out[i] = d;
        }
        out
    }

    fn from_words(w: &[u32]) -> BigUint {
        let mut bytes = Vec::new();
        for x in w {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn edge_cases() -> Vec<BigUint> {
        let p = prime();
        let one = BigUint::from(1u8);
        let mut v = vec![
            BigUint::from(0u8),
            one.clone(),
            BigUint::from(2u8),
            &p - &one,
            &p - BigUint::from(2u8),
            (BigUint::from(1u8) << 255u32),
            (BigUint::from(1u8) << 128u32) - &one,
            BigUint::from(0xffff_ffffu32),
            BigUint::from(0x1_0000_03D0u64),
        ];
        let mut x = 0x9E3779B97F4A7C15u64;
        for _ in 0..180 {
            let mut bytes = [0u8; 32];
            for b in bytes.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = (x >> 24) as u8;
            }
            v.push(BigUint::from_bytes_be(&bytes) % &p);
        }
        v
    }

    #[test]
    fn field_arithmetic_matches_big_integers() {
        let Some(gpu) = crate::wgpu::test_gpu(0) else {
            return;
        };
        let p = prime();
        let values = edge_cases();
        let pairs: Vec<(BigUint, BigUint)> = values
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), values[(i * 7 + 3) % values.len()].clone()))
            .chain(
                values
                    .iter()
                    .take(9)
                    .flat_map(|a| values.iter().take(9).map(move |b| (a.clone(), b.clone()))),
            )
            .collect();
        let n = pairs.len();
        let mut input = Vec::new();
        for (a, b) in &pairs {
            input.extend_from_slice(&to_words(a));
            input.extend_from_slice(&to_words(b));
        }
        let device = &gpu.device;
        let make = |usage, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (size * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let inp = make(
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            input.len(),
        );
        let res = make(
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            n * 40,
        );
        let read = make(
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            n * 40,
        );
        gpu.queue.write_buffer(
            &inp,
            0,
            &input
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(selftest_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&res, 0, &read, 0, (n * 40 * 4) as u64);
        gpu.queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let words: Vec<u32> = read
            .slice(..)
            .get_mapped_range()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let mut bad = Vec::new();
        for (i, (a, b)) in pairs.iter().enumerate() {
            let got = |k: usize| from_words(&words[i * 40 + k * 8..i * 40 + k * 8 + 8]);
            let expect_inv = if a == &BigUint::from(0u8) {
                BigUint::from(0u8)
            } else {
                a.modpow(&(&p - BigUint::from(2u8)), &p)
            };
            let checks = [
                ("add", got(0), (a + b) % &p),
                ("sub", got(1), (a + &p - b) % &p),
                ("mul", got(2), (a * b) % &p),
                ("sqr", got(3), (a * a) % &p),
                ("inv", got(4), expect_inv),
            ];
            for (name, g, e) in checks {
                if g != e {
                    bad.push(format!("{name}({a:x}, {b:x}) = {g:x}, expected {e:x}"));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {} wrong, first: {}",
            bad.len(),
            n * 5,
            bad[0]
        );
    }
}
