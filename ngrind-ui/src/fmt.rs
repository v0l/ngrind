const PREFIXES: [&str; 7] = ["", "k", "M", "G", "T", "P", "E"];

pub fn scaled(v: f64) -> (String, &'static str) {
    if !v.is_finite() || v <= 0.0 {
        return ("0".into(), "");
    }
    let mut x = v;
    let mut i = 0;
    while x >= 1000.0 && i + 1 < PREFIXES.len() {
        x /= 1000.0;
        i += 1;
    }
    if x >= 1000.0 {
        return (format!("{v:.2e}"), "");
    }
    let text = if x >= 100.0 || i == 0 && x >= 10.0 {
        format!("{x:.0}")
    } else if x >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    };
    (text, PREFIXES[i])
}

pub fn count(v: f64) -> String {
    let (n, p) = scaled(v);
    if p.is_empty() { n } else { format!("{n} {p}") }
}

pub fn duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "unknown".into();
    }
    if secs < 1.0 {
        return "under 1 s".into();
    }
    let s = secs as u64;
    if s < 60 {
        return format!("{s} s");
    }
    if s < 3600 {
        return format!("{} m {:02} s", s / 60, s % 60);
    }
    if s < 86_400 {
        return format!("{} h {:02} m", s / 3600, s % 3600 / 60);
    }
    if s < 86_400 * 365 {
        return format!("{} d {} h", s / 86_400, s % 86_400 / 3600);
    }
    let years = secs / (86_400.0 * 365.25);
    if years < 1000.0 {
        format!("{years:.1} years")
    } else {
        format!("{} years", count(years))
    }
}

pub fn chance(done: f64, expected: f64) -> f64 {
    if expected <= 0.0 {
        1.0
    } else {
        -(-done / expected).exp_m1()
    }
}

pub fn median_seconds(expected: f64, rate: f64) -> f64 {
    if rate <= 0.0 {
        f64::INFINITY
    } else {
        std::f64::consts::LN_2 * expected / rate
    }
}

pub fn clock(secs: f64) -> String {
    let s = secs as u64;
    format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_scale() {
        assert_eq!(count(950.0), "950");
        assert_eq!(count(1_500_000.0), "1.50 M");
        assert_eq!(count(33_554_432.0), "33.6 M");
        assert!(count(1e30).contains('e'));
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration(5.2), "5 s");
        assert_eq!(duration(125.0), "2 m 05 s");
        assert_eq!(duration(3.0 * 3600.0 + 120.0), "3 h 02 m");
        assert_eq!(duration(f64::INFINITY), "unknown");
    }

    #[test]
    fn chance_is_half_at_the_median() {
        let e = 1e6;
        let n = std::f64::consts::LN_2 * e;
        assert!((chance(n, e) - 0.5).abs() < 1e-9);
        assert!(chance(0.0, e).abs() < 1e-12);
    }
}
