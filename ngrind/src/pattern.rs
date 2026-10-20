use serde::{Deserialize, Serialize};

pub const BECH32: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
pub const NPUB_CHARS: usize = 52;
pub const HEX_MAX: usize = 16;
const LAST_CHAR_VALUES: [u8; 2] = [0, 16];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Alphabet {
    Npub,
    Hex,
}

impl Alphabet {
    pub fn radix(self) -> usize {
        match self {
            Alphabet::Npub => 32,
            Alphabet::Hex => 16,
        }
    }

    pub fn key_chars(self) -> usize {
        match self {
            Alphabet::Npub => NPUB_CHARS,
            Alphabet::Hex => 64,
        }
    }

    fn value(self, c: char) -> Option<u8> {
        match self {
            Alphabet::Npub => BECH32.iter().position(|b| *b as char == c).map(|i| i as u8),
            Alphabet::Hex => c.to_digit(16).map(|d| d as u8),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub alphabet: Alphabet,
    pub prefix: String,
    pub suffix: String,
    prefix_values: Vec<u8>,
    suffix_values: Vec<u8>,
}

impl Pattern {
    pub fn parse(alphabet: Alphabet, prefix: &str, suffix: &str) -> Result<Pattern, String> {
        let prefix = prefix.trim().to_ascii_lowercase();
        let suffix = suffix.trim().to_ascii_lowercase();
        if prefix.is_empty() && suffix.is_empty() {
            return Err("enter a prefix or a suffix".into());
        }
        let values = |text: &str| -> Result<Vec<u8>, String> {
            text.chars()
                .map(|c| {
                    alphabet.value(c).ok_or_else(|| match alphabet {
                        Alphabet::Npub => format!(
                            "'{c}' is not in the npub alphabet (1, b, i and o are never used)"
                        ),
                        Alphabet::Hex => format!("'{c}' is not a hex digit"),
                    })
                })
                .collect()
        };
        let prefix_values = values(&prefix)?;
        let suffix_values = values(&suffix)?;
        let room = match alphabet {
            Alphabet::Npub => NPUB_CHARS,
            Alphabet::Hex => HEX_MAX,
        };
        if prefix.len() + suffix.len() > room {
            return Err(format!("at most {room} characters in total"));
        }
        if alphabet == Alphabet::Npub
            && let Some(last) = suffix_values.last()
            && !LAST_CHAR_VALUES.contains(last)
        {
            return Err(
                "an npub suffix is matched before the checksum and always ends in q or s".into(),
            );
        }
        Ok(Pattern {
            alphabet,
            prefix,
            suffix,
            prefix_values,
            suffix_values,
        })
    }

    pub fn chars(&self) -> usize {
        self.prefix.len() + self.suffix.len()
    }

    pub fn bits(&self) -> f64 {
        let radix_bits = (self.alphabet.radix() as f64).log2();
        let mut bits = self.chars() as f64 * radix_bits;
        if self.alphabet == Alphabet::Npub && !self.suffix.is_empty() {
            bits -= radix_bits - 1.0;
        }
        bits
    }

    pub fn expected(&self) -> f64 {
        self.bits().exp2()
    }

    pub fn matches(&self, public: &[u8; 32]) -> bool {
        let at: fn(&[u8; 32], usize) -> u8 = match self.alphabet {
            Alphabet::Npub => group,
            Alphabet::Hex => nibble,
        };
        let total = self.alphabet.key_chars();
        self.prefix_values
            .iter()
            .enumerate()
            .all(|(i, v)| at(public, i) == *v)
            && self
                .suffix_values
                .iter()
                .enumerate()
                .all(|(i, v)| at(public, total - self.suffix_values.len() + i) == *v)
    }

    pub fn prefilter_hex(&self) -> Option<String> {
        if self.alphabet != Alphabet::Npub || !self.suffix.is_empty() {
            return None;
        }
        let bytes = (self.prefix_values.len() * 5 / 8).min(8);
        if bytes == 0 {
            return None;
        }
        let mut bits = Vec::with_capacity(bytes * 8);
        for v in &self.prefix_values {
            for shift in (0..5).rev() {
                bits.push((v >> shift) & 1);
            }
        }
        Some(
            bits.chunks(8)
                .take(bytes)
                .map(|byte| format!("{:02x}", byte.iter().fold(0u8, |acc, b| acc << 1 | b)))
                .collect(),
        )
    }

    pub fn symbol_bits(&self) -> u32 {
        match self.alphabet {
            Alphabet::Npub => 5,
            Alphabet::Hex => 4,
        }
    }

    pub fn symbols(&self) -> Vec<(u32, u32)> {
        let total = self.alphabet.key_chars();
        let head = self.prefix_values.iter().enumerate().map(|(i, v)| (i, *v));
        let from = total - self.suffix_values.len();
        let tail = self
            .suffix_values
            .iter()
            .enumerate()
            .map(move |(i, v)| (from + i, *v));
        head.chain(tail)
            .map(|(i, v)| (i as u32, v as u32))
            .collect()
    }

    pub fn describe(&self) -> String {
        let head = match self.alphabet {
            Alphabet::Npub => "npub1",
            Alphabet::Hex => "",
        };
        match (self.prefix.is_empty(), self.suffix.is_empty()) {
            (false, true) => format!("{head}{}…", self.prefix),
            (true, false) => format!("{head}…{}", self.suffix),
            _ => format!("{head}{}…{}", self.prefix, self.suffix),
        }
    }
}

fn group(public: &[u8; 32], i: usize) -> u8 {
    let bit = i * 5;
    let byte = bit / 8;
    let window = (public[byte] as u16) << 8 | public.get(byte + 1).copied().unwrap_or(0) as u16;
    ((window >> (11 - (bit & 7))) & 31) as u8
}

fn nibble(public: &[u8; 32], i: usize) -> u8 {
    let byte = public[i / 2];
    if i.is_multiple_of(2) {
        byte >> 4
    } else {
        byte & 15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys;

    fn sample() -> [u8; 32] {
        let mut sk = [0u8; 32];
        sk[31] = 7;
        keys::public_of(&sk).unwrap()
    }

    #[test]
    fn groups_agree_with_bech32() {
        let public = sample();
        let text = keys::npub(&public);
        let data = &text.as_bytes()[5..5 + NPUB_CHARS];
        for (i, c) in data.iter().enumerate() {
            assert_eq!(BECH32[group(&public, i) as usize], *c, "char {i} of {text}");
        }
    }

    #[test]
    fn npub_prefix_and_suffix_match_the_encoded_key() {
        let public = sample();
        let text = keys::npub(&public);
        let data = &text[5..5 + NPUB_CHARS];
        let prefix = &data[..4];
        let suffix = &data[NPUB_CHARS - 3..];
        assert!(suffix.ends_with('q') || suffix.ends_with('s'));
        assert!(
            Pattern::parse(Alphabet::Npub, prefix, "")
                .unwrap()
                .matches(&public)
        );
        assert!(
            Pattern::parse(Alphabet::Npub, "", suffix)
                .unwrap()
                .matches(&public)
        );
        assert!(
            Pattern::parse(Alphabet::Npub, prefix, suffix)
                .unwrap()
                .matches(&public)
        );
        let other = if prefix.starts_with('q') { "p" } else { "q" };
        assert!(
            !Pattern::parse(Alphabet::Npub, other, "")
                .unwrap()
                .matches(&public)
        );
    }

    #[test]
    fn prefilter_is_the_start_of_the_key() {
        let public = sample();
        let text = keys::npub(&public);
        for n in 2..=14 {
            let prefix = &text[5..5 + n];
            let pattern = Pattern::parse(Alphabet::Npub, prefix, "").unwrap();
            let filter = pattern.prefilter_hex().unwrap();
            assert!(
                keys::hex(&public).starts_with(&filter),
                "{n} chars: {filter}"
            );
            assert!(filter.len() <= 16);
        }
        assert!(
            Pattern::parse(Alphabet::Npub, "q", "")
                .unwrap()
                .prefilter_hex()
                .is_none()
        );
        assert!(
            Pattern::parse(Alphabet::Npub, "", "as")
                .unwrap()
                .prefilter_hex()
                .is_none()
        );
        assert!(
            Pattern::parse(Alphabet::Hex, "ab", "")
                .unwrap()
                .prefilter_hex()
                .is_none()
        );
    }

    #[test]
    fn hex_prefix_and_suffix_match() {
        let public = sample();
        let text = keys::hex(&public);
        assert!(
            Pattern::parse(Alphabet::Hex, &text[..5], &text[60..])
                .unwrap()
                .matches(&public)
        );
        assert!(
            !Pattern::parse(Alphabet::Hex, "", "0000")
                .unwrap()
                .matches(&public)
        );
    }

    #[test]
    fn rejects_what_can_never_match() {
        assert!(Pattern::parse(Alphabet::Npub, "b", "").is_err());
        assert!(Pattern::parse(Alphabet::Npub, "", "ab").is_err());
        assert!(Pattern::parse(Alphabet::Npub, "", "ap").is_err());
        assert!(Pattern::parse(Alphabet::Npub, "", "aq").is_ok());
        assert!(Pattern::parse(Alphabet::Npub, "", "as").is_ok());
        assert!(Pattern::parse(Alphabet::Hex, "xyz", "").is_err());
        assert!(Pattern::parse(Alphabet::Hex, "", "").is_err());
    }

    #[test]
    fn odds_follow_the_alphabet() {
        assert_eq!(
            Pattern::parse(Alphabet::Npub, "acd", "").unwrap().bits(),
            15.0
        );
        assert_eq!(
            Pattern::parse(Alphabet::Hex, "abc", "").unwrap().bits(),
            12.0
        );
        assert_eq!(
            Pattern::parse(Alphabet::Npub, "", "2s").unwrap().bits(),
            6.0
        );
    }
}
