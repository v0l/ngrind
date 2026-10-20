use crate::keys;
use secp256k1::Keypair;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub type Tags = Vec<Vec<String>>;

#[derive(Clone, Debug)]
pub struct Template {
    pub pubkey: String,
    pub kind: u16,
    pub tags: Tags,
    pub content: String,
    pub difficulty: u32,
    pub fast: bool,
}

pub struct Split {
    pub prefix: String,
    pub suffix: String,
}

pub struct Mined {
    pub tags: Tags,
    pub content: String,
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn parse_tags(text: &str) -> Result<Tags, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let value: Value = serde_json::from_str(text).map_err(|e| format!("tags are not JSON: {e}"))?;
    let list = value.as_array().ok_or("tags must be an array of arrays")?;
    let mut out = Vec::with_capacity(list.len());
    for tag in list {
        let values = tag
            .as_array()
            .ok_or_else(|| format!("every tag must be an array, got {tag}"))?;
        if values.is_empty() {
            return Err("a tag cannot be empty".into());
        }
        let values = values
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("tag values must be strings, got {v}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values[0] != "nonce" {
            out.push(values);
        }
    }
    Ok(out)
}

fn with_nonce(tags: &Tags, nonce: &str, difficulty: u32) -> Tags {
    let mut tags = tags.clone();
    tags.push(vec!["nonce".into(), nonce.into(), difficulty.to_string()]);
    tags
}

fn serialize(pubkey: &str, created_at: u64, kind: u16, tags: &Tags, content: &str) -> String {
    json!([0, pubkey, created_at, kind, tags, content]).to_string()
}

impl Template {
    pub fn split(&self, created_at: u64) -> Split {
        if self.fast {
            let full = serialize(
                &self.pubkey,
                created_at,
                self.kind,
                &with_nonce(&self.tags, "0", self.difficulty),
                &format!("{}\n", self.content),
            );
            let suffix = "\"]".to_string();
            let prefix = full[..full.len() - suffix.len()].to_string();
            return Split { prefix, suffix };
        }
        let full = serialize(
            &self.pubkey,
            created_at,
            self.kind,
            &with_nonce(&self.tags, "0", self.difficulty),
            &self.content,
        );
        let content = serde_json::to_string(&self.content).expect("a string always serialises");
        let suffix = format!("\",\"{}\"]],{content}]", self.difficulty);
        let cut = full.len() - suffix.len() - 1;
        Split {
            prefix: full[..cut].to_string(),
            suffix,
        }
    }

    pub fn mined(&self, nonce: u64) -> Mined {
        if self.fast {
            Mined {
                tags: with_nonce(&self.tags, "0", self.difficulty),
                content: format!("{}\n{nonce}", self.content),
            }
        } else {
            Mined {
                tags: with_nonce(&self.tags, &nonce.to_string(), self.difficulty),
                content: self.content.clone(),
            }
        }
    }

    pub fn id_of(&self, created_at: u64, mined: &Mined) -> [u8; 32] {
        let text = serialize(
            &self.pubkey,
            created_at,
            self.kind,
            &mined.tags,
            &mined.content,
        );
        Sha256::digest(text.as_bytes()).into()
    }

    pub fn event(
        &self,
        created_at: u64,
        nonce: u64,
        signer: Option<&Keypair>,
    ) -> (String, [u8; 32], bool) {
        let mined = self.mined(nonce);
        let id = self.id_of(created_at, &mined);
        let mut event = json!({
            "id": keys::hex(&id),
            "pubkey": self.pubkey,
            "created_at": created_at,
            "kind": self.kind,
            "tags": mined.tags,
            "content": mined.content,
        });
        let signed = match signer {
            Some(pair) => {
                let sig = keys::secp().sign_schnorr_no_aux_rand(&id, pair);
                event["sig"] = json!(keys::hex(&sig.to_byte_array()));
                true
            }
            None => false,
        };
        (event.to_string(), id, signed)
    }
}

#[derive(Clone, Debug)]
pub struct PowFound {
    pub id: String,
    pub nonce: u64,
    pub bits: u32,
    pub created_at: u64,
    pub event: String,
    pub signed: bool,
}

impl PowFound {
    pub fn verified(
        template: &Template,
        created_at: u64,
        nonce: u64,
        signer: Option<&Keypair>,
    ) -> Result<PowFound, String> {
        let mined = template.mined(nonce);
        let id = template.id_of(created_at, &mined);
        let bits = leading_zero_bits(&id);
        if bits < template.difficulty {
            return Err("a nonce from the search does not reproduce the event id".into());
        }
        let (event, id, signed) = template.event(created_at, nonce, signer);
        Ok(PowFound {
            id: keys::hex(&id),
            nonce,
            bits,
            created_at,
            event,
            signed,
        })
    }
}

pub fn leading_zero_bits(hash: &[u8]) -> u32 {
    let mut total = 0;
    for &b in hash {
        if b == 0 {
            total += 8;
        } else {
            total += b.leading_zeros();
            break;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(fast: bool, content: &str, tags: Tags) -> Template {
        Template {
            pubkey: "ab".repeat(32),
            kind: 1,
            tags,
            content: content.into(),
            difficulty: 12,
            fast,
        }
    }

    fn joined(t: &Template, created_at: u64, nonce: u64) -> String {
        let s = t.split(created_at);
        format!("{}{nonce}{}", s.prefix, s.suffix)
    }

    #[test]
    fn split_reproduces_the_event_for_any_nonce() {
        let samples = [
            ("hello", vec![]),
            (
                "quote \" backslash \\ newline \n tab \t",
                vec![vec!["t".to_string(), "x\"y".to_string()]],
            ),
            (
                "日本語 😀 العربية",
                vec![vec!["e".to_string(), "00".to_string()]],
            ),
            ("", vec![]),
        ];
        for fast in [false, true] {
            for (content, tags) in samples.clone() {
                let t = template(fast, content, tags);
                for nonce in [0u64, 7, 123_456_789_012, u64::MAX / 3] {
                    let mined = t.mined(nonce);
                    let expect = serialize(
                        &t.pubkey,
                        1_700_000_000,
                        t.kind,
                        &mined.tags,
                        &mined.content,
                    );
                    assert_eq!(
                        joined(&t, 1_700_000_000, nonce),
                        expect,
                        "fast {fast} content {content:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn tags_reject_nonsense_and_drop_nonce() {
        assert!(parse_tags("").unwrap().is_empty());
        assert!(parse_tags("{}").is_err());
        assert!(parse_tags("[[1]]").is_err());
        assert!(parse_tags("[[]]").is_err());
        assert_eq!(
            parse_tags(r#"[["nonce","1","2"],["t","x"]]"#).unwrap(),
            vec![vec!["t".to_string(), "x".to_string()]]
        );
    }

    #[test]
    fn zero_bits_count_across_bytes() {
        assert_eq!(leading_zero_bits(&[0, 0, 0x10]), 19);
        assert_eq!(leading_zero_bits(&[0x80]), 0);
        assert_eq!(leading_zero_bits(&[0, 0]), 16);
    }

    #[test]
    fn signed_event_verifies() {
        let secret = keys::random_secret();
        let pair = Keypair::from_secret_key(keys::secp(), &secret);
        let mut t = template(false, "hi", vec![]);
        t.pubkey = keys::hex(&pair.x_only_public_key().0.serialize());
        let (text, id, signed) = t.event(1_700_000_000, 42, Some(&pair));
        assert!(signed);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["id"], keys::hex(&id));
        assert_eq!(v["sig"].as_str().unwrap().len(), 128);
        assert_eq!(v["tags"][0][1], "42");
    }
}
