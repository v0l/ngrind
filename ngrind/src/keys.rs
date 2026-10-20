use bech32::{Bech32, Hrp};
use secp256k1::{Keypair, Secp256k1, SecretKey, SignOnly};
use std::sync::OnceLock;

pub fn secp() -> &'static Secp256k1<SignOnly> {
    static CTX: OnceLock<Secp256k1<SignOnly>> = OnceLock::new();
    CTX.get_or_init(Secp256k1::signing_only)
}

pub fn encode(hrp: &str, bytes: &[u8; 32]) -> String {
    let hrp = Hrp::parse(hrp).expect("static hrp is valid");
    bech32::encode::<Bech32>(hrp, bytes).expect("32 bytes always encode")
}

pub fn npub(public: &[u8; 32]) -> String {
    encode("npub", public)
}

pub fn nsec(secret: &[u8; 32]) -> String {
    encode("nsec", secret)
}

pub fn public_of(secret: &[u8; 32]) -> Result<[u8; 32], String> {
    let sk = SecretKey::from_byte_array(*secret).map_err(|e| e.to_string())?;
    let pair = Keypair::from_secret_key(secp(), &sk);
    Ok(pair.x_only_public_key().0.serialize())
}

pub fn random_secret() -> SecretKey {
    loop {
        if let Ok(sk) = SecretKey::from_byte_array(rand::random::<[u8; 32]>()) {
            return sk;
        }
    }
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn parse_public(text: &str) -> Result<[u8; 32], String> {
    let text = text.trim();
    if let Some(bytes) = hex32(text) {
        return Ok(bytes);
    }
    let (hrp, data) =
        bech32::decode(text).map_err(|_| "not an npub or a 64 digit hex key".to_string())?;
    match (hrp.as_str(), <[u8; 32]>::try_from(data.as_slice())) {
        ("npub", Ok(bytes)) => Ok(bytes),
        _ => Err("expected an npub or a 64 digit hex key".to_string()),
    }
}

pub fn parse_secret(text: &str) -> Result<[u8; 32], String> {
    let (hrp, data) = bech32::decode(text.trim()).map_err(|_| "not an nsec".to_string())?;
    match (hrp.as_str(), <[u8; 32]>::try_from(data.as_slice())) {
        ("nsec", Ok(bytes)) => Ok(bytes),
        _ => Err("expected an nsec".to_string()),
    }
}

pub fn parse_secret_any(text: &str) -> Result<[u8; 32], String> {
    let text = text.trim();
    hex32(text).map(Ok).unwrap_or_else(|| parse_secret(text))
}

pub fn data_dir() -> Option<std::path::PathBuf> {
    dirs::data_dir().map(|d| d.join("ngrind"))
}

pub fn record(text: &str) -> Result<std::path::PathBuf, String> {
    use std::io::Write;
    let dir = data_dir().ok_or("no data directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("found.txt");
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(&path).map_err(|e| e.to_string())?;
    writeln!(file, "{text}").map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npub_round_trips() {
        let pk = [7u8; 32];
        assert_eq!(parse_public(&npub(&pk)).unwrap(), pk);
        assert_eq!(parse_public(&hex(&pk)).unwrap(), pk);
    }

    #[test]
    fn nsec_is_not_a_public_key() {
        let sk = [9u8; 32];
        assert!(parse_public(&nsec(&sk)).is_err());
        assert_eq!(parse_secret(&nsec(&sk)).unwrap(), sk);
        assert!(parse_secret(&npub(&sk)).is_err());
    }

    #[test]
    fn known_public_key() {
        let mut sk = [0u8; 32];
        sk[31] = 1;
        assert_eq!(
            hex(&public_of(&sk).unwrap()),
            "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
        );
    }
}
