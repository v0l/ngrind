use secp256k1::{PublicKey, SecretKey};
use std::sync::{Arc, OnceLock};

pub const CHUNKS: usize = 16;
pub const VALUES: usize = 65536;
pub const POINT_BYTES: usize = 32;
pub const TABLE_BYTES: usize = CHUNKS * VALUES * POINT_BYTES;

pub struct GTable {
    pub x: Vec<u8>,
    pub y: Vec<u8>,
}

impl GTable {
    pub fn generate() -> GTable {
        let mut x = vec![0u8; TABLE_BYTES];
        let mut y = vec![0u8; TABLE_BYTES];
        std::thread::scope(|scope| {
            let rows = x
                .chunks_mut(VALUES * POINT_BYTES)
                .zip(y.chunks_mut(VALUES * POINT_BYTES));
            for (chunk, (xs, ys)) in rows.enumerate() {
                scope.spawn(move || fill(chunk, xs, ys));
            }
        });
        GTable { x, y }
    }

    pub fn shared() -> Arc<GTable> {
        static TABLE: OnceLock<Arc<GTable>> = OnceLock::new();
        TABLE.get_or_init(|| Arc::new(GTable::generate())).clone()
    }
}

fn fill(chunk: usize, xs: &mut [u8], ys: &mut [u8]) {
    let bit = 16 * chunk;
    let mut scalar = [0u8; 32];
    scalar[31 - bit / 8] = 1 << (bit % 8);
    let base = PublicKey::from_secret_key(
        &SecretKey::from_secret_bytes(scalar).expect("a power of two below the curve order"),
    );
    let mut point = base;
    for j in 0..VALUES - 1 {
        let bytes = point.serialize_uncompressed();
        for b in 0..POINT_BYTES {
            xs[j * POINT_BYTES + b] = bytes[32 - b];
            ys[j * POINT_BYTES + b] = bytes[64 - b];
        }
        point = point
            .combine(&base)
            .expect("multiples of a point below the order never cancel");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys;

    #[test]
    fn first_entries_are_small_multiples_of_g() {
        let table = GTable::generate();
        let at = |element: usize| -> [u8; 32] {
            let mut x: [u8; 32] = table.x[element * 32..element * 32 + 32].try_into().unwrap();
            x.reverse();
            x
        };
        for k in [1u8, 2, 3, 200] {
            let mut sk = [0u8; 32];
            sk[31] = k;
            assert_eq!(at(k as usize - 1), keys::public_of(&sk).unwrap(), "{k} * G");
        }
        let mut sk = [0u8; 32];
        sk[29] = 1;
        assert_eq!(
            at(VALUES),
            keys::public_of(&sk).unwrap(),
            "chunk 1 starts at 2^16 * G"
        );
    }

    #[test]
    fn matches_the_cpp_table_byte_for_byte() {
        let (Ok(x), Ok(y)) = (
            std::env::var("NGRIND_CPP_GTABLE_X"),
            std::env::var("NGRIND_CPP_GTABLE_Y"),
        ) else {
            return;
        };
        let table = GTable::generate();
        assert!(std::fs::read(x).unwrap() == table.x, "x differs");
        assert!(std::fs::read(y).unwrap() == table.y, "y differs");
    }
}
