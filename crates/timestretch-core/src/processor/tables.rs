//! Immutable window seeds embedded for identical values on every target.
//! See tables/README.md for their format and the Rust regeneration/check commands.
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
pub(super) fn flank() -> Vec<f32> {
    decode(include_bytes!("tables/flank.bin"))
}
pub(super) fn cosine(n: usize) -> Vec<f32> {
    decode(match n {
        512 => include_bytes!("tables/cosine-512.bin"),
        1024 => include_bytes!("tables/cosine-1024.bin"),
        2048 => include_bytes!("tables/cosine-2048.bin"),
        4096 => include_bytes!("tables/cosine-4096.bin"),
        8192 => include_bytes!("tables/cosine-8192.bin"),
        _ => unreachable!("validated power-of-two FFT geometry"),
    })
}
