use crate::core::types::{AudioBuffer, StretchError};

/// WAV file header constants
const RIFF_HEADER: &[u8; 4] = b"RIFF";
const WAVE_FORMAT: &[u8; 4] = b"WAVE";
const FMT_CHUNK_ID: &[u8; 4] = b"fmt ";
const DATA_CHUNK_ID: &[u8; 4] = b"data";

/// PCM format code
const PCM_FORMAT: u16 = 1;
/// IEEE float format code
const IEEE_FLOAT_FORMAT: u16 = 3;

/// Read a WAV file from bytes and return an AudioBuffer.
///
/// Supports 16-bit PCM and 32-bit IEEE float formats.
///
/// # Errors
/// Returns `StretchError::IoError` if the file format is invalid.
pub fn read_wav(data: &[u8]) -> Result<AudioBuffer, StretchError> {
    if data.len() < 44 {
        return Err(StretchError::IoError(
            "WAV file too short (minimum 44 bytes)".to_string(),
        ));
    }

    // Check RIFF header
    if &data[0..4] != RIFF_HEADER {
        return Err(StretchError::IoError(
            "Not a valid RIFF file".to_string(),
        ));
    }

    // Check WAVE format
    if &data[8..12] != WAVE_FORMAT {
        return Err(StretchError::IoError(
            "Not a valid WAVE file".to_string(),
        ));
    }

    // Find fmt chunk
    let (fmt_offset, _fmt_size) = find_chunk(data, FMT_CHUNK_ID)?;

    let audio_format = read_u16_le(data, fmt_offset);
    let num_channels = read_u16_le(data, fmt_offset + 2);
    let sample_rate = read_u32_le(data, fmt_offset + 4);
    let _byte_rate = read_u32_le(data, fmt_offset + 8);
    let _block_align = read_u16_le(data, fmt_offset + 12);
    let bits_per_sample = read_u16_le(data, fmt_offset + 14);

    if audio_format != PCM_FORMAT && audio_format != IEEE_FLOAT_FORMAT {
        return Err(StretchError::IoError(format!(
            "Unsupported audio format: {audio_format}. Only PCM (1) and IEEE float (3) are supported."
        )));
    }

    if num_channels == 0 || num_channels > 2 {
        return Err(StretchError::InvalidChannels(num_channels));
    }

    // Find data chunk
    let (data_offset, data_size) = find_chunk(data, DATA_CHUNK_ID)?;

    let samples = match (audio_format, bits_per_sample) {
        (PCM_FORMAT, 16) => read_pcm16(&data[data_offset..data_offset + data_size as usize]),
        (IEEE_FLOAT_FORMAT, 32) | (PCM_FORMAT, 32) => {
            if audio_format == IEEE_FLOAT_FORMAT {
                read_float32(&data[data_offset..data_offset + data_size as usize])
            } else {
                // 32-bit PCM
                read_pcm32(&data[data_offset..data_offset + data_size as usize])
            }
        }
        _ => {
            return Err(StretchError::IoError(format!(
                "Unsupported format: {audio_format} with {bits_per_sample} bits per sample"
            )));
        }
    };

    AudioBuffer::new(samples, num_channels, sample_rate)
}

/// Write an AudioBuffer as WAV format bytes.
///
/// Writes 32-bit IEEE float format by default.
pub fn write_wav_float(buffer: &AudioBuffer) -> Vec<u8> {
    let bits_per_sample: u16 = 32;
    let byte_rate = buffer.sample_rate * buffer.channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = buffer.channels * (bits_per_sample / 8);
    let data_size = (buffer.data.len() * 4) as u32;
    let file_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    out.extend_from_slice(RIFF_HEADER);
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(WAVE_FORMAT);

    // fmt chunk
    out.extend_from_slice(FMT_CHUNK_ID);
    out.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&IEEE_FLOAT_FORMAT.to_le_bytes());
    out.extend_from_slice(&buffer.channels.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(DATA_CHUNK_ID);
    out.extend_from_slice(&data_size.to_le_bytes());
    for &sample in &buffer.data {
        out.extend_from_slice(&sample.to_le_bytes());
    }

    out
}

/// Write an AudioBuffer as 16-bit PCM WAV format bytes.
pub fn write_wav_pcm16(buffer: &AudioBuffer) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = buffer.sample_rate * buffer.channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = buffer.channels * (bits_per_sample / 8);
    let data_size = (buffer.data.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    out.extend_from_slice(RIFF_HEADER);
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(WAVE_FORMAT);

    // fmt chunk
    out.extend_from_slice(FMT_CHUNK_ID);
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&PCM_FORMAT.to_le_bytes());
    out.extend_from_slice(&buffer.channels.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(DATA_CHUNK_ID);
    out.extend_from_slice(&data_size.to_le_bytes());
    for &sample in &buffer.data {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_val = (clamped * 32767.0) as i16;
        out.extend_from_slice(&int_val.to_le_bytes());
    }

    out
}

/// Find a chunk in the WAV data by its ID.
/// Returns (data_offset, data_size).
fn find_chunk(data: &[u8], chunk_id: &[u8; 4]) -> Result<(usize, u32), StretchError> {
    let mut offset = 12; // Skip RIFF header + file size + WAVE
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let size = read_u32_le(data, offset + 4);
        if id == chunk_id {
            return Ok((offset + 8, size));
        }
        offset += 8 + size as usize;
        // Chunks must be word-aligned
        if offset % 2 != 0 {
            offset += 1;
        }
    }
    Err(StretchError::IoError(format!(
        "Chunk '{}' not found in WAV file",
        String::from_utf8_lossy(chunk_id)
    )))
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_pcm16(data: &[u8]) -> Vec<f32> {
    let num_samples = data.len() / 2;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let raw = i16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        samples.push(raw as f32 / 32768.0);
    }
    samples
}

fn read_float32(data: &[u8]) -> Vec<f32> {
    let num_samples = data.len() / 4;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        samples.push(f32::from_le_bytes(bytes));
    }
    samples
}

fn read_pcm32(data: &[u8]) -> Vec<f32> {
    let num_samples = data.len() / 4;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let raw = i32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        samples.push(raw as f32 / 2_147_483_648.0);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_buffer() -> AudioBuffer {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        AudioBuffer::new(samples, 1, 44100).unwrap()
    }

    #[test]
    fn test_wav_float_roundtrip() {
        let original = make_test_buffer();
        let wav_data = write_wav_float(&original);
        let decoded = read_wav(&wav_data).unwrap();

        assert_eq!(decoded.channels, original.channels);
        assert_eq!(decoded.sample_rate, original.sample_rate);
        assert_eq!(decoded.data.len(), original.data.len());
        for (i, (&a, &b)) in original.data.iter().zip(decoded.data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Float roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_wav_pcm16_roundtrip() {
        let original = make_test_buffer();
        let wav_data = write_wav_pcm16(&original);
        let decoded = read_wav(&wav_data).unwrap();

        assert_eq!(decoded.channels, original.channels);
        assert_eq!(decoded.sample_rate, original.sample_rate);
        assert_eq!(decoded.data.len(), original.data.len());
        // 16-bit PCM has limited precision
        for (i, (&a, &b)) in original.data.iter().zip(decoded.data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.001,
                "PCM16 roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_wav_stereo_roundtrip() {
        let samples: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
        let original = AudioBuffer::new(samples, 2, 48000).unwrap();
        let wav_data = write_wav_float(&original);
        let decoded = read_wav(&wav_data).unwrap();

        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 48000);
        assert_eq!(decoded.num_frames(), original.num_frames());
    }

    #[test]
    fn test_wav_invalid_data() {
        assert!(read_wav(&[]).is_err());
        assert!(read_wav(&[0u8; 10]).is_err());
        assert!(read_wav(b"NOT_A_WAV_FILE_AT_ALL_______________").is_err());
    }

    #[test]
    fn test_wav_empty_buffer() {
        let buf = AudioBuffer::new(vec![], 1, 44100).unwrap();
        let wav_data = write_wav_float(&buf);
        let decoded = read_wav(&wav_data).unwrap();
        assert!(decoded.is_empty());
    }
}
