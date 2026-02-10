//! WAV file reading and writing (16-bit PCM, 32-bit PCM, 32-bit IEEE float).

use crate::core::types::{AudioBuffer, Channels, Sample};
use crate::error::StretchError;
use std::io::{Read, Write};

/// WAV audio format codes.
const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;

/// Scaling factor for converting 16-bit PCM samples to/from f32.
const PCM_16BIT_SCALE: f32 = 32768.0;
/// Maximum representable value for 16-bit PCM output (32767, not 32768, to avoid clipping).
const PCM_16BIT_MAX_OUT: f32 = 32767.0;
/// Scaling factor for converting 24-bit PCM samples to f32.
const PCM_24BIT_SCALE: f32 = 8388608.0;
/// Sign bit mask for 24-bit PCM sign extension.
const PCM_24BIT_SIGN_BIT: i32 = 0x800000;
/// Bitmask for 24-bit PCM values.
const PCM_24BIT_MASK: i32 = 0xFFFFFF;
/// Minimum size for a valid WAV file (RIFF + fmt + data headers).
const WAV_MIN_HEADER_SIZE: usize = 44;
/// Size of the fmt chunk's fixed fields.
const WAV_FMT_MIN_SIZE: usize = 16;

/// Parsed WAV format and data chunk info.
struct WavChunks<'a> {
    format_code: u16,
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    audio_data: &'a [u8],
}

/// Validates the RIFF/WAVE header and returns the offset past it.
fn validate_riff_header(data: &[u8]) -> Result<usize, StretchError> {
    if data.len() < WAV_MIN_HEADER_SIZE {
        return Err(StretchError::InvalidFormat(
            "WAV file too short".to_string(),
        ));
    }
    if &data[0..4] != b"RIFF" {
        return Err(StretchError::InvalidFormat(
            "Missing RIFF header".to_string(),
        ));
    }
    if &data[8..12] != b"WAVE" {
        return Err(StretchError::InvalidFormat(
            "Missing WAVE identifier".to_string(),
        ));
    }
    Ok(12)
}

/// Iterates WAV chunks to find fmt and data, returning parsed info.
fn parse_wav_chunks(data: &[u8], start: usize) -> Result<WavChunks<'_>, StretchError> {
    let mut cursor = start;
    let mut format_code: u16 = 0;
    let mut num_channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut audio_data: &[u8] = &[];

    while cursor + 8 <= data.len() {
        let chunk_id = &data[cursor..cursor + 4];
        cursor += 4;
        let chunk_size = read_u32_le(data, cursor) as usize;
        cursor += 4;

        if chunk_id == b"fmt " {
            if cursor + WAV_FMT_MIN_SIZE > data.len() {
                return Err(StretchError::InvalidFormat(
                    "fmt chunk too short".to_string(),
                ));
            }
            format_code = read_u16_le(data, cursor);
            num_channels = read_u16_le(data, cursor + 2);
            sample_rate = read_u32_le(data, cursor + 4);
            // skip byte rate (4 bytes) and block align (2 bytes)
            bits_per_sample = read_u16_le(data, cursor + 14);
        } else if chunk_id == b"data" {
            if cursor + chunk_size > data.len() {
                audio_data = &data[cursor..];
            } else {
                audio_data = &data[cursor..cursor + chunk_size];
            }
        }

        cursor += chunk_size;
        // WAV chunks are word-aligned
        if chunk_size % 2 != 0 {
            cursor += 1;
        }
    }

    if sample_rate == 0 {
        return Err(StretchError::InvalidFormat(
            "No fmt chunk found".to_string(),
        ));
    }

    Ok(WavChunks {
        format_code,
        num_channels,
        sample_rate,
        bits_per_sample,
        audio_data,
    })
}

/// Converts 16-bit PCM audio bytes to f32 samples.
fn convert_pcm_16bit(audio_data: &[u8]) -> Vec<Sample> {
    let num_samples = audio_data.len() / 2;
    let mut result = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let raw = read_i16_le(audio_data, i * 2);
        result.push(raw as f32 / PCM_16BIT_SCALE);
    }
    result
}

/// Converts 24-bit PCM audio bytes to f32 samples.
fn convert_pcm_24bit(audio_data: &[u8]) -> Vec<Sample> {
    let num_samples = audio_data.len() / 3;
    let mut result = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 3;
        let raw = (audio_data[offset] as i32)
            | ((audio_data[offset + 1] as i32) << 8)
            | ((audio_data[offset + 2] as i32) << 16);
        // Sign extend from 24-bit to 32-bit
        let raw = if raw & PCM_24BIT_SIGN_BIT != 0 {
            raw | !PCM_24BIT_MASK
        } else {
            raw
        };
        result.push(raw as f32 / PCM_24BIT_SCALE);
    }
    result
}

/// Converts 32-bit IEEE float audio bytes to f32 samples.
fn convert_ieee_float_32bit(audio_data: &[u8]) -> Vec<Sample> {
    let num_samples = audio_data.len() / 4;
    let mut result = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let bytes = [
            audio_data[i * 4],
            audio_data[i * 4 + 1],
            audio_data[i * 4 + 2],
            audio_data[i * 4 + 3],
        ];
        result.push(f32::from_le_bytes(bytes));
    }
    result
}

/// Converts raw audio bytes to f32 samples based on format and bit depth.
fn convert_samples(
    audio_data: &[u8],
    format_code: u16,
    bits_per_sample: u16,
) -> Result<Vec<Sample>, StretchError> {
    match (format_code, bits_per_sample) {
        (WAV_FORMAT_PCM, 16) => Ok(convert_pcm_16bit(audio_data)),
        (WAV_FORMAT_PCM, 24) => Ok(convert_pcm_24bit(audio_data)),
        (WAV_FORMAT_IEEE_FLOAT, 32) => Ok(convert_ieee_float_32bit(audio_data)),
        (fmt, bits) => Err(StretchError::InvalidFormat(format!(
            "Unsupported WAV format: code={}, bits={}",
            fmt, bits
        ))),
    }
}

/// Reads a WAV file from a byte slice.
pub fn read_wav(data: &[u8]) -> Result<AudioBuffer, StretchError> {
    let cursor = validate_riff_header(data)?;
    let chunks = parse_wav_chunks(data, cursor)?;

    let channels = match chunks.num_channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => {
            return Err(StretchError::InvalidFormat(format!(
                "Unsupported channel count: {}",
                n
            )))
        }
    };

    let samples = convert_samples(
        chunks.audio_data,
        chunks.format_code,
        chunks.bits_per_sample,
    )?;
    Ok(AudioBuffer::new(samples, chunks.sample_rate, channels))
}

/// Creates an I/O error with the file path prepended to the message.
fn io_error(path: &str, err: std::io::Error) -> StretchError {
    StretchError::IoError(format!("{}: {}", path, err))
}

/// Reads a WAV file from disk.
pub fn read_wav_file(path: &str) -> Result<AudioBuffer, StretchError> {
    let mut file = std::fs::File::open(path).map_err(|e| io_error(path, e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|e| io_error(path, e))?;
    read_wav(&data)
}

/// Writes RIFF/WAVE header and fmt+data chunk headers to a buffer.
fn write_wav_header(
    out: &mut Vec<u8>,
    format_code: u16,
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data_size: u32,
) {
    let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let file_size = 36 + data_size;

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_code.to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
}

/// Allocates a WAV output buffer with the header already written.
fn init_wav_buffer(buffer: &AudioBuffer, format_code: u16, bits_per_sample: u16) -> Vec<u8> {
    let num_channels = buffer.channels.count() as u16;
    let bytes_per_sample = bits_per_sample as usize / 8;
    let data_size = (buffer.data.len() * bytes_per_sample) as u32;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    write_wav_header(
        &mut out,
        format_code,
        num_channels,
        buffer.sample_rate,
        bits_per_sample,
        data_size,
    );
    out
}

/// Writes an audio buffer as a WAV file (16-bit PCM).
pub fn write_wav_16bit(buffer: &AudioBuffer) -> Vec<u8> {
    let mut out = init_wav_buffer(buffer, WAV_FORMAT_PCM, 16);
    for &sample in &buffer.data {
        let clamped = sample.clamp(-1.0, 1.0);
        let raw = (clamped * PCM_16BIT_MAX_OUT) as i16;
        out.extend_from_slice(&raw.to_le_bytes());
    }
    out
}

/// Maximum representable value for 24-bit PCM output (8388607, not 8388608, to avoid clipping).
const PCM_24BIT_MAX_OUT: f32 = 8388607.0;

/// Writes an audio buffer as a WAV file (24-bit PCM).
pub fn write_wav_24bit(buffer: &AudioBuffer) -> Vec<u8> {
    let mut out = init_wav_buffer(buffer, WAV_FORMAT_PCM, 24);
    for &sample in &buffer.data {
        let clamped = sample.clamp(-1.0, 1.0);
        let raw = (clamped * PCM_24BIT_MAX_OUT) as i32;
        out.push(raw as u8);
        out.push((raw >> 8) as u8);
        out.push((raw >> 16) as u8);
    }
    out
}

/// Writes an audio buffer as a WAV file (32-bit float).
pub fn write_wav_float(buffer: &AudioBuffer) -> Vec<u8> {
    let mut out = init_wav_buffer(buffer, WAV_FORMAT_IEEE_FLOAT, 32);
    for &sample in &buffer.data {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Writes WAV data to disk.
fn write_wav_file(path: &str, data: &[u8]) -> Result<(), StretchError> {
    let mut file = std::fs::File::create(path).map_err(|e| io_error(path, e))?;
    file.write_all(data).map_err(|e| io_error(path, e))?;
    Ok(())
}

/// Writes a WAV file to disk (16-bit PCM).
pub fn write_wav_file_16bit(path: &str, buffer: &AudioBuffer) -> Result<(), StretchError> {
    write_wav_file(path, &write_wav_16bit(buffer))
}

/// Writes a WAV file to disk (24-bit PCM).
pub fn write_wav_file_24bit(path: &str, buffer: &AudioBuffer) -> Result<(), StretchError> {
    write_wav_file(path, &write_wav_24bit(buffer))
}

/// Writes a WAV file to disk (32-bit float).
pub fn write_wav_file_float(path: &str, buffer: &AudioBuffer) -> Result<(), StretchError> {
    write_wav_file(path, &write_wav_float(buffer))
}

#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[inline]
fn read_i16_le(data: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([data[offset], data[offset + 1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_roundtrip_16bit() {
        let original = AudioBuffer::from_mono(vec![0.0, 0.5, -0.5, 1.0, -1.0], 44100);
        let wav_data = write_wav_16bit(&original);
        let decoded = read_wav(&wav_data).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.channels, Channels::Mono);
        assert_eq!(decoded.data.len(), 5);
        // 16-bit has quantization error
        for i in 0..5 {
            assert!(
                (decoded.data[i] - original.data[i]).abs() < 0.001,
                "sample {}: {} vs {}",
                i,
                decoded.data[i],
                original.data[i]
            );
        }
    }

    #[test]
    fn test_wav_roundtrip_float() {
        let original = AudioBuffer::from_stereo(vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6], 48000);
        let wav_data = write_wav_float(&original);
        let decoded = read_wav(&wav_data).unwrap();
        assert_eq!(decoded.sample_rate, 48000);
        assert_eq!(decoded.channels, Channels::Stereo);
        assert_eq!(decoded.data.len(), 6);
        for i in 0..6 {
            assert!(
                (decoded.data[i] - original.data[i]).abs() < 1e-6,
                "sample {}: {} vs {}",
                i,
                decoded.data[i],
                original.data[i]
            );
        }
    }

    #[test]
    fn test_wav_roundtrip_24bit() {
        let original = AudioBuffer::from_mono(vec![0.0, 0.5, -0.5, 1.0, -1.0], 44100);
        let wav_data = write_wav_24bit(&original);
        let decoded = read_wav(&wav_data).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.channels, Channels::Mono);
        assert_eq!(decoded.data.len(), 5);
        // 24-bit has very small quantization error
        for i in 0..5 {
            assert!(
                (decoded.data[i] - original.data[i]).abs() < 0.0001,
                "sample {}: {} vs {}",
                i,
                decoded.data[i],
                original.data[i]
            );
        }
    }

    #[test]
    fn test_wav_24bit_stereo() {
        let data = vec![0.25, -0.25, 0.5, -0.5, 0.75, -0.75];
        let original = AudioBuffer::from_stereo(data, 48000);
        let wav_data = write_wav_24bit(&original);
        let decoded = read_wav(&wav_data).unwrap();
        assert_eq!(decoded.channels, Channels::Stereo);
        assert_eq!(decoded.sample_rate, 48000);
        assert_eq!(decoded.num_frames(), 3);
        for i in 0..6 {
            assert!(
                (decoded.data[i] - original.data[i]).abs() < 0.0001,
                "sample {}: {} vs {}",
                i,
                decoded.data[i],
                original.data[i]
            );
        }
    }

    #[test]
    fn test_wav_invalid_data() {
        assert!(read_wav(&[]).is_err());
        assert!(read_wav(b"NOT_RIFF_HEADER_AT_ALL______________________").is_err());
    }

    #[test]
    fn test_wav_stereo_16bit() {
        let data = vec![0.25, -0.25, 0.5, -0.5];
        let original = AudioBuffer::from_stereo(data, 44100);
        let wav = write_wav_16bit(&original);
        let decoded = read_wav(&wav).unwrap();
        assert_eq!(decoded.channels, Channels::Stereo);
        assert_eq!(decoded.num_frames(), 2);
    }
}
