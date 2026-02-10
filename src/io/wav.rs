use crate::core::types::{AudioBuffer, Channels, Sample};
use crate::error::StretchError;
use std::io::{Read, Write};

/// WAV audio format codes.
const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;

/// Reads a WAV file from a byte slice.
pub fn read_wav(data: &[u8]) -> Result<AudioBuffer, StretchError> {
    let mut cursor = 0;

    // RIFF header
    if data.len() < 44 {
        return Err(StretchError::InvalidFormat(
            "WAV file too short".to_string(),
        ));
    }

    let riff = &data[0..4];
    if riff != b"RIFF" {
        return Err(StretchError::InvalidFormat(
            "Missing RIFF header".to_string(),
        ));
    }
    cursor += 4;

    let _file_size = read_u32_le(data, cursor);
    cursor += 4;

    let wave = &data[cursor..cursor + 4];
    if wave != b"WAVE" {
        return Err(StretchError::InvalidFormat(
            "Missing WAVE identifier".to_string(),
        ));
    }
    cursor += 4;

    // Find fmt and data chunks
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
            if cursor + 16 > data.len() {
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
                // Use whatever data is available
                audio_data = &data[cursor..];
            } else {
                audio_data = &data[cursor..cursor + chunk_size];
            }
        }

        cursor += chunk_size;
        // WAV chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            cursor += 1;
        }
    }

    if sample_rate == 0 {
        return Err(StretchError::InvalidFormat(
            "No fmt chunk found".to_string(),
        ));
    }

    let channels = match num_channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => {
            return Err(StretchError::InvalidFormat(format!(
                "Unsupported channel count: {}",
                n
            )))
        }
    };

    // Convert audio data to f32 samples
    let samples: Vec<Sample> = match (format_code, bits_per_sample) {
        (WAV_FORMAT_PCM, 16) => {
            let num_samples = audio_data.len() / 2;
            let mut result = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let raw = read_i16_le(audio_data, i * 2);
                result.push(raw as f32 / 32768.0);
            }
            result
        }
        (WAV_FORMAT_PCM, 24) => {
            let num_samples = audio_data.len() / 3;
            let mut result = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let offset = i * 3;
                let raw = (audio_data[offset] as i32)
                    | ((audio_data[offset + 1] as i32) << 8)
                    | ((audio_data[offset + 2] as i32) << 16);
                // Sign extend
                let raw = if raw & 0x800000 != 0 {
                    raw | !0xFFFFFF
                } else {
                    raw
                };
                result.push(raw as f32 / 8388608.0);
            }
            result
        }
        (WAV_FORMAT_IEEE_FLOAT, 32) => {
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
        (fmt, bits) => {
            return Err(StretchError::InvalidFormat(format!(
                "Unsupported WAV format: code={}, bits={}",
                fmt, bits
            )))
        }
    };

    Ok(AudioBuffer::new(samples, sample_rate, channels))
}

/// Reads a WAV file from disk.
pub fn read_wav_file(path: &str) -> Result<AudioBuffer, StretchError> {
    let mut file =
        std::fs::File::open(path).map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    read_wav(&data)
}

/// Writes an audio buffer as a WAV file (16-bit PCM).
pub fn write_wav_16bit(buffer: &AudioBuffer) -> Vec<u8> {
    let num_channels = buffer.channels.count() as u16;
    let bits_per_sample: u16 = 16;
    let byte_rate = buffer.sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = (buffer.data.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut out = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&WAV_FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &sample in &buffer.data {
        let clamped = sample.clamp(-1.0, 1.0);
        let raw = (clamped * 32767.0) as i16;
        out.extend_from_slice(&raw.to_le_bytes());
    }

    out
}

/// Writes an audio buffer as a WAV file (32-bit float).
pub fn write_wav_float(buffer: &AudioBuffer) -> Vec<u8> {
    let num_channels = buffer.channels.count() as u16;
    let bits_per_sample: u16 = 32;
    let byte_rate = buffer.sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = (buffer.data.len() * 4) as u32;
    let file_size = 36 + data_size;

    let mut out = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&WAV_FORMAT_IEEE_FLOAT.to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &sample in &buffer.data {
        out.extend_from_slice(&sample.to_le_bytes());
    }

    out
}

/// Writes a WAV file to disk (16-bit PCM).
pub fn write_wav_file_16bit(path: &str, buffer: &AudioBuffer) -> Result<(), StretchError> {
    let data = write_wav_16bit(buffer);
    let mut file = std::fs::File::create(path)
        .map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    file.write_all(&data)
        .map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    Ok(())
}

/// Writes a WAV file to disk (32-bit float).
pub fn write_wav_file_float(path: &str, buffer: &AudioBuffer) -> Result<(), StretchError> {
    let data = write_wav_float(buffer);
    let mut file = std::fs::File::create(path)
        .map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    file.write_all(&data)
        .map_err(|e| StretchError::IoError(format!("{}: {}", path, e)))?;
    Ok(())
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
        let original = AudioBuffer::from_mono(
            vec![0.0, 0.5, -0.5, 1.0, -1.0],
            44100,
        );
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
        let original = AudioBuffer::from_stereo(
            vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            48000,
        );
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
