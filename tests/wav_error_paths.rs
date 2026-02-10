//! Tests for WAV I/O error paths and corrupt file handling.
//!
//! Validates that the WAV parser correctly rejects malformed, truncated, and
//! unsupported WAV files with descriptive error messages.

use timestretch::io::wav::{read_wav, write_wav_16bit, write_wav_24bit, write_wav_float};
use timestretch::{AudioBuffer, Channels, StretchError};

// ---------- RIFF/WAVE header validation ----------

#[test]
fn test_wav_empty_input() {
    let result = read_wav(&[]);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => assert!(msg.contains("too short"), "msg: {}", msg),
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_truncated_riff_header() {
    // Only 8 bytes — valid RIFF but no WAVE, too short overall
    let data = b"RIFF\x00\x00\x00\x00";
    let result = read_wav(data);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => assert!(msg.contains("too short"), "msg: {}", msg),
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_missing_riff_magic() {
    // 44+ bytes but not starting with RIFF
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(b"NOPE");
    data[8..12].copy_from_slice(b"WAVE");
    let result = read_wav(&data);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => assert!(msg.contains("RIFF"), "msg: {}", msg),
        e => panic!("Expected InvalidFormat with RIFF, got {:?}", e),
    }
}

#[test]
fn test_wav_missing_wave_identifier() {
    // Valid RIFF but "AVI " instead of "WAVE"
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(b"RIFF");
    data[4..8].copy_from_slice(&36u32.to_le_bytes());
    data[8..12].copy_from_slice(b"AVI ");
    let result = read_wav(&data);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => assert!(msg.contains("WAVE"), "msg: {}", msg),
        e => panic!("Expected InvalidFormat with WAVE, got {:?}", e),
    }
}

#[test]
fn test_wav_various_wrong_identifiers() {
    // Test several wrong identifiers
    for wrong_id in &[b"AIFF", b"OGGx", b"FLAC"] {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(b"RIFF");
        data[4..8].copy_from_slice(&40u32.to_le_bytes());
        data[8..12].copy_from_slice(*wrong_id);
        let result = read_wav(&data);
        assert!(result.is_err(), "Should reject identifier: {:?}", wrong_id);
    }
}

// ---------- fmt chunk validation ----------

#[test]
fn test_wav_no_fmt_chunk() {
    // Valid RIFF/WAVE header, but contains only a data chunk, no fmt
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&44u32.to_le_bytes()); // file size
    data.extend_from_slice(b"WAVE");
    // data chunk only (no fmt)
    data.extend_from_slice(b"data");
    data.extend_from_slice(&4u32.to_le_bytes()); // 4 bytes of audio
    data.extend_from_slice(&[0u8; 4]); // audio data
                                       // Pad to minimum header size
    while data.len() < 44 {
        data.push(0);
    }
    let result = read_wav(&data);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(
                msg.contains("fmt") || msg.contains("No fmt"),
                "msg: {}",
                msg
            );
        }
        e => panic!("Expected InvalidFormat about fmt chunk, got {:?}", e),
    }
}

#[test]
fn test_wav_fmt_chunk_too_short() {
    // RIFF/WAVE header then fmt chunk with size < 16
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&100u32.to_le_bytes());
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&8u32.to_le_bytes()); // Only 8 bytes of fmt (needs 16)
    data.extend_from_slice(&[0u8; 8]); // 8 bytes of partial fmt data
                                       // data chunk
    data.extend_from_slice(b"data");
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]);
    // Pad
    while data.len() < 44 {
        data.push(0);
    }
    let result = read_wav(&data);
    // This should fail because either the fmt is too short, or the format is unsupported
    assert!(result.is_err(), "Should reject truncated fmt chunk");
}

// ---------- Unsupported format combinations ----------

fn build_wav_with_format(format_code: u16, bits_per_sample: u16, audio_bytes: &[u8]) -> Vec<u8> {
    let num_channels: u16 = 1;
    let sample_rate: u32 = 44100;
    let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = audio_bytes.len() as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&format_code.to_le_bytes());
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(audio_bytes);
    wav
}

#[test]
fn test_wav_unsupported_8bit_pcm() {
    let audio = vec![128u8; 100]; // 8-bit unsigned PCM
    let wav = build_wav_with_format(1, 8, &audio);
    let result = read_wav(&wav);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(msg.contains("Unsupported"), "msg: {}", msg);
            assert!(msg.contains("8"), "Should mention 8-bit: {}", msg);
        }
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_unsupported_32bit_pcm() {
    let audio = vec![0u8; 400]; // 100 samples * 4 bytes
    let wav = build_wav_with_format(1, 32, &audio);
    let result = read_wav(&wav);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(msg.contains("Unsupported"), "msg: {}", msg);
        }
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_unsupported_ieee_float_16bit() {
    let audio = vec![0u8; 200];
    let wav = build_wav_with_format(3, 16, &audio); // IEEE float with 16 bits
    let result = read_wav(&wav);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(msg.contains("Unsupported"), "msg: {}", msg);
        }
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_unsupported_ieee_float_24bit() {
    let audio = vec![0u8; 300];
    let wav = build_wav_with_format(3, 24, &audio);
    let result = read_wav(&wav);
    assert!(result.is_err());
}

#[test]
fn test_wav_unsupported_adpcm_format() {
    let audio = vec![0u8; 100];
    let wav = build_wav_with_format(2, 4, &audio); // ADPCM
    let result = read_wav(&wav);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(msg.contains("Unsupported"), "msg: {}", msg);
        }
        e => panic!("Expected InvalidFormat, got {:?}", e),
    }
}

#[test]
fn test_wav_unsupported_alaw_format() {
    let audio = vec![0u8; 100];
    let wav = build_wav_with_format(6, 8, &audio); // A-law
    let result = read_wav(&wav);
    assert!(result.is_err());
}

#[test]
fn test_wav_unsupported_mulaw_format() {
    let audio = vec![0u8; 100];
    let wav = build_wav_with_format(7, 8, &audio); // mu-law
    let result = read_wav(&wav);
    assert!(result.is_err());
}

// ---------- Channel count validation ----------

#[test]
fn test_wav_unsupported_channel_count() {
    // Build a WAV with 6 channels (5.1 surround)
    let num_channels: u16 = 6;
    let sample_rate: u32 = 44100;
    let bits: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * 2;
    let block_align = num_channels * 2;
    let audio = vec![0u8; 120]; // 10 frames * 6 channels * 2 bytes
    let data_size = audio.len() as u32;

    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_size).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&audio);

    let result = read_wav(&wav);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::InvalidFormat(msg) => {
            assert!(msg.contains("channel"), "msg: {}", msg);
        }
        e => panic!("Expected InvalidFormat about channels, got {:?}", e),
    }
}

#[test]
fn test_wav_zero_channels() {
    // 0 channels should be rejected
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&36u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&0u16.to_le_bytes()); // 0 channels
    wav.extend_from_slice(&44100u32.to_le_bytes());
    wav.extend_from_slice(&0u32.to_le_bytes()); // byte rate
    wav.extend_from_slice(&0u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&0u32.to_le_bytes());

    let result = read_wav(&wav);
    assert!(result.is_err(), "Should reject 0 channels");
}

// ---------- Truncated audio data ----------

#[test]
fn test_wav_truncated_data_chunk() {
    // data chunk claims 1000 bytes but file ends early
    let _wav = build_wav_with_format(1, 16, &[0u8; 20]);
    // Manually create a WAV where chunk_size > actual data
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&100u32.to_le_bytes());
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&44100u32.to_le_bytes());
    data.extend_from_slice(&88200u32.to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&1000u32.to_le_bytes()); // claims 1000 bytes
    data.extend_from_slice(&[0u8; 20]); // only 20 bytes present

    // Parser should handle gracefully (either error or use available data)
    let result = read_wav(&data);
    // The parser uses fallback to available data, so this should succeed with fewer samples
    if let Ok(buf) = &result {
        assert!(
            buf.data.len() <= 500,
            "Should have at most 500 samples from 1000 bytes claim"
        );
    }
    // Either way, no panic is the key requirement
}

#[test]
fn test_wav_empty_data_chunk() {
    let wav = build_wav_with_format(1, 16, &[]);
    let result = read_wav(&wav);
    // Empty audio data is valid — 0 samples
    if let Ok(buf) = result {
        assert!(buf.data.is_empty());
    }
}

#[test]
fn test_wav_single_sample_16bit() {
    let wav = build_wav_with_format(1, 16, &[0x00, 0x40]); // Single 16-bit sample
    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 1);
    assert!(result.data[0] > 0.0); // 0x4000 = 16384 -> ~0.5
}

#[test]
fn test_wav_odd_byte_count_16bit() {
    // 16-bit with 3 bytes (1.5 samples) — parser should read only 1 sample
    let wav = build_wav_with_format(1, 16, &[0x00, 0x40, 0xFF]);
    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 1);
}

#[test]
fn test_wav_incomplete_24bit_sample() {
    // 24-bit with 5 bytes (1 complete + 2 extra) — parser should read only 1 sample
    let wav = build_wav_with_format(1, 24, &[0x00, 0x00, 0x40, 0xFF, 0xFF]);
    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 1);
}

#[test]
fn test_wav_incomplete_float_sample() {
    // 32-bit float with 6 bytes (1 complete + 2 extra) — parser should read only 1
    let mut audio = Vec::new();
    audio.extend_from_slice(&0.5f32.to_le_bytes());
    audio.extend_from_slice(&[0xFF, 0xFF]); // incomplete second sample
    let wav = build_wav_with_format(3, 32, &audio);
    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 1);
    assert!((result.data[0] - 0.5).abs() < 1e-6);
}

// ---------- WAV write/read round-trip edge cases ----------

#[test]
fn test_wav_roundtrip_empty_buffer() {
    let buf = AudioBuffer::from_mono(vec![], 44100);
    let wav_16 = write_wav_16bit(&buf);
    let decoded = read_wav(&wav_16).unwrap();
    assert!(decoded.data.is_empty());
    assert_eq!(decoded.sample_rate, 44100);
}

#[test]
fn test_wav_roundtrip_single_sample_all_formats() {
    let buf = AudioBuffer::from_mono(vec![0.5], 44100);

    // 16-bit
    let decoded = read_wav(&write_wav_16bit(&buf)).unwrap();
    assert_eq!(decoded.data.len(), 1);
    assert!((decoded.data[0] - 0.5).abs() < 0.001);

    // 24-bit
    let decoded = read_wav(&write_wav_24bit(&buf)).unwrap();
    assert_eq!(decoded.data.len(), 1);
    assert!((decoded.data[0] - 0.5).abs() < 0.0001);

    // Float
    let decoded = read_wav(&write_wav_float(&buf)).unwrap();
    assert_eq!(decoded.data.len(), 1);
    assert!((decoded.data[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_wav_clipping_boundary_values() {
    // Values outside [-1, 1] should be clamped by writer
    let buf = AudioBuffer::from_mono(vec![-2.0, -1.0, 0.0, 1.0, 2.0], 44100);

    // 16-bit clamps
    let decoded_16 = read_wav(&write_wav_16bit(&buf)).unwrap();
    assert!(decoded_16.data[0] >= -1.0 && decoded_16.data[0] <= -0.99);
    assert!(decoded_16.data[4] >= 0.99 && decoded_16.data[4] <= 1.0);

    // 24-bit clamps
    let decoded_24 = read_wav(&write_wav_24bit(&buf)).unwrap();
    assert!(decoded_24.data[0] >= -1.0 && decoded_24.data[0] <= -0.99);
    assert!(decoded_24.data[4] >= 0.99 && decoded_24.data[4] <= 1.0);
}

#[test]
fn test_wav_16bit_quantization_accuracy() {
    // 16-bit has 1/32768 ≈ 3e-5 resolution
    let values: Vec<f32> = (-10..=10).map(|i| i as f32 / 10.0).collect();
    let buf = AudioBuffer::from_mono(values.clone(), 44100);
    let decoded = read_wav(&write_wav_16bit(&buf)).unwrap();
    for (i, (&orig, &dec)) in values.iter().zip(decoded.data.iter()).enumerate() {
        let clamped = orig.clamp(-1.0, 1.0);
        assert!(
            (dec - clamped).abs() < 0.001,
            "16-bit quantization error at {}: {} vs {}",
            i,
            dec,
            clamped
        );
    }
}

#[test]
fn test_wav_24bit_quantization_accuracy() {
    // 24-bit has 1/8388608 ≈ 1.2e-7 resolution
    let values: Vec<f32> = (-10..=10).map(|i| i as f32 / 10.0).collect();
    let buf = AudioBuffer::from_mono(values.clone(), 44100);
    let decoded = read_wav(&write_wav_24bit(&buf)).unwrap();
    for (i, (&orig, &dec)) in values.iter().zip(decoded.data.iter()).enumerate() {
        let clamped = orig.clamp(-1.0, 1.0);
        assert!(
            (dec - clamped).abs() < 0.0001,
            "24-bit quantization error at {}: {} vs {}",
            i,
            dec,
            clamped
        );
    }
}

#[test]
fn test_wav_float_exact_roundtrip() {
    // 32-bit float should be bit-exact
    let values = vec![
        0.0,
        1.0,
        -1.0,
        0.123_456_79,
        -0.987_654_3,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    let buf = AudioBuffer::from_mono(values.clone(), 48000);
    let decoded = read_wav(&write_wav_float(&buf)).unwrap();
    for (i, (&orig, &dec)) in values.iter().zip(decoded.data.iter()).enumerate() {
        assert!(
            (dec - orig).abs() < f32::EPSILON,
            "Float roundtrip mismatch at {}: {} vs {}",
            i,
            dec,
            orig
        );
    }
}

#[test]
fn test_wav_stereo_roundtrip_all_formats() {
    let data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let buf = AudioBuffer::from_stereo(data.clone(), 44100);

    // 16-bit stereo
    let decoded = read_wav(&write_wav_16bit(&buf)).unwrap();
    assert_eq!(decoded.channels, Channels::Stereo);
    assert_eq!(decoded.data.len(), 8);

    // 24-bit stereo
    let decoded = read_wav(&write_wav_24bit(&buf)).unwrap();
    assert_eq!(decoded.channels, Channels::Stereo);
    assert_eq!(decoded.data.len(), 8);

    // Float stereo
    let decoded = read_wav(&write_wav_float(&buf)).unwrap();
    assert_eq!(decoded.channels, Channels::Stereo);
    assert_eq!(decoded.data.len(), 8);
}

#[test]
fn test_wav_48khz_sample_rate_preservation() {
    let buf = AudioBuffer::from_mono(vec![0.5; 48000], 48000);
    let decoded = read_wav(&write_wav_float(&buf)).unwrap();
    assert_eq!(decoded.sample_rate, 48000);
}

#[test]
fn test_wav_24bit_negative_values() {
    // Specifically test 24-bit sign extension for negative values
    let values = vec![-1.0, -0.5, -0.25, -0.125];
    let buf = AudioBuffer::from_mono(values.clone(), 44100);
    let decoded = read_wav(&write_wav_24bit(&buf)).unwrap();
    for (i, (&orig, &dec)) in values.iter().zip(decoded.data.iter()).enumerate() {
        assert!(dec < 0.0, "Sample {} should be negative, got {}", i, dec);
        assert!(
            (dec - orig).abs() < 0.0001,
            "24-bit negative value error at {}: {} vs {}",
            i,
            dec,
            orig
        );
    }
}

// ---------- WAV file I/O error paths ----------

#[test]
fn test_wav_file_read_nonexistent() {
    let result = timestretch::io::wav::read_wav_file("/nonexistent/path/test.wav");
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::IoError(msg) => {
            assert!(msg.contains("/nonexistent"), "msg: {}", msg);
        }
        e => panic!("Expected IoError, got {:?}", e),
    }
}

#[test]
fn test_wav_file_write_invalid_directory() {
    let buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
    let result = timestretch::io::wav::write_wav_file_16bit("/nonexistent/dir/out.wav", &buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        StretchError::IoError(msg) => {
            assert!(msg.contains("/nonexistent"), "msg: {}", msg);
        }
        e => panic!("Expected IoError, got {:?}", e),
    }
}

#[test]
fn test_wav_file_write_24bit_invalid_directory() {
    let buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
    let result = timestretch::io::wav::write_wav_file_24bit("/nonexistent/dir/out.wav", &buf);
    assert!(result.is_err());
}

#[test]
fn test_wav_file_write_float_invalid_directory() {
    let buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
    let result = timestretch::io::wav::write_wav_file_float("/nonexistent/dir/out.wav", &buf);
    assert!(result.is_err());
}

// ---------- WAV with extra/unknown chunks ----------

#[test]
fn test_wav_with_unknown_chunks_before_data() {
    // Some WAV files have INFO, LIST, or other chunks between fmt and data
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&200u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&44100u32.to_le_bytes());
    wav.extend_from_slice(&88200u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());

    // Unknown chunk (should be skipped)
    wav.extend_from_slice(b"LIST");
    wav.extend_from_slice(&8u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 8]); // 8 bytes of list data

    // data chunk with one sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&2u32.to_le_bytes());
    wav.extend_from_slice(&0x4000u16.to_le_bytes()); // ~0.5

    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 1);
    assert!(result.data[0] > 0.0);
}

#[test]
fn test_wav_with_odd_sized_chunk() {
    // WAV chunks are word-aligned — odd-sized chunks should be padded
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&200u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&44100u32.to_le_bytes());
    wav.extend_from_slice(&88200u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());

    // Odd-sized unknown chunk (size 3 — should have 1 byte padding)
    wav.extend_from_slice(b"JUNK");
    wav.extend_from_slice(&3u32.to_le_bytes());
    wav.extend_from_slice(&[0xAA; 3]);
    wav.push(0x00); // padding byte for word alignment

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4u32.to_le_bytes());
    wav.extend_from_slice(&0x4000u16.to_le_bytes());
    wav.extend_from_slice(&0xC000u16.to_le_bytes()); // negative value

    let result = read_wav(&wav).unwrap();
    assert_eq!(result.data.len(), 2);
}

// ---------- Error display ----------

#[test]
fn test_stretch_error_display_formats() {
    let e = StretchError::InvalidFormat("test message".to_string());
    assert!(format!("{}", e).contains("test message"));

    let e = StretchError::InvalidRatio("bad ratio".to_string());
    assert!(format!("{}", e).contains("bad ratio"));

    let e = StretchError::IoError("disk full".to_string());
    assert!(format!("{}", e).contains("disk full"));

    let e = StretchError::InputTooShort {
        provided: 100,
        minimum: 4096,
    };
    let msg = format!("{}", e);
    assert!(msg.contains("100"));
    assert!(msg.contains("4096"));

    let e = StretchError::BpmDetectionFailed("no beats".to_string());
    assert!(format!("{}", e).contains("no beats"));

    let e = StretchError::NonFiniteInput;
    assert!(format!("{}", e).contains("NaN") || format!("{}", e).contains("non-finite"));
}

#[test]
fn test_stretch_error_is_std_error() {
    // Verify StretchError implements std::error::Error
    let e: Box<dyn std::error::Error> = Box::new(StretchError::NonFiniteInput);
    assert!(!format!("{}", e).is_empty());
}

#[test]
fn test_stretch_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let stretch_err: StretchError = io_err.into();
    match stretch_err {
        StretchError::IoError(msg) => assert!(msg.contains("not found"), "msg: {}", msg),
        e => panic!("Expected IoError, got {:?}", e),
    }
}

#[test]
fn test_stretch_error_clone_and_eq() {
    let e1 = StretchError::NonFiniteInput;
    let e2 = e1.clone();
    assert_eq!(e1, e2);

    let e3 = StretchError::InvalidFormat("test".to_string());
    let e4 = StretchError::InvalidFormat("test".to_string());
    assert_eq!(e3, e4);

    assert_ne!(e1, e3);
}
