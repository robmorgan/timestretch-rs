use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio data.
pub struct DecodedAudio {
    /// Interleaved stereo f32 samples.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    /// Total frames (samples per channel).
    pub num_frames: usize,
}

/// Decode an audio file to interleaved stereo f32 samples.
pub fn decode_file(path: &Path) -> Result<DecodedAudio, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("Failed to probe format: {e}"))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| "No audio track found".to_string())?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let sample_rate = codec_params.sample_rate.ok_or("Unknown sample rate")?;
    let src_channels = codec_params
        .channels
        .map(|c| c.count() as u32)
        .unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {e}"))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(format!("Error reading packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Decode error: {e}")),
        };

        append_samples(&decoded, src_channels, &mut all_samples);
    }

    // If mono, convert to interleaved stereo
    let (samples, out_channels) = if src_channels == 1 {
        let stereo: Vec<f32> = all_samples
            .iter()
            .flat_map(|&s| [s, s])
            .collect();
        (stereo, 2)
    } else {
        (all_samples, src_channels.min(2))
    };

    let num_frames = samples.len() / out_channels as usize;

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels: out_channels,
        num_frames,
    })
}

fn append_samples(buf: &AudioBufferRef, src_channels: u32, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().min(2);
            for f in 0..frames {
                for c in 0..chans {
                    out.push(*b.chan(c).get(f).unwrap_or(&0.0));
                }
                // If source is mono, duplicate for stereo output handled later
                if chans == 1 && src_channels == 1 {
                    // mono samples go in as mono; stereo conversion happens in decode_file
                }
            }
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().min(2);
            for f in 0..frames {
                for c in 0..chans {
                    let sample = *b.chan(c).get(f).unwrap_or(&0);
                    out.push(sample as f32 / 32768.0);
                }
            }
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().min(2);
            for f in 0..frames {
                for c in 0..chans {
                    let sample = *b.chan(c).get(f).unwrap_or(&0);
                    out.push(sample as f32 / 2_147_483_648.0);
                }
            }
        }
        AudioBufferRef::U8(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().min(2);
            for f in 0..frames {
                for c in 0..chans {
                    let sample = *b.chan(c).get(f).unwrap_or(&128);
                    out.push((sample as f32 - 128.0) / 128.0);
                }
            }
        }
        _ => {
            // For other formats, try to get F32 data
            log::warn!("Unsupported sample format, skipping packet");
        }
    }
}
