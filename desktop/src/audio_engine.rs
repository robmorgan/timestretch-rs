use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::brake::BrakeResampler;
use crate::scrub::ScrubVoice;
use crate::state::{AtomicRate, AtomicVolume, ScrubPhase, ScrubState};

/// Scrub voice crossfade time in seconds (engage and release).
const SCRUB_MIX_SECS: f32 = 0.005;

pub struct AudioEngine {
    _stream: Stream,
}

impl AudioEngine {
    /// Queries the default output device's sample rate without opening a
    /// stream. Used at startup to size the UI before any track is loaded.
    pub fn default_sample_rate() -> Option<u32> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let config = device.default_output_config().ok()?;
        Some(config.sample_rate().0)
    }

    /// Create an audio engine whose callback owns an
    /// [`timestretch::engine::EngineProcessor`] and reads audio from it
    /// directly — no intermediate processed-audio ring. The feed thread
    /// keeps the engine's source ring topped up and signals hard restarts
    /// via `reset_request`, which the callback acknowledges by clearing the
    /// flag after resetting.
    ///
    /// While `scrub` is active the callback crossfades to a raw varispeed
    /// read of `source_audio` chasing the published pointer position — the
    /// audible CDJ-style scrub — and back to the engine on release.
    ///
    /// `brake` is the Wide range's sub-floor fader factor (1.0 = none):
    /// below 1.0 the callback reads the engine through a
    /// [`BrakeResampler`], slowing and pitch-dropping the keylocked output
    /// down to a frozen stop at 0.0.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        volume: Arc<AtomicVolume>,
        stream_active: Arc<AtomicBool>,
        desired_sample_rate: Option<u32>,
        mut processor: timestretch::engine::EngineProcessor,
        reset_request: Arc<AtomicBool>,
        scrub: Arc<ScrubState>,
        source_audio: Arc<Vec<f32>>,
        brake: Arc<AtomicRate>,
    ) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No audio output device found".to_string())?;

        let default_config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default output config: {e}"))?;

        let sample_rate = desired_sample_rate.unwrap_or(default_config.sample_rate().0);
        let config = StreamConfig {
            channels: 2,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut voice = ScrubVoice::new(sample_rate);
        let mut brake_resampler = BrakeResampler::new(sample_rate);
        let mix_alpha = 1.0 - (-1.0 / (SCRUB_MIX_SECS * sample_rate as f32)).exp();
        let mut scrub_mix: f32 = 0.0;
        let mut prev_phase = ScrubPhase::Idle;
        let mut scratch: Vec<f32> = Vec::with_capacity(8192);

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Acknowledge a pending hard restart before anything
                    // else so seeks work even while output is inactive.
                    if reset_request.load(Ordering::Acquire) {
                        processor.reset();
                        // A warm-start seek must not replay pre-seek
                        // frames buffered in the brake FIFO.
                        brake_resampler.reset();
                        reset_request.store(false, Ordering::Release);
                    }

                    let phase = scrub.phase();
                    // Phase edges: seed on a fresh engage; start the release
                    // glide (and publish its predicted landing for the UI's
                    // parallel engine warm-start) on entry to Settling. A
                    // press+release inside one block arrives as Idle →
                    // Settling — seed first, then glide.
                    if prev_phase == ScrubPhase::Idle && phase != ScrubPhase::Idle {
                        voice.seed(scrub.target());
                    }
                    if prev_phase != ScrubPhase::Settling && phase == ScrubPhase::Settling {
                        let landing = voice.begin_settle(scrub.settle_rate_target(), &source_audio);
                        scrub.publish_landing(landing);
                    }
                    prev_phase = phase;

                    let scrubbing = phase != ScrubPhase::Idle;
                    let engine_live = stream_active.load(Ordering::Relaxed);
                    if !engine_live && !scrubbing && scrub_mix == 0.0 {
                        data.fill(0.0);
                        return;
                    }

                    // Engine path. While the scrub voice fully owns the
                    // output, the engine is left unconsumed (frozen) — its
                    // state is discarded by the release-time warm-start seek.
                    if engine_live && scrub_mix < 1.0 {
                        let b = brake.load();
                        if b < 1.0 || brake_resampler.engaged() {
                            brake_resampler.render(b, data, |buf| processor.process(buf));
                        } else {
                            processor.process(data);
                        }
                    } else {
                        data.fill(0.0);
                    }

                    // Scrub voice, crossfaded in per frame while engaged and
                    // back out after release.
                    if scrubbing || scrub_mix > 0.0 {
                        scratch.resize(data.len(), 0.0);
                        match phase {
                            ScrubPhase::Active => {
                                voice.render(scrub.target(), &source_audio, &mut scratch);
                            }
                            // Settling and the post-settle mix ramp-out both
                            // follow the glide: past the landing the voice
                            // holds the settle rate, time-aligned with the
                            // engine warm-started there.
                            ScrubPhase::Settling | ScrubPhase::Idle => {
                                if voice.render_settle(&source_audio, &mut scratch)
                                    && phase == ScrubPhase::Settling
                                {
                                    scrub.finish_settle();
                                    prev_phase = ScrubPhase::Idle;
                                }
                            }
                        }
                        scrub.publish_voice_frame(voice.position());
                        scrub.publish_voice_rate(voice.rate());
                        let mix_target: f32 = if scrubbing { 1.0 } else { 0.0 };
                        for (frame, voice_frame) in
                            data.chunks_exact_mut(2).zip(scratch.chunks_exact(2))
                        {
                            scrub_mix += (mix_target - scrub_mix) * mix_alpha;
                            for (out, &v) in frame.iter_mut().zip(voice_frame) {
                                *out = *out * (1.0 - scrub_mix) + v * scrub_mix;
                            }
                        }
                        // Snap the asymptotic one-pole at the rails: without
                        // this, mix stalls one ulp below 1.0 and the engine
                        // keeps being consumed (unfed, at inaudible gain) for
                        // the whole drag, draining its ring into underruns.
                        if scrubbing && scrub_mix > 0.999 {
                            scrub_mix = 1.0;
                        } else if !scrubbing && scrub_mix < 1e-4 {
                            scrub_mix = 0.0;
                        }
                    }

                    // Lock-free: a realtime thread must never wait on the
                    // UI's state mutex.
                    let volume = volume.load();
                    for sample in data.iter_mut() {
                        *sample *= volume;
                    }
                },
                move |err| {
                    log::error!("Audio output error: {err}");
                },
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start audio stream: {e}"))?;

        Ok(AudioEngine { _stream: stream })
    }
}
