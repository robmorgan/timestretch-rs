//! Prepared application delivery state. `render` is the device callback body;
//! no DSP session, thread scheduling, allocation, locks or logging live here.
use crate::{
    output_queue::Consumer,
    scrub::ScrubVoice,
    state::{AtomicVolume, ScrubPhase, ScrubState},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
pub struct Delivery {
    consumer: Consumer,
    volume: Arc<AtomicVolume>,
    stream_active: Arc<AtomicBool>,
    scrub: Arc<ScrubState>,
    source_audio: Arc<Vec<f32>>,
    voice: ScrubVoice,
    mix_alpha: f32,
    scrub_mix: f32,
    prev_phase: ScrubPhase,
    scratch: Box<[f32; 4096]>,
    transition: timestretch::TransitionEnvelope,
    last_frame: [f32; 2],
    was_live: bool,
}
impl Delivery {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: u32,
        consumer: Consumer,
        volume: Arc<AtomicVolume>,
        stream_active: Arc<AtomicBool>,
        scrub: Arc<ScrubState>,
        source_audio: Arc<Vec<f32>>,
    ) -> Self {
        Self {
            consumer,
            volume,
            stream_active,
            scrub,
            source_audio,
            voice: ScrubVoice::new(sample_rate),
            mix_alpha: 1.0 - (-1.0 / (0.005 * sample_rate as f32)).exp(),
            scrub_mix: 0.0,
            prev_phase: ScrubPhase::Idle,
            scratch: Box::new([0.0; 4096]),
            transition: Default::default(),
            last_frame: [0.0; 2],
            was_live: false,
        }
    }
    pub fn render(&mut self, data: &mut [f32]) {
        // Chunk arbitrarily large device callbacks through fixed storage.
        for data in data.chunks_mut(4096) {
            if self.consumer.sync_epoch() {
                self.transition.begin(self.last_frame);
            }
            let phase = self.scrub.phase();
            // Phase edges: seed on a fresh engage; start the release
            // glide (and publish its predicted landing for the UI's
            // parallel engine warm-start) on entry to Settling. A
            // press+release inside one block arrives as Idle →
            // Settling — seed first, then glide.
            if self.prev_phase == ScrubPhase::Idle && phase != ScrubPhase::Idle {
                self.voice.seed(self.scrub.target());
            }
            if self.prev_phase != ScrubPhase::Settling && phase == ScrubPhase::Settling {
                let landing = self
                    .voice
                    .begin_settle(self.scrub.settle_rate_target(), &self.source_audio);
                self.scrub.publish_landing(landing);
            }
            self.prev_phase = phase;

            let scrubbing = phase != ScrubPhase::Idle;
            let engine_live = self.stream_active.load(Ordering::Relaxed);
            if engine_live != self.was_live {
                self.transition.begin(if engine_live {
                    [0.0; 2]
                } else {
                    self.last_frame
                });
                self.was_live = engine_live;
            }
            if !engine_live && !scrubbing && self.scrub_mix == 0.0 {
                data.fill(0.0);
                self.transition.apply(data, 2);
                if data.len() >= 2 {
                    self.last_frame.copy_from_slice(&data[data.len() - 2..]);
                }
                let volume = self.volume.load();
                for sample in data.iter_mut() {
                    *sample *= volume;
                }
                continue;
            }

            // Engine path. While the scrub voice fully owns the
            // output, the engine is left unconsumed (frozen) — its
            // state is discarded by the release-time warm-start seek.
            if engine_live && self.scrub_mix < 1.0 {
                let consumed = self.consumer.read(data);
                self.consumer.acknowledge_frames(consumed);
            } else {
                data.fill(0.0);
            }

            // Scrub voice, crossfaded in per frame while engaged and
            // back out after release.
            if scrubbing || self.scrub_mix > 0.0 {
                let scratch = &mut self.scratch[..data.len()];
                match phase {
                    ScrubPhase::Active => {
                        self.voice
                            .render(self.scrub.target(), &self.source_audio, scratch);
                    }
                    // Settling and the post-settle mix ramp-out both
                    // follow the glide: past the landing the voice
                    // holds the settle rate, time-aligned with the
                    // engine warm-started there.
                    ScrubPhase::Settling | ScrubPhase::Idle => {
                        if self.voice.render_settle(&self.source_audio, scratch)
                            && phase == ScrubPhase::Settling
                        {
                            self.scrub.finish_settle();
                            self.prev_phase = ScrubPhase::Idle;
                        }
                    }
                }
                self.scrub.publish_voice_frame(self.voice.position());
                self.scrub.publish_voice_rate(self.voice.rate());
                let mix_target: f32 = if scrubbing { 1.0 } else { 0.0 };
                for (frame, voice_frame) in data.chunks_exact_mut(2).zip(scratch.chunks_exact(2)) {
                    self.scrub_mix += (mix_target - self.scrub_mix) * self.mix_alpha;
                    for (out, &v) in frame.iter_mut().zip(voice_frame) {
                        *out = *out * (1.0 - self.scrub_mix) + v * self.scrub_mix;
                    }
                }
                // Snap the asymptotic one-pole at the rails: without
                // this, mix stalls one ulp below 1.0 and the engine
                // keeps being consumed (unfed, at inaudible gain) for
                // the whole drag, draining its ring into underruns.
                if scrubbing && self.scrub_mix > 0.999 {
                    self.scrub_mix = 1.0;
                } else if !scrubbing && self.scrub_mix < 1e-4 {
                    self.scrub_mix = 0.0;
                }
            }

            self.transition.apply(data, 2);
            if data.len() >= 2 {
                self.last_frame.copy_from_slice(&data[data.len() - 2..]);
            }
            // Lock-free: a realtime thread must never wait on the
            // UI's state mutex.
            let volume = self.volume.load();
            for sample in data.iter_mut() {
                *sample *= volume;
            }
        }
    }
}
