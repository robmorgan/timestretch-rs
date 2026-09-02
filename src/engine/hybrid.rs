//! PROTO (ROADMAP Stage 25 kill experiment, env-gated off by default).
//!
//! Hybrid transient/tonal decomposition on the offline Keylock ratios —
//! the mechanism class Elastique and Rubber Band R3 share and the one this
//! crate never built. Not a shipping feature: the gate is
//! `TIMESTRETCH_PROTO_HYBRID=2|3`, read once at render time, and the whole
//! DSP lives in this file so a kill verdict is a one-file revert.
//!
//! Mechanism (whole buffer in hand, artifact guaranteed):
//!
//! 1. **Events.** Each artifact onset becomes an event span
//!    `[onset - PRE, onset + POST)` in source samples. Onsets closer than
//!    `MIN_SPACING` are merged (the stronger wins) so every residual segment
//!    is long enough for one full analysis window.
//! 2. **Timeline.** Event `k` is copied *verbatim* (no stretch) so that its
//!    onset lands at `round(onset * ratio)` — the sample-exact position the
//!    engine's timeline map would report. Everything between two events is
//!    a residual segment whose output span is fixed by the neighbouring
//!    events, so its stretch ratio is `out_len / in_len`, slightly above
//!    the nominal ratio (the events took none of the stretch).
//! 3. **Tonal path.** Every residual segment is stretched by a *fresh*
//!    identity-locked PV in the shipped wide-head configuration (FFT ≈ 46
//!    ms, hop = FFT/8, Hann, `PhaseLockingMode::Identity`). A fresh PV
//!    seeds its phases from the segment's own first analysis frame, so at
//!    the seam right after an event the PV phase is the original phase —
//!    Röbel's window-centre alignment, achieved by restarting instead of
//!    by resetting mid-stream. The seam *before* an event carries the
//!    stretched PV's drifted phase against the verbatim chunk; the attack
//!    itself masks that discontinuity, which is why `PRE` is short.
//! 4. **Three-path** (`=3`): each residual segment is first split by HPSS
//!    into harmonic and percussive parts; the harmonic part takes the
//!    identity-locked PV, the percussive remainder (hats, tails, air — the
//!    classic "underwater" source under locked phase) goes through a
//!    magnitude-only noise vocoder with a seeded random phase per bin per
//!    frame. Deterministic by construction (splitmix64, fixed seed).
//! 5. **Recombination.** Amplitude-complementary raised cosines over
//!    `XFADE` at every chunk/residual seam (the sola.rs argument: the two
//!    sides are correlated, equal-power would bulge). Each residual
//!    segment is RMS-matched to its own input so the PV's coherence loss
//!    does not skew the level balance against the verbatim events.
//!
//! Stereo runs as M/S like the wide head; other channel counts run per
//! channel. Output is exactly `round(frames * ratio)` frames.

use std::sync::Arc;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::analysis::hpss::{HpssParams, hpss};
use crate::core::fft::COMPLEX_ZERO;
use crate::core::preanalysis::PreAnalysisArtifact;
use crate::core::window::{WindowType, generate_window};
use crate::stretch::PhaseLockingMode;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Event span before the onset, in seconds.
const PRE_SECS: f64 = 0.005;
/// Default event span after the onset, in seconds. Covers the attack and
/// the early body of a kick; the decay is tonal enough to stretch.
const POST_SECS: f64 = 0.040;
/// Raised-cosine seam length, in seconds.
const XFADE_SECS: f64 = 0.006;
/// Onsets closer than this are merged so residual segments stay longer
/// than one analysis window.
const MIN_SPACING_SECS: f64 = 0.120;
/// Analysis window length in seconds; FFT is the next power of two.
const WINDOW_SECS: f64 = 0.046;
/// HPSS analysis size relative to the PV FFT.
const HPSS_HOP_DIV: usize = 4;
/// Seed for the noise path's phase generator.
const NOISE_SEED: u64 = 0x5EED_5EED_1234_ABCD;

/// The prototype's configuration.
#[derive(Debug, Clone)]
pub struct ProtoHybrid {
    /// 2 = transient + tonal; 3 = transient + tonal + noise.
    pub paths: u8,
    pre: usize,
    post: usize,
    xfade: usize,
    min_spacing: usize,
    fft: usize,
    hop: usize,
}

impl ProtoHybrid {
    /// Reads `TIMESTRETCH_PROTO_HYBRID` (`2` or `3`); anything else is off.
    /// `TIMESTRETCH_PROTO_HYBRID_POST_MS` overrides the post-onset span.
    pub fn from_env(sample_rate: u32) -> Option<Self> {
        let paths: u8 = std::env::var("TIMESTRETCH_PROTO_HYBRID")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        if paths != 2 && paths != 3 {
            return None;
        }
        let mut me = Self::new(paths, sample_rate);
        if let Some(ms) = std::env::var("TIMESTRETCH_PROTO_HYBRID_POST_MS")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|ms| ms.is_finite() && *ms > 0.0)
        {
            me.post = (ms / 1000.0 * sample_rate as f64).round() as usize;
        }
        Some(me)
    }

    /// Explicit constructor (tests and harnesses; no env involved).
    pub fn new(paths: u8, sample_rate: u32) -> Self {
        let sr = sample_rate as f64;
        let fft = (WINDOW_SECS * sr).round().max(256.0) as usize;
        let fft = fft.next_power_of_two();
        Self {
            paths: if paths == 3 { 3 } else { 2 },
            pre: (PRE_SECS * sr).round() as usize,
            post: (POST_SECS * sr).round() as usize,
            xfade: (XFADE_SECS * sr).round().max(8.0) as usize,
            min_spacing: (MIN_SPACING_SECS * sr).round() as usize,
            fft,
            hop: fft / 8,
        }
    }

    /// Renders `input` (interleaved) at `ratio`; returns exactly
    /// `round(frames * ratio) * channels` samples.
    pub fn render(
        &self,
        input: &[f32],
        channels: usize,
        sample_rate: u32,
        ratio: f64,
        artifact: &Arc<PreAnalysisArtifact>,
    ) -> Vec<f32> {
        let frames = input.len() / channels;
        let expected = (frames as f64 * ratio).round() as usize;
        if frames == 0 || expected == 0 {
            return vec![0.0; expected * channels];
        }

        // Processing channels: M/S for stereo (the wide head's convention),
        // plain per-channel otherwise.
        let mut procs: Vec<Vec<f32>> = (0..channels)
            .map(|c| input.iter().skip(c).step_by(channels).copied().collect())
            .collect();
        let ms = channels == 2;
        if ms {
            let (left, right) = procs.split_at_mut(1);
            for (a, b) in left[0].iter_mut().zip(right[0].iter_mut()) {
                let (l, r) = (*a, *b);
                *a = 0.5 * (l + r);
                *b = 0.5 * (l - r);
            }
        }

        let plan = self.plan(frames, expected, ratio, artifact);

        let rendered: Vec<Vec<f32>> = procs
            .iter()
            .enumerate()
            .map(|(ci, x)| self.render_channel(x, sample_rate, expected, &plan, ci as u64))
            .collect();

        let mut out = vec![0.0f32; expected * channels];
        for i in 0..expected {
            if ms {
                let (m, s) = (rendered[0][i], rendered[1][i]);
                out[2 * i] = m + s;
                out[2 * i + 1] = m - s;
            } else {
                for (c, ch) in rendered.iter().enumerate() {
                    out[i * channels + c] = ch[i];
                }
            }
        }
        out
    }

    /// Event list and residual segments on both timelines.
    fn plan(
        &self,
        frames: usize,
        expected: usize,
        ratio: f64,
        artifact: &PreAnalysisArtifact,
    ) -> Plan {
        // Merge onsets closer than MIN_SPACING, keeping the stronger.
        let mut onsets: Vec<(usize, f32)> = artifact
            .transient_onsets
            .iter()
            .enumerate()
            .filter(|(_, o)| **o < frames)
            .map(|(i, o)| (*o, artifact.strength_at(i)))
            .collect();
        onsets.sort_by_key(|&(o, _)| o);
        let mut merged: Vec<(usize, f32)> = Vec::with_capacity(onsets.len());
        for (o, s) in onsets {
            match merged.last_mut() {
                Some(last) if o.saturating_sub(last.0) < self.min_spacing => {
                    if s > last.1 {
                        *last = (o, s);
                    }
                }
                _ => merged.push((o, s)),
            }
        }

        let mut events: Vec<Event> = Vec::with_capacity(merged.len());
        let mut prev_end_out = 0usize;
        let mut prev_end_in = 0usize;
        for (o, _) in merged {
            let a = o.saturating_sub(self.pre);
            let b = (o + self.post).min(frames);
            if b <= a + 2 * self.xfade {
                continue;
            }
            let out_on = (o as f64 * ratio).round() as usize;
            let c0 = out_on.saturating_sub(o - a);
            let c1 = c0 + (b - a);
            if c1 > expected {
                break;
            }
            // Residual before this event must hold a full window on both
            // timelines, or the event is dropped.
            let res_in = a + self.xfade - prev_end_in.saturating_sub(self.xfade);
            let res_out = c0 + self.xfade - prev_end_out.saturating_sub(self.xfade);
            if a < prev_end_in || c0 < prev_end_out || res_in < self.fft || res_out < self.fft {
                continue;
            }
            events.push(Event { a, b, c0, c1 });
            prev_end_in = b;
            prev_end_out = c1;
        }

        // Residual segments between events (and the two ends).
        let mut segments = Vec::with_capacity(events.len() + 1);
        let mut in_s = 0usize;
        let mut out_s = 0usize;
        let mut fade_in = false;
        for ev in &events {
            segments.push(Segment {
                in_s,
                in_e: (ev.a + self.xfade).min(frames),
                out_s,
                out_e: (ev.c0 + self.xfade).min(expected),
                fade_in,
                fade_out: true,
            });
            in_s = ev.b.saturating_sub(self.xfade);
            out_s = ev.c1.saturating_sub(self.xfade);
            fade_in = true;
        }
        segments.push(Segment {
            in_s,
            in_e: frames,
            out_s,
            out_e: expected,
            fade_in,
            fade_out: false,
        });
        Plan { events, segments }
    }

    fn render_channel(
        &self,
        x: &[f32],
        sample_rate: u32,
        expected: usize,
        plan: &Plan,
        chan_seed: u64,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; expected];

        // Three-path: split the whole channel once; segments index into it.
        let (harm, perc) = if self.paths == 3 {
            let hp = hpss(x, self.fft, self.fft / HPSS_HOP_DIV, &HpssParams::default());
            (Some(hp.0), Some(hp.1))
        } else {
            (None, None)
        };

        for (si, seg) in plan.segments.iter().enumerate() {
            let in_len = seg.in_e.saturating_sub(seg.in_s);
            let out_len = seg.out_e.saturating_sub(seg.out_s);
            if in_len == 0 || out_len == 0 {
                continue;
            }
            let r = out_len as f64 / in_len as f64;
            let mut y = match (&harm, &perc) {
                (Some(h), Some(p)) => {
                    let mut yh =
                        self.stretch_tonal(&h[seg.in_s..seg.in_e], sample_rate, r, out_len);
                    let yn = self.stretch_noise(
                        &p[seg.in_s..seg.in_e],
                        r,
                        out_len,
                        NOISE_SEED ^ chan_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ si as u64,
                    );
                    for (a, b) in yh.iter_mut().zip(yn) {
                        *a += b;
                    }
                    yh
                }
                _ => self.stretch_tonal(&x[seg.in_s..seg.in_e], sample_rate, r, out_len),
            };
            // Level: the PV loses coherence energy; match the segment's own
            // input so the verbatim events do not read louder by contrast.
            let g = rms(&x[seg.in_s..seg.in_e]) / rms(&y).max(1e-9);
            let g = if g.is_finite() {
                g.clamp(0.5, 2.0)
            } else {
                1.0
            };
            for v in y.iter_mut() {
                *v *= g;
            }
            let xf = self.xfade.min(out_len / 2);
            if seg.fade_in {
                fade(&mut y[..xf], true);
            }
            if seg.fade_out {
                let n = y.len();
                fade(&mut y[n - xf..], false);
            }
            for (o, v) in out[seg.out_s..seg.out_e].iter_mut().zip(y) {
                *o += v;
            }
        }

        for ev in &plan.events {
            let mut chunk = x[ev.a..ev.b].to_vec();
            let xf = self.xfade.min(chunk.len() / 2);
            fade(&mut chunk[..xf], true);
            let n = chunk.len();
            fade(&mut chunk[n - xf..], false);
            for (o, v) in out[ev.c0..ev.c1].iter_mut().zip(chunk) {
                *o += v;
            }
        }
        out
    }

    /// Fresh identity-locked PV over one residual segment, fitted to
    /// `out_len`.
    fn stretch_tonal(&self, seg: &[f32], sample_rate: u32, r: f64, out_len: usize) -> Vec<f32> {
        if seg.len() < self.fft {
            return resample_linear(seg, out_len);
        }
        let mut pv = PhaseVocoder::with_options(
            self.fft,
            self.hop,
            r,
            sample_rate,
            100.0,
            WindowType::Hann,
            PhaseLockingMode::Identity,
        );
        let mut y = pv.process(seg).unwrap_or_default();
        y.resize(out_len, 0.0);
        y
    }

    /// Magnitude-only noise vocoder: analysis STFT at `hop`, synthesis
    /// frames placed at `hop * r` with a fresh seeded random phase per bin,
    /// Hann both sides, window-sum normalised.
    fn stretch_noise(&self, seg: &[f32], r: f64, out_len: usize, seed: u64) -> Vec<f32> {
        let fft = self.fft;
        let hop = self.hop;
        if seg.len() < fft {
            return resample_linear(seg, out_len);
        }
        let window = generate_window(WindowType::Hann, fft);
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(fft);
        let inv = planner.plan_fft_inverse(fft);
        let bins = fft / 2 + 1;

        // Analysis magnitudes (zero-padded half a window at both ends so
        // the segment edges are covered).
        let pad = fft / 2;
        let padded_len = seg.len() + 2 * pad;
        let n_frames = (padded_len - fft) / hop + 1;
        let mut mags: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
        let mut buf = vec![COMPLEX_ZERO; fft];
        for f in 0..n_frames {
            let pos = f * hop;
            for i in 0..fft {
                let p = pos + i;
                let v = if p >= pad && p - pad < seg.len() {
                    seg[p - pad]
                } else {
                    0.0
                };
                buf[i] = Complex::new(v * window[i], 0.0);
            }
            fwd.process(&mut buf);
            mags.push(buf[..bins].iter().map(|c| c.norm()).collect());
        }

        let mut rng = SplitMix64(seed);
        let mut out = vec![0.0f32; out_len + fft];
        let mut wsum = vec![0.0f32; out_len + fft];
        let norm = 1.0 / fft as f32;
        let syn_hop = hop as f64 * r;
        let mut pos = 0.0f64;
        let mut k = 0usize;
        loop {
            let out_pos = pos.round() as isize - pad as isize;
            if out_pos >= out_len as isize {
                break;
            }
            let a_frame = ((k as f64 * syn_hop / r) / hop as f64).round() as usize;
            let a_frame = a_frame.min(n_frames - 1);
            let m = &mags[a_frame];
            buf[0] = Complex::new(m[0], 0.0);
            for b in 1..bins - 1 {
                let ph = rng.next_f32() * std::f32::consts::TAU;
                let c = Complex::new(m[b] * ph.cos(), m[b] * ph.sin());
                buf[b] = c;
                buf[fft - b] = c.conj();
            }
            buf[bins - 1] = Complex::new(m[bins - 1], 0.0);
            inv.process(&mut buf);
            for i in 0..fft {
                let o = out_pos + i as isize;
                if o >= 0 && (o as usize) < out.len() {
                    out[o as usize] += buf[i].re * norm * window[i];
                    wsum[o as usize] += window[i] * window[i];
                }
            }
            pos += syn_hop;
            k += 1;
        }
        let floor = wsum.iter().cloned().fold(0.0f32, f32::max) * 0.1;
        for (o, w) in out.iter_mut().zip(&wsum) {
            *o /= w.max(floor).max(1e-6);
        }
        out.truncate(out_len);
        out
    }
}

struct Event {
    a: usize,
    b: usize,
    c0: usize,
    c1: usize,
}

struct Segment {
    in_s: usize,
    in_e: usize,
    out_s: usize,
    out_e: usize,
    fade_in: bool,
    fade_out: bool,
}

struct Plan {
    events: Vec<Event>,
    segments: Vec<Segment>,
}

/// Amplitude-complementary raised cosine over `buf` (in place).
fn fade(buf: &mut [f32], fade_in: bool) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    for (i, v) in buf.iter_mut().enumerate() {
        let progress = i as f64 / (n - 1) as f64;
        let g_in = (0.5 - 0.5 * (std::f64::consts::PI * progress).cos()) as f32;
        *v *= if fade_in { g_in } else { 1.0 - g_in };
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Linear resample of a short segment to `out_len` (segments shorter than
/// one analysis window; a pitch shift of at most the nominal deviation over
/// under 46 ms).
fn resample_linear(seg: &[f32], out_len: usize) -> Vec<f32> {
    if seg.is_empty() || out_len == 0 {
        return vec![0.0; out_len];
    }
    if seg.len() == 1 {
        return vec![seg[0]; out_len];
    }
    let step = (seg.len() - 1) as f64 / (out_len.max(2) - 1) as f64;
    (0..out_len)
        .map(|i| {
            let p = i as f64 * step;
            let j = (p.floor() as usize).min(seg.len() - 2);
            let t = (p - j as f64) as f32;
            seg[j] * (1.0 - t) + seg[j + 1] * t
        })
        .collect()
}

/// Deterministic PRNG for the noise path (no runtime randomness: the
/// determinism harness compares runs bit for bit).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze;

    fn click_pad(sr: u32, secs: f64, period_secs: f64) -> (Vec<f32>, Vec<usize>) {
        let n = (sr as f64 * secs) as usize;
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f64 / sr as f64;
            *v = 0.2 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32
                + 0.1 * (2.0 * std::f64::consts::PI * 331.0 * t).sin() as f32;
        }
        let mut clicks = Vec::new();
        let mut t = 0.25;
        while t < secs - 0.1 {
            let i = (t * sr as f64) as usize;
            for k in 0..8 {
                x[i + k] += 0.7 * (1.0 - k as f32 / 8.0);
            }
            clicks.push(i);
            t += period_secs;
        }
        (x, clicks)
    }

    fn count_peaks(y: &[f32], thresh: f32, min_gap: usize) -> Vec<usize> {
        let mut peaks = Vec::new();
        let mut i = 0;
        while i < y.len() {
            if y[i].abs() > thresh {
                peaks.push(i);
                i += min_gap;
            } else {
                i += 1;
            }
        }
        peaks
    }

    #[test]
    fn output_length_is_exact_and_events_land_on_the_mapped_timeline() {
        let sr = 44_100;
        let (mono, clicks) = click_pad(sr, 3.0, 0.5);
        // The event timeline is the artifact's; pin it to the known clicks
        // so the test exercises the mapping, not the detector.
        let mut art = analyze(&mono, sr);
        art.transient_onsets = clicks.clone();
        art.transient_strengths = vec![1.0; clicks.len()];
        art.onset_band_flux.clear();
        let artifact = Arc::new(art);
        let stereo: Vec<f32> = mono.iter().flat_map(|&v| [v, -0.5 * v]).collect();
        for paths in [2u8, 3] {
            for ratio in [0.926, 1.0417, 1.08] {
                let proto = ProtoHybrid::new(paths, sr);
                let out = proto.render(&stereo, 2, sr, ratio, &artifact);
                let expected = (mono.len() as f64 * ratio).round() as usize;
                assert_eq!(out.len(), expected * 2, "paths {paths} ratio {ratio}");
                let left: Vec<f32> = out.iter().step_by(2).copied().collect();
                let peaks = count_peaks(&left, 0.45, sr as usize / 10);
                assert_eq!(
                    peaks.len(),
                    clicks.len(),
                    "paths {paths} ratio {ratio}: clicks survive verbatim"
                );
                for (p, c) in peaks.iter().zip(&clicks) {
                    let want = (*c as f64 * ratio).round() as usize;
                    let err = p.abs_diff(want);
                    assert!(
                        err <= 4,
                        "paths {paths} ratio {ratio}: click at {c} mapped to {p}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn render_is_deterministic_across_runs() {
        let sr = 44_100;
        let (mono, _) = click_pad(sr, 2.0, 0.4);
        let artifact = Arc::new(analyze(&mono, sr));
        let proto = ProtoHybrid::new(3, sr);
        let a = proto.render(&mono, 1, sr, 1.06, &artifact);
        let b = proto.render(&mono, 1, sr, 1.06, &artifact);
        assert_eq!(a, b);
    }

    #[test]
    fn no_events_degrades_to_one_pv_segment_with_matched_level() {
        let sr = 44_100;
        let n = sr as usize * 2;
        let x: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin() as f32)
            .collect();
        let artifact = Arc::new(analyze(&x, sr));
        let proto = ProtoHybrid::new(2, sr);
        let y = proto.render(&x, 1, sr, 1.08, &artifact);
        assert_eq!(y.len(), (n as f64 * 1.08).round() as usize);
        let (rx, ry) = (rms(&x), rms(&y[sr as usize / 4..y.len() - sr as usize / 4]));
        assert!((ry / rx - 1.0).abs() < 0.1, "rms {ry} vs {rx}");
    }
}
