//! Measured streaming latency: first-sample-out and control-to-audio.
//!
//! These tests measure actual observed latency and hold it against
//! `StreamProcessor::latency_samples()` / `latency_report()` — the reported
//! numbers must describe reality, not just the buffering formula.

use timestretch::{ControlPath, EdmPreset, StreamProcessor, StretchParams};

const SR: u32 = 44_100;
const CHUNK: usize = 256;

/// Deterministic full-scale pseudo-noise (LCG) so first-output detection is
/// unambiguous.
fn noise(len: usize) -> Vec<f32> {
    let mut state = 0x2545_f491u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn sine(freq: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR as f32).sin())
        .collect()
}

/// Streams `input` in CHUNK-frame chunks; returns the number of input frames
/// pushed when the first output sample above the energy floor appeared.
fn measure_first_sample_out(processor: &mut StreamProcessor, input: &[f32]) -> Option<usize> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    let mut pushed = 0usize;
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
        pushed += chunk.len();
        if output.iter().any(|s| s.abs() > 1e-4) {
            return Some(pushed);
        }
    }
    None
}

/// Sliding zero-crossing frequency estimate over `window` samples starting at
/// `start`.
fn freq_at(signal: &[f32], start: usize, window: usize) -> Option<f64> {
    if start + window > signal.len() {
        return None;
    }
    let slice = &signal[start..start + window];
    let mut crossings = 0usize;
    for i in 1..slice.len() {
        if slice[i - 1] <= 0.0 && slice[i] > 0.0 {
            crossings += 1;
        }
    }
    Some(crossings as f64 * SR as f64 / window as f64)
}

fn config(fft: usize, preset: Option<EdmPreset>, ratio: f64) -> StretchParams {
    let mut params = StretchParams::new(ratio)
        .with_sample_rate(SR)
        .with_channels(1);
    if let Some(preset) = preset {
        params = params.with_preset(preset);
    } else {
        params = params.with_fft_size(fft);
    }
    params
}

#[test]
fn first_sample_out_matches_reported_in_band() {
    // (fft, preset) per streaming configuration; ratio 1.05 is off-unity
    // (defeats the bit-exact passthrough) but inside the [0.9, 1.1] band.
    let cases: [(usize, Option<EdmPreset>); 3] = [
        (1024, None),
        (2048, None),
        (4096, Some(EdmPreset::DjBeatmatch)),
    ];
    let input = noise(SR as usize * 2);

    for (fft, preset) in cases {
        let processor = StreamProcessor::new(config(fft, preset, 1.05));
        assert_first_sample_out_honest(processor, &input, &format!("fft={}", fft));
    }
}

#[test]
fn first_sample_out_matches_reported_per_profile() {
    let input = noise(SR as usize * 2);
    for &profile in timestretch::StreamProfile::ALL {
        let processor = StreamProcessor::new(
            StretchParams::new(1.05)
                .with_sample_rate(SR)
                .with_channels(1)
                .with_stream_profile(profile),
        );
        assert_first_sample_out_honest(processor, &input, profile.label());
    }
}

#[test]
fn first_sample_out_matches_reported_multi_res() {
    use timestretch::StreamingEngine;

    // Multi-resolution needs a longer runway: the sub-bass gate is 1.5x the
    // derived sub-bass FFT (Club 8192 -> 12288 frames, Quality 16384 ->
    // 24576 frames).
    let input = noise(SR as usize * 2);
    for profile in [
        timestretch::StreamProfile::Club,
        timestretch::StreamProfile::Quality,
    ] {
        let mut processor = StreamProcessor::new(
            StretchParams::new(1.05)
                .with_sample_rate(SR)
                .with_channels(1)
                .with_stream_profile(profile),
        );
        processor
            .set_streaming_engine(StreamingEngine::MultiResolution)
            .expect("Club/Quality profiles support multi-resolution");
        let mid_fft = processor.params().fft_size;
        let expected_gate = (mid_fft * 4).min(16384) * 3 / 2;
        assert_eq!(
            processor.latency_samples(),
            expected_gate,
            "{}: multi-res gate must be keyed to the sub-bass FFT",
            profile.label()
        );
        assert_first_sample_out_honest(
            processor,
            &input,
            &format!("multi-res {}", profile.label()),
        );
    }
}

#[test]
fn multi_res_rejected_at_live_profile_keeps_deterministic_gate() {
    use timestretch::StreamingEngine;

    let mut processor = StreamProcessor::new(
        StretchParams::new(1.05)
            .with_sample_rate(SR)
            .with_channels(1)
            .with_stream_profile(timestretch::StreamProfile::Live),
    );
    let gate_before = processor.latency_samples();
    assert!(
        processor
            .set_streaming_engine(StreamingEngine::MultiResolution)
            .is_err(),
        "Live profile must reject the multi-resolution engine"
    );
    assert_eq!(
        processor.latency_samples(),
        gate_before,
        "failed engine selection must not change the reported gate"
    );
}

fn assert_first_sample_out_honest(mut processor: StreamProcessor, input: &[f32], label: &str) {
    let reported = processor.latency_samples();
    let hop = processor.params().hop_size;
    let measured = measure_first_sample_out(&mut processor, input)
        .unwrap_or_else(|| panic!("{}: no output produced", label));
    eprintln!(
        "{} hop={} reported={} measured={} delta={}",
        label,
        hop,
        reported,
        measured,
        measured as isize - reported as isize
    );
    let tolerance = CHUNK + hop;
    assert!(
        measured.abs_diff(reported) <= tolerance,
        "{}: measured first-sample-out {} vs reported {} exceeds tolerance {}",
        label,
        measured,
        reported,
        tolerance
    );
}

#[test]
fn first_sample_out_off_band_reports_wider_gate() {
    let mut processor = StreamProcessor::new(config(1024, None, 1.3));
    let reported = processor.latency_samples();
    let report = processor.latency_report();
    assert_eq!(
        report.effective_gate_frames,
        1024 * 2,
        "off-band target ratio must widen the gate to fft*2"
    );
    assert!(report.effective_gate_frames > report.base_gate_frames);

    let input = noise(SR as usize * 2);
    let measured = measure_first_sample_out(&mut processor, &input).expect("no output produced");
    let hop = processor.params().hop_size;
    eprintln!(
        "off-band: reported={} measured={} delta={}",
        reported,
        measured,
        measured as isize - reported as isize
    );
    assert!(
        measured.abs_diff(reported) <= CHUNK + hop,
        "off-band measured {} vs reported {}",
        measured,
        reported
    );
}

#[test]
fn control_to_audio_pitch_change() {
    let mut params = config(1024, None, 1.02);
    params = params.with_hop_size(256);
    let mut processor = StreamProcessor::new(params);
    let reported = processor.latency_samples();

    let input = sine(440.0, SR as usize * 6);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    let mut chunks = input.chunks(CHUNK);

    // Reach steady state (~1 s), then change pitch.
    for _ in 0..(SR as usize / CHUNK) {
        processor
            .process_into(chunks.next().unwrap(), &mut output)
            .expect("process");
    }
    let out_mark = output.len();
    processor.set_pitch_scale(1.06).expect("pitch change");

    for chunk in chunks {
        processor.process_into(chunk, &mut output).expect("process");
    }

    // Find where the output frequency has moved >30% toward the target.
    let window = 2048usize;
    let start_freq: f64 = 440.0; // pitch was unity before the change
    let target_freq: f64 = 440.0 * 1.06;
    let threshold = start_freq + 0.3 * (target_freq - start_freq);
    let mut moved_at: Option<usize> = None;
    let mut pos = out_mark;
    while let Some(freq) = freq_at(&output, pos, window) {
        if freq > threshold {
            moved_at = Some(pos - out_mark);
            break;
        }
        pos += window / 4;
    }
    let moved_at = moved_at.expect("output frequency never moved toward the pitch target");

    // Budget: buffering gate + 3 glide time constants + measurement window.
    let glide = (3.0 * 0.050 * SR as f64) as usize;
    let budget = reported + glide + window + CHUNK;
    eprintln!(
        "pitch control-to-audio: moved_at={} budget={} (reported={} glide={})",
        moved_at, budget, reported, glide
    );
    assert!(
        moved_at <= budget,
        "pitch change became audible after {} samples; budget {}",
        moved_at,
        budget
    );

    // And it settles at the target.
    let tail_start = output.len().saturating_sub(window * 2);
    let settled = freq_at(&output, tail_start, window).expect("tail window");
    assert!(
        (settled - target_freq).abs() / target_freq < 0.02,
        "output frequency settled at {:.1} Hz, expected ~{:.1} Hz",
        settled,
        target_freq
    );
}

#[test]
fn control_to_audio_ratio_change() {
    let mut params = config(1024, None, 1.02);
    params = params.with_hop_size(256);
    let mut processor = StreamProcessor::new(params);
    let reported = processor.latency_samples();

    let input = noise(SR as usize * 6);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    let mut consumed = 0usize;
    let mut out_lens: Vec<(usize, usize)> = Vec::new(); // (input_pushed, output_len)

    let mut changed_at_input = None;
    for chunk in input.chunks(CHUNK) {
        if consumed >= SR as usize && changed_at_input.is_none() {
            processor.set_stretch_ratio(1.08).expect("ratio change");
            changed_at_input = Some(consumed);
        }
        processor.process_into(chunk, &mut output).expect("process");
        consumed += chunk.len();
        out_lens.push((consumed, output.len()));
    }
    let changed_at_input = changed_at_input.expect("ratio change never applied");

    // Local output/input slope over a trailing window; find where it passes
    // the midpoint between the old (1.02) and new (1.08) rates.
    let window_points = 32; // 32 chunks = 8192 input samples
    let mut transition_input = None;
    for i in window_points..out_lens.len() {
        let (in_a, out_a) = out_lens[i - window_points];
        let (in_b, out_b) = out_lens[i];
        if in_a < changed_at_input {
            continue;
        }
        let slope = (out_b - out_a) as f64 / (in_b - in_a) as f64;
        if slope > 1.05 {
            transition_input = Some(in_b - changed_at_input);
            break;
        }
    }
    let transition_input = transition_input.expect("output rate never reached the new ratio");

    let glide = (3.0 * 0.050 * SR as f64) as usize;
    let budget = reported + glide + window_points * CHUNK + 2 * CHUNK;
    eprintln!(
        "ratio control-to-audio: transition_at={} budget={} (reported={} glide={})",
        transition_input, budget, reported, glide
    );
    assert!(
        transition_input <= budget,
        "ratio change reached output after {} input samples; budget {}",
        transition_input,
        budget
    );
}

#[test]
fn cliff_regression_gate_tracks_target_not_glide() {
    // Start out-of-band (gate = fft*2)...
    let mut processor = StreamProcessor::new(config(1024, None, 1.3));
    assert_eq!(processor.latency_report().effective_gate_frames, 2048);

    let input = noise(SR as usize * 2);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    for chunk in input.chunks(CHUNK).take(32) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    assert!(!output.is_empty(), "warmup should produce output");

    // ...then retarget in-band: the report must drop at the control call,
    // while current_ratio is still gliding down through the 1.1 boundary.
    processor.set_stretch_ratio(1.05).expect("ratio change");
    assert!(
        processor.current_stretch_ratio() > 1.1,
        "test premise: glide still above the band boundary"
    );
    assert_eq!(
        processor.latency_report().effective_gate_frames,
        1536,
        "gate must follow the target ratio immediately, not the glide"
    );

    // Output continues without stalling across the boundary crossing.
    let before = output.len();
    for chunk in input.chunks(CHUNK).skip(32).take(64) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    assert!(
        output.len() > before,
        "stream stalled across the gate-band transition"
    );

    // Converse: an out-of-band pitch target widens the gate immediately.
    let mut processor = StreamProcessor::new(config(1024, None, 1.05));
    assert_eq!(processor.latency_report().effective_gate_frames, 1536);
    processor.set_pitch_scale(1.15).expect("pitch change");
    assert_eq!(
        processor.latency_report().effective_gate_frames,
        2048,
        "target ratio*pitch of 1.2075 must widen the gate"
    );
}

#[test]
fn gate_after_reset_and_seek_matches_fresh() {
    let fresh = StreamProcessor::new(config(1024, None, 1.05));
    let fresh_report = fresh.latency_report();

    let mut processor = StreamProcessor::new(config(1024, None, 1.05));
    let input = noise(SR as usize);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    processor.reset();
    processor.set_source_position(12_345).expect("seek");
    assert_eq!(processor.latency_report(), fresh_report);
    assert_eq!(processor.latency_samples(), fresh.latency_samples());
}

#[test]
fn varispeed_tempo_step_control_to_audio() {
    // Stage 15 exit criterion: in varispeed-first mode a tempo step reaches
    // the output within one callback plus the resampler lookahead — no
    // buffering-gate term, no 50 ms glide term.
    let mut params = config(1024, None, 1.02);
    params = params.with_hop_size(256);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_control_path(ControlPath::VarispeedFirst)
        .expect("varispeed path");

    let report = processor.latency_report();
    assert!(
        report.control_to_audio_frames <= 80,
        "varispeed control-to-audio must be resampler lookahead only, got {}",
        report.control_to_audio_frames
    );

    let input = noise(SR as usize * 4);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    let mut consumed = 0usize;
    let mut out_lens: Vec<(usize, usize)> = Vec::new();

    let mut changed_at_input = None;
    for chunk in input.chunks(CHUNK) {
        if consumed >= SR as usize && changed_at_input.is_none() {
            processor.set_stretch_ratio(1.08).expect("ratio change");
            changed_at_input = Some(consumed);
        }
        processor.process_into(chunk, &mut output).expect("process");
        consumed += chunk.len();
        out_lens.push((consumed, output.len()));
    }
    let changed_at_input = changed_at_input.expect("ratio change never applied");

    // A short slope window keeps the budget meaningful: 8 chunks of input.
    let window_points = 8;
    let mut transition_input = None;
    for i in window_points..out_lens.len() {
        let (in_a, out_a) = out_lens[i - window_points];
        let (in_b, out_b) = out_lens[i];
        if in_a < changed_at_input {
            continue;
        }
        let slope = (out_b - out_a) as f64 / (in_b - in_a) as f64;
        if slope > 1.05 {
            transition_input = Some(in_b - changed_at_input);
            break;
        }
    }
    let transition_input = transition_input.expect("output rate never reached the new ratio");

    // Budget: the slope window itself + one callback of phase slack + the
    // varispeed kernel lookahead. Deliberately NO gate and NO glide terms.
    let budget = window_points * CHUNK + 2 * CHUNK + report.varispeed_lookahead_samples.max(80);
    eprintln!(
        "varispeed tempo control-to-audio: transition_at={} budget={}",
        transition_input, budget
    );
    assert!(
        transition_input <= budget,
        "varispeed tempo step reached output after {} input samples; budget {}",
        transition_input,
        budget
    );
}

#[test]
fn varispeed_pipeline_delay_constant_under_ride() {
    // The gate is a constant content delay under tempo rides (that is the
    // property the host compensates once); only the honest sinc kernel
    // half-span may wiggle by a few samples as the step moves.
    let mut params = config(1024, None, 1.04);
    params = params.with_hop_size(256);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_control_path(ControlPath::VarispeedFirst)
        .expect("varispeed path");

    let baseline = processor.latency_report();
    let input = noise(SR as usize * 3);
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4);
    let mut min_delay = usize::MAX;
    let mut max_delay = 0usize;
    for (i, chunk) in input.chunks(CHUNK).enumerate() {
        let t = i as f64 * CHUNK as f64 / SR as f64;
        let ratio = 1.04 + 0.04 * (2.0 * std::f64::consts::PI * t / 2.0).sin();
        processor.set_stretch_ratio(ratio).expect("ride ratio");
        processor.process_into(chunk, &mut output).expect("process");

        let report = processor.latency_report();
        assert_eq!(
            report.effective_gate_frames, baseline.effective_gate_frames,
            "gate must stay constant through an in-band tempo ride"
        );
        assert!(
            report.control_to_audio_frames <= 80,
            "tempo control latency must stay at resampler lookahead, got {}",
            report.control_to_audio_frames
        );
        min_delay = min_delay.min(report.pipeline_delay_frames);
        max_delay = max_delay.max(report.pipeline_delay_frames);
    }
    eprintln!(
        "varispeed pipeline delay under ride: {}..{} frames",
        min_delay, max_delay
    );
    assert!(
        max_delay - min_delay <= 8,
        "pipeline delay must be constant up to kernel-span wiggle: {}..{}",
        min_delay,
        max_delay
    );
}

#[test]
fn varispeed_first_sample_out_matches_reported() {
    // The gate is in resampled frames: at ratio r the input side fills it
    // after ~gate/r source frames (plus the resampler lookaheads).
    for ratio in [0.94, 1.05] {
        let mut params = config(1024, None, ratio);
        params = params.with_hop_size(256);
        let mut processor = StreamProcessor::new(params);
        processor
            .set_control_path(ControlPath::VarispeedFirst)
            .expect("varispeed path");
        let report = processor.latency_report();
        let hop = processor.params().hop_size;

        let input = noise(SR as usize * 2);
        let measured = measure_first_sample_out(&mut processor, &input).expect("no output");
        let expected = (report.effective_gate_frames as f64 / ratio).ceil() as usize
            + report.varispeed_lookahead_samples
            + report.pitch_lookahead_samples;
        eprintln!(
            "varispeed ratio={} measured={} expected={} delta={}",
            ratio,
            measured,
            expected,
            measured as isize - expected as isize
        );
        assert!(
            measured.abs_diff(expected) <= CHUNK + hop,
            "ratio {}: measured first-sample-out {} vs expected {} exceeds tolerance {}",
            ratio,
            measured,
            expected,
            CHUNK + hop
        );
    }
}

#[test]
fn latency_report_includes_pitch_lookahead() {
    let mut processor = StreamProcessor::new(config(1024, None, 1.05));
    assert_eq!(processor.latency_report().pitch_lookahead_samples, 0);

    processor.set_pitch_scale(1.04).expect("pitch change");
    let report = processor.latency_report();
    assert!(
        report.pitch_lookahead_samples >= 16,
        "engaged sinc pitch must report its kernel lookahead, got {}",
        report.pitch_lookahead_samples
    );
    assert_eq!(
        report.total_frames,
        report.effective_gate_frames + report.pitch_lookahead_samples
    );
    assert!(report.total_secs() > 0.0);
}
