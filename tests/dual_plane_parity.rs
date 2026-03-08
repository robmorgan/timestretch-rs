use timestretch::analysis::comparison;
use timestretch::{RtConfig, RtProcessor, StreamProcessor, StretchParams};

fn synth_stereo_signal(frames: usize, sample_rate: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        let env = (1.0 - (i as f32 / frames.max(1) as f32) * 0.15).max(0.2);
        let beat_env = if i % (sample_rate as usize / 2).max(1) < 192 {
            let x = (i % (sample_rate as usize / 2).max(1)) as f32 / 192.0;
            (1.0 - x).max(0.0)
        } else {
            0.0
        };
        let left = env
            * (0.35 * (2.0 * std::f32::consts::PI * 110.0 * t).sin()
                + 0.24 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.12 * beat_env);
        let right = env
            * (0.33 * (2.0 * std::f32::consts::PI * 165.0 * t).sin()
                + 0.20 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
                + 0.10 * beat_env);
        out.push(left);
        out.push(right);
    }
    out
}

fn run_stream_deterministic(
    input_interleaved: &[f32],
    params: StretchParams,
    chunk_frames: usize,
) -> Vec<f32> {
    let channels = params.channels.count().max(1);
    let mut processor = StreamProcessor::new(params);
    let mut out = Vec::with_capacity(input_interleaved.len() * 2);
    for chunk in input_interleaved.chunks(chunk_frames * channels) {
        processor.process_into(chunk, &mut out).unwrap();
    }
    processor.flush_into(&mut out).unwrap();
    out
}

fn run_stream_deterministic_modulated(
    input_interleaved: &[f32],
    params: StretchParams,
    chunk_frames: usize,
    dual_plane: bool,
) -> Vec<f32> {
    let channels = params.channels.count().max(1);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_dual_plane_deterministic(dual_plane)
        .expect("fresh-stream dual-plane toggle should succeed");

    let mut out = Vec::with_capacity(input_interleaved.len() * 2);
    for (chunk_idx, chunk) in input_interleaved
        .chunks(chunk_frames * channels)
        .enumerate()
    {
        let phase = chunk_idx as f64 * 0.15;
        let ratio = (1.0 + 0.22 * phase.sin()).clamp(0.75, 1.40);
        let pitch = (1.0 + 0.09 * phase.cos()).clamp(0.85, 1.20);
        processor
            .set_stretch_ratio(ratio)
            .expect("ratio sweep value should be valid");
        processor
            .set_pitch_scale(pitch)
            .expect("pitch sweep value should be valid");
        processor.process_into(chunk, &mut out).unwrap();
    }
    processor.flush_into(&mut out).unwrap();
    out
}

fn run_rt_plane(input_interleaved: &[f32], params: StretchParams, block_frames: usize) -> Vec<f32> {
    let channels = params.channels.count().max(1);
    let cfg = RtConfig::new(params, block_frames);
    let mut rt = RtProcessor::prepare(cfg).unwrap();

    let mut input_planar: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(block_frames))
        .collect();
    let mut output_planar: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0; 8192]).collect();

    let mut out = Vec::with_capacity(input_interleaved.len() * 2);
    for chunk in input_interleaved.chunks(block_frames * channels) {
        let frames = chunk.len() / channels;
        for ch in 0..channels {
            input_planar[ch].clear();
        }
        for frame in 0..frames {
            let base = frame * channels;
            for ch in 0..channels {
                input_planar[ch].push(chunk[base + ch]);
            }
        }

        let input_refs: Vec<&[f32]> = input_planar.iter().map(|ch| ch.as_slice()).collect();
        let mut output_refs: Vec<&mut [f32]> = output_planar
            .iter_mut()
            .map(|ch| ch.as_mut_slice())
            .collect();
        let (_consumed, produced) = rt.process(&input_refs, &mut output_refs);
        for frame in 0..produced {
            for ch in 0..channels {
                out.push(output_refs[ch][frame]);
            }
        }
    }

    rt.flush(&mut out).unwrap();
    out
}

fn align_by_offset<'a>(
    reference: &'a [f32],
    candidate: &'a [f32],
    offset: isize,
) -> (&'a [f32], &'a [f32]) {
    let mut ref_start = 0usize;
    let mut cand_start = 0usize;
    if offset > 0 {
        cand_start = offset as usize;
    } else if offset < 0 {
        ref_start = (-offset) as usize;
    }
    if ref_start >= reference.len() || cand_start >= candidate.len() {
        return (&[], &[]);
    }
    let aligned_len = (reference.len() - ref_start).min(candidate.len() - cand_start);
    (
        &reference[ref_start..ref_start + aligned_len],
        &candidate[cand_start..cand_start + aligned_len],
    )
}

fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let energy = signal
        .iter()
        .map(|&s| {
            let x = s as f64;
            x * x
        })
        .sum::<f64>()
        / signal.len() as f64;
    energy.sqrt()
}

#[test]
fn deterministic_stream_and_rt_plane_match_unity_passthrough() {
    let sample_rate = 48_000u32;
    let input = synth_stereo_signal(256 * 48, sample_rate);
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256);

    let stream = run_stream_deterministic(&input, params.clone(), 256);
    let rt = run_rt_plane(&input, params, 256);

    assert_eq!(stream.len(), input.len());
    assert_eq!(rt.len(), input.len());
    assert_eq!(stream, input);
    assert_eq!(rt, input);
    assert_eq!(stream, rt);
}

#[test]
fn deterministic_stream_and_rt_plane_parity_under_ratio_change() {
    let sample_rate = 44_100u32;
    let input = synth_stereo_signal(sample_rate as usize * 8, sample_rate);
    let params = StretchParams::new(1.03)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256);

    let stream = run_stream_deterministic(&input, params.clone(), 256);
    let rt = run_rt_plane(&input, params, 256);

    assert!(!stream.is_empty());
    assert!(!rt.is_empty());

    let len_diff_pct = stream.len().abs_diff(rt.len()) as f64 / stream.len().max(1) as f64 * 100.0;
    assert!(
        len_diff_pct <= 1.0,
        "deterministic-vs-rt parity length diff too high: {:.4}% (stream={}, rt={})",
        len_diff_pct,
        stream.len(),
        rt.len()
    );

    let compare_len = stream.len().min(rt.len()).min(44_100 * 2 * 6);
    let xcorr = comparison::cross_correlation(&stream[..compare_len], &rt[..compare_len]);
    let (stream_aligned, rt_aligned) = align_by_offset(
        &stream[..compare_len],
        &rt[..compare_len],
        xcorr.peak_offset,
    );
    assert!(
        !stream_aligned.is_empty() && !rt_aligned.is_empty(),
        "deterministic-vs-rt alignment window is empty"
    );

    let stream_rms = rms(stream_aligned).max(1e-9);
    let rt_rms = rms(rt_aligned).max(1e-9);
    let rms_ratio = (stream_rms / rt_rms).max(rt_rms / stream_rms);

    assert!(
        xcorr.peak_value >= 0.45,
        "deterministic-vs-rt parity correlation too low: {:.4}",
        xcorr.peak_value
    );
    assert!(
        rms_ratio <= 1.35,
        "deterministic-vs-rt parity RMS drift too high: ratio={:.4} stream_rms={:.6} rt_rms={:.6}",
        rms_ratio,
        stream_rms,
        rt_rms
    );
}

#[test]
fn deterministic_dual_plane_matches_legacy_under_tempo_pitch_modulation_sweeps() {
    let sample_rate = 44_100u32;
    let input = synth_stereo_signal(sample_rate as usize * 10, sample_rate);
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256);

    let legacy = run_stream_deterministic_modulated(&input, params.clone(), 256, false);
    let dual_plane = run_stream_deterministic_modulated(&input, params, 256, true);

    assert!(!legacy.is_empty());
    assert!(!dual_plane.is_empty());

    let len_diff_pct =
        legacy.len().abs_diff(dual_plane.len()) as f64 / legacy.len().max(1) as f64 * 100.0;
    assert!(
        len_diff_pct <= 1.0,
        "legacy-vs-dual modulation sweep length diff too high: {:.4}% (legacy={}, dual={})",
        len_diff_pct,
        legacy.len(),
        dual_plane.len()
    );

    let compare_len = legacy.len().min(dual_plane.len()).min(44_100 * 2 * 8);
    let xcorr = comparison::cross_correlation(&legacy[..compare_len], &dual_plane[..compare_len]);
    let (legacy_aligned, dual_aligned) = align_by_offset(
        &legacy[..compare_len],
        &dual_plane[..compare_len],
        xcorr.peak_offset,
    );
    assert!(
        !legacy_aligned.is_empty() && !dual_aligned.is_empty(),
        "legacy-vs-dual modulation sweep alignment window is empty"
    );

    let legacy_rms = rms(legacy_aligned).max(1e-9);
    let dual_rms = rms(dual_aligned).max(1e-9);
    let rms_ratio = (legacy_rms / dual_rms).max(dual_rms / legacy_rms);

    assert!(
        xcorr.peak_value >= 0.30,
        "legacy-vs-dual modulation sweep correlation too low: {:.4}",
        xcorr.peak_value
    );
    assert!(
        rms_ratio <= 1.45,
        "legacy-vs-dual modulation sweep RMS drift too high: ratio={:.4} legacy_rms={:.6} dual_rms={:.6}",
        rms_ratio,
        legacy_rms,
        dual_rms
    );
}
