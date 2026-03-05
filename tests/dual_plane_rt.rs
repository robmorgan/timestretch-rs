use std::sync::Arc;

use timestretch::{
    DualPlaneProcessor, RtConfig, RtProcessor, StretchParams, TimeWarpMap, WarpAnchor,
};

fn mono_sine_block(frames: usize, sample_rate: u32, hz: f32, phase: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * hz * t + phase).sin() * 0.25
        })
        .collect()
}

#[test]
fn rt_processor_handles_fixed_blocks() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(48_000)
        .with_channels(1)
        .with_fft_size(1024)
        .with_hop_size(256);
    let cfg = RtConfig::new(params, 256);
    let mut rt = RtProcessor::prepare(cfg).unwrap();

    let mut out_planar = vec![vec![0.0f32; 2048]];
    let mut out = Vec::with_capacity(256 * 40);
    for i in 0..20 {
        let block = mono_sine_block(256, 48_000, 220.0, i as f32 * 0.2);
        let input_refs = [&block[..]];
        let mut output_refs = [out_planar[0].as_mut_slice()];
        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);
        assert_eq!(consumed, 256);
        out.extend_from_slice(&output_refs[0][..produced]);
    }
    rt.flush(&mut out).unwrap();
    assert!(!out.is_empty());
}

#[test]
fn dual_plane_accepts_warp_and_analysis_updates() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44_100)
        .with_channels(1)
        .with_fft_size(1024)
        .with_hop_size(256);
    let cfg = RtConfig::new(params, 256);
    let mut proc = DualPlaneProcessor::prepare(cfg).unwrap();

    let warp = Arc::new(
        TimeWarpMap::from_anchors(vec![
            WarpAnchor::new(0.0, 0.0).unwrap(),
            WarpAnchor::new(44_100.0, 48_510.0).unwrap(),
            WarpAnchor::new(88_200.0, 92_610.0).unwrap(),
        ])
        .unwrap(),
    );
    assert!(proc.publish_warp_map(warp));

    let mut out = Vec::with_capacity(256 * 32);
    let mut out_planar = vec![vec![0.0f32; 4096]];
    for i in 0..16 {
        let block = mono_sine_block(256, 44_100, 110.0, i as f32 * 0.1);
        let input_refs = [&block[..]];
        let mut output_refs = [out_planar[0].as_mut_slice()];
        let (_consumed, produced) = proc.process(&input_refs, &mut output_refs);
        out.extend_from_slice(&output_refs[0][..produced]);
    }
    proc.flush(&mut out).unwrap();
    assert!(!out.is_empty());
}
