//! Real-time pull-engine example.
//!
//! Demonstrates the pull engine's controller/processor/source split for
//! real-time DJ use, simulating a pitch fader that changes the tempo on
//! the fly while an audio callback pulls fixed-size buffers.
//!
//! Run with: cargo run --example realtime_stream

use std::f32::consts::PI;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};

fn main() {
    let sample_rate = 44_100u32;
    let callback_frames = 256usize;

    // A 4-second 220 Hz tone stands in for the deck's audio file.
    let source_audio: Vec<f32> = (0..sample_rate as usize * 4)
        .map(|i| 0.5 * (2.0 * PI * 220.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    // Build the engine: keylock profile (pitch stays put while tempo
    // moves), stereo-capable but mono here.
    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels: 1,
        profile: EngineProfile::Keylock,
        ..EngineConfig::default()
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);

    // In a real app these three run on separate threads:
    // - the feed thread pushes source audio and watches `demand_hint`,
    // - the UI thread writes tempo through the lock-free controller,
    // - the audio callback calls `process` (infallible, allocation-free).
    let mut feed_cursor = 0usize;
    let mut out = vec![0.0f32; callback_frames];
    let mut rendered = 0usize;

    let mut callback_index = 0usize;
    while feed_cursor < source_audio.len() {
        // Simulate a DJ riding the pitch fader from 0 to +6%.
        let t = callback_index as f64 * callback_frames as f64 / sample_rate as f64;
        controller.set_tempo_rate(1.0 + 0.06 * (t / 4.0).min(1.0));

        // Feed thread: keep the ring comfortably ahead of demand.
        while feed_cursor < source_audio.len()
            && source.occupied_frames() < source.demand_hint(callback_frames, 4.0)
        {
            let end = (feed_cursor + 4_096).min(source_audio.len());
            feed_cursor += source.push(&source_audio[feed_cursor..end]);
        }

        // Audio callback: fills exactly the requested frames.
        processor.process(&mut out);
        rendered += out.len();
        callback_index += 1;
    }

    println!(
        "rendered {} frames across {} callbacks (source position {:.0}, underruns {})",
        rendered,
        callback_index,
        controller.source_position(),
        controller.underrun_frames()
    );
}
