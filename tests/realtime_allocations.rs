use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use timestretch::{EdmPreset, StreamProcessor, StretchParams};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static REALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static REALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_TEST_MUTEX: Mutex<()> = Mutex::new(());

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            REALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn reset_alloc_counters() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn begin_alloc_tracking() {
    reset_alloc_counters();
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
}

fn end_alloc_tracking() -> (usize, usize, usize, usize) {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    (
        ALLOC_CALLS.load(Ordering::Relaxed),
        REALLOC_CALLS.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
        REALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn test_chunk_stereo(frames: usize, sample_rate: f32) -> Vec<f32> {
    let freq_l = 95.0f32;
    let freq_r = 142.0f32;
    let mut out = Vec::with_capacity(frames * 2);
    for n in 0..frames {
        let t = n as f32 / sample_rate;
        let l = (2.0 * std::f32::consts::PI * freq_l * t).sin();
        let r = (2.0 * std::f32::consts::PI * freq_r * t).sin();
        out.push(l);
        out.push(r);
    }
    out
}

#[test]
fn process_into_steady_state_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;
    const CHUNK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(CHANNELS)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let max_samples = chunk.len() * (WARMUP_ITERS + MEASURE_ITERS) * 8;
    let mut output = Vec::with_capacity(max_samples);

    for _ in 0..WARMUP_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process_into should succeed");
    }
    output.clear();

    begin_alloc_tracking();
    for _ in 0..MEASURE_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("steady-state process_into should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "steady-state process_into allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}

#[test]
fn process_into_pitch_scaled_steady_state_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;
    const CHUNK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(CHANNELS)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    // Engage the (default sinc) pitch resampler path before warmup.
    processor
        .set_pitch_scale(1.06)
        .expect("pitch scale should be accepted");

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let max_samples = chunk.len() * (WARMUP_ITERS + MEASURE_ITERS) * 8;
    let mut output = Vec::with_capacity(max_samples);

    for _ in 0..WARMUP_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process_into should succeed");
    }
    output.clear();

    begin_alloc_tracking();
    for _ in 0..MEASURE_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("steady-state pitch process_into should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "pitch-scaled process_into allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}

#[test]
fn process_into_pitch_sweep_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;
    const CHUNK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(CHANNELS)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_pitch_scale(1.02)
        .expect("pitch scale should be accepted");

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let max_samples = chunk.len() * (WARMUP_ITERS + MEASURE_ITERS) * 8;
    let mut output = Vec::with_capacity(max_samples);

    for _ in 0..WARMUP_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process_into should succeed");
    }
    output.clear();

    // Sweep the pitch control inside the tracked region: the DJ use case is
    // continuous pitch modulation, and set_pitch_scale must stay alloc-free.
    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        let scale = 1.02 + 0.08 * (i as f64 / MEASURE_ITERS as f64);
        processor
            .set_pitch_scale(scale)
            .expect("pitch scale should be accepted");
        processor
            .process_into(&chunk, &mut output)
            .expect("pitch-sweep process_into should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "pitch-sweep process_into allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}

#[test]
fn process_into_with_preanalysis_artifact_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;
    const CHUNK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    // A large artifact (thousands of onsets) so the cursor path is exercised
    // across the measured span, including several scheduled resets.
    let onset_spacing = 2048usize;
    let transient_onsets: Vec<usize> = (1..4000).map(|i| i * onset_spacing).collect();
    let transient_strengths = vec![0.9f32; transient_onsets.len()];
    let beat_positions: Vec<usize> = (1..2000).map(|i| i * 20_671).collect();
    let artifact = timestretch::PreAnalysisArtifact {
        version: timestretch::PREANALYSIS_VERSION,
        sample_rate: SAMPLE_RATE,
        bpm: 128.0,
        downbeat_offset_samples: 0,
        confidence: 0.95,
        beat_positions,
        transient_onsets,
        transient_strengths,
        onset_band_flux: Vec::new(),
        analysis_hop_size: 512,
        source_len_samples: 0,
        content_hash: 0,
    };

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(CHANNELS)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_pre_analysis(artifact)
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let max_samples = chunk.len() * (WARMUP_ITERS + MEASURE_ITERS) * 8;
    let mut output = Vec::with_capacity(max_samples);

    for _ in 0..WARMUP_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process_into should succeed");
    }
    output.clear();

    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        // Exercise the modulation-hold branch of artifact scheduling too.
        if i == MEASURE_ITERS / 2 {
            processor
                .set_stretch_ratio(1.08)
                .expect("ratio change should be accepted");
        }
        processor
            .process_into(&chunk, &mut output)
            .expect("steady-state process_into should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    let stats = processor.transient_reset_stats();
    assert!(
        stats.artifact_events_scheduled_total > 0,
        "artifact scheduling must have been active during the measurement"
    );
    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "artifact-driven steady-state process_into allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}

#[test]
fn process_into_live_profile_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHUNK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(2)
        .with_stream_profile(timestretch::StreamProfile::Live);
    let mut processor = StreamProcessor::new(params);

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let max_samples = chunk.len() * (WARMUP_ITERS + MEASURE_ITERS) * 8;
    let mut output = Vec::with_capacity(max_samples);

    for _ in 0..WARMUP_ITERS {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process_into should succeed");
    }
    output.clear();

    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        if i == MEASURE_ITERS / 2 {
            processor
                .set_stretch_ratio(1.08)
                .expect("ratio change should be accepted");
        }
        processor
            .process_into(&chunk, &mut output)
            .expect("steady-state process_into should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "Live-profile steady-state process_into allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}

#[test]
fn warm_start_seek_no_heap_growth_after_warmup() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u32 = 2;
    const CHUNK_FRAMES: usize = 256;

    let params = StretchParams::new(1.05)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(CHANNELS)
        .with_stream_profile(timestretch::StreamProfile::Live);
    let mut processor = StreamProcessor::new(params);

    let chunk = test_chunk_stereo(CHUNK_FRAMES, SAMPLE_RATE as f32);
    let mut output = Vec::with_capacity(chunk.len() * 512);

    // A cue point ~1 s in, with a full preroll of preceding audio.
    let preroll_frames = processor.warm_start_preroll_frames();
    let cue = SAMPLE_RATE as usize;
    let preroll = test_chunk_stereo(cue, SAMPLE_RATE as f32);
    let preroll_slice = &preroll[(cue - preroll_frames) * CHANNELS as usize..];

    // Warm up the processor and every buffer the seek touches (one seek plus
    // steady processing) so capacities are grown before measuring.
    for _ in 0..64 {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process");
    }
    processor
        .warm_start_seek(cue, preroll_slice)
        .expect("warmup seek");
    for _ in 0..64 {
        processor
            .process_into(&chunk, &mut output)
            .expect("warmup process");
    }
    output.clear();

    // The seek itself must not allocate in the deterministic engine.
    begin_alloc_tracking();
    for _ in 0..32 {
        processor
            .warm_start_seek(cue, preroll_slice)
            .expect("steady-state warm_start_seek should succeed");
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "warm_start_seek allocated: alloc_calls={}, realloc_calls={}, alloc_bytes={}, realloc_bytes={}",
        alloc_calls,
        realloc_calls,
        alloc_bytes,
        realloc_bytes
    );
}
