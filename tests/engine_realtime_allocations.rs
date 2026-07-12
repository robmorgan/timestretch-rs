//! New-engine port of the counting-allocator gate (ROADMAP new Stage 1):
//! zero heap activity in steady state on the audio-thread `process` path,
//! including under per-callback tempo retargets. The host-side source feed
//! and control writes are also covered — the whole runtime loop must be
//! allocation-free after construction.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

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
            if std::env::var_os("ALLOC_TRACE").is_some() {
                TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
                eprintln!(
                    "realloc {} -> {} bytes\n{}",
                    layout.size(),
                    new_size,
                    std::backtrace::Backtrace::force_capture()
                );
                TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
            }
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn begin_alloc_tracking() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_BYTES.store(0, Ordering::Relaxed);
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

fn test_chunk_stereo(frames: usize, sample_rate: f32, phase_frames: usize) -> Vec<f32> {
    let freq_l = 95.0f32;
    let freq_r = 142.0f32;
    let mut out = Vec::with_capacity(frames * 2);
    for n in 0..frames {
        let t = (phase_frames + n) as f32 / sample_rate;
        out.push((2.0 * std::f32::consts::PI * freq_l * t).sin());
        out.push((2.0 * std::f32::consts::PI * freq_r * t).sin());
    }
    out
}

#[test]
fn engine_process_steady_state_no_heap_activity() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CALLBACK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;

    let handles = Engine::build(EngineConfig::default()).expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    // Preallocate the feed chunk and output outside the tracked region.
    let feed = test_chunk_stereo(2048, SAMPLE_RATE as f32, 0);
    let mut out = vec![0.0f32; CALLBACK_FRAMES * 2];

    source.push(&feed);
    for _ in 0..WARMUP_ITERS {
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 1.1) {
            source.push(&feed);
        }
        processor.process(&mut out);
    }

    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        // Per-callback tempo retargets: the DJ ride case this path exists
        // for. Both the control write and the audio pull must be silent.
        let t = i as f64 / MEASURE_ITERS as f64;
        controller.set_tempo_rate(1.0 + 0.06 * (2.0 * std::f64::consts::PI * t).sin());
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 1.1) {
            source.push(&feed);
        }
        processor.process(&mut out);
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        controller.underrun_frames(),
        0,
        "steady state must not underrun"
    );
    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "engine steady state allocated: alloc_calls={alloc_calls}, realloc_calls={realloc_calls}, \
         alloc_bytes={alloc_bytes}, realloc_bytes={realloc_bytes}"
    );
}

#[test]
fn engine_process_varied_callback_sizes_no_heap_activity() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const WARMUP_ITERS: usize = 64;
    const MEASURE_ITERS: usize = 96;
    // The block scheduler must adapt caller sizes 64–1024 without touching
    // the heap.
    const SIZES: [usize; 5] = [64, 128, 333, 512, 1024];

    let handles = Engine::build(EngineConfig::default()).expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    let feed = test_chunk_stereo(4096, SAMPLE_RATE as f32, 0);
    let mut out = vec![0.0f32; 1024 * 2];

    source.push(&feed);
    for i in 0..WARMUP_ITERS {
        let frames = SIZES[i % SIZES.len()];
        if source.occupied_frames() < source.demand_hint(frames, 1.1) {
            source.push(&feed);
        }
        processor.process(&mut out[..frames * 2]);
    }

    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        let frames = SIZES[i % SIZES.len()];
        controller.set_tempo_rate(if i % 2 == 0 { 1.08 } else { 0.92 });
        if source.occupied_frames() < source.demand_hint(frames, 1.1) {
            source.push(&feed);
        }
        processor.process(&mut out[..frames * 2]);
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "varied-callback steady state allocated: alloc_calls={alloc_calls}, \
         realloc_calls={realloc_calls}, alloc_bytes={alloc_bytes}, realloc_bytes={realloc_bytes}"
    );
    let _ = controller;
}

#[test]
fn engine_keylock_steady_state_no_heap_activity() {
    // ROADMAP Stage 2 exit criterion: zero-alloc steady state holds with
    // keylock engaged (band split + PV corrector + post-resampler), under
    // per-callback tempo retargets.
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const SAMPLE_RATE: u32 = 44_100;
    const CALLBACK_FRAMES: usize = 256;
    const WARMUP_ITERS: usize = 128;
    const MEASURE_ITERS: usize = 96;

    let handles = Engine::build(EngineConfig {
        profile: EngineProfile::Keylock,
        ..EngineConfig::default()
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    let feed = test_chunk_stereo(2048, SAMPLE_RATE as f32, 0);
    let mut out = vec![0.0f32; CALLBACK_FRAMES * 2];

    source.push(&feed);
    for i in 0..WARMUP_ITERS {
        // Warm every transposition-dependent buffer before measuring.
        let t = i as f64 / WARMUP_ITERS as f64;
        controller.set_tempo_rate(1.0 + 0.2 * (2.0 * std::f64::consts::PI * t).sin());
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 1.3) {
            source.push(&feed);
        }
        processor.process(&mut out);
    }

    begin_alloc_tracking();
    for i in 0..MEASURE_ITERS {
        let t = i as f64 / MEASURE_ITERS as f64;
        controller.set_tempo_rate(1.0 + 0.08 * (2.0 * std::f64::consts::PI * t).sin());
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 1.1) {
            source.push(&feed);
        }
        processor.process(&mut out);
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert_eq!(
        controller.underrun_frames(),
        0,
        "keylock steady state must not underrun"
    );
    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "keylock steady state allocated: alloc_calls={alloc_calls}, realloc_calls={realloc_calls}, \
         alloc_bytes={alloc_bytes}, realloc_bytes={realloc_bytes}"
    );
}

#[test]
fn engine_underrun_and_recovery_no_heap_activity() {
    let _guard = ALLOC_TEST_MUTEX
        .lock()
        .expect("allocation test mutex poisoned");
    const CALLBACK_FRAMES: usize = 256;

    let handles = Engine::build(EngineConfig::default()).expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    let feed = test_chunk_stereo(1024, 44_100.0, 0);
    let mut out = vec![0.0f32; CALLBACK_FRAMES * 2];

    source.push(&feed);
    for _ in 0..16 {
        processor.process(&mut out);
    }

    // Starve the ring, then refill: the underrun path (silence fill,
    // counters) and the recovery path must both be allocation-free.
    begin_alloc_tracking();
    for _ in 0..8 {
        processor.process(&mut out);
    }
    source.push(&feed);
    for _ in 0..8 {
        processor.process(&mut out);
    }
    let (alloc_calls, realloc_calls, alloc_bytes, realloc_bytes) = end_alloc_tracking();

    assert!(
        controller.underrun_frames() > 0,
        "starvation must be counted"
    );
    assert_eq!(
        alloc_calls + realloc_calls,
        0,
        "underrun/recovery allocated: alloc_calls={alloc_calls}, realloc_calls={realloc_calls}, \
         alloc_bytes={alloc_bytes}, realloc_bytes={realloc_bytes}"
    );
}
