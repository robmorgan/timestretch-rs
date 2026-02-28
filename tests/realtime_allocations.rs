use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use timestretch::{EdmPreset, StreamProcessor, StretchParams};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static REALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static REALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

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

    // This test exercises the default low-latency streaming path.
    processor.set_hybrid_mode(false);

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
