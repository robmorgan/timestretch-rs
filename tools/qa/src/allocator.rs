// Same counting-allocator pattern as tests/engine_realtime_allocations.rs.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
struct Counter;
static TRACK: AtomicBool = AtomicBool::new(false);
static FREES: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOC: Counter = Counter;
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Relaxed);
            if TRACK.load(Relaxed) {
                COUNT.fetch_add(1, Relaxed);
            }
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Relaxed);
            if TRACK.load(Relaxed) {
                COUNT.fetch_add(1, Relaxed);
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if TRACK.load(Relaxed) {
            FREES.fetch_add(1, Relaxed);
        }
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_add(n, Relaxed);
            LIVE.fetch_sub(l.size(), Relaxed);
            if TRACK.load(Relaxed) {
                COUNT.fetch_add(1, Relaxed);
            }
        }
        q
    }
}
pub fn begin() {
    #[cfg(target_os = "macos")]
    if let Some(probe) = SYSTEM_PROBE.get().copied().flatten() {
        probe.begin();
    }
    COUNT.store(0, Relaxed);
    FREES.store(0, Relaxed);
    TRACK.store(true, Relaxed);
}
pub fn end() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(probe) = SYSTEM_PROBE.get().copied().flatten() {
        SYSTEM_COUNT.store(probe.end(), Relaxed);
    }
    TRACK.store(false, Relaxed);
    COUNT.load(Relaxed)
}
pub fn live() -> usize {
    LIVE.load(Relaxed)
}

#[cfg(target_os = "macos")]
static SYSTEM_PROBE: std::sync::OnceLock<Option<timestretch_accelerate::AllocationProbe>> =
    std::sync::OnceLock::new();
#[cfg(target_os = "macos")]
static SYSTEM_COUNT: AtomicUsize = AtomicUsize::new(0);
pub fn init() {
    #[cfg(target_os = "macos")]
    SYSTEM_PROBE.get_or_init(timestretch_accelerate::AllocationProbe::load);
}
pub fn system_count() -> Option<usize> {
    #[cfg(target_os = "macos")]
    if SYSTEM_PROBE.get().copied().flatten().is_some() {
        return Some(SYSTEM_COUNT.load(Relaxed));
    }
    None
}

pub fn deallocations() -> usize {
    FREES.load(Relaxed)
}
