//! Allocator observer is test-only; library crates remain safe Rust.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use timestretch::*;
struct Audit;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};static COUNTS:Cell<(usize,usize)>=const{Cell::new((0,0))};}
fn event(allocation: bool) {
    let _ = ACTIVE.try_with(|a| {
        if a.get() {
            COUNTS.with(|c| {
                let (x, y) = c.get();
                c.set((x + usize::from(allocation), y + usize::from(!allocation)));
            });
        }
    });
}
unsafe impl GlobalAlloc for Audit {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        event(true);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        event(true);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        event(true);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        event(false);
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Audit = Audit;
fn begin() {
    COUNTS.set((0, 0));
    ACTIVE.set(true);
}
fn end() -> (usize, usize) {
    ACTIVE.set(false);
    COUNTS.get()
}
#[test]
fn prepared_processing_controls_seek_and_eof_do_not_allocate_or_free() {
    begin();
    let allocation = std::hint::black_box(vec![0u8; 256]);
    drop(allocation);
    assert_eq!(end(), (1, 1));
    for backend in [Backend::Portable, Backend::Accelerate] {
        if backend.resolve().is_err() {
            continue;
        }
        for (source_rate, output_rate) in [
            (44100, 44100),
            (48000, 48000),
            (96000, 96000),
            (32000, 192000),
            (192000, 32000),
        ] {
            let source = vec![0.1f32; 24000 * 2];
            let mut out = [0.0; 514];
            let config = SessionConfig {
                backend,
                source_sample_rate: source_rate,
                output_sample_rate: output_rate,
                source_access: SourceAccess::Seekable,
                source_frames: Some(24000),
                ..Default::default()
            };
            let mut s = PreparedSession::new(config).unwrap().activate();
            let memory = s.owned_memory();
            begin();
            s.schedule(1, 1000, Parameters::keylock(TempoRate::MIN))
                .unwrap();
            s.schedule(
                2,
                8000,
                Parameters::linked(TempoRate::new(2.0).unwrap()).unwrap(),
            )
            .unwrap();
            let mut seeks = 0;
            for i in 0..100000 {
                if i == 300 {
                    s.set_transport(Transport::Paused);
                    s.seek(9000).unwrap();
                    s.seek(1000).unwrap();
                    seeks += 1;
                    s.set_transport(Transport::Playing);
                }
                let p = s.advance(3);
                assert_ne!(p.status, ProcessStatus::Fault);
                if p.status == ProcessStatus::NeedSource {
                    let r = s.source_request().unwrap();
                    s.supply(
                        r,
                        &source
                            [r.start_frame as usize * 2..(r.start_frame as usize + r.frames) * 2],
                    )
                    .unwrap();
                }
                while s.next_applied().is_some() {}
                let stamp = s.read_output(&mut out).unwrap();
                s.acknowledge(stamp.consumed(stamp.frames).unwrap())
                    .unwrap();
                if s.finished() && seeks > 0 {
                    break;
                }
                if s.finished() {
                    s.seek(0).unwrap();
                    s.set_transport(Transport::Playing);
                }
            }
            let counts = end();
            assert_eq!(counts, (0, 0), "{backend:?} {source_rate}/{output_rate}");
            assert_eq!(s.owned_memory(), memory);
            assert!(s.finished());
        }
    }
}

#[test]
fn joint_ratio_boundary_commands_reuse_prepared_storage() {
    for backend in [Backend::Portable, Backend::Accelerate] {
        if backend.resolve().is_err() {
            continue;
        }
        for rate in [44100, 48000, 96000] {
            let source: Vec<_> = (0..8192 * 2)
                .map(|i| (i as f32 * 0.017).sin() * 0.2)
                .collect();
            for (tempo, pitch) in [
                (1.0 / 32.0, 0.25),
                (1.0 / 32.0, 1.0),
                (1.0 / 16.0, 2.0),
                (1.0 / 8.0, 4.0),
                (0.25, 0.25),
                (0.25, 1.0),
                (0.25, 4.0),
                (1.0, 0.25),
                (1.0, 4.0),
                (4.0, 1.0),
                (4.0, 4.0),
            ] {
                let mut s = PreparedSession::new(SessionConfig {
                    backend,
                    source_sample_rate: rate,
                    output_sample_rate: rate,
                    source_frames: Some(8192),
                    ..Default::default()
                })
                .unwrap()
                .activate();
                let memory = s.owned_memory();
                let parameters = Parameters {
                    tempo: TempoRate::new(tempo).unwrap(),
                    pitch: PitchRatio::new(pitch).unwrap(),
                    ..Default::default()
                };
                let mut out = [0.0; 514];
                let mut applied = false;
                begin();
                s.schedule(1, 0, parameters).unwrap();
                for _ in 0..100000 {
                    assert_ne!(s.advance(1).status, ProcessStatus::Fault);
                    if let Some(r) = s.source_request() {
                        let start = r.start_frame as usize * 2;
                        s.supply(r, &source[start..start + r.frames * 2]).unwrap();
                    }
                    while let Some(event) = s.next_applied() {
                        applied |= event.command_id == 1 && event.requested == parameters;
                    }
                    let stamp = s.read_output(&mut out).unwrap();
                    assert!(out[..stamp.frames * 2].iter().all(|x| x.is_finite()));
                    s.acknowledge(stamp.consumed(stamp.frames).unwrap())
                        .unwrap();
                    if s.finished() {
                        break;
                    }
                }
                let counts = end();
                assert_eq!(counts, (0, 0), "{backend:?} {rate}/{tempo}/{pitch}");
                assert_eq!(s.owned_memory(), memory);
                assert!(s.finished() && applied);
            }
        }
    }
}
