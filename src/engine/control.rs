//! Control plane: the lock-free timestamped parameter mailbox and the
//! [`EngineController`] handle.
//!
//! The engine splits into two halves at construction: an
//! [`EngineController`] owned by the control thread (UI, MIDI, network) and
//! an [`EngineProcessor`](crate::engine::EngineProcessor) owned by the audio
//! thread. They communicate only through this module's wait-free SPSC
//! structures — no locks, no allocation, and nothing on the audio side that
//! can fail.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Tempo rate hard floor (matches the varispeed control-path range).
pub const MIN_TEMPO_RATE: f64 = 0.25;
/// Tempo rate hard ceiling.
pub const MAX_TEMPO_RATE: f64 = 4.0;

/// Mailbox capacity in events. Drained every `process()` call, so overflow
/// needs more distinct control writes inside one audio callback than slots;
/// on overflow the event is dropped (counted) and the latest-value register
/// still converges the target by the next block.
const MAILBOX_SLOTS: usize = 256;

/// Timestamp value meaning "apply at the next block boundary".
pub const APPLY_ASAP: u64 = u64::MAX;

/// A control parameter carried by a mailbox event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    /// Playback tempo as a rate multiplier (1.0 = original tempo, 1.02 = 2%
    /// faster). In tape mode pitch follows this rate.
    TempoRate,
    /// Warm-start priming request: the value is the number of preroll
    /// source frames at the head of the ring to run through the graph with
    /// output discarded, so playback resumes converged (no cold-start
    /// silence-then-warmup at the seek target).
    WarmStart,
    /// Keylock (pitch correction) enable: 1.0 corrects the high band, 0.0
    /// bypasses to delay-matched varispeed. The keylock stage crossfades
    /// between the two over a short ramp, so toggling live is click-free;
    /// profiles without a keylock stage ignore it.
    Keylock,
}

impl Param {
    fn from_code(code: u64) -> Option<Self> {
        match code {
            0 => Some(Param::TempoRate),
            1 => Some(Param::WarmStart),
            2 => Some(Param::Keylock),
            _ => None,
        }
    }

    fn code(self) -> u64 {
        match self {
            Param::TempoRate => 0,
            Param::WarmStart => 1,
            Param::Keylock => 2,
        }
    }
}

/// One timestamped control event.
#[derive(Debug, Clone, Copy)]
pub struct ControlEvent {
    /// Engine-output frame index at which to apply, or [`APPLY_ASAP`].
    pub at_frame: u64,
    pub param: Param,
    pub value: f64,
}

/// One mailbox slot. Fields are plain atomics; the SPSC head/tail protocol
/// (write fields, then publish tail with `Release`; read tail with `Acquire`,
/// then fields) makes each slot's contents visible before it is readable.
#[derive(Debug)]
struct Slot {
    at_frame: AtomicU64,
    param: AtomicU64,
    value_bits: AtomicU64,
}

impl Slot {
    fn new() -> Self {
        Self {
            at_frame: AtomicU64::new(0),
            param: AtomicU64::new(0),
            value_bits: AtomicU64::new(0),
        }
    }
}

/// State shared between the controller and processor halves.
#[derive(Debug)]
pub(crate) struct EngineShared {
    slots: Box<[Slot]>,
    /// Consumer cursor (monotonic event count).
    head: AtomicUsize,
    /// Producer cursor (monotonic event count).
    tail: AtomicUsize,
    /// Events dropped because the mailbox was full.
    dropped_events: AtomicU64,
    /// Latest tempo target (f64 bits): convergence backstop if events drop.
    tempo_latest_bits: AtomicU64,
    /// Latest keylock target (f64 bits): convergence backstop if events drop.
    keylock_latest_bits: AtomicU64,
    /// Output frames the processor filled with silence due to source underrun.
    underrun_frames: AtomicU64,
    /// Fractional source position (f64 bits) of the engine's output head.
    source_position_bits: AtomicU64,
    /// Total frames delivered to the audio callback.
    delivered_frames: AtomicU64,
}

impl EngineShared {
    pub(crate) fn new(initial_tempo_rate: f64) -> Arc<Self> {
        let slots: Vec<Slot> = (0..MAILBOX_SLOTS).map(|_| Slot::new()).collect();
        Arc::new(Self {
            slots: slots.into_boxed_slice(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            dropped_events: AtomicU64::new(0),
            tempo_latest_bits: AtomicU64::new(initial_tempo_rate.to_bits()),
            keylock_latest_bits: AtomicU64::new(1.0f64.to_bits()),
            underrun_frames: AtomicU64::new(0),
            source_position_bits: AtomicU64::new(0.0f64.to_bits()),
            delivered_frames: AtomicU64::new(0),
        })
    }

    /// Producer side: appends one event; drops (and counts) it when full.
    fn push_event(&self, event: ControlEvent) {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= MAILBOX_SLOTS {
            self.dropped_events.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let slot = &self.slots[tail % MAILBOX_SLOTS];
        slot.at_frame.store(event.at_frame, Ordering::Relaxed);
        slot.param.store(event.param.code(), Ordering::Relaxed);
        slot.value_bits
            .store(event.value.to_bits(), Ordering::Relaxed);
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
    }

    /// Consumer side: pops the oldest pending event, if any.
    pub(crate) fn pop_event(&self) -> Option<ControlEvent> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let slot = &self.slots[head % MAILBOX_SLOTS];
        let event = ControlEvent {
            at_frame: slot.at_frame.load(Ordering::Relaxed),
            param: Param::from_code(slot.param.load(Ordering::Relaxed))?,
            value: f64::from_bits(slot.value_bits.load(Ordering::Relaxed)),
        };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(event)
    }

    pub(crate) fn tempo_latest(&self) -> f64 {
        f64::from_bits(self.tempo_latest_bits.load(Ordering::Relaxed))
    }

    pub(crate) fn keylock_latest(&self) -> f64 {
        f64::from_bits(self.keylock_latest_bits.load(Ordering::Relaxed))
    }

    /// Processor-side write of the latest-tempo register (used when a full
    /// pending queue degrades a scheduled retarget to ASAP).
    pub(crate) fn store_tempo_latest(&self, rate: f64) {
        self.tempo_latest_bits
            .store(rate.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn add_underrun_frames(&self, frames: u64) {
        if frames > 0 {
            self.underrun_frames.fetch_add(frames, Ordering::Relaxed);
        }
    }

    pub(crate) fn publish_position(&self, source_pos: f64, delivered_frames: u64) {
        self.source_position_bits
            .store(source_pos.to_bits(), Ordering::Relaxed);
        self.delivered_frames
            .store(delivered_frames, Ordering::Relaxed);
    }
}

/// Clamps a requested tempo rate into the supported varispeed range,
/// mapping non-finite input to 1.0.
#[inline]
pub fn clamp_tempo_rate(rate: f64) -> f64 {
    if rate.is_finite() {
        rate.clamp(MIN_TEMPO_RATE, MAX_TEMPO_RATE)
    } else {
        1.0
    }
}

/// Control-thread handle to a running engine.
///
/// All methods are wait-free and never block the audio thread. Out-of-range
/// values are clamped, never rejected: the deck must always respond.
#[derive(Debug)]
pub struct EngineController {
    shared: Arc<EngineShared>,
    sample_rate: u32,
}

impl EngineController {
    pub(crate) fn new(shared: Arc<EngineShared>, sample_rate: u32) -> Self {
        Self {
            shared,
            sample_rate,
        }
    }

    /// Retargets the tempo rate at the next block boundary. The rate is
    /// clamped to `[0.25, 4.0]`; in tape mode pitch follows it.
    pub fn set_tempo_rate(&self, rate: f64) {
        let rate = clamp_tempo_rate(rate);
        self.shared
            .tempo_latest_bits
            .store(rate.to_bits(), Ordering::Relaxed);
        self.shared.push_event(ControlEvent {
            at_frame: APPLY_ASAP,
            param: Param::TempoRate,
            value: rate,
        });
    }

    /// Retargets the tempo rate at an exact output frame (the axis
    /// [`delivered_frames`](Self::delivered_frames) advances on): the new
    /// rate's first output sample is exactly `at_output_frame`. Timestamps
    /// in the past apply immediately; multiple pending retargets apply in
    /// timestamp order.
    pub fn set_tempo_rate_at(&self, rate: f64, at_output_frame: u64) {
        // Scheduled retargets do not touch the latest-value register — it
        // backs ASAP semantics only.
        self.shared.push_event(ControlEvent {
            at_frame: at_output_frame,
            param: Param::TempoRate,
            value: clamp_tempo_rate(rate),
        });
    }

    /// The most recently requested (clamped) tempo rate.
    pub fn tempo_rate_target(&self) -> f64 {
        self.shared.tempo_latest()
    }

    /// Requests warm-start priming: the next `preroll_frames` source frames
    /// in the ring are run through the graph with output discarded, so the
    /// audio that follows plays through fully converged DSP state.
    ///
    /// Seek protocol: reset the processor (host-mediated), re-anchor the
    /// track position, send this, then push `preroll_frames` of the audio
    /// PRECEDING the target followed by the target's audio. Priming is
    /// budgeted across a few callbacks (silence during, ~1 ms of work per
    /// callback) and ends with a declick fade-in.
    pub fn warm_start(&self, preroll_frames: u32) {
        self.shared.push_event(ControlEvent {
            at_frame: APPLY_ASAP,
            param: Param::WarmStart,
            value: preroll_frames as f64,
        });
    }

    /// Enables or disables keylock (high-band pitch correction) at the next
    /// block boundary. The keylock stage crossfades over a short ramp
    /// (~12 ms), so this is safe to toggle during live playback: disabled
    /// output is delay-matched pure varispeed — pitch follows tempo — at
    /// the chain's unchanged constant latency. Profiles without a keylock
    /// stage ignore it.
    pub fn set_keylock(&self, enabled: bool) {
        let value: f64 = if enabled { 1.0 } else { 0.0 };
        self.shared
            .keylock_latest_bits
            .store(value.to_bits(), Ordering::Relaxed);
        self.shared.push_event(ControlEvent {
            at_frame: APPLY_ASAP,
            param: Param::Keylock,
            value,
        });
    }

    /// The most recently requested keylock state.
    pub fn keylock_target(&self) -> bool {
        self.shared.keylock_latest() >= 0.5
    }

    /// Fractional source position (frames since the first pushed source
    /// frame) of the audio currently leaving the engine.
    pub fn source_position(&self) -> f64 {
        f64::from_bits(self.shared.source_position_bits.load(Ordering::Relaxed))
    }

    /// Total frames the processor has delivered to the audio callback.
    pub fn delivered_frames(&self) -> u64 {
        self.shared.delivered_frames.load(Ordering::Relaxed)
    }

    /// Output frames filled with silence because the source ring ran dry.
    pub fn underrun_frames(&self) -> u64 {
        self.shared.underrun_frames.load(Ordering::Relaxed)
    }

    /// Control events dropped due to mailbox overflow (should stay 0).
    pub fn dropped_events(&self) -> u64 {
        self.shared.dropped_events.load(Ordering::Relaxed)
    }

    /// Engine sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_roundtrip_in_order() {
        let shared = EngineShared::new(1.0);
        let controller = EngineController::new(Arc::clone(&shared), 44_100);
        controller.set_tempo_rate(1.05);
        controller.set_tempo_rate(0.98);

        let first = shared.pop_event().expect("first event");
        assert_eq!(first.param, Param::TempoRate);
        assert!((first.value - 1.05).abs() < 1e-12);
        let second = shared.pop_event().expect("second event");
        assert!((second.value - 0.98).abs() < 1e-12);
        assert!(shared.pop_event().is_none());
        assert_eq!(controller.dropped_events(), 0);
    }

    #[test]
    fn mailbox_overflow_drops_and_counts_but_latest_converges() {
        let shared = EngineShared::new(1.0);
        let controller = EngineController::new(Arc::clone(&shared), 44_100);
        for i in 0..(MAILBOX_SLOTS + 10) {
            controller.set_tempo_rate(1.0 + i as f64 * 1e-4);
        }
        assert_eq!(controller.dropped_events(), 10);
        let expected_latest = clamp_tempo_rate(1.0 + (MAILBOX_SLOTS + 9) as f64 * 1e-4);
        assert!((shared.tempo_latest() - expected_latest).abs() < 1e-12);

        let mut drained = 0;
        while shared.pop_event().is_some() {
            drained += 1;
        }
        assert_eq!(drained, MAILBOX_SLOTS);
    }

    #[test]
    fn keylock_roundtrip_and_latest_register() {
        let shared = EngineShared::new(1.0);
        let controller = EngineController::new(Arc::clone(&shared), 44_100);
        assert!(controller.keylock_target());

        controller.set_keylock(false);
        let event = shared.pop_event().expect("keylock event");
        assert_eq!(event.param, Param::Keylock);
        assert_eq!(event.value, 0.0);
        // The latest-value register backstops dropped events.
        assert_eq!(shared.keylock_latest(), 0.0);
        assert!(!controller.keylock_target());

        controller.set_keylock(true);
        assert_eq!(shared.pop_event().expect("re-enable").value, 1.0);
        assert!(controller.keylock_target());
    }

    #[test]
    fn tempo_rate_clamped() {
        assert_eq!(clamp_tempo_rate(9.0), MAX_TEMPO_RATE);
        assert_eq!(clamp_tempo_rate(0.0), MIN_TEMPO_RATE);
        assert_eq!(clamp_tempo_rate(f64::NAN), 1.0);
        assert_eq!(clamp_tempo_rate(f64::INFINITY), 1.0);
        assert_eq!(clamp_tempo_rate(1.07), 1.07);
    }

    #[test]
    fn mailbox_cross_thread_smoke() {
        let shared = EngineShared::new(1.0);
        let controller = EngineController::new(Arc::clone(&shared), 44_100);
        let producer = std::thread::spawn(move || {
            for i in 0..10_000u32 {
                controller.set_tempo_rate(1.0 + (i % 100) as f64 * 1e-3);
            }
        });
        let mut seen = 0u64;
        while !producer.is_finished() {
            while let Some(event) = shared.pop_event() {
                assert!(event.value >= MIN_TEMPO_RATE && event.value <= MAX_TEMPO_RATE);
                seen += 1;
            }
        }
        while let Some(_event) = shared.pop_event() {
            seen += 1;
        }
        let dropped = shared.dropped_events.load(Ordering::Relaxed);
        assert_eq!(seen + dropped, 10_000);
    }
}
