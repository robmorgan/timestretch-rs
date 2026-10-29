//! Typed Direct → HybridCore → Direct → SpectralCore schedule. Direct manages
//! buffering and timing around each core; the hybrid core coordinates spectral
//! processing, pitch-mark analysis, and grain synthesis.
use super::{
    buffering::{Planar, Source, outputs, planes, views},
    *,
};
use crate::StretchError;
type Result<T> = std::result::Result<T, StretchError>;
#[derive(Default)]
struct Trace {
    #[cfg(test)]
    events: Vec<(i32, i32)>,
}
impl Trace {
    fn record(&mut self, kind: i32, step: i32) {
        #[cfg(test)]
        self.events.push((kind, step));
        #[cfg(not(test))]
        let _ = (kind, step);
    }
}
#[derive(Clone, Debug)]
struct Base {
    combined: f32,
    pitch: f32,
    real_pitch: f32,
    inv_pitch: f32,
    inv_combined: f32,
    correction: f32,
    cutoff: f32,
    rate: f32,
    ch: usize,
    n: i32,
    hi: i32,
    ho: i32,
    span_in: i32,
    span_out: i32,
    initial_hi: i32,
    next: i32,
    calls: i32,
    first: i32,
    time_adjust: i32,
    initial_adjust: i32,
    input_time: f64,
    reference: f64,
    real: f64,
    target: f64,
    fresh: bool,
    bypass: bool,
    hold: bool,
    keep: bool,
    modulus: i32,
    toggle: i32,
}
impl Base {
    fn new(ch: usize, rate: u32) -> Self {
        let n = if rate < 64000 { 4096 } else { 8192 };
        Self {
            combined: 1.,
            pitch: 1.,
            real_pitch: 1.,
            inv_pitch: 1.,
            inv_combined: 1.,
            correction: 0.,
            cutoff: rate as f32 * 0.5,
            rate: rate as f32,
            ch,
            n,
            hi: n / 8,
            ho: n / 8,
            span_in: n / 8,
            span_out: n / 8,
            initial_hi: n / 8,
            next: 0,
            calls: 0,
            first: 0,
            time_adjust: 0,
            initial_adjust: 0,
            input_time: -f64::from(n / 2),
            reference: 0.,
            real: 0.,
            target: 0.,
            fresh: true,
            bypass: false,
            hold: false,
            keep: false,
            modulus: 1,
            toggle: 0,
        }
    }
    fn real_factor(&mut self, error: f64) -> f32 {
        if error as i32 == 0 {
            self.correction = 0.;
            return self.pitch;
        }
        self.correction = self.correction.max(error.abs() as f32 * 1.25);
        let horizon = (3 * self.span_out) as f32;
        let horizon = f64::from(horizon.max(self.correction));
        (((horizon + error) / horizon).clamp(0.5, 2.) * f64::from(self.pitch)) as f32
    }
    fn quantize(&mut self, ratio: &mut f32) {
        let shift = if self.modulus > 1 || *ratio > 6. {
            5
        } else if f64::from(*ratio) <= 2.2 {
            3
        } else if f64::from(*ratio) <= 3.2 {
            4
        } else if f64::from(*ratio) <= 5.2 {
            5
        } else if *ratio <= 10. {
            6
        } else {
            2
        };
        self.hi = self.n >> shift;
        self.ho = (*ratio * self.hi as f32) as i32;
        *ratio = self.ho as f32 / self.hi as f32;
    }
    fn advance(&mut self) {
        self.real_pitch = self.real_factor(self.real - self.target);
        let advance = f64::from(self.span_out as f32 / self.pitch);
        self.target += advance;
        self.real += f64::from(self.span_out as f32 / self.real_pitch);
        self.reference += advance;
        self.fresh = false;
    }
}
/// Phase-vocoder processing and resampling with shared scheduling state.
struct SpectralCore {
    b: Base,
    pv: PhaseVocoder,
    resampler: Resampler,
    scratch: Planar,
}
/// Coordinates spectral processing, pitch-mark analysis, and grain synthesis.
struct HybridCore {
    b: Base,
    inner: Box<Direct>,
    splitter: Splitter,
    analysis: MarkAnalysis,
    marks: PitchMarks,
    synth: GrainSynthesis,
    resampler: Resampler,
    a: Planar,
    band: Planar,
    source: Source,
    split_frames: usize,
    offset: usize,
    internal_steps: i32,
    extra: i32,
    batch_start: i32,
    hop_scale: f32,
    shortage: bool,
}
enum Core {
    Spectral(Box<SpectralCore>),
    Hybrid(Box<HybridCore>),
}
impl Core {
    fn base(&self) -> &Base {
        match self {
            Self::Spectral(e) => &e.b,
            Self::Hybrid(p) => &p.b,
        }
    }
    fn base_mut(&mut self) -> &mut Base {
        match self {
            Self::Spectral(e) => &mut e.b,
            Self::Hybrid(p) => &mut p.b,
        }
    }
    fn spectral(&self) -> &SpectralCore {
        match self {
            Self::Spectral(e) => e,
            Self::Hybrid(p) => p.inner.core.spectral(),
        }
    }
    fn spectral_mut(&mut self) -> &mut SpectralCore {
        match self {
            Self::Spectral(e) => e,
            Self::Hybrid(p) => p.inner.core.spectral_mut(),
        }
    }
    fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }
    fn blocks(&self) -> i32 {
        let b = self.base();
        b.span_in.max(b.hi) / b.hi
    }
    fn needed(&self, block: i32) -> bool {
        let e = &self.spectral().b;
        match self {
            Self::Spectral(_) => (e.toggle + block) % e.modulus == 0 || e.keep,
            Self::Hybrid(p) => {
                if e.keep {
                    return true;
                }
                for b in self.blocks() * block..self.blocks() * (block + 1) {
                    if (e.toggle + b) % e.modulus == 0 {
                        return (p.source.available() as i32) < p.b.span_in * (block + 1);
                    }
                }
                false
            }
        }
    }
    fn samples_in(&self) -> usize {
        let b = self.base();
        if self.is_hybrid() {
            if self.needed(0) {
                b.span_in.max(b.hi) as usize
            } else {
                0
            }
        } else {
            b.span_in as usize
        }
    }
    fn samples_out(&self) -> i32 {
        self.base().span_out.max(self.base().ho)
    }
    fn time(&self, internal: bool) -> i64 {
        match self {
            Self::Hybrid(p) => p.inner.time(internal) - i64::from(p.b.n / 8),
            Self::Spectral(e) => {
                let b = &e.b;
                let t = (b.input_time
                    + f64::from(
                        ((b.toggle + b.modulus - 1) % b.modulus) as f32 / b.modulus as f32
                            * b.span_in as f32,
                    )) as i64;
                t + if internal {
                    0
                } else {
                    i64::from(b.time_adjust)
                }
            }
        }
    }
    fn unused(&self) -> i32 {
        let b = self.base();
        match self {
            Self::Spectral(_) => {
                (if b.bypass { 0 } else { 2 })
                    + if b.target == b.real {
                        (b.inv_pitch * (b.n >> 1) as f32) as i32
                    } else {
                        b.n >> 1
                    }
            }
            Self::Hybrid(p) => {
                let base = p.inner.core.unused();
                let extra = if b.hi == b.n / 8 {
                    0.
                } else {
                    (b.n / 8) as f32
                };
                if b.target == b.real {
                    ((b.inv_pitch * extra * b.combined)
                        .mul_add(b.inv_pitch, (f64::from(base) - b.target) as f32)
                        + 2.0) as i32
                } else {
                    (b.combined * (-extra)).mul_add(b.inv_pitch, base as f32) as i32 + 2
                }
            }
        }
    }
    fn extra(&self) -> i32 {
        let b = &self.spectral().b;
        (0..self.blocks())
            .map(|k| {
                let full = if b.hold {
                    b.toggle == -k
                } else {
                    (b.toggle + k) % b.modulus == 0
                };
                (if full { 5 } else { 2 }) * b.ch as i32 + 1
            })
            .sum()
    }
    fn num_calls(&mut self, frames: i32) -> i32 {
        let b = self.base();
        if frames == 0 {
            return b.calls
                + match self {
                    Self::Hybrid(p) => p.extra,
                    _ => 0,
                };
        }
        if frames < 0 {
            return 0;
        }
        let (mut remaining, mut real, mut target) = (-f64::from(frames), b.real, b.target);
        let (mut calls, mut block) = (0, 0);
        loop {
            let b = self.base();
            let e = &self.spectral().b;
            if self.is_hybrid() {
                for k in 0..self.blocks() {
                    calls += (if (e.toggle + block + k) % e.modulus == 0 {
                        5
                    } else {
                        2
                    }) * b.ch as i32
                        + 1;
                }
                block += self.blocks();
                calls += b.calls;
            } else {
                calls += (if (e.toggle + block) % e.modulus == 0 {
                    5
                } else {
                    2
                }) * b.ch as i32;
                block += 1;
            }
            let r = self.base_mut().real_factor(real - target);
            let b = self.base();
            target += f64::from(b.span_out as f32 / b.pitch);
            real += f64::from(b.span_out) / f64::from(r);
            remaining += f64::from(self.samples_out()) / f64::from(r);
            if remaining as i32 >= 1 {
                break;
            }
        }
        calls
    }
    fn cutoff(&mut self, hz: f32, preserve: bool) {
        let e = self.spectral_mut();
        if !preserve {
            e.b.cutoff = hz;
        }
        e.pv.active =
            (e.b.n / 2).min(((hz / e.b.rate).clamp(0., 1.) * e.b.n as f32) as i32) as usize;
    }
    fn crossover(&mut self, hz: f32) {
        let e = self.spectral_mut();
        e.pv.crossover = ((hz / e.b.rate).clamp(0., 1.) * e.b.n as f32) as usize;
    }
    fn update_spans(&mut self) -> Result<()> {
        if let Self::Hybrid(p) = self {
            let e = p.inner.core.base();
            p.b.hi = e.hi;
            p.b.ho = e.ho;
            p.b.span_in = (p.b.n / 8).max(e.hi);
            p.b.span_out = e.ho * (p.b.span_in / e.hi).max(1);
            p.internal_steps = p.inner.calls();
        }
        let extra = self.extra();
        if let Self::Hybrid(p) = self {
            p.extra = extra;
        }
        Ok(())
    }
    fn quantize(&mut self, ratio: &mut f32) -> Result<()> {
        self.spectral_mut().b.quantize(ratio);
        if self.is_hybrid() {
            self.update_spans()?;
        }
        Ok(())
    }
    fn scale_hop(&mut self, factor: f32) -> Result<()> {
        let e = self.spectral_mut();
        let b = &mut e.b;
        b.ho = (b.ho as f32 * factor) as i32;
        b.hi = (b.hi as f32 * factor) as i32;
        b.span_in = b.hi;
        b.span_out = b.ho;
        e.pv.set_hops(b.hi as usize, b.ho as usize, b.combined)?;
        if let Self::Hybrid(p) = self {
            p.hop_scale = factor;
            self.update_spans()?;
        }
        Ok(())
    }
    fn set_ratio(
        &mut self,
        stretch: &mut f32,
        pitch: &mut f32,
        adjust: bool,
        continuity: bool,
    ) -> Result<i32> {
        let requested = *stretch * *pitch;
        if requested < 0.25 {
            return Ok(2);
        }
        if let Self::Hybrid(p) = self {
            p.b.pitch = *pitch;
            p.b.inv_pitch = 1. / p.b.pitch;
            p.b.combined = requested;
            let allow = p.inner.core.spectral().pv.allow_aux;
            p.inner.core.spectral_mut().pv.allow_aux = true;
            let mut one = 1.;
            if p.inner
                .set_ratio(&mut p.b.combined, &mut one, false, false)?
                != 0
            {
                return Ok(2);
            }
            p.inner.core.spectral_mut().pv.allow_aux = allow;
            let e = p.inner.core.base();
            p.b.hi = e.hi;
            p.b.ho = e.ho;
            p.b.inv_combined = 1. / p.b.combined;
            if adjust {
                p.b.pitch = p.b.combined / *stretch;
                p.b.inv_pitch = 1. / p.b.pitch;
                *pitch = p.b.pitch;
            } else {
                *stretch = p.b.combined * p.b.inv_pitch;
            }
            p.synth.set_stretch(p.b.combined)?;
            p.inner.core.cutoff(p.b.rate * 0.5 * p.b.inv_pitch, false);
            p.b.span_in = (p.b.n / 8).max(p.inner.core.base().hi);
            p.b.span_out = p.b.ho * (p.b.span_in / p.b.hi).max(1);
            let cross = if *stretch > 2. {
                let f = (*stretch - 2.) + 1.;
                (f * f * f * 1500.).min(p.b.rate * 0.5)
            } else {
                1500.
            };
            p.inner.core.crossover(cross);
            let b = &mut p.b;
            let delay = (b.n / 2 - b.initial_hi + b.hi) as f32;
            b.target = b.reference - f64::from(b.inv_pitch.mul_add(delay, -delay));
            if continuity && !b.fresh {
                b.real_pitch = b.real_factor(b.real - b.target);
            } else {
                b.real = b.target;
                b.real_pitch = b.pitch;
            }
            p.internal_steps = p.inner.calls();
            p.b.combined = p.inner.core.base().combined;
            let extra = self.extra();
            if let Self::Hybrid(p) = self {
                p.extra = extra;
            }
            return Ok(0);
        }
        let e = self.spectral_mut();
        let b = &mut e.b;
        let modulus = modulus_for(requested);
        let restart = if requested <= 6. {
            !b.hold && b.modulus != 1
        } else {
            b.modulus != modulus
        };
        if restart {
            b.toggle = 0;
            b.next = 0;
            b.calls = b.ch as i32 * 5;
        }
        b.modulus = modulus;
        b.pitch = *pitch;
        b.inv_pitch = 1. / b.pitch;
        b.combined = requested / modulus as f32;
        let mut combined = b.combined;
        b.quantize(&mut combined);
        b.combined = combined;
        b.inv_combined = 1. / combined;
        if adjust {
            b.pitch = combined * b.modulus as f32 / *stretch;
            b.inv_pitch = 1. / b.pitch;
            *pitch = b.pitch;
        } else {
            *stretch = combined * b.inv_pitch * b.modulus as f32;
        }
        if b.fresh {
            b.input_time = -f64::from(b.n / 2);
            b.initial_hi = b.hi;
        }
        e.pv.set_hops(b.hi as usize, b.ho as usize, b.combined)?;
        e.pv.active = (b.n / 2).min(
            ((b.rate * 0.5 * b.inv_pitch).min(b.cutoff) / b.rate)
                .clamp(0., 1.)
                .mul_add(b.n as f32, 0.) as i32,
        ) as usize;
        e.pv.infinite = b.modulus >= 7 || b.hold;
        if e.pv.infinite {
            e.pv.state.original_spectrum_bypass = false;
        }
        b.span_in = b.hi;
        b.span_out = b.ho;
        let delay = (b.hi - b.initial_hi + b.n / 2) as f32;
        b.target = b.reference - f64::from(b.inv_pitch.mul_add(delay, -delay));
        let offset = (-(b.n / 2) as f32).mul_add(b.inv_combined, delay);
        if continuity && !b.fresh {
            b.time_adjust = (offset - b.initial_adjust as f32) as i32;
            b.real_pitch = b.real_factor(b.real - b.target);
        } else {
            b.real = b.target;
            if b.fresh {
                b.time_adjust = 0;
                b.initial_adjust = offset as i32;
            } else {
                b.time_adjust = (offset - b.initial_adjust as f32) as i32;
            }
            b.real_pitch = b.pitch;
        }
        if b.hold {
            b.modulus = 2;
        }
        Ok(0)
    }
    fn hold(&mut self, value: bool, keep: bool) -> Result<i32> {
        let e = self.spectral_mut();
        let b = &mut e.b;
        if b.hold != value {
            b.hold = value;
            if value {
                if b.modulus > 1 {
                    b.hold = false;
                    b.keep = false;
                    return Ok(0x4c4b43);
                }
                e.pv.infinite = true;
                e.pv.state.original_spectrum_bypass = false;
                b.modulus = 2;
            } else {
                e.pv.infinite = false;
                b.modulus = 1;
                b.toggle = 0;
                b.first = 0;
                b.next = 0;
            }
            b.calls = (5 - b.first) * b.ch as i32;
        }
        b.keep = b.hold && keep;
        if self.is_hybrid() {
            self.update_spans()?;
        }
        Ok(0)
    }
    fn reset(&mut self) -> Result<()> {
        let ch = self.base().ch;
        let rate = self.base().rate as u32;
        *self.base_mut() = Base::new(ch, rate);
        match self {
            Self::Spectral(e) => {
                e.pv.reset();
                e.pv.allow_aux = true;
                e.pv.set_hops(e.b.hi as usize, e.b.ho as usize, 1.)?;
                e.pv.allow_aux = false;
                e.resampler.reset();
                e.b.calls = 5 * ch as i32;
            }
            Self::Hybrid(p) => {
                p.resampler.reset();
                p.inner.reset()?;
                p.marks.reset();
                p.analysis.reset();
                p.splitter.reset();
                p.synth.reset();
                p.source.reset();
                for v in p.a.iter_mut().chain(p.band.iter_mut()) {
                    v.fill(0.);
                }
                p.inner.core.base_mut().bypass = true;
                p.inner.core.crossover(1500.);
                p.b.calls = 4;
                p.hop_scale = 1.;
                p.internal_steps = p.inner.calls();
                let extra = self.extra();
                if let Self::Hybrid(p) = self {
                    p.extra = extra;
                }
            }
        }
        Ok(())
    }
    fn fill(&mut self, input: &mut Planar, trace: &mut Trace) -> Result<()> {
        trace.record(if self.is_hybrid() { 3 } else { 4 }, 0);
        match self {
            Self::Spectral(e) => {
                scramble(input, e.b.hi as usize, e.b.ch);
                e.b.input_time += f64::from(e.b.span_in);
                for ch in 0..e.b.ch {
                    e.pv.process_step(
                        &views(input, e.b.hi as usize)[..e.b.ch],
                        None,
                        &mut [],
                        -1 - ch as i32,
                    );
                }
            }
            Self::Hybrid(p) => {
                let n = p.b.span_in as usize;
                p.split_frames = p.splitter.process(
                    &views(input, n)[..p.b.ch],
                    &mut outputs(&mut p.a)[..p.b.ch],
                    &mut outputs(&mut p.band)[..p.b.ch],
                )?;
                p.analysis
                    .process(&views(&p.band, p.split_frames)[..p.b.ch], &mut p.marks)?;
                p.synth.frames_needed(0, &p.marks)?;
                p.synth.push(&views(&p.band, p.split_frames)[..p.b.ch])?;
                p.synth
                    .process(&mut outputs(&mut p.band)[..p.b.ch], &p.marks)?;
                p.source.set(&views(&p.a, p.split_frames), p.split_frames)?;
                while p.source.take(&mut outputs(&mut p.band), p.b.hi as usize) {
                    p.inner.core.fill(&mut p.band, trace)?;
                }
                p.source.release()?;
            }
        }
        Ok(())
    }
    fn process(
        &mut self,
        input: &mut Planar,
        aux: Option<&mut Planar>,
        step: i32,
        trace: &mut Trace,
    ) -> Result<usize> {
        trace.record(if self.is_hybrid() { 1 } else { 2 }, step);
        if let Self::Spectral(e) = self {
            let b = &mut e.b;
            let mut aux = aux;
            b.next = step;
            if step == 0 {
                b.input_time += f64::from(b.span_in);
                scramble(input, b.hi as usize, b.ch);
                if let Some(a) = aux.as_mut() {
                    scramble(a, b.ho as usize, b.ch);
                }
            }
            if b.keep && b.toggle != 0 && b.next == b.first {
                scramble(input, b.hi as usize, b.ch);
                b.input_time += f64::from(b.span_in);
                for ch in 0..b.ch {
                    e.pv.process_step(
                        &views(input, b.hi as usize)[..b.ch],
                        None,
                        &mut [],
                        -1 - ch as i32,
                    );
                }
            }
            let av = aux.as_ref().map(|a| views(a, b.ho as usize));
            e.pv.process_step(
                &views(input, b.hi as usize)[..b.ch],
                av.as_ref().map(|a| &a[..b.ch]),
                &mut outputs(&mut e.scratch)[..b.ch],
                b.next,
            );
            if step % 5 == 4 {
                let ch = step as usize / 5;
                input[ch][..b.ho as usize].copy_from_slice(&e.scratch[ch][..b.ho as usize]);
            }
            let total = b.ch as i32 * 5;
            if b.next % total == total - 1 {
                b.toggle = if b.hold {
                    1
                } else {
                    (b.toggle + 1) % b.modulus
                };
            }
            if b.toggle != 0 && b.next % 5 == 4 {
                b.next += 3;
            }
            b.first = if b.toggle != 0 { 3 } else { 0 };
            b.next = (b.next + 1) % total;
            if b.next != b.first {
                return Ok(0);
            }
            b.calls = (5 - b.first) * b.ch as i32;
            b.advance();
            descramble(input, b.ho as usize, b.ch);
            if b.bypass {
                return Ok(b.span_out as usize);
            }
            let n = e.resampler.process(
                &views(input, b.span_out as usize)[..b.ch],
                &mut outputs(&mut e.scratch)[..b.ch],
                b.real_pitch,
            )?;
            for (dest, src) in input.iter_mut().zip(&e.scratch).take(b.ch) {
                dest[..n].copy_from_slice(&src[..n]);
            }
            return Ok(n);
        }
        let calls = self.num_calls(0);
        let incoming = self.samples_in();
        let Self::Hybrid(p) = self else {
            unreachable!()
        };
        p.b.next = (step + 1) % calls;
        match step {
            0 => {
                p.split_frames = p.splitter.process(
                    &views(input, incoming)[..p.b.ch],
                    &mut outputs(&mut p.a)[..p.b.ch],
                    &mut outputs(&mut p.band)[..p.b.ch],
                )?;
                p.source.set(&views(&p.a, p.split_frames), p.split_frames)?;
                let e = p.inner.core.spectral_mut();
                if e.b.modulus > 1 || e.b.keep || p.hop_scale > 1. {
                    e.pv.allow_aux = false;
                } else if e.b.toggle == 0 && p.hop_scale == 1. {
                    e.pv.allow_aux = true;
                }
                if p.hop_scale <= 1. {
                    e.pv.do_aux = true;
                }
                p.offset = 0;
                return Ok(0);
            }
            1 => {
                p.analysis
                    .process(&views(&p.band, p.split_frames)[..p.b.ch], &mut p.marks)?;
                if p.marks.records().len() >= 257 && p.synth.state().last_mark_index >= 129 {
                    p.marks.retain_last(p.marks.records().len() - 128)?;
                    p.synth.discard_mark_prefix(128)?;
                }
                return Ok(0);
            }
            2 => {
                p.synth
                    .set_play_time(p.inner.time(true) as f64 + f64::from(p.inner.core.base().hi))?;
                p.synth.frames_needed(p.b.span_out as usize, &p.marks)?;
                p.synth.push(&views(&p.band, p.split_frames)[..p.b.ch])?;
                p.synth
                    .process(&mut outputs(&mut p.band)[..p.b.ch], &p.marks)?;
                p.split_frames = 0;
                p.shortage = false;
                p.batch_start = 3;
                return Ok(0);
            }
            _ => {}
        }
        if p.shortage {
            return Ok(0);
        }
        if step < calls - 1 {
            let local = (step - p.batch_start) % (p.internal_steps + 1);
            if local == 0 {
                let n = p.inner.needed(0) as usize;
                if !p.source.take(&mut outputs(input), n) {
                    p.shortage = true;
                    return Ok(0);
                }
                let band = [&p.band[0][p.offset..], &p.band[1][p.offset..]];
                p.inner.queue_aux(&views(input, n), &band, n)?;
            } else if local == p.internal_steps {
                let [a, b] = &mut p.band;
                let mut band = [&mut a[p.offset..], &mut b[p.offset..]];
                p.inner.get(&mut band)?;
                p.batch_start += p.internal_steps + 1;
                p.internal_steps = p.inner.calls();
                p.offset += p.b.ho as usize;
                return Ok(0);
            }
            p.inner.step(trace)?;
            return Ok(0);
        }
        p.source.release()?;
        p.internal_steps = p.inner.calls();
        // Release the mutable HybridCore borrow before querying the shared schedule.
        let extra = self.extra();
        let Self::Hybrid(p) = self else {
            unreachable!()
        };
        p.extra = extra;
        p.b.advance();
        if p.hop_scale < 1. {
            p.hop_scale = 1.;
        }
        p.resampler.process(
            &views(&p.band, p.b.span_out as usize)[..p.b.ch],
            &mut outputs(input)[..p.b.ch],
            p.b.real_pitch,
        )
    }
}
fn modulus_for(ratio: f32) -> i32 {
    if ratio <= 6. {
        return 1;
    }
    let mut modulus = 2;
    let mut local = ratio * 0.5;
    while local > 4. || local - local as i32 as f32 == 0. {
        modulus += 1;
        local = ratio / modulus as f32;
    }
    modulus
}
fn scramble(p: &mut Planar, n: usize, ch: usize) {
    if ch == 2 {
        let [left, right] = p;
        for (l, r) in left.iter_mut().zip(right).take(n) {
            let (a, b) = (*l, *r);
            *l = a.mul_add(0.6, b * 0.4);
            *r = a.mul_add(0.6, b * (-0.4));
        }
    }
}
fn descramble(p: &mut Planar, n: usize, ch: usize) {
    if ch == 2 {
        let [left, right] = p;
        for (l, r) in left.iter_mut().zip(right).take(n) {
            let (a, b) = (*l, *r);
            *l = (a + b) / 1.2;
            *r = (a - b) / 0.8;
        }
    }
}
struct Direct {
    core: Core,
    input: Source,
    aux_source: Source,
    output: Source,
    main: Planar,
    aux: Planar,
    fade: Planar,
    fade_scratch: Planar,
    processed: i32,
    flush_remaining: i32,
    fade_frames: i32,
    output_time: i32,
    pending_offset: i32,
    time_offset: i32,
    initial_hop: i32,
    delivered: i64,
    deferred: i32,
    have_aux: bool,
    buffered: bool,
    queued: bool,
    flush_fresh: bool,
    advance_pending: bool,
    hop_scale: f32,
}
impl Direct {
    fn new(core: Core, ch: usize) -> Self {
        let n = core.base().n as usize;
        let capacity = n * if core.is_hybrid() { 25 } else { 21 } / 8;
        Self {
            core,
            input: Source::new(ch, capacity, capacity),
            aux_source: Source::new(ch, capacity, capacity),
            output: Source::new(ch, n * 3, capacity),
            main: planes(capacity),
            aux: planes(capacity),
            fade: planes(capacity),
            fade_scratch: planes(capacity),
            processed: 0,
            flush_remaining: -1,
            fade_frames: 128,
            output_time: 0,
            pending_offset: 0,
            time_offset: 0,
            initial_hop: (n / 8) as i32,
            delivered: 0,
            deferred: 0,
            have_aux: false,
            buffered: false,
            queued: false,
            flush_fresh: true,
            advance_pending: false,
            hop_scale: 1.,
        }
    }
    fn time(&self, internal: bool) -> i64 {
        self.core.time(internal) + i64::from(self.time_offset)
    }
    fn queue(&mut self, src: &[&[f32]], n: usize) -> Result<()> {
        self.queued = true;
        self.input.set(src, n)?;
        self.input
            .take(&mut outputs(&mut self.main), self.core.samples_in());
        self.input.release()
    }
    fn queue_aux(&mut self, src: &[&[f32]], aux: &[&[f32]], n: usize) -> Result<()> {
        let ho = self.core.base().ho as usize;
        self.aux_source.set(aux, ho)?;
        self.aux_source.take(&mut outputs(&mut self.aux), ho);
        self.aux_source.release()?;
        self.have_aux = true;
        self.queue(src, n)
    }
    fn step(&mut self, trace: &mut Trace) -> Result<()> {
        if self.advance_pending {
            self.output_time += self.deferred;
            self.advance_pending = false;
        }
        self.processed = self.core.process(
            &mut self.main,
            if self.have_aux {
                Some(&mut self.aux)
            } else {
                None
            },
            self.core.base().next,
            trace,
        )? as i32;
        if !self.buffered || self.processed == 0 {
            return Ok(());
        }
        let prior = self.output_time;
        self.output_time += self.processed;
        if self.output_time > 0 {
            if prior < 0 {
                let discard = (self.processed - self.output_time) as usize;
                for ch in &mut self.main {
                    ch.copy_within(discard.., 0);
                }
                self.processed = self.output_time;
            }
            self.output.set(
                &views(&self.main, self.processed as usize),
                self.processed as usize,
            )?;
            self.output.release()?;
            self.processed = self.output.available() as i32;
            self.time_offset = self.pending_offset;
            if !self.core.needed(0) || self.input.available() < self.core.samples_in() {
                return Ok(());
            }
        } else if !self.core.needed(0) {
            return Ok(());
        }
        self.input
            .take(&mut outputs(&mut self.main), self.core.samples_in());
        Ok(())
    }
    fn get(&mut self, dst: &mut [&mut [f32]]) -> Result<usize> {
        if !self.buffered {
            self.output.set(
                &views(&self.main, self.processed as usize),
                self.processed as usize,
            )?;
            self.output.take(dst, self.processed as usize);
            self.output.release()?;
        } else {
            if self.hop_scale != 1. {
                let (hi, ho) = (self.core.base().hi, self.core.base().ho);
                self.core.scale_hop(1. / self.hop_scale)?;
                let discard = ((ho - self.core.base().ho) as f32 / self.core.base().pitch) as i32;
                self.processed -= discard;
                self.time_offset += self.core.base().hi - hi;
                self.output.take(dst, discard as usize);
            }
            self.output.take(dst, self.processed as usize);
            self.buffered = false;
            let n = self.processed.min(self.fade_frames) as usize;
            let reciprocal = 1. / n as f32;
            for i in 0..n {
                for (ch, dest) in dst.iter_mut().enumerate().take(self.core.base().ch) {
                    dest[i] =
                        reciprocal * (i as f32).mul_add(dest[i], self.fade[ch][i] * (n - i) as f32);
                }
            }
        }
        self.delivered += i64::from(self.processed);
        Ok(self.processed as usize)
    }
    fn needed(&self, step: i32) -> i32 {
        if !self.core.needed(step) {
            return 0;
        }
        let b = self.core.base();
        if !self.buffered {
            return b.span_in;
        }
        let time = self.output_time - self.fade_frames;
        let r = b.real_pitch.max(b.pitch);
        let (mut n, mut block) = (0, 0);
        if time <= 0 {
            let mut t = f64::from(time);
            loop {
                if self.core.needed(block) {
                    n += self.core.samples_in() as i32;
                }
                t = f64::from(self.core.samples_out()).mul_add(1. / f64::from(r), t);
                block += 1;
                if t as i32 >= 1 {
                    break;
                }
            }
        }
        n
    }
    fn calls(&mut self) -> i32 {
        self.core.num_calls(if self.buffered {
            self.fade_frames - self.output_time
        } else {
            0
        })
    }
    fn frames_processed(&self) -> i32 {
        let b = self.core.base();
        if self.buffered {
            self.processed
                - ((b.ho - (b.ho as f32 / self.hop_scale) as i32) as f32 / b.pitch) as i32
        } else {
            self.processed
        }
    }
    fn pre_needed(&self) -> i32 {
        if self.queued {
            0
        } else {
            self.core.base().n / 2
                + if self.core.is_hybrid() {
                    self.core.base().n / 8
                } else {
                    0
                }
        }
    }
    fn reset_local(&mut self) {
        self.input.reset();
        self.output.reset();
        self.fade_frames = self.core.base().hi >> 2;
        for ch in &mut self.main {
            ch.fill(0.);
        }
        self.output_time = 0;
        self.pending_offset = 0;
        self.processed = 0;
        self.flush_remaining = -1;
        self.have_aux = false;
        self.deferred = 0;
        self.buffered = false;
        self.queued = false;
        self.flush_fresh = true;
        self.advance_pending = false;
        self.hop_scale = 1.;
        self.initial_hop = self.core.base().hi;
        self.delivered = -i64::from(self.core.unused());
        self.time_offset = 0;
    }
    fn reset(&mut self) -> Result<()> {
        self.core.reset()?;
        self.reset_local();
        Ok(())
    }
    fn set_ratio(
        &mut self,
        s: &mut f32,
        p: &mut f32,
        adjust: bool,
        continuity: bool,
    ) -> Result<i32> {
        if self.buffered {
            Ok(2)
        } else {
            self.core.set_ratio(s, p, adjust, continuity)
        }
    }
    #[cfg(test)]
    fn block(
        &mut self,
        src: &[&[f32]],
        n: usize,
        dst: Option<&mut [&mut [f32]]>,
        trace: &mut Trace,
    ) -> Result<usize> {
        self.queue(src, n)?;
        let calls = self.calls();
        for _ in 0..calls {
            self.step(trace)?;
        }
        match dst {
            Some(dst) => self.get(dst),
            None => Ok(0),
        }
    }
    fn prefill(
        &mut self,
        src: &[&[f32]],
        n: usize,
        dst: Option<&mut [&mut [f32]]>,
        trace: &mut Trace,
    ) -> Result<usize> {
        let unused = self.core.unused();
        if self.queued {
            return Ok(0);
        }
        let batches = self.pre_needed() / self.core.samples_in() as i32;
        self.initial_hop = self.core.base().hi;
        self.delivered = -i64::from(unused);
        self.input.set(src, n)?;
        for _ in 0..batches - 1 {
            if self
                .input
                .take(&mut outputs(&mut self.main), self.core.samples_in())
            {
                self.core.fill(&mut self.main, trace)?;
            }
        }
        self.input
            .take(&mut outputs(&mut self.main), self.core.samples_in());
        self.input.release()?;
        let count = self.calls();
        for _ in 0..count {
            self.step(trace)?;
        }
        self.flush_fresh = true;
        let written = match dst {
            Some(dst) => self.get(dst)?,
            None => 0,
        };
        let b = self.core.base();
        self.time_offset = ((-(unused - written as i32) as f32 / (b.inv_pitch * b.ho as f32))
            * (b.hi as f32 / self.core.spectral().b.modulus as f32)
            - self.core.time(true) as f32) as i32;
        self.queued = true;
        Ok(written)
    }
    fn preprocess(
        &mut self,
        src: &[&[f32]],
        n: usize,
        dst: &mut [&mut [f32]],
        option: i32,
        trace: &mut Trace,
    ) -> Result<usize> {
        if self.queued {
            return Ok(0);
        }
        let batches = self.pre_needed() / self.core.samples_in() as i32;
        let unused = self.core.unused();
        let prior_out = self.core.base().ho;
        self.input.set(src, n)?;
        for _ in 0..batches {
            if self
                .input
                .take(&mut outputs(&mut self.main), self.core.samples_in())
            {
                self.core.fill(&mut self.main, trace)?;
            }
        }
        self.input.release()?;
        self.initial_hop = self.core.base().hi;
        if (option == 0 || option == 2) && self.core.spectral().b.modulus == 1 {
            self.hop_scale = 2.;
            let b = self.core.base();
            let compression = b.ho < b.hi;
            loop {
                let input_limit = b.n >> if option == 0 && !compression { 2 } else { 1 };
                if (input_limit as f32) < self.hop_scale * b.hi as f32 {
                    break;
                }
                let out = if compression {
                    self.hop_scale.mul_add(b.ho as f32, self.fade_frames as f32)
                } else {
                    self.hop_scale * b.ho as f32
                };
                let limit = (b.n >> 1) as f32;
                if if compression && option == 0 {
                    limit <= out
                } else {
                    limit < out
                } {
                    break;
                }
                self.hop_scale += self.hop_scale;
            }
            self.hop_scale = (self.hop_scale * 0.5).min(if option == 2 { 4. } else { 2. });
            self.core.scale_hop(self.hop_scale)?;
        }
        self.buffered = true;
        self.output_time = 2 - unused;
        self.deferred = 0;
        let b = self.core.base();
        if option == 2 {
            if self.fade_frames + b.ho > n as i32 {
                self.deferred = b.ho - n as i32 + self.fade_frames;
            }
        } else if self.hop_scale != 1. && b.combined > 1. {
            self.deferred = (self.core.samples_in() as i32)
                .min(b.ho - prior_out + self.fade_frames)
                .max(b.ho - n as i32 + self.fade_frames);
        }
        self.flush_fresh = true;
        self.advance_pending = true;
        self.output
            .set(src, (self.fade_frames + b.ho - self.deferred) as usize)?;
        self.output.take(
            &mut outputs(&mut self.main),
            (b.ho - self.deferred) as usize,
        );
        self.output
            .take(&mut outputs(&mut self.fade), self.fade_frames as usize);
        self.output.release()?;
        let count = (b.ho - self.deferred) as usize;
        let pitch = b.pitch;
        let written = match &mut self.core {
            Core::Hybrid(p) => {
                p.resampler
                    .process(&views(&self.main, count)[..p.b.ch], dst, pitch)?
            }
            Core::Spectral(e) => {
                e.resampler
                    .process(&views(&self.main, count)[..e.b.ch], dst, pitch)?
            }
        };
        // Separate owned scratch preserves the main buffer during resampling.
        let fade_count = self.fade_frames as usize;
        for ch in 0..self.core.base().ch {
            self.fade_scratch[ch][..fade_count].copy_from_slice(&self.fade[ch][..fade_count]);
        }
        self.fade_frames = match &mut self.core {
            Core::Hybrid(p) => p.resampler.process(
                &views(&self.fade_scratch, fade_count)[..p.b.ch],
                &mut outputs(&mut self.fade)[..p.b.ch],
                pitch,
            )?,
            Core::Spectral(e) => e.resampler.process(
                &views(&self.fade_scratch, fade_count)[..e.b.ch],
                &mut outputs(&mut self.fade)[..e.b.ch],
                pitch,
            )?,
        } as i32;
        self.delivered = written as i64;
        self.deferred = (self.deferred as f32 / pitch) as i32;
        let b = self.core.base();
        let time = self.core.time(true) as f32;
        self.time_offset = (written as f32
            / ((b.combined * self.core.spectral().b.modulus as f32) / pitch)
            - time) as i32;
        let advance = -(-(written as i32 + self.output_time) - self.deferred) as f32
            / (b.inv_pitch * b.ho as f32);
        let offset =
            advance.mul_add(b.hi as f32 / self.core.spectral().b.modulus as f32, -time) as i32;
        self.pending_offset = (offset as f32 - (pitch + pitch) / b.combined) as i32;
        self.time_offset -= ((pitch + pitch) / b.combined) as i32;
        self.deferred += ((b.ho - prior_out) as f32 / pitch) as i32;
        self.output_time -= ((b.ho - prior_out) as f32 / pitch) as i32;
        let read = [
            &dst[0][..written],
            if dst.len() > 1 {
                &dst[1][..written]
            } else {
                &[]
            },
        ];
        self.output.set(&read, written)?;
        self.output.advance(2);
        self.output.take(dst, written - 2);
        self.output.release()?;
        self.output.reset();
        Ok(written - 2)
    }
    fn flush(&mut self, dst: &mut [&mut [f32]], trace: &mut Trace) -> Result<Option<usize>> {
        let b = self.core.base();
        if self.flush_fresh {
            self.flush_remaining =
                (self.input.available() as i32) + (b.n >> 1) + b.hi + b.hi * ((b.n >> 1) / b.ho);
            self.flush_fresh = false;
        }
        if self.flush_remaining < 0 {
            return Ok(None);
        }
        for ch in &mut self.main {
            ch.fill(0.);
        }
        let need = self.needed(0) as usize;
        self.flush_remaining -= need as i32;
        if !self.input.take(&mut outputs(&mut self.main), need) {
            for ch in &mut self.main {
                ch.fill(0.);
            }
        }
        let count = self.calls();
        for _ in 0..count {
            self.step(trace)?;
        }
        Ok(Some(self.get(dst)?))
    }
    fn sync(&mut self, source: i32, output: i32, pitch: &mut f32, align: bool) -> Result<i32> {
        let b = self.core.base();
        let half = b.n >> 1;
        let current = self.core.time(true);
        if !self.queued {
            return Ok(0x4c4b42);
        }
        let source_now = i64::from(half) + current + i64::from(self.time_offset);
        let out_now = self.delivered + i64::from(half);
        let source = i64::from(source);
        let output = i64::from(output);
        let distance = source - source_now;
        if distance < i64::from(half) || output - out_now < i64::from(half) {
            return Ok(0x4c4b43);
        }
        let origin = if source_now - out_now == -1 {
            source_now
        } else {
            out_now
        };
        let nominal = (output - origin) as f32 / distance as f32;
        let mut candidate = nominal * *pitch;
        let mut combined = *pitch * candidate;
        let mut quantized = candidate;
        self.core.quantize(&mut combined)?;
        let mut hop = self.core.base().hi / modulus_for(combined);
        let mut correction = 0.;
        if candidate.abs() > 0.001 {
            for _ in 0..257 {
                combined = candidate * *pitch;
                self.core.quantize(&mut quantized)?;
                hop = (hop + self.core.base().hi / modulus_for(quantized)) >> 1;
                let ratio = combined / *pitch;
                correction =
                    (-ratio).mul_add((self.initial_hop - hop) as f32, ratio * 0.) / distance as f32;
                quantized = nominal + correction;
                let change = (quantized - candidate).abs();
                candidate = quantized;
                if change <= 0.001 {
                    break;
                }
            }
        }
        let pv = &mut self.core.spectral_mut().pv;
        pv.state.transient_counter =
            i32::try_from(distance + i64::from(half) + i64::from(self.time_offset))
                .map_err(|_| super::invalid("sync distance exceeds transient counter"))?;
        pv.sync_enabled = align;
        let mid = pv.state.input_hop as i32 >> 1;
        pv.state.transient_target =
            ((mid + pv.state.transient_counter) % pv.state.input_hop as i32 - mid).max(-mid);
        let mut stretch = nominal + correction;
        self.set_ratio(&mut stretch, pitch, true, true)
    }
}

/// Which ratio argument is adjusted to the scheduler's quantized hop sizes.
#[derive(Clone, Copy, Debug)]
pub enum RatioAdjustment {
    /// Preserve requested pitch; report achieved duration.
    Stretch,
    /// Preserve requested duration; report achieved pitch.
    Pitch,
}
/// Startup strategies for filling history and temporarily enlarging hops.
#[derive(Clone, Copy, Debug)]
pub enum StartupOptimization {
    /// Moderate temporary hop enlargement.
    Moderate,
    /// Fill history without enlarging hops.
    History,
    /// Up to fourfold temporary hop enlargement.
    Extended,
}
/// Scheduler state available without inspecting implementation buffers.
#[derive(Clone, Debug)]
pub struct SchedulerState {
    /// Current source frames per internal analysis frame.
    pub input_hop: usize,
    /// Current output frames per internal synthesis frame.
    pub output_hop: usize,
    /// Current spectral repetition cycle length.
    pub toggle_modulus: i32,
    /// Current repetition position.
    pub toggle_count: i32,
    /// Next hybrid processing stage.
    pub next_step: i32,
    /// Whether optimized startup still has output awaiting collection.
    pub buffered_startup: bool,
    /// Actual quantized pitch factor.
    pub pitch: f32,
    /// Source position including Direct's startup correction.
    pub source_time: i64,
}
/// Streaming scheduler for spectral processing, pitch-mark analysis, and grain
/// synthesis. File and live hosts queue planar fragments, run the requested
/// number of steps, and collect output through this same scheduler. All DSP
/// storage is allocated during construction.
///
/// Scope: native 44.1/48/96 kHz mono/stereo, unity spectral envelope, ordinary grains.
/// Control status codes: 0 succeeds, 2 rejects a ratio, 0x4c4b42 requires startup,
/// and 0x4c4b43 rejects hold/sync timing.
pub struct NativeProcessor {
    direct: Direct,
    trace: Trace,
    // Hosts may fragment input freely. Direct itself receives one complete
    // demand per cycle, since its queue call also consumes a main input block.
    frame_input: Planar,
    queued_frames: usize,
    input_ready: bool,
}
impl NativeProcessor {
    /// Construct native geometry for 44.1, 48, or 96 kHz.
    pub fn new(sample_rate: u32, channels: usize) -> Result<Self> {
        Self::new_with_numerics(sample_rate, channels, crate::Numerics::Portable)
    }
    pub fn new_with_numerics(
        sample_rate: u32,
        channels: usize,
        numerics: crate::Numerics,
    ) -> Result<Self> {
        if !numerics.available() {
            return Err(super::invalid("numerical backend unavailable"));
        }
        super::check_channels(channels)?;
        if ![44100, 48000, 96000].contains(&sample_rate) {
            return Err(super::invalid(
                "scheduled timestretch supports 44100, 48000 or 96000 Hz",
            ));
        }
        let n = if sample_rate < 64000 { 4096 } else { 8192 };
        let capacity = n * 25 / 8;
        let synthesis_parameter = (capacity * 4) as f32;
        let synthesis_capacity = ((synthesis_parameter * 131072.0) / 44100.0) as usize;
        let synthesis_pad = ((synthesis_parameter * 8192.0) / 44100.0) as usize;
        let spectral = Core::Spectral(Box::new(SpectralCore {
            b: Base::new(channels, sample_rate),
            pv: PhaseVocoder::new_with_numerics(n, channels, numerics)?,
            resampler: Resampler::new(channels, n, Interpolation::Hermite)?,
            scratch: planes(n * 21 / 8),
        }));
        let core = Core::Hybrid(Box::new(HybridCore {
            b: Base::new(channels, sample_rate),
            inner: Box::new(Direct::new(spectral, channels)),
            splitter: Splitter::new_with_numerics(
                n / 4,
                sample_rate,
                channels,
                capacity,
                numerics,
            )?,
            analysis: MarkAnalysis::new_with_numerics(n / 8, channels, capacity, numerics)?,
            marks: PitchMarks::new(2048)?,
            synth: GrainSynthesis::new(channels, synthesis_capacity, synthesis_pad)?,
            resampler: Resampler::new(channels, n, Interpolation::Hermite)?,
            a: planes(n),
            band: planes(n),
            source: Source::new(channels, n / 2, n),
            split_frames: 0,
            offset: 0,
            internal_steps: 0,
            extra: 0,
            batch_start: 0,
            hop_scale: 1.,
            shortage: false,
        }));
        let mut result = Self {
            direct: Direct::new(core, channels),
            trace: Trace::default(),
            frame_input: planes(capacity),
            queued_frames: 0,
            input_ready: false,
        };
        result.reset()?;
        if let Core::Hybrid(p) = &mut result.direct.core {
            p.synth.frames_needed(n / 8, &p.marks)?;
        }
        Ok(result)
    }
    /// Allocating diagnostic snapshot; QA calls this outside processing/timing.
    #[cfg(feature = "qa")]
    #[doc(hidden)]
    pub fn qa_snapshot(&self) -> serde_json::Value {
        let p = match &self.direct.core {
            Core::Hybrid(p) => p,
            _ => unreachable!(),
        };
        let pv = &p.inner.core.spectral().pv;
        let hash = |values: &[f32]| {
            values
                .iter()
                .fold(14695981039346656037u64, |h, x| {
                    (h ^ u64::from(x.to_bits())).wrapping_mul(1099511628211)
                })
                .to_string()
        };
        let marks: Vec<_> = p
            .marks
            .records()
            .iter()
            .map(|m| {
                serde_json::json!([
                    m.position,
                    m.energy.to_bits(),
                    m.transient.to_bits(),
                    m.tonality.to_bits(),
                    m.onset.to_bits(),
                    m.estimated_period.to_bits()
                ])
            })
            .collect();
        serde_json::json!({"frame":pv.state.frame_counter,"marks":marks,"classifier":pv.current_state[..pv.n/2],"spectrum_hash":hash(&pv.original[0]),"magnitude_hash":hash(&pv.magnitude[0][..pv.n/2+1]),
        "splitter_window":p.splitter.window.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
        "stages":{
          "main_history":(0..p.b.ch).map(|c|hash(&pv.main_history[c][..pv.n])).collect::<Vec<_>>(),
          "aux_history":(0..p.b.ch).map(|c|hash(&pv.aux_history[c][..pv.n])).collect::<Vec<_>>(),
          "output_ola":(0..p.b.ch).map(|c|hash(&pv.output_ola[c][..pv.n])).collect::<Vec<_>>(),
          "input_phase":(0..p.b.ch).map(|c|hash(&pv.input_phase[c][..pv.n/2+1])).collect::<Vec<_>>(),
          "output_phase":(0..p.b.ch).map(|c|hash(&pv.output_phase[c][..pv.n/2+1])).collect::<Vec<_>>(),
          "stored":(0..p.b.ch).map(|c|hash(&pv.stored[c][..pv.n/2+1])).collect::<Vec<_>>(),
          "aux_ola":(0..p.b.ch).map(|c|hash(&pv.aux_ola[c][..pv.n])).collect::<Vec<_>>()
        }})
    }
    fn check_input(&self, input: &[&[f32]]) -> Result<usize> {
        let n = super::input_frames(input, self.direct.core.base().ch)?;
        if n > self.frame_input[0].len() {
            return Err(super::invalid(
                "scheduler input fragment exceeds factory capacity",
            ));
        }
        Ok(n)
    }
    fn check_output(&self, output: &[&mut [f32]]) -> Result<()> {
        super::output_capacity(output, self.direct.core.base().ch, self.output_capacity())
    }
    /// Required output capacity per channel for every collection/startup call.
    pub fn output_capacity(&self) -> usize {
        self.direct.main[0].len()
    }
    /// Native sample rate used by all DSP components.
    pub fn sample_rate(&self) -> u32 {
        self.direct.core.base().rate as u32
    }
    /// Input capacity allocated during construction.
    pub fn input_capacity(&self) -> usize {
        self.frame_input[0].len()
    }
    /// Initial output frames omitted by the standalone transport policy.
    pub fn initial_unused_frames(&self) -> usize {
        self.direct.core.unused().max(0) as usize
    }
    /// PV frame length selected by the stream rate.
    pub fn block_size(&self) -> usize {
        self.direct.core.base().n as usize
    }
    /// Required startup history; zero once input has been queued.
    pub fn prefill_frames(&self) -> usize {
        self.direct.pre_needed() as usize
    }
    /// Source frames needed for the current or a future output block.
    pub fn frames_needed(&self, future_block: usize) -> Result<usize> {
        if future_block > 1024 {
            return Err(super::invalid("prediction exceeds 1024 blocks"));
        }
        if future_block == 0 && self.input_ready {
            return Ok(0);
        }
        Ok((self.direct.needed(future_block as i32) as usize).saturating_sub(self.queued_frames))
    }
    /// Number of individual stages required to complete the next output block.
    pub fn process_calls(&mut self) -> usize {
        self.direct.calls() as usize
    }
    /// Queue up to frames_needed samples, in arbitrary fragments. A preallocated
    /// adapter forwards the complete demand to Direct once; empty calls cannot
    /// consume another startup block. Call stages after satisfying the demand.
    pub fn queue(&mut self, input: &[&[f32]]) -> Result<()> {
        let n = self.check_input(input)?;
        let remaining = self.frames_needed(0)?;
        if n > remaining {
            return Err(super::invalid("queue exceeds current scheduler demand"));
        }
        if self.input_ready {
            return Ok(());
        }
        if self.queued_frames + n > self.frame_input[0].len() {
            return Err(super::invalid(
                "startup demand exceeds input adapter capacity",
            ));
        }
        for (ch, src) in input.iter().enumerate() {
            self.frame_input[ch][self.queued_frames..self.queued_frames + n].copy_from_slice(src);
        }
        self.queued_frames += n;
        if n == remaining {
            self.direct.queue(
                &views(&self.frame_input, self.queued_frames),
                self.queued_frames,
            )?;
            self.queued_frames = 0;
            self.input_ready = true;
        }
        Ok(())
    }
    /// Execute one scheduler-selected processing stage.
    pub fn step(&mut self) -> Result<()> {
        if !self.input_ready {
            if self.frames_needed(0)? != 0 {
                return Err(super::invalid(
                    "satisfy frames_needed before processing stages",
                ));
            }
            let empty: [&[f32]; 2] = [&[], &[]];
            self.queue(&empty[..self.direct.core.base().ch])?;
        }
        self.direct.step(&mut self.trace)
    }
    /// Frames ready after the completed schedule (including startup trimming).
    pub fn frames_processed(&self) -> usize {
        self.direct.frames_processed().max(0) as usize
    }
    /// Collect the completed block into preallocated planar output.
    pub fn read(&mut self, output: &mut [&mut [f32]]) -> Result<usize> {
        self.check_output(output)?;
        if self.direct.frames_processed() < 0 {
            return Err(super::invalid(
                "complete startup schedule before collection",
            ));
        }
        let n = self.direct.get(output)?;
        self.input_ready = false;
        Ok(n)
    }
    /// Queue a fragment and, once the input demand is met, run and collect a block.
    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) -> Result<usize> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(0);
        self.check_output(output)?;
        self.queue(input)?;
        if !self.input_ready {
            return Ok(0);
        }
        for _ in 0..self.process_calls() {
            self.step()?;
        }
        self.read(output)
    }
    /// Fill history and run the first scheduled block. `None` defers collection.
    pub fn prefill(
        &mut self,
        input: &[&[f32]],
        output: Option<&mut [&mut [f32]]>,
    ) -> Result<usize> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(0);
        let n = self.check_input(input)?;
        if let Some(o) = output.as_ref() {
            self.check_output(o)?;
        }
        if self.queued_frames != 0 || self.input_ready {
            return Err(super::invalid(
                "startup requires an empty host input adapter",
            ));
        }
        self.direct.prefill(input, n, output, &mut self.trace)
    }
    /// Optimized startup using the same queues, DSP states, and processing steps.
    pub fn preprocess(
        &mut self,
        input: &[&[f32]],
        output: &mut [&mut [f32]],
        option: StartupOptimization,
    ) -> Result<usize> {
        let n = self.check_input(input)?;
        self.check_output(output)?;
        if self.queued_frames != 0 || self.input_ready {
            return Err(super::invalid(
                "startup requires an empty host input adapter",
            ));
        }
        let option = match option {
            StartupOptimization::Moderate => 0,
            StartupOptimization::History => 1,
            StartupOptimization::Extended => 2,
        };
        self.direct
            .preprocess(input, n, output, option, &mut self.trace)
    }
    /// Quantize stretch/pitch, returning (control status, actual stretch, pitch).
    pub fn set_ratio(
        &mut self,
        mut stretch: f32,
        mut pitch: f32,
        adjust: RatioAdjustment,
        continuity: bool,
    ) -> Result<(i32, f32, f32)> {
        if !stretch.is_finite()
            || !(0.001..=32.).contains(&stretch)
            || !pitch.is_finite()
            || !(0.25..=4.).contains(&pitch)
            || stretch * pitch > 32.
        {
            return Err(super::invalid(
                "scheduler stretch must be 0.001..=32, pitch 0.25..=4, and their product at most 32",
            ));
        }
        if self.queued_frames != 0 || self.input_ready {
            return Ok((2, stretch, pitch));
        }
        let status = self.direct.set_ratio(
            &mut stretch,
            &mut pitch,
            matches!(adjust, RatioAdjustment::Pitch),
            continuity,
        )?;
        Ok((status, stretch, pitch))
    }
    /// Hold/release audio, optionally advancing source time during the hold.
    pub fn set_hold(&mut self, hold: bool, keep_time: bool) -> Result<i32> {
        if self.queued_frames != 0 || self.input_ready {
            return Ok(0x4c4b43);
        }
        self.direct.core.hold(hold, keep_time)
    }
    /// Drain retained audio. `None` indicates the scheduler has finished flushing.
    pub fn flush(&mut self, output: &mut [&mut [f32]]) -> Result<Option<usize>> {
        self.check_output(output)?;
        if self.queued_frames != 0 || self.input_ready {
            return Err(super::invalid(
                "complete or pad the queued input cycle before flushing",
            ));
        }
        self.direct.flush(output, &mut self.trace)
    }
    /// Clear queues and reset DSP counters. Analysis retains sample/filter
    /// storage, grain synthesis retains sample storage and window scale, and the
    /// splitter clears only its leading half-frame and spectrum. The vocoder
    /// clears histories while retaining windows and provider tables.
    pub fn reset(&mut self) -> Result<()> {
        self.queued_frames = 0;
        self.input_ready = false;
        self.direct.reset()
    }
    /// Restart transport from fresh zeroed sample storage, without allocating.
    /// Clear analysis, synthesis, and splitter storage before resetting counters
    /// so audio from a previous seek cannot leak into the new stream.
    pub fn restart(&mut self) -> Result<()> {
        if let Core::Hybrid(p) = &mut self.direct.core {
            p.analysis.clear_storage();
            p.synth.clear_storage();
            p.splitter.clear_storage();
            p.inner.core.spectral_mut().pv.state = super::VocoderState::default();
        }
        self.reset()?;
        if let Core::Hybrid(p) = &mut self.direct.core {
            p.synth
                .frames_needed(self.direct.initial_hop as usize, &p.marks)?;
        }
        Ok(())
    }
    /// Set a future source/output synchronization point and return status/pitch.
    pub fn set_sync_point(
        &mut self,
        source: i32,
        output: i32,
        mut pitch: f32,
        align: bool,
    ) -> Result<(i32, f32)> {
        if !pitch.is_finite() || !(0.5..=2.).contains(&pitch) {
            return Err(super::invalid("sync pitch must be 0.5..=2"));
        }
        if self.queued_frames != 0 || self.input_ready {
            return Ok((2, pitch));
        }
        let code = self.direct.sync(source, output, &mut pitch, align)?;
        Ok((code, pitch))
    }
    /// Minimum source/output sync positions and whether a previous sync is done.
    pub fn sync_status(&self) -> (i64, i64, bool) {
        let d = &self.direct;
        let pv = &d.core.spectral().pv;
        (
            i64::from(d.time_offset) + d.core.time(true) + i64::from(d.core.base().n),
            d.delivered + i64::from(d.core.base().n),
            pv.state.transient_counter <= pv.state.transient_target && !d.buffered,
        )
    }
    /// Read-only mark records for timestretch diagnostics.
    pub fn pitch_marks(&self) -> &[PitchMark] {
        let Core::Hybrid(p) = &self.direct.core else {
            unreachable!()
        };
        p.marks.records()
    }
    /// Read-only ordinary-grain state for timestretch diagnostics.
    pub fn synthesis_state(&self) -> &SynthesisState {
        let Core::Hybrid(p) = &self.direct.core else {
            unreachable!()
        };
        p.synth.state()
    }
    /// Read-only shared vocoder, including reference bins and saved states.
    pub fn vocoder(&self) -> &PhaseVocoder {
        &self.direct.core.spectral().pv
    }
    /// Current scalar scheduler state; this query allocates nothing.
    pub fn state(&self) -> SchedulerState {
        let d = &self.direct;
        let b = d.core.base();
        let e = &d.core.spectral().b;
        SchedulerState {
            input_hop: b.hi as usize,
            output_hop: b.ho as usize,
            toggle_modulus: e.modulus,
            toggle_count: e.toggle,
            next_step: b.next,
            buffered_startup: d.buffered,
            pitch: b.pitch,
            source_time: d.time(true),
        }
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod native_tests;

use crate::memory::HeapBytes;
impl HeapBytes for SpectralCore {
    fn heap_bytes(&self) -> usize {
        self.pv.heap_bytes() + self.resampler.heap_bytes() + self.scratch.heap_bytes()
    }
}
impl HeapBytes for HybridCore {
    fn heap_bytes(&self) -> usize {
        self.inner.heap_bytes()
            + self.splitter.heap_bytes()
            + self.analysis.heap_bytes()
            + self.marks.heap_bytes()
            + self.synth.heap_bytes()
            + self.resampler.heap_bytes()
            + self.a.heap_bytes()
            + self.band.heap_bytes()
            + self.source.heap_bytes()
    }
}
impl HeapBytes for Direct {
    fn heap_bytes(&self) -> usize {
        self.core.heap_bytes()
            + self.input.heap_bytes()
            + self.aux_source.heap_bytes()
            + self.output.heap_bytes()
            + self.main.heap_bytes()
            + self.aux.heap_bytes()
            + self.fade.heap_bytes()
            + self.fade_scratch.heap_bytes()
    }
}
impl HeapBytes for NativeProcessor {
    fn heap_bytes(&self) -> usize {
        self.direct.heap_bytes() + self.frame_input.heap_bytes()
    }
}
impl HeapBytes for Core {
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Spectral(e) => e.heap_bytes(),
            Self::Hybrid(p) => p.heap_bytes(),
        }
    }
}
impl NativeProcessor {
    /// Owned buffers and boxed states; provider FFT plans remain opaque.
    pub fn owned_buffer_bytes(&self) -> usize {
        self.heap_bytes()
    }
}
