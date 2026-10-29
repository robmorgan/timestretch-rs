//! QA owns reference adapters, allocation probes and scoring; no application uses it.
#![recursion_limit = "256"]

mod allocator;
mod callback_audit;
mod core;
#[allow(dead_code)]
#[path = "../../../desktop/src/delivery.rs"]
mod delivery;
mod device;
mod diagnostics;
#[allow(dead_code)]
#[path = "../../../desktop/src/output_queue.rs"]
mod output_queue;
mod playback;
mod quality;
#[path = "../../../desktop/src/render_worker.rs"]
mod render_worker;
#[allow(dead_code)]
#[path = "../../../desktop/src/scrub.rs"]
mod scrub;
mod session_perf;
#[allow(dead_code)]
#[path = "../../../desktop/src/state.rs"]
mod state;
#[path = "../../../desktop/src/worker_priority.rs"]
mod worker_priority;
use serde_json::{Value, json};
use std::{error::Error, time::Instant};
use timestretch::io::{read_wav_file, write_wav_file_float};
use timestretch::{AudioBuffer, Backend, DurationRatio, Parameters, SessionConfig};
use timestretch_core::{
    Numerics,
    processor::{NativeProcessor, RatioAdjustment},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn numerics(arm: &str) -> Result<Numerics> {
    Ok(match arm {
        "rust" | "portable" => Numerics::Portable,
        "apple" | "accelerate" => Numerics::Accelerate,
        "fft" => Numerics::AppleFft,
        "math" => Numerics::AppleMath,
        _ => return Err("unknown arm".into()),
    })
}
fn backend(arm: &str) -> Result<Backend> {
    Ok(match numerics(arm)? {
        Numerics::Portable => Backend::Portable,
        Numerics::Accelerate => Backend::Accelerate,
        _ => return Err("split providers are QA core experiments only".into()),
    })
}
fn render(a: &[String]) -> Result<Value> {
    let input = read_wav_file(&a[1])?;
    let ratio: f64 = a[3].parse()?;
    if matches!(a[0].as_str(), "fft" | "math") {
        let (samples, actual) = diagnostics::kernel_render(&input, &a[0], ratio as f32)?;
        let frames = samples.len() / input.channels.count();
        write_wav_file_float(
            &a[2],
            &AudioBuffer::try_new(samples, input.sample_rate, input.channels)?,
        )?;
        return Ok(
            json!({"arm":a[0],"actual_stretch":actual,"output_frames":frames,"underruns":0,"host":"QA native kernel"}),
        );
    }
    let result = timestretch::render(
        &input.data,
        SessionConfig {
            backend: backend(&a[0])?,
            source_sample_rate: input.sample_rate,
            output_sample_rate: input.sample_rate,
            channels: input.channels.count(),
            parameters: Parameters::keylock(DurationRatio::new(ratio)?.tempo()),
            ..Default::default()
        },
    )?;
    let r = result.report;
    write_wav_file_float(
        &a[2],
        &AudioBuffer::try_new(result.samples, input.sample_rate, input.channels)?,
    )?;
    Ok(
        json!({"arm":a[0],"sample_rate":input.sample_rate,"channels":input.channels.count(),
        "source_frames":r.source_frames,"output_frames":r.output_frames,"requested_stretch":ratio,
        "actual_stretch":r.parameters.achieved_stretch,"achieved_pitch":r.parameters.achieved_pitch,
        "underruns":0,"host":"StreamSession","startup":"standard prefill; initial-unused trim; ordered EOF"}),
    )
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    let v=match a.first().map(String::as_str) {
        Some("audit-probe") => { allocator::init();allocator::begin();let data=std::hint::black_box(vec![0u8;4096]);let n=allocator::end();std::hint::black_box(&data);json!({"host_allocations":n,"system_allocations":allocator::system_count()}) },
        Some("render") if a.len()==5 => render(&a[1..])?,
        Some("core") if a.len()>=8 => core::perf(&a[1..])?,
        Some("trace") if a.len()==5 => diagnostics::trace(&a[1..])?,
        Some("score") if a.len()==3 => diagnostics::score(&a[1..])?,
        Some("micro") if a.len()==2 => timestretch_core::processor::qa::microbench(numerics(&a[1])?)?,
        Some("device") if (5..=7).contains(&a.len()) => device::run(&a[1..])?,
        Some("callback-audit") => callback_audit::run()?,
        Some("quality") => quality::run()?,
        Some("session") if a.len()>=7 => session_perf::run(&a[1..])?,
        Some("playback") if a.len()>=7 => playback::run(&a[1..])?,
        _=>return Err("usage: timestretch-qa render ARM INPUT OUTPUT DURATION | core ARM SR CH CALLBACK TEMPO N RAW [audit] | playback ARM SR CH CALLBACK TEMPO N [RESERVE] [stress] | trace ARM INPUT JSONL DURATION | score REFERENCE TEST | micro ARM | audit-probe".into())
    };
    println!("{v}");
    if a.first().is_some_and(|s| s == "quality")
        && a.iter().any(|s| s == "--strict")
        && v.as_array().unwrap().iter().any(|r| r["pass"] != true)
    {
        return Err("retained quality gates failed".into());
    }
    Ok(())
}
