use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use timestretch::{Epoch, OutputStamp};
pub(super) fn run() -> Result<Value> {
    allocator::init();
    let mut rows = Vec::new();
    for rate in [44100, 48000, 96000] {
        let source = Arc::new(vec![0.2; rate as usize * 8]);
        let (mut producer, consumer) = output_queue::channel(32768);
        let _keep = consumer.keep_alive();
        let active = Arc::new(AtomicBool::new(true));
        let volume = Arc::new(state::AtomicVolume::new(0.5));
        let scrub = Arc::new(state::ScrubState::new());
        let mut delivery = delivery::Delivery::new(
            rate,
            consumer,
            volume.clone(),
            active.clone(),
            scrub.clone(),
            source,
        );
        let samples = [0.2; 65536];
        let mut out = [0.0; 8192];
        let mut epoch = Epoch(1);
        let mut next = 0;
        let mut worst_ratio = 0.0f64;
        let mut maximum_ns = 0;
        let mut finite = true;
        allocator::begin();
        for i in 0..1200 {
            match i % 160 {
                0 => active.store(false, Ordering::Release),
                10 => active.store(true, Ordering::Release),
                70 => scrub.begin(3000.0),
                75 => scrub.update_target(80000.0),
                80 => scrub.release(1.0),
                100 => scrub.cancel(),
                110 => {
                    epoch.0 += 1;
                    producer.publish_epoch(epoch);
                    next = 0;
                }
                _ => {}
            }
            let n = producer.free_frames();
            assert!(producer.push(
                &samples[..n * 2],
                OutputStamp {
                    epoch,
                    start_frame: next,
                    frames: n
                }
            ));
            next += n as u64;
            let frames = [64, 256, 1024, 4096][i % 4];
            let t = Instant::now();
            delivery.render(&mut out[..frames * 2]);
            let ns = t.elapsed().as_nanos() as u64;
            maximum_ns = maximum_ns.max(ns);
            worst_ratio = worst_ratio.max(ns as f64 / (frames as f64 / rate as f64 * 1e9));
            finite &= out[..frames * 2].iter().all(|x| x.is_finite());
        }
        let allocations = allocator::end();
        let deallocations = allocator::deallocations();
        let system = allocator::system_count();
        rows.push(json!({"sample_rate":rate,"allocations":allocations,"deallocations":deallocations,"system_allocations":system,
            "max_callback_ns":maximum_ns,"max_callback_budget_ratio":worst_ratio,"finite":finite,
            "scope":"actual Delivery::render: startup, pause, seek epoch, scrub, release, oversized callbacks; intentional starvation included"}));
        if allocations != 0 || deallocations != 0 || !finite || system.is_some_and(|n| n != 0) {
            return Err(format!("callback audit failed: {}", rows.last().unwrap()).into());
        }
    }
    Ok(json!(rows))
}
