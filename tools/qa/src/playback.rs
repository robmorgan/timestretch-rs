use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
use timestretch::{PitchRatio, PreparedSession, SourceAccess, TempoRate, TransitionEnvelope};

// Valid corners/edges of the joint domain, followed by a seek to exercise
// warmup after independent pitch and linked-mode changes.
const BOUNDARY_RATIOS: [(f64, f64); 12] = [
    (1.0 / 32.0, 1.0),
    (1.0 / 32.0, 0.25),
    (1.0 / 16.0, 2.0),
    (1.0 / 8.0, 4.0),
    (0.25, 0.25),
    (0.25, 1.0),
    (0.25, 4.0),
    (1.0, 0.25),
    (1.0, 4.0),
    (4.0, 1.0),
    (4.0, 4.0),
    (1.0, 1.0),
];

/// Real wall-clock callbacks and the actual desktop render pump/queue. No file
/// I/O, source generation or scoring is included in timed work.
pub(super) fn run(a: &[String]) -> Result<Value> {
    let arm = &a[0];
    let sr: u32 = a[1].parse()?;
    let ch: usize = a[2].parse()?;
    let cb: usize = a[3].parse()?;
    let tempo: f64 = a[4].parse()?;
    let n: usize = a[5].parse()?;
    let reserve: usize = a
        .get(6)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or_else(|| render_worker::output_reserve_frames(sr));
    let boundary_stress = a.get(7).is_some_and(|s| s == "boundary");
    let stress = boundary_stress || a.get(7).is_some_and(|s| s == "stress");
    if n == 0
        || cb == 0
        || cb > 8192
        || !(1..=2).contains(&ch)
        || !(1024..=16384).contains(&reserve)
    {
        return Err("invalid playback workload".into());
    }
    let source: Vec<f32> = (0..sr as usize * 2)
        .flat_map(|i| {
            (0..ch).map(move |c| {
                (i as f32 * (0.031 + c as f32 * 0.013)).sin() * 0.2 + (i as f32 * 0.173).cos() * 0.1
            })
        })
        .collect();
    let mut times = vec![0u64; n];
    let mut wake_lateness = vec![0u64; n];
    let mut output = vec![0.0; cb * ch];
    let mut worker_scratch = vec![0.0; 1024 * ch];
    let duration = n as f64 * cb as f64 / sr as f64;
    let mut worker_times = Vec::with_capacity((duration * 2200.0) as usize + 2048);
    allocator::init();
    let before = allocator::live();
    let init = Instant::now();
    let mut session = PreparedSession::new(SessionConfig {
        backend: backend(arm)?,
        channels: ch,
        source_sample_rate: sr,
        output_sample_rate: sr,
        source_frames: Some((source.len() / ch) as u64),
        source_access: SourceAccess::Seekable,
        parameters: Parameters::keylock(TempoRate::new(tempo)?),
        ..Default::default()
    })?
    .activate();
    session.set_loop(Some((0, (source.len() / ch) as u64)))?;
    let init_ns = init.elapsed().as_nanos() as u64;
    let session_bytes = allocator::live() - before;
    let before = allocator::live();
    let (mut producer, mut consumer) = output_queue::channel_with_channels(32768, ch);
    let queue_bytes = allocator::live() - before;
    producer.publish_epoch(session.epoch());
    let stop = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(AtomicBool::new(false));
    let command = Arc::new(AtomicUsize::new(0));
    let cmd = command.clone();
    let halt = stop.clone();
    let primed = ready.clone();
    let epoch = session.epoch();
    let worker_start = Instant::now();
    let worker = std::thread::spawn(move || {
        let mut priority = worker_priority::AudioPriority::new(sr, reserve);
        let mut scheduling_error = priority.as_ref().err().cloned();
        let mut seen = 0;
        let mut error = None;
        let mut truncated = 0;
        let mut prime_ns = 0;
        while !halt.load(Ordering::Acquire) {
            if let Some(receipt) = producer.receipt() {
                if let Err(e) = session.acknowledge(receipt) {
                    error = Some(e);
                    break;
                }
            }
            let c = cmd.load(Ordering::Acquire);
            if c != seen {
                let result =
                    if c % (if boundary_stress {
                        BOUNDARY_RATIOS.len() + 1
                    } else {
                        3
                    }) == 0
                    {
                        primed.store(false, Ordering::Release);
                        session
                            .seek((c as u64 * 997) % sr as u64)
                            .map(|epoch| producer.publish_epoch(epoch))
                    } else if boundary_stress {
                        let (tempo, pitch) = BOUNDARY_RATIOS[(c - 1) % (BOUNDARY_RATIOS.len() + 1)];
                        session.schedule(
                            c as u64,
                            session.commit_horizon(),
                            Parameters {
                                tempo: TempoRate::new(tempo).unwrap(),
                                pitch: PitchRatio::new(pitch).unwrap(),
                                ..Default::default()
                            },
                        )
                    } else {
                        let rate = if c % 2 == 0 { 0.5 } else { 2.0 };
                        let p = if c % 3 == 1 {
                            Parameters::keylock(TempoRate::new(rate).unwrap())
                        } else {
                            Parameters::linked(TempoRate::new(rate).unwrap()).unwrap()
                        };
                        session.schedule(c as u64, session.commit_horizon(), p)
                    };
                if let Err(e) = result {
                    error = Some(e);
                    break;
                }
                seen = c;
            }
            let t = Instant::now();
            let r = render_worker::fill_reserve(
                &mut session,
                &source,
                &mut producer,
                &mut worker_scratch,
                reserve,
                8,
            );
            let elapsed = t.elapsed().as_nanos() as u64;
            if worker_times.len() < worker_times.capacity() {
                worker_times.push(elapsed);
            } else {
                truncated += 1;
            }
            if let Err(e) = r {
                error = Some(e);
                break;
            }
            if producer.occupied_frames() >= reserve {
                if prime_ns == 0 {
                    prime_ns = worker_start.elapsed().as_nanos() as u64;
                }
                primed.store(true, Ordering::Release);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Ok(guard) = priority.as_mut() {
            if let Err(e) = guard.release() {
                scheduling_error = Some(e);
            }
        }
        (
            producer,
            worker_times,
            error,
            truncated,
            prime_ns,
            session.counters(),
            scheduling_error,
        )
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ready.load(Ordering::Acquire) && Instant::now() < deadline {
        consumer.sync_epoch();
        std::thread::sleep(Duration::from_micros(100));
    }
    if !ready.load(Ordering::Acquire) {
        stop.store(true, Ordering::Release);
        let _ = worker.join();
        return Err("worker did not prime".into());
    }
    let mut transition = TransitionEnvelope::default();
    let mut last = [0.0; 2];
    // This is a synthetic device clock, not a CPAL callback. Give its thread
    // the declared audio policy before timing; general scheduler stalls in the
    // clock driver must remain a separate diagnostic experiment.
    let mut callback_priority = worker_priority::AudioPriority::new(sr, cb);
    let mut callback_scheduling_error = callback_priority.as_ref().err().cloned();
    let period = Duration::from_secs_f64(cb as f64 / sr as f64);
    let clock = Instant::now();
    let mut shortages = 0;
    let mut transition_silence = 0;
    let mut finite = true;
    let mut min_ready = usize::MAX;
    for i in 0..n + 64 {
        let when = clock + period.mul_f64(i as f64);
        loop {
            let now = Instant::now();
            if now >= when {
                break;
            }
            let wait = when - now;
            if wait > Duration::from_micros(150) {
                std::thread::sleep(wait - Duration::from_micros(100));
            } else {
                std::hint::spin_loop();
            }
        }
        if i == 64 {
            allocator::begin();
        }
        if i >= 64 {
            wake_lateness[i - 64] =
                Instant::now().saturating_duration_since(when).as_nanos() as u64;
        }
        if stress && i >= 64 && (i - 64) % 500 == 0 {
            command.fetch_add(1, Ordering::Release);
        }
        let t = Instant::now();
        if consumer.sync_epoch() {
            transition.begin(last);
        }
        let available = if ready.load(Ordering::Acquire) {
            let got = consumer.read(&mut output);
            consumer.acknowledge_frames(got);
            got
        } else {
            output.fill(0.0);
            if i >= 64 {
                transition_silence += cb;
            }
            cb
        };
        transition.apply(&mut output, ch);
        for c in 0..ch {
            last[c] = output[output.len() - ch + c];
        }
        let elapsed = t.elapsed().as_nanos() as u64;
        if i >= 64 {
            times[i - 64] = elapsed;
            shortages += cb - available;
            min_ready = min_ready.min(available);
        }
        finite &= output.iter().all(|x| x.is_finite());
        std::hint::black_box(&output);
    }
    let allocations = allocator::end();
    let system_allocations = allocator::system_count();
    if let Ok(guard) = callback_priority.as_mut() {
        if let Err(e) = guard.release() {
            callback_scheduling_error = Some(e);
        }
    }
    stop.store(true, Ordering::Release);
    let (producer, worker_times, error, truncated, prime_ns, counters, scheduling_error) =
        worker.join().map_err(|_| "worker panicked")?;
    let budget = cb as f64 / sr as f64 * 1e9;
    let mut sorted = times.clone();
    sorted.sort_unstable();
    let pct = |p: f64| sorted[((n - 1) as f64 * p).ceil() as usize];
    let mut worker_sorted = worker_times.clone();
    worker_sorted.sort_unstable();
    let total: u64 = times.iter().sum();
    Ok(
        json!({"arm":arm,"sample_rate":sr,"channels":ch,"callback":cb,"tempo":tempo,"callbacks":n,
        "warmup":64,"reserve_frames":reserve,"reserve_ms":1000.0*reserve as f64/sr as f64,"work_quota":8,"worker_sleep_ms":1,
        "stress":stress,"boundary_stress":boundary_stress,"boundary_tempo_pitch":if boundary_stress {Some(BOUNDARY_RATIOS)} else {None},"init_ns":init_ns,"initial_prime_ns":prime_ns,"session_owned_rust_bytes":session_bytes,"queue_owned_rust_bytes":queue_bytes,
        "worker_scheduling":worker_priority::AudioPriority::label(),"worker_scheduling_error":scheduling_error,
        "callback_scheduling":worker_priority::AudioPriority::label(),"callback_scheduling_error":callback_scheduling_error,
        "opaque_system_plan_bytes":null,"allocations":allocations,"system_allocations":system_allocations,"finite":finite,
        "underruns":shortages,"queue_underruns_including_warmup":producer.underrun_frames(),"transition_silence_frames":transition_silence,
        "callback_ns":times,"worker_ns":worker_times,"worker_samples_truncated":truncated,"wake_lateness_ns":wake_lateness,
        "budget_ns":budget,"median_ns":pct(0.5),"p95_ns":pct(0.95),"p99_ns":pct(0.99),"p999_ns":if n>=10000 {Some(pct(0.999))}else{None},
        "max_ns":sorted.last(),"deadline_misses":times.iter().filter(|&&x|x as f64>budget).count(),
        "frames_per_callback_cpu_second":(n*cb) as f64/(total as f64/1e9),"min_delivered_frames":min_ready,
        "worker_max_ns":worker_sorted.last(),"worker_error":error.map(|e|format!("{e}")),"epoch_start":epoch.0,
        "consumed_frames":counters.consumed_frames,"scope":"desktop render pump + SPSC queue + delivery; synthetic device clock with recorded audio scheduling policy; no physical device"}),
    )
}
