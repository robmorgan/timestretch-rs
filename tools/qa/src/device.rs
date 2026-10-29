//! Muted physical-device soak using the actual desktop delivery and render pump.
use super::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;
use timestretch::{PreparedSession, SourceAccess, TempoRate};
pub(super) fn run(a: &[String]) -> Result<Value> {
    let backend = backend(&a[0])?;
    let source_rate: u32 = a[1].parse()?;
    let cb: u32 = a[2].parse()?;
    let seconds: u64 = a[3].parse()?;
    let audio_priority = !a.get(5).is_some_and(|s| s == "normal");
    if !(1..=120).contains(&seconds) || ![64, 256, 1024].contains(&cb) {
        return Err("device: seconds 1..120; callbacks 64/256/1024".into());
    }
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let rate = device.default_output_config()?.sample_rate().0;
    let reserve: usize = a
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or_else(|| render_worker::output_reserve_frames(rate));
    if !(1024..=16384).contains(&reserve) {
        return Err("device: reserve 1024..16384 frames".into());
    }
    let source: Arc<Vec<f32>> = Arc::new(
        (0..source_rate as usize * 2)
            .flat_map(|i| {
                let v = (i as f32 * 0.031).sin() * 0.2;
                [v, v * 0.7]
            })
            .collect(),
    );
    let mut session = PreparedSession::new(SessionConfig {
        backend,
        channels: 2,
        source_sample_rate: source_rate,
        output_sample_rate: rate,
        source_frames: Some((source.len() / 2) as u64),
        source_access: SourceAccess::Seekable,
        ..Default::default()
    })?
    .activate();
    session.set_loop(Some((0, (source.len() / 2) as u64)))?;
    let (mut producer, consumer) = output_queue::channel(32768);
    let _keep = consumer.keep_alive();
    producer.publish_epoch(session.epoch());
    let ready = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let command = Arc::new(AtomicUsize::new(0));
    let halt = stop.clone();
    let primed = ready.clone();
    let cmds = command.clone();
    let worker_source = source.clone();
    let worker = std::thread::spawn(move || {
        let mut priority =
            audio_priority.then(|| worker_priority::AudioPriority::new(rate, reserve));
        let mut scheduling_error = priority.as_ref().and_then(|p| p.as_ref().err()).cloned();
        let mut scratch = [0.0; 2048];
        let mut last = 0;
        let mut error = None;
        let mut worst = 0u64;
        let mut wake_gap = 0u64;
        let mut slept_at = Instant::now();
        let mut max_pending = 0usize;
        while !halt.load(Ordering::Acquire) {
            wake_gap = wake_gap.max(slept_at.elapsed().as_nanos() as u64);
            if let Some(receipt) = producer.receipt() {
                if let Err(e) = session.acknowledge(receipt) {
                    error = Some(e);
                    break;
                }
            }
            let c = cmds.load(Ordering::Acquire);
            if c != last {
                let p = if c % 2 == 0 {
                    Parameters::keylock(TempoRate::new(0.5).unwrap())
                } else {
                    Parameters::linked(TempoRate::new(2.0).unwrap()).unwrap()
                };
                if let Err(e) = session.schedule(c as u64, session.commit_horizon(), p) {
                    error = Some(e);
                    break;
                }
                last = c;
            }
            let t = Instant::now();
            let result = render_worker::fill_reserve(
                &mut session,
                &worker_source,
                &mut producer,
                &mut scratch,
                reserve,
                8,
            );
            worst = worst.max(t.elapsed().as_nanos() as u64);
            if let Err(e) = result {
                error = Some(e);
                break;
            }
            max_pending = max_pending.max(session.output_ready());
            if producer.occupied_frames() >= reserve {
                primed.store(true, Ordering::Release);
            }
            slept_at = Instant::now();
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(Ok(guard)) = priority.as_mut() {
            if let Err(e) = guard.release() {
                scheduling_error = Some(e);
            }
        }
        (
            producer.underrun_frames(),
            worst,
            wake_gap,
            max_pending,
            error,
            scheduling_error,
        )
    });
    let mut delivery = delivery::Delivery::new(
        rate,
        consumer,
        Arc::new(state::AtomicVolume::new(0.0)),
        ready.clone(),
        Arc::new(state::ScrubState::new()),
        source,
    );
    let capacity = (seconds as usize + 2) * rate as usize / 32;
    let times: Arc<Vec<AtomicU64>> = Arc::new((0..capacity).map(|_| AtomicU64::new(0)).collect());
    let frames: Arc<Vec<AtomicU64>> = Arc::new((0..capacity).map(|_| AtomicU64::new(0)).collect());
    let count = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let ts = times.clone();
    let sizes = frames.clone();
    let calls = count.clone();
    let errs = errors.clone();
    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(rate),
        buffer_size: cpal::BufferSize::Fixed(cb),
    };
    let built = device.build_output_stream(
        &config,
        move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let t = Instant::now();
            delivery.render(out);
            let ns = t.elapsed().as_nanos() as u64;
            let i = calls.fetch_add(1, Ordering::Relaxed);
            if i < ts.len() {
                ts[i].store(ns, Ordering::Relaxed);
                sizes[i].store((out.len() / 2) as u64, Ordering::Relaxed);
            }
        },
        move |_| {
            errs.fetch_add(1, Ordering::Relaxed);
        },
        None,
    );
    let stream = match built {
        Ok(s) => s,
        Err(e) => {
            stop.store(true, Ordering::Release);
            let _ = worker.join();
            return Err(e.into());
        }
    };
    let start = Instant::now();
    while !ready.load(Ordering::Acquire) && start.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(1));
    }
    if !ready.load(Ordering::Acquire) {
        stop.store(true, Ordering::Release);
        drop(stream);
        let (_, _, _, _, error, _) = worker.join().map_err(|_| "device worker panicked")?;
        return Err(format!("device priming failed: {error:?}").into());
    }
    if let Err(e) = stream.play() {
        stop.store(true, Ordering::Release);
        drop(stream);
        let _ = worker.join();
        return Err(e.into());
    }
    let start = Instant::now();
    let mut next = 1;
    while start.elapsed() < Duration::from_secs(seconds) {
        if start.elapsed() >= Duration::from_millis(next * 500) {
            command.fetch_add(1, Ordering::Release);
            next += 1;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let paused = stream.pause();
    drop(stream);
    stop.store(true, Ordering::Release);
    let (underruns, worst, wake_gap, max_pending, error, scheduling_error) =
        worker.join().map_err(|_| "device worker panicked")?;
    paused?;
    let n = count.load(Ordering::Relaxed).min(capacity);
    let measured: Vec<_> = times[..n]
        .iter()
        .map(|x| x.load(Ordering::Relaxed))
        .collect();
    let sizes: Vec<_> = frames[..n]
        .iter()
        .map(|x| x.load(Ordering::Relaxed))
        .collect();
    let mut ratios: Vec<_> = measured
        .iter()
        .zip(&sizes)
        .map(|(&ns, &f)| ns as f64 / (f as f64 / rate as f64 * 1e9))
        .collect();
    ratios.sort_by(f64::total_cmp);
    Ok(
        json!({"backend":format!("{backend:?}"),"source_rate":source_rate,"device_rate":rate,"device":device.name()?,"requested_callback":cb,
        "seconds":seconds,"callbacks":n,"callback_ns":measured,"callback_frames":sizes,"deadline_misses":ratios.iter().filter(|&&r|r>1.0).count(),
        "p999_budget_ratio":if n>=10000{Some(ratios[(n-1)*999/1000])}else{None},"max_budget_ratio":ratios.last(),
        "reserve_frames":reserve,"reserve_ms":1000.0*reserve as f64/rate as f64,"worker_sleep_ms":1,
        "worker_scheduling":if audio_priority {worker_priority::AudioPriority::label()} else {"platform default (diagnostic)"},"worker_scheduling_error":scheduling_error,
        "queue_underruns":underruns,"worker_max_ns":worst,"worker_max_sleep_ns":wake_gap,"max_pending_session_frames":max_pending,"worker_error":error.map(|e|e.to_string()),"device_errors":errors.load(Ordering::Relaxed),
        "muted":true,"hardware_xruns":null,"scope":"physical CPAL output at unchanged default device rate; actual desktop Delivery and render pump; tempo/linked-mode rides; no loopback capture"}),
    )
}
