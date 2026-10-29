use timestretch::{
    Backend, DurationRatio, Parameters, PitchRatio, PreparedSession, ProcessStatus, SessionConfig,
    SessionError, SourceAccess, StreamSession, TempoRate, Transport,
};

fn signal(rate: usize, seconds: f64) -> Vec<f32> {
    (0..(rate as f64 * seconds) as usize)
        .flat_map(|i| {
            let x = (i as f32 * 0.063).sin() * 0.3 + (i as f32 * 0.271).cos() * 0.1;
            [x, x * 0.7 + (i as f32 * 0.137).sin() * 0.2]
        })
        .collect()
}
fn config(rate: u32, frames: usize) -> SessionConfig {
    SessionConfig {
        source_sample_rate: rate,
        output_sample_rate: rate,
        source_frames: Some(frames as u64),
        source_access: SourceAccess::Seekable,
        backend: Backend::Portable,
        ..Default::default()
    }
}
fn drive(
    session: &mut StreamSession,
    input: &[f32],
    block: usize,
    quota: usize,
    input_chunk: usize,
) -> Vec<f32> {
    let channels = session.config().channels;
    let mut scratch = vec![0.0; block * channels];
    let mut result = Vec::new();
    for _ in 0..2_000_000 {
        let report = session.advance(quota);
        assert_ne!(report.status, ProcessStatus::Fault, "{:?}", session.fault());
        if report.status == ProcessStatus::NeedSource {
            let request = session
                .source_request()
                .expect("source demand must be satisfiable");
            let start = request.start_frame as usize * channels;
            let end = (start + request.frames.min(input_chunk) * channels).min(input.len());
            if start == end {
                session.finish_input().unwrap();
            } else {
                session.supply(request, &input[start..end]).unwrap();
            }
        }
        let stamp = session.read_output(&mut scratch).unwrap();
        if stamp.frames > 0 {
            result.extend_from_slice(&scratch[..stamp.frames * channels]);
            session
                .acknowledge(stamp.consumed(stamp.frames).unwrap())
                .unwrap();
        }
        while session.next_applied().is_some() {}
        if session.finished() {
            return result;
        }
    }
    panic!("session failed to terminate: {:?}", session.counters());
}

#[test]
fn public_ratio_bounds_are_reciprocal_and_validate_combined_factors() {
    for duration in [0.25, 0.5, 1.0, 4.0, 8.0, 16.0, 32.0] {
        let d = DurationRatio::new(duration).unwrap();
        assert_eq!(d.tempo().duration(), d);
        Parameters::keylock(d.tempo()).validate().unwrap();
    }
    for invalid in [0.0, -1.0, 1.0 / 33.0, 4.01, f64::NAN, f64::INFINITY] {
        assert!(TempoRate::new(invalid).is_err());
    }
    assert!(DurationRatio::new(32.01).is_err());
    assert!(DurationRatio::new(0.24).is_err());
    assert!(Parameters::linked(TempoRate::MIN).is_err());
    for pitch in [0.25, 0.5, 1.0, 1.123, 2.0, 4.0] {
        let pitch = PitchRatio::new(pitch).unwrap();
        let (min, max) = pitch.tempo_bounds();
        for tempo in [min, max] {
            Parameters {
                tempo,
                pitch,
                ..Default::default()
            }
            .validate()
            .unwrap();
        }
    }
    for semitones in [
        -24.0, -23.99, -23.98, -12.1, -0.01, 0.01, 0.19, 0.29, 7.13, 11.99, 23.99, 24.0,
    ] {
        let pitch = PitchRatio::from_semitones(semitones).unwrap();
        let (min, max) = pitch.tempo_bounds();
        for tempo in [min, max] {
            PreparedSession::new(SessionConfig {
                parameters: Parameters {
                    tempo,
                    pitch,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("boundary {semitones}/{tempo:?}: {e}"));
        }
    }
    let mut session = PreparedSession::new(SessionConfig::default())
        .unwrap()
        .activate();
    assert_eq!(
        session.schedule(
            1,
            0,
            Parameters {
                tempo: TempoRate::MIN,
                pitch: PitchRatio::MAX,
                ..Default::default()
            }
        ),
        Err(SessionError::InvalidParameter)
    );
    session
        .schedule(2, 0, Parameters::keylock(TempoRate::MIN))
        .unwrap();
}

#[test]
fn slow_keylock_preserves_pitch_on_both_backends() {
    for backend in [Backend::Portable, Backend::Accelerate] {
        if backend.resolve().is_err() {
            continue;
        }
        for rate in [44100, 48000, 96000] {
            for channels in [1, 2] {
                let input: Vec<f32> = (0..rate / 2)
                    .flat_map(|i| {
                        let value = (std::f64::consts::TAU * 440.0 * i as f64 / rate as f64).sin()
                            as f32
                            * 0.3;
                        // Antiphase stereo exercises linked channel processing.
                        (0..channels).map(move |ch| if ch == 0 { value } else { -value })
                    })
                    .collect();
                for duration in [8.0, 16.0, 32.0] {
                    let rendered = timestretch::render(
                        &input,
                        SessionConfig {
                            source_sample_rate: rate,
                            output_sample_rate: rate,
                            channels,
                            backend,
                            parameters: Parameters::keylock(
                                DurationRatio::new(duration).unwrap().tempo(),
                            ),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let p = rendered.report.parameters;
                    assert_eq!(p.achieved_pitch, 1.0);
                    assert_eq!(
                        rendered.samples.len(),
                        ((input.len() / channels) as f64 * p.achieved_stretch).round() as usize
                            * channels
                    );
                    assert!(rendered.samples.iter().all(|v| v.is_finite()));
                    let frames = rendered.samples.len() / channels;
                    let mut crossings = Vec::new();
                    for i in frames / 4..frames * 3 / 4 {
                        let a = rendered.samples[(i - 1) * channels] as f64;
                        let b = rendered.samples[i * channels] as f64;
                        if a < 0.0 && b >= 0.0 {
                            crossings.push(i as f64 - 1.0 - a / (b - a));
                        }
                    }
                    assert!(crossings.len() > 100, "silent or stalled output");
                    let mut periods: Vec<_> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
                    periods.sort_by(f64::total_cmp);
                    let hz = rate as f64 / periods[periods.len() / 2];
                    assert!(
                        (hz - 440.0).abs() < 1.0,
                        "{backend:?}/{rate}/{channels}/{duration}: {hz} Hz"
                    );
                    if channels == 2 {
                        let error = rendered
                            .samples
                            .chunks_exact(2)
                            .map(|f| (f[0] + f[1]).abs())
                            .fold(0.0f32, f32::max);
                        assert!(error < 1e-5, "antiphase error {error}");
                    }
                }
            }
        }
    }
}

#[test]
fn extreme_stretches_preserve_partition_eof_seek_and_control_semantics() {
    let input = signal(48000, 0.3);
    for duration in [8.0, 16.0, 32.0] {
        let mut cfg = config(48000, input.len() / 2);
        cfg.parameters = Parameters::keylock(DurationRatio::new(duration).unwrap().tempo());
        let mut baseline = PreparedSession::new(cfg.clone()).unwrap().activate();
        let expected = drive(&mut baseline, &input, 1024, 32, 2048);
        cfg.source_frames = None;
        let mut fragmented = PreparedSession::new(cfg.clone()).unwrap().activate();
        assert_eq!(drive(&mut fragmented, &input, 73, 1, 17), expected);
        assert!(fragmented.finished());
        cfg.source_frames = Some((input.len() / 2) as u64);
        let run = |block, quota, chunk| {
            let mut session = PreparedSession::new(cfg.clone()).unwrap().activate();
            session.set_transport(Transport::Paused);
            session.seek(12000).unwrap();
            assert_eq!(session.advance(1).status, ProcessStatus::Paused);
            session.set_transport(Transport::Playing);
            session
                .schedule(1, 5000, Parameters::keylock(TempoRate::new(0.5).unwrap()))
                .unwrap();
            session.schedule(2, 10000, cfg.parameters).unwrap();
            let result = drive(&mut session, &input, block, quota, chunk);
            assert_eq!(session.position(), (input.len() / 2) as f64);
            result
        };
        assert_eq!(run(64, 1, 13), run(1024, 32, 2048));
    }
}

#[test]
fn native_stream_is_partition_and_work_quota_independent() {
    for rate in [44100, 48000, 96000] {
        let input = signal(rate as usize, 0.35);
        let mut cfg = config(rate, input.len() / 2);
        cfg.parameters = Parameters::keylock(TempoRate::new(1.3).unwrap());
        let mut baseline = PreparedSession::new(cfg.clone()).unwrap().activate();
        let expected_length = (input.len() as f64 / 2.0
            * baseline.applied_parameters().achieved_stretch)
            .round() as usize;
        let expected = drive(&mut baseline, &input, 1024, 32, 2048);
        assert_eq!(expected.len(), expected_length * 2);
        assert!(expected.iter().all(|x| x.is_finite()));
        for (block, quota, input_chunk) in [(1, 1, 7), (73, 3, 13), (256, 11, 1024)] {
            let mut session = PreparedSession::new(cfg.clone()).unwrap().activate();
            let actual = drive(&mut session, &input, block, quota, input_chunk);
            assert!(
                actual == expected,
                "partition divergence at {rate}: block {block}, quota {quota}"
            );
        }
    }
}

#[test]
fn zero_short_and_unknown_length_streams_finish_without_invented_samples() {
    for frames in [0, 1, 17, 2047, 9000] {
        let input = signal(48000, frames as f64 / 48000.0);
        for tempo in [TempoRate::UNITY, TempoRate::MIN] {
            for unknown in [false, true] {
                let mut cfg = config(48000, input.len() / 2);
                cfg.parameters = Parameters::keylock(tempo);
                if unknown {
                    cfg.source_frames = None;
                }
                let mut session = PreparedSession::new(cfg).unwrap().activate();
                let expected_frames = (frames as f64
                    * session.applied_parameters().achieved_stretch)
                    .round() as usize;
                let result = drive(&mut session, &input, 73, 5, 2048);
                assert_eq!(
                    result.len(),
                    expected_frames * 2,
                    "frames {frames}, unknown {unknown}"
                );
                assert!(session.finished());
                session.finish_input().unwrap();
            }
        }
    }
}

#[test]
fn collection_does_not_advance_playhead_and_old_receipts_do_not_cross_seek() {
    let input = signal(48000, 0.5);
    let mut session = PreparedSession::new(config(48000, input.len() / 2))
        .unwrap()
        .activate();
    let mut out = [0.0; 512];
    let stamp = loop {
        let report = session.advance(32);
        if report.status == ProcessStatus::NeedSource {
            let r = session.source_request().unwrap();
            session
                .supply(
                    r,
                    &input[r.start_frame as usize * 2..(r.start_frame as usize + r.frames) * 2],
                )
                .unwrap();
        }
        let stamp = session.read_output(&mut out).unwrap();
        if stamp.frames > 0 {
            break stamp;
        }
    };
    assert_eq!(session.position(), 0.0);
    session.acknowledge(stamp.consumed(100).unwrap()).unwrap();
    assert!(session.position() > 0.0);
    session.set_transport(Transport::Paused);
    let epoch = session.seek(10000).unwrap();
    assert_ne!(epoch, stamp.epoch);
    assert_eq!(session.position(), 10000.0);
    assert_eq!(session.transport(), Transport::Paused);
    assert_eq!(session.advance(32).status, ProcessStatus::Paused);
    assert!(
        !session
            .acknowledge(stamp.consumed(stamp.frames).unwrap())
            .unwrap()
    );
    assert_eq!(session.position(), 10000.0);
    session.set_transport(Transport::Playing);
    let result = drive(&mut session, &input, 256, 7, 127);
    assert_eq!(result.len(), (input.len() / 2 - 10000) * 2);
}

#[test]
fn queued_controls_apply_at_identical_algorithm_boundaries() {
    let input = signal(48000, 0.8);
    let run = |block, quota| {
        let mut session = PreparedSession::new(config(48000, input.len() / 2))
            .unwrap()
            .activate();
        session
            .schedule(1, 3000, Parameters::keylock(TempoRate::new(0.7).unwrap()))
            .unwrap();
        session
            .schedule(
                2,
                9000,
                Parameters {
                    pitch: PitchRatio::new(1.2).unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
        drive(&mut session, &input, block, quota, 91)
    };
    assert!(run(64, 1) == run(1024, 64));
}

#[test]
fn errors_are_explicit_and_do_not_poison_valid_input() {
    assert!(
        PreparedSession::new(SessionConfig {
            channels: 3,
            ..Default::default()
        })
        .is_err()
    );
    assert!(TempoRate::new(f64::NAN).is_err());
    assert!(PitchRatio::new(0.0).is_err());
    let mut s = PreparedSession::new(SessionConfig::default())
        .unwrap()
        .activate();
    assert_eq!(s.seek(1), Err(SessionError::UnsupportedOperation));
    let r = s.source_request().unwrap();
    assert_eq!(
        s.supply(r, &[f32::NAN, 0.0]),
        Err(SessionError::NonFiniteInput)
    );
    assert_eq!(s.source_request().unwrap().start_frame, 0);
    assert!(s.supply(r, &[0.0, 0.0]).is_ok());
}

#[test]
fn converted_rates_are_partition_independent_and_report_native_rate() {
    for (source, output) in [
        (32000, 44100),
        (48000, 96000),
        (192000, 32000),
        (44100, 192000),
    ] {
        for tempo in [TempoRate::UNITY, TempoRate::MIN] {
            let input = signal(source, 0.3);
            let mut cfg = config(source as u32, input.len() / 2);
            cfg.output_sample_rate = output;
            cfg.parameters = Parameters::keylock(tempo);
            let mut a = PreparedSession::new(cfg.clone()).unwrap().activate();
            let achieved = a.applied_parameters().achieved_stretch;
            let expected = drive(&mut a, &input, 1024, 32, 2048);
            let mut b = PreparedSession::new(cfg).unwrap().activate();
            let actual = drive(&mut b, &input, 73, 1, 7);
            assert_eq!(
                actual.len(),
                (input.len() as f64 / 2.0 * achieved * output as f64 / source as f64).round()
                    as usize
                    * 2
            );
            assert!(
                actual == expected,
                "converted partition divergence: {source}/{output}"
            );
            assert!(actual.iter().all(|x| x.is_finite()));
        }
    }
}

#[test]
fn joint_ratio_boundaries_and_independent_pitch_drain() {
    for rate in [44100, 48000, 96000] {
        for tempo in [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 0.25, 1.0, 4.0] {
            for pitch in [0.25, 1.0, 4.0] {
                let input = signal(rate as usize, 0.3);
                let mut cfg = config(rate, input.len() / 2);
                cfg.parameters = Parameters {
                    tempo: TempoRate::new(tempo).unwrap(),
                    pitch: PitchRatio::new(pitch).unwrap(),
                    ..Default::default()
                };
                if !(0.25..=32.0).contains(&(pitch / tempo)) {
                    assert!(matches!(
                        PreparedSession::new(cfg),
                        Err(SessionError::InvalidParameter)
                    ));
                    continue;
                }
                let mut session = PreparedSession::new(cfg)
                    .unwrap_or_else(|e| panic!("{rate}/{tempo}/{pitch}: {e}"))
                    .activate();
                let expected = (input.len() as f64 / 2.0
                    * session.applied_parameters().achieved_stretch)
                    .round() as usize
                    * 2;
                let actual = drive(&mut session, &input, 256, 4, 2048);
                assert_eq!(actual.len(), expected, "{rate}/{tempo}/{pitch}");
                assert!(actual.iter().all(|v| v.is_finite()));
            }
        }
    }
}

#[test]
fn repeated_seek_and_prepared_replacement_preserve_pause_and_parameters() {
    let input = signal(48000, 0.6);
    let mut cfg = config(48000, input.len() / 2);
    cfg.parameters = Parameters::keylock(TempoRate::new(0.75).unwrap());
    let mut session = PreparedSession::new(cfg.clone()).unwrap().activate();
    session.set_transport(Transport::Paused);
    session.seek(13000).unwrap();
    session.seek(7000).unwrap();
    let previous = session.epoch();
    session
        .replace_prepared(PreparedSession::new(cfg.clone()).unwrap())
        .unwrap();
    assert!(session.epoch().0 > previous.0);
    assert_eq!(session.transport(), Transport::Paused);
    assert_eq!(session.position(), 7000.0);
    assert_eq!(session.applied_parameters().requested, cfg.parameters);
    session.set_transport(Transport::Playing);
    let actual = drive(&mut session, &input, 73, 3, 31);
    let mut fresh = PreparedSession::new(cfg).unwrap().activate();
    fresh.seek(7000).unwrap();
    assert_eq!(actual, drive(&mut fresh, &input, 256, 32, 2048));
    assert_eq!(session.transport(), Transport::Stopped);
}

#[test]
fn bounded_commands_and_unconsumed_output_apply_backpressure() {
    let mut cfg = config(48000, 480000);
    cfg.max_commands = 1;
    cfg.max_inflight_frames = 65536;
    let mut s = PreparedSession::new(cfg).unwrap().activate();
    s.schedule(1, 0, Parameters::default()).unwrap();
    assert_eq!(
        s.schedule(2, 0, Parameters::default()),
        Err(SessionError::Capacity)
    );
    let source = [0.0; 4096];
    let mut out = [0.0; 2048];
    let mut last = None;
    for _ in 0..10000 {
        let p = s.advance(4);
        if p.status == ProcessStatus::NeedSource {
            let r = s.source_request().unwrap();
            s.supply(r, &source[..r.frames * 2]).unwrap();
        }
        if p.status == ProcessStatus::NeedOutput {
            break;
        }
        let stamp = s.read_output(&mut out).unwrap();
        if stamp.frames > 0 {
            last = Some(stamp);
        }
    }
    assert_eq!(s.advance(1).status, ProcessStatus::NeedOutput);
    assert_eq!(s.position(), 0.0);
    let stamp = last.unwrap();
    s.acknowledge(stamp.consumed(stamp.frames).unwrap())
        .unwrap();
    assert_ne!(s.advance(1).status, ProcessStatus::NeedOutput);
    assert!(s.counters().collected_frames < 65536);
}

#[test]
fn continuous_loop_requests_wrap_and_stale_source_is_rejected() {
    let input = signal(48000, 0.2);
    let mut s = PreparedSession::new(config(48000, input.len() / 2))
        .unwrap()
        .activate();
    let stale = s.source_request().unwrap();
    s.set_loop(Some((1000, 3000))).unwrap();
    assert_eq!(s.supply(stale, &[0.0; 2]), Err(SessionError::StaleSource));
    let mut out = [0.0; 146];
    let mut supplied = 0;
    let mut consumed = 0;
    for _ in 0..30000 {
        let p = s.advance(3);
        assert_ne!(p.status, ProcessStatus::Fault);
        if p.status == ProcessStatus::NeedSource {
            let r = s.source_request().unwrap();
            assert!(r.start_frame < 3000);
            s.supply(
                r,
                &input[r.start_frame as usize * 2..(r.start_frame as usize + r.frames) * 2],
            )
            .unwrap();
            supplied += r.frames;
        }
        let stamp = s.read_output(&mut out).unwrap();
        s.acknowledge(stamp.consumed(stamp.frames).unwrap())
            .unwrap();
        consumed += stamp.frames;
        if consumed > 20000 {
            break;
        }
    }
    assert!(supplied > 20000);
    assert!(consumed > 20000);
    assert!(!s.finished());
    assert!((1000.0..3000.0).contains(&s.position()));
    s.set_loop(None).unwrap();
    assert!(s.loop_region().is_none());
    let remaining = drive(&mut s, &input, 256, 4, 2048);
    assert!(!remaining.is_empty());
}

#[test]
fn simultaneous_controls_do_not_depend_on_source_fragment_size() {
    let input = signal(48000, 0.4);
    let run = |chunk| {
        let mut s = PreparedSession::new(config(48000, input.len() / 2))
            .unwrap()
            .activate();
        for (id, tempo) in [(1, 0.8), (2, 1.4), (3, 0.6)] {
            s.schedule(id, 0, Parameters::keylock(TempoRate::new(tempo).unwrap()))
                .unwrap();
        }
        drive(&mut s, &input, 73, 3, chunk)
    };
    assert_eq!(run(7), run(2048));
}

#[test]
fn prepared_session_can_move_threads_without_backend_leakage() {
    let input = signal(48000, 0.3);
    let expected = timestretch::render(&input, config(48000, input.len() / 2)).unwrap();
    let prepared = PreparedSession::new(config(48000, input.len() / 2)).unwrap();
    let actual = std::thread::spawn(move || drive(&mut prepared.activate(), &input, 73, 3, 31))
        .join()
        .unwrap();
    assert_eq!(actual, expected.samples);
}

#[cfg(feature = "io")]
#[test]
fn audio_buffer_constructors_validate_external_audio() {
    use timestretch::{AudioBuffer, Channels};
    assert!(AudioBuffer::new(vec![0.0; 3], 48000, Channels::Stereo).is_err());
    assert!(AudioBuffer::from_mono(vec![f32::NAN], 48000).is_err());
    assert!(AudioBuffer::from_mono(vec![0.0], 0).is_err());
}

#[test]
fn finite_overload_returns_a_fault_or_finite_audio() {
    for rate in [44100, 48000, 96000] {
        for channels in [1, 2] {
            let mut source = vec![f32::MAX; 8192 * channels];
            for frame in source.chunks_exact_mut(channels).step_by(2) {
                frame.fill(-f32::MAX);
            }
            let result = timestretch::render(
                &source,
                SessionConfig {
                    backend: Backend::Portable,
                    channels,
                    source_sample_rate: rate,
                    output_sample_rate: rate,
                    ..Default::default()
                },
            );
            if let Ok(rendered) = result {
                assert!(rendered.samples.iter().all(|x| x.is_finite()));
            }
        }
    }
}
