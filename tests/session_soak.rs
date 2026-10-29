use timestretch::*;
/// Deterministic event campaign. The seed is printed and can be replayed in CI.
fn campaign(events: usize) {
    let mut seed = std::env::var("TIMESTRETCH_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(20261028);
    eprintln!("session campaign seed={seed}");
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let source: Vec<_> = (0..48000 * 2)
        .map(|i| (i as f32 * 0.03).sin() * 0.2)
        .collect();
    let mut s = PreparedSession::new(SessionConfig {
        backend: Backend::Portable,
        source_sample_rate: 48000,
        output_sample_rate: 48000,
        source_frames: Some(48000),
        source_access: SourceAccess::Seekable,
        ..Default::default()
    })
    .unwrap()
    .activate();
    let mut out = [0.0; 1026];
    let mut prior = None;
    for event in 0..events {
        let value = random();
        match value % 31 {
            0 => {
                s.set_loop(None).unwrap();
                s.seek(value / 31 % 48001).unwrap();
            }
            1 => {
                s.set_transport(Transport::Paused);
            }
            2 => {
                s.set_transport(Transport::Playing);
            }
            3 => {
                s.set_loop(Some((1000, 13000))).unwrap();
            }
            4 => {
                if let Some(receipt) = prior {
                    let _ = s.acknowledge(receipt).unwrap();
                }
            }
            5 => {
                let tempo = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 0.25, 0.5, 1.0, 2.0, 4.0]
                    [(value / 31 % 8) as usize];
                let _ = s.schedule(
                    event as u64,
                    s.commit_horizon() + 100,
                    Parameters::keylock(TempoRate::new(tempo).unwrap()),
                );
            }
            _ => {}
        }
        let p = s.advance((value % 13 + 1) as usize);
        assert_ne!(p.status, ProcessStatus::Fault, "{:?}", s.fault());
        if p.status == ProcessStatus::NeedSource {
            let r = s.source_request().unwrap();
            let n = r.frames.min((value % 127 + 1) as usize);
            s.supply(
                r,
                &source[r.start_frame as usize * 2..(r.start_frame as usize + n) * 2],
            )
            .unwrap();
        }
        while s.next_applied().is_some() {}
        let n = (value % 513 + 1) as usize;
        let stamp = s.read_output(&mut out[..n * 2]).unwrap();
        assert!(out[..stamp.frames * 2].iter().all(|x| x.is_finite()));
        let receipt = stamp.consumed(stamp.frames).unwrap();
        s.acknowledge(receipt).unwrap();
        prior = Some(receipt);
        assert!(s.position() >= 0.0 && s.position() <= 48000.0);
        assert!(
            s.timing().pending_output_frames + s.timing().unacknowledged_output_frames
                <= s.config().max_inflight_frames
        );
    }
}
#[test]
fn seeded_transport_campaign() {
    campaign(5000);
}
#[test]
#[ignore = "extended randomized transport campaign"]
fn extended_transport_campaign() {
    campaign(1_000_000);
}
