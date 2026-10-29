//! The desktop's bounded render pump, shared with the complete playback harness.
use crate::output_queue::Producer;
use timestretch::{ProcessStatus, SessionError, StreamSession};

/// Additional host reserve, separate from algorithm lookahead and device
/// latency. The 50 ms target covers the measured worker wake/burst stalls;
/// qualification still records every shortage rather than masking one.
pub fn output_reserve_frames(output_rate: u32) -> usize {
    (output_rate as usize).div_ceil(20)
}

/// Fill the application's reserve. All source demand and media semantics come
/// from the session. The host chooses this reserve, work quota and wakeups.
pub fn fill_reserve(
    session: &mut StreamSession,
    source: &[f32],
    output: &mut Producer,
    scratch: &mut [f32],
    reserve: usize,
    quota: usize,
) -> Result<(), SessionError> {
    let ch = session.config().channels;
    for _ in 0..128 {
        let occupied = output.occupied_frames();
        if occupied >= reserve {
            break;
        }
        // Publish completed media before starting another indivisible DSP
        // operation. Otherwise a pending block can sit behind unrelated work,
        // starving delivery and unnecessarily extending the control horizon.
        let capacity = output
            .free_frames()
            .min(reserve - occupied)
            .min(scratch.len() / ch);
        if capacity == 0 {
            break;
        }
        if session.output_ready() > 0 {
            let stamp = session.read_output(&mut scratch[..capacity * ch])?;
            if !output.push(&scratch[..stamp.frames * ch], stamp) {
                return Err(SessionError::InvalidState);
            }
            continue;
        }
        if session.rendered_finished() {
            output.finish();
            break;
        }
        let progress = session.advance(quota);
        if progress.status == ProcessStatus::Fault {
            return Err(session.fault().unwrap_or(SessionError::InvalidState));
        }
        if progress.status == ProcessStatus::NeedSource {
            let r = session.source_request().ok_or(SessionError::InvalidState)?;
            let start = (r.start_frame as usize)
                .checked_mul(ch)
                .ok_or(SessionError::InvalidFormat)?;
            let end = (start + r.frames * ch).min(source.len());
            if start > end {
                return Err(SessionError::InvalidFormat);
            }
            if start == end {
                session.finish_input()?;
            } else {
                session.supply(r, &source[start..end])?;
            }
        }
        while session.next_applied().is_some() {}
        if session.rendered_finished() && session.output_ready() == 0 {
            output.finish();
            break;
        }
        if progress.status == ProcessStatus::Paused {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output_queue;
    use timestretch::{
        Backend, Parameters, PreparedSession, SessionConfig, SourceAccess, TempoRate, Transport,
    };
    #[test]
    fn pending_media_fills_reserve_without_advancing_the_control_horizon() {
        let source: Vec<_> = (0..48000).map(|i| (i as f32 * 0.031).sin() * 0.2).collect();
        let mut session = PreparedSession::new(SessionConfig {
            channels: 1,
            backend: Backend::Portable,
            source_frames: Some(source.len() as u64),
            ..Default::default()
        })
        .unwrap()
        .activate();
        while session.output_ready() < 128 {
            let report = session.advance(8);
            assert_ne!(report.status, ProcessStatus::Fault);
            if let Some(r) = session.source_request() {
                let start = r.start_frame as usize;
                session.supply(r, &source[start..start + r.frames]).unwrap();
            }
        }
        let ready = session.output_ready();
        let horizon = session.commit_horizon();
        let (mut producer, _consumer) = output_queue::channel_with_channels(2048, 1);
        let mut scratch = [0.0; 257];
        fill_reserve(&mut session, &source, &mut producer, &mut scratch, 128, 32).unwrap();
        assert_eq!(producer.occupied_frames(), 128);
        assert_eq!(session.output_ready(), ready - 128);
        assert_eq!(session.commit_horizon(), horizon);
    }
    // Fake device clock: delayed wakes, paused delivery, tiny partial receipts,
    // backpressure and output partitions vary, while explicit media commands do not.
    #[test]
    fn fake_clock_preserves_samples_and_consumed_source_mapping() {
        let source: Vec<_> = (0..48000).map(|i| (i as f32 * 0.031).sin() * 0.2).collect();
        let run = |reserve, quota, chunk, delayed| {
            let mut s = PreparedSession::new(SessionConfig {
                channels: 1,
                backend: Backend::Portable,
                source_sample_rate: 48000,
                output_sample_rate: 48000,
                source_frames: Some(source.len() as u64),
                source_access: SourceAccess::Seekable,
                ..Default::default()
            })
            .unwrap()
            .activate();
            s.set_transport(Transport::Paused);
            s.seek(3000).unwrap();
            for (id, at, tempo) in [(1, 0, 0.8), (2, 10000, 1.4), (3, 20000, 0.6)] {
                s.schedule(id, at, Parameters::keylock(TempoRate::new(tempo).unwrap()))
                    .unwrap();
            }
            let (mut producer, mut consumer) = output_queue::channel_with_channels(2048, 1);
            producer.publish_epoch(s.epoch());
            consumer.sync_epoch();
            s.set_transport(Transport::Playing);
            let mut scratch = [0.0; 257];
            let mut block = vec![0.0; chunk];
            let mut samples = Vec::new();
            let mut positions = Vec::new();
            for tick in 0..100000 {
                let pause = delayed && tick % 37 < 3;
                s.set_transport(if pause {
                    Transport::Paused
                } else {
                    Transport::Playing
                });
                if !pause && (!delayed || tick % 5 != 0) {
                    fill_reserve(&mut s, &source, &mut producer, &mut scratch, reserve, quota)
                        .unwrap();
                }
                if !pause {
                    let n = consumer.read(&mut block);
                    samples.extend_from_slice(&block[..n]);
                    for _ in 0..n {
                        consumer.acknowledge_frames(1);
                        s.acknowledge(producer.receipt().unwrap()).unwrap();
                        positions.push(s.position());
                    }
                }
                if s.finished() {
                    break;
                }
            }
            assert!(s.finished());
            (samples, positions)
        };
        let baseline = run(1024, 32, 64, false);
        assert_eq!(baseline, run(2048, 1, 73, true));
        assert_eq!(baseline, run(256, 3, 17, true));
    }
}
