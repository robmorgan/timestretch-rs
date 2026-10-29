//! Host-owned worker and bounded delivery. Channels here are for an illustrative
//! consumer, not a realtime callback; desktop uses its preallocated SPSC queue.
use std::sync::mpsc::sync_channel;
use timestretch::{
    OutputConsumption, PreparedSession, ProcessStatus, SessionConfig, SourceAccess, Transport,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = vec![0.0f32; 48000];
    let prepared = PreparedSession::new(SessionConfig {
        channels: 1,
        source_sample_rate: 48000,
        output_sample_rate: 48000,
        source_frames: Some(input.len() as u64),
        source_access: SourceAccess::Seekable,
        ..Default::default()
    })?;
    let (send_audio, receive_audio) = sync_channel(2);
    let (send_receipt, receive_receipt) = sync_channel::<OutputConsumption>(2);
    let worker = std::thread::spawn(move || -> Result<(), timestretch::SessionError> {
        let mut s = prepared.activate();
        // A paused seek is a library transaction, even on a host worker.
        s.set_transport(Transport::Paused);
        s.seek(12000)?;
        s.set_transport(Transport::Playing);
        let mut block = [0.0; 256];
        while !s.finished() {
            if s.advance(4).status == ProcessStatus::Fault {
                return Err(s.fault().unwrap());
            }
            if let Some(r) = s.source_request() {
                s.supply(
                    r,
                    &input[r.start_frame as usize..r.start_frame as usize + r.frames],
                )?;
            }
            let stamp = s.read_output(&mut block)?;
            if stamp.frames > 0 {
                if send_audio.send((block, stamp)).is_err() {
                    break;
                }
                let Ok(receipt) = receive_receipt.recv() else {
                    break;
                };
                s.acknowledge(receipt)?;
            }
        }
        Ok(())
    });
    let mut delivered = 0;
    while let Ok((samples, stamp)) = receive_audio.recv() {
        // A real host delivers these frames before returning this receipt.
        std::hint::black_box(&samples[..stamp.frames]);
        delivered += stamp.frames;
        send_receipt.send(stamp.consumed(stamp.frames)?)?;
    }
    worker.join().map_err(|_| "worker panicked")??;
    println!("delivered {delivered} frames after seeking");
    Ok(())
}
