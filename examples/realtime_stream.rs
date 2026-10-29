//! A synchronous host. Move this function onto a host-owned worker for playback;
//! enqueue stamped blocks and return the device's cumulative consumption receipts.
use timestretch::{PreparedSession, ProcessStatus, SessionConfig};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = vec![0.0; 48000];
    let mut session = PreparedSession::new(SessionConfig {
        channels: 1,
        source_sample_rate: 48000,
        output_sample_rate: 48000,
        source_frames: Some(input.len() as u64),
        ..Default::default()
    })?
    .activate();
    let mut block = [0.0; 256];
    while !session.finished() {
        match session.advance(4).status {
            ProcessStatus::NeedSource => {
                let request = session.source_request().ok_or("missing source request")?;
                let start = request.start_frame as usize;
                session.supply(request, &input[start..start + request.frames])?;
            }
            ProcessStatus::Fault => return Err(session.fault().unwrap().into()),
            _ => {}
        }
        let stamp = session.read_output(&mut block)?;
        // This example consumes immediately. A device host returns this receipt
        // after delivery, not after rendering or placing samples in its queue.
        session.acknowledge(stamp.consumed(stamp.frames)?)?;
    }
    println!("consumed {} source frames", session.position());
    Ok(())
}
