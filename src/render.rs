//! Offline rendering is a driver of the same session used by playback.
use crate::{
    AppliedParameters, Backend, PreparedSession, ProcessStatus, SessionConfig, SessionCounters,
    SessionError, SessionResult, SourceAccess, TimingInfo,
};

#[derive(Clone, Debug)]
pub struct RenderReport {
    pub source_frames: u64,
    pub output_frames: u64,
    pub source_sample_rate: u32,
    pub output_sample_rate: u32,
    pub channels: usize,
    pub backend: Backend,
    pub parameters: AppliedParameters,
    pub timing: TimingInfo,
    pub counters: SessionCounters,
    pub memory: crate::OwnedMemory,
}
#[derive(Clone, Debug)]
pub struct RenderedAudio {
    pub samples: Vec<f32>,
    pub report: RenderReport,
}

/// Render interleaved samples without normalization or a second DSP path.
pub fn render(input: &[f32], mut config: SessionConfig) -> SessionResult<RenderedAudio> {
    if !(1..=2).contains(&config.channels) || input.len() % config.channels != 0 {
        return Err(SessionError::InvalidFormat);
    }
    if input.iter().any(|v| !v.is_finite()) {
        return Err(SessionError::NonFiniteInput);
    }
    let frames = (input.len() / config.channels) as u64;
    if config.source_frames.is_some_and(|length| length != frames) {
        return Err(SessionError::InvalidFormat);
    }
    config.source_frames = Some(frames);
    config.source_access = SourceAccess::Seekable;
    let mut session = PreparedSession::new(config)?.activate();
    let channels = session.config().channels;
    let parameters = session.applied_parameters();
    let expected =
        (frames as f64 * parameters.achieved_stretch * session.config().output_sample_rate as f64
            / session.config().source_sample_rate as f64)
            .round() as usize;
    let mut samples = Vec::with_capacity(expected.saturating_mul(channels));
    let mut block = vec![0.0; 1024 * channels];
    while !session.finished() {
        let progress = session.advance(32);
        match progress.status {
            ProcessStatus::Fault => {
                return Err(session.fault().unwrap_or(SessionError::InvalidState));
            }
            ProcessStatus::NeedSource => {
                let request = session.source_request().ok_or(SessionError::InvalidState)?;
                let start = request.start_frame as usize * channels;
                let end = (start + request.frames * channels).min(input.len());
                if start == end {
                    session.finish_input()?;
                } else {
                    session.supply(request, &input[start..end])?;
                }
            }
            _ => {}
        }
        let stamp = session.read_output(&mut block)?;
        samples.extend_from_slice(&block[..stamp.frames * channels]);
        session.acknowledge(stamp.consumed(stamp.frames)?)?;
    }
    Ok(RenderedAudio {
        report: RenderReport {
            source_frames: frames,
            output_frames: (samples.len() / channels) as u64,
            source_sample_rate: session.config().source_sample_rate,
            output_sample_rate: session.config().output_sample_rate,
            channels,
            backend: session.config().backend,
            parameters,
            timing: session.timing(),
            counters: session.counters(),
            memory: session.owned_memory(),
        },
        samples,
    })
}
