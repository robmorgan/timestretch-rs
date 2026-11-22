//! JavaScript bindings for the `timestretch` session protocol.
//!
//! The classes mirror `timestretch::StreamSession` one-to-one; the host page
//! still owns scheduling, queues and the audio device. Frame counts and epochs
//! cross the boundary as JS numbers (exact up to 2^53; the library limit is
//! 2^48 frames). `wasm-bindgen` generates the unsafe glue, which is why this
//! lives outside the library crates that forbid unsafe code.
use timestretch as ts;
use wasm_bindgen::prelude::*;

fn err(e: ts::SessionError) -> JsError {
    JsError::new(&format!("{e:?}"))
}

/// A non-negative integer carried in a JS number.
fn frames(value: f64) -> Result<u64, JsError> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= (1u64 << 53) as f64 {
        Ok(value as u64)
    } else {
        Err(err(ts::SessionError::InvalidParameter))
    }
}

fn parameters(tempo: f64, pitch: f64) -> Result<ts::Parameters, JsError> {
    let parameters = ts::Parameters {
        tempo: ts::TempoRate::new(tempo).map_err(err)?,
        pitch: ts::PitchRatio::new(pitch).map_err(err)?,
        ..Default::default()
    };
    parameters.validate().map_err(err)?;
    Ok(parameters)
}

/// Session configuration. `new SessionOptions()` holds the library defaults:
/// 44.1 kHz stereo, unity tempo and pitch, sequential source of unknown length.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct SessionOptions {
    #[wasm_bindgen(js_name = sourceSampleRate)]
    pub source_sample_rate: u32,
    #[wasm_bindgen(js_name = outputSampleRate)]
    pub output_sample_rate: u32,
    pub channels: u32,
    /// Source time / output time, 1/32 to 4.
    pub tempo: f64,
    /// Frequency multiplier, 0.25 to 4. `pitch / tempo` must be in 0.25..=32.
    pub pitch: f64,
    /// Required for `seek` and `setLoop`.
    pub seekable: bool,
    /// Known source length; EOF then follows the last frame automatically.
    #[wasm_bindgen(js_name = sourceFrames)]
    pub source_frames: Option<f64>,
    /// Bound on pending plus unacknowledged output, 65,536 to 1,048,576.
    #[wasm_bindgen(js_name = maxInflightFrames)]
    pub max_inflight_frames: u32,
    /// Bound on scheduled controls, 1 to 4,096.
    #[wasm_bindgen(js_name = maxCommands)]
    pub max_commands: u32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        let d = ts::SessionConfig::default();
        Self {
            source_sample_rate: d.source_sample_rate,
            output_sample_rate: d.output_sample_rate,
            channels: d.channels as u32,
            tempo: 1.0,
            pitch: 1.0,
            seekable: false,
            source_frames: None,
            max_inflight_frames: d.max_inflight_frames as u32,
            max_commands: d.max_commands as u32,
        }
    }
}

#[wasm_bindgen]
impl SessionOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionOptions {
    fn config(&self) -> Result<ts::SessionConfig, JsError> {
        Ok(ts::SessionConfig {
            source_sample_rate: self.source_sample_rate,
            output_sample_rate: self.output_sample_rate,
            channels: self.channels as usize,
            source_access: if self.seekable {
                ts::SourceAccess::Seekable
            } else {
                ts::SourceAccess::Sequential
            },
            source_frames: self.source_frames.map(frames).transpose()?,
            parameters: parameters(self.tempo, self.pitch)?,
            max_inflight_frames: self.max_inflight_frames as usize,
            max_commands: self.max_commands as usize,
            ..Default::default()
        })
    }
}

/// Render interleaved samples in one call and return the interleaved result.
/// Drives the same session as streaming playback.
#[wasm_bindgen]
pub fn render(interleaved: &[f32], options: &SessionOptions) -> Result<Vec<f32>, JsError> {
    let mut config = options.config()?;
    // `render` derives the length from the input.
    config.source_frames = None;
    Ok(ts::render(interleaved, config).map_err(err)?.samples)
}

/// Result of `Session.advance`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStatus {
    Priming,
    Progress,
    NeedSource,
    NeedOutput,
    NeedAcknowledgement,
    Paused,
    Rendered,
    Finished,
    Fault,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Stopped,
    Paused,
    Playing,
}

/// The source span the session wants next. Supply any nonempty prefix.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct SourceRequest {
    #[wasm_bindgen(readonly)]
    pub epoch: f64,
    #[wasm_bindgen(readonly, js_name = startFrame)]
    pub start_frame: f64,
    #[wasm_bindgen(readonly)]
    pub frames: u32,
}

/// Where a block returned by `readOutput` sits in the session's output domain.
/// Queue it with the samples; acknowledge `startFrame + consumed` on delivery.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct OutputStamp {
    #[wasm_bindgen(readonly)]
    pub epoch: f64,
    #[wasm_bindgen(readonly, js_name = startFrame)]
    pub start_frame: f64,
    #[wasm_bindgen(readonly)]
    pub frames: u32,
}

/// A control that reached an algorithm boundary, with the achieved ratios.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct AppliedParameters {
    #[wasm_bindgen(readonly, js_name = commandId)]
    pub command_id: f64,
    #[wasm_bindgen(readonly, js_name = requestedAt)]
    pub requested_at: f64,
    #[wasm_bindgen(readonly, js_name = appliedAt)]
    pub applied_at: f64,
    /// Achieved source time / output time.
    #[wasm_bindgen(readonly)]
    pub tempo: f64,
    /// Achieved frequency multiplier.
    #[wasm_bindgen(readonly)]
    pub pitch: f64,
}

impl From<ts::AppliedParameters> for AppliedParameters {
    fn from(a: ts::AppliedParameters) -> Self {
        Self {
            command_id: a.command_id as f64,
            requested_at: a.requested_at as f64,
            applied_at: a.applied_at as f64,
            tempo: a.tempo(),
            pitch: a.achieved_pitch,
        }
    }
}

/// One host-driven stream. Single owner: keep it on one thread (a Web Worker
/// for playback) and do not call it from an AudioWorklet's `process()`.
#[wasm_bindgen]
pub struct Session {
    inner: ts::StreamSession,
}

#[wasm_bindgen]
impl Session {
    /// Validates the options and allocates all session storage.
    #[wasm_bindgen(constructor)]
    pub fn new(options: &SessionOptions) -> Result<Session, JsError> {
        Ok(Self {
            inner: ts::PreparedSession::new(options.config()?)
                .map_err(err)?
                .activate(),
        })
    }

    /// The span to supply next, or `undefined` when no source is wanted.
    #[wasm_bindgen(js_name = sourceRequest)]
    pub fn source_request(&self) -> Option<SourceRequest> {
        self.inner.source_request().map(|r| SourceRequest {
            epoch: r.epoch.0 as f64,
            start_frame: r.start_frame as f64,
            frames: r.frames as u32,
        })
    }

    /// Supply interleaved finite samples for a prefix of `request`. Returns
    /// the frames accepted. A request from before a seek is rejected.
    pub fn supply(&mut self, request: &SourceRequest, interleaved: &[f32]) -> Result<u32, JsError> {
        let request = ts::SourceRequest {
            epoch: ts::Epoch(frames(request.epoch)?),
            start_frame: frames(request.start_frame)?,
            frames: request.frames as usize,
        };
        Ok(self.inner.supply(request, interleaved).map_err(err)? as u32)
    }

    /// End of a sequential source of unknown length.
    #[wasm_bindgen(js_name = finishInput)]
    pub fn finish_input(&mut self) -> Result<(), JsError> {
        self.inner.finish_input().map_err(err)
    }

    /// Run up to `quota` bounded stages. A stage is not a time guarantee.
    pub fn advance(&mut self, quota: u32) -> ProcessStatus {
        match self.inner.advance(quota as usize).status {
            ts::ProcessStatus::Priming => ProcessStatus::Priming,
            ts::ProcessStatus::Progress => ProcessStatus::Progress,
            ts::ProcessStatus::NeedSource => ProcessStatus::NeedSource,
            ts::ProcessStatus::NeedOutput => ProcessStatus::NeedOutput,
            ts::ProcessStatus::NeedAcknowledgement => ProcessStatus::NeedAcknowledgement,
            ts::ProcessStatus::Paused => ProcessStatus::Paused,
            ts::ProcessStatus::Rendered => ProcessStatus::Rendered,
            ts::ProcessStatus::Finished => ProcessStatus::Finished,
            ts::ProcessStatus::Fault => ProcessStatus::Fault,
        }
    }

    /// Copy up to `interleaved.length / channels` ready frames into
    /// `interleaved`. Only `stamp.frames` frames are written. Collected
    /// output is not playback: report delivery with `acknowledge`.
    #[wasm_bindgen(js_name = readOutput)]
    pub fn read_output(&mut self, interleaved: &mut [f32]) -> Result<OutputStamp, JsError> {
        let stamp = self.inner.read_output(interleaved).map_err(err)?;
        Ok(OutputStamp {
            epoch: stamp.epoch.0 as f64,
            start_frame: stamp.start_frame as f64,
            frames: stamp.frames as u32,
        })
    }

    /// Cumulative delivery receipt: every output frame of `epoch` before
    /// `endFrame` has been played. Returns whether the position advanced;
    /// repeated and older-epoch receipts are ignored.
    pub fn acknowledge(&mut self, epoch: f64, end_frame: f64) -> Result<bool, JsError> {
        self.inner
            .acknowledge(ts::OutputConsumption {
                epoch: ts::Epoch(frames(epoch)?),
                end_frame: frames(end_frame)?,
            })
            .map_err(err)
    }

    /// Schedule tempo and pitch at an output frame. Read the outcome from
    /// `nextApplied`.
    pub fn schedule(
        &mut self,
        command_id: f64,
        at_output_frame: f64,
        tempo: f64,
        pitch: f64,
    ) -> Result<(), JsError> {
        self.inner
            .schedule(
                frames(command_id)?,
                frames(at_output_frame)?,
                parameters(tempo, pitch)?,
            )
            .map_err(err)
    }

    /// The next applied control event, or `undefined`. Drain these: the
    /// acknowledgement queue is bounded.
    #[wasm_bindgen(js_name = nextApplied)]
    pub fn next_applied(&mut self) -> Option<AppliedParameters> {
        self.inner.next_applied().map(Into::into)
    }

    /// Parameters currently in effect.
    #[wasm_bindgen(js_name = appliedParameters)]
    pub fn applied_parameters(&self) -> AppliedParameters {
        self.inner.applied_parameters().into()
    }

    /// Seek to a source frame. Starts a new epoch, which is returned; drop
    /// output queued under the old one. Needs `seekable`.
    pub fn seek(&mut self, source_frame: f64) -> Result<f64, JsError> {
        Ok(self.inner.seek(frames(source_frame)?).map_err(err)?.0 as f64)
    }

    /// Loop the half-open source range `[start, end)`. Returns the new epoch.
    #[wasm_bindgen(js_name = setLoop)]
    pub fn set_loop(&mut self, start: f64, end: f64) -> Result<f64, JsError> {
        let region = Some((frames(start)?, frames(end)?));
        Ok(self.inner.set_loop(region).map_err(err)?.0 as f64)
    }

    /// Leave the loop at the consumed position. Returns the new epoch.
    #[wasm_bindgen(js_name = clearLoop)]
    pub fn clear_loop(&mut self) -> Result<f64, JsError> {
        Ok(self.inner.set_loop(None).map_err(err)?.0 as f64)
    }

    #[wasm_bindgen(js_name = setTransport)]
    pub fn set_transport(&mut self, transport: Transport) {
        self.inner.set_transport(match transport {
            Transport::Stopped => ts::Transport::Stopped,
            Transport::Paused => ts::Transport::Paused,
            Transport::Playing => ts::Transport::Playing,
        });
    }

    #[wasm_bindgen(getter)]
    pub fn transport(&self) -> Transport {
        match self.inner.transport() {
            ts::Transport::Stopped => Transport::Stopped,
            ts::Transport::Paused => Transport::Paused,
            ts::Transport::Playing => Transport::Playing,
        }
    }

    /// Source frame corresponding to the last acknowledged output.
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> f64 {
        self.inner.position()
    }

    #[wasm_bindgen(getter)]
    pub fn epoch(&self) -> f64 {
        self.inner.epoch().0 as f64
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u32 {
        self.inner.config().channels as u32
    }

    /// Frames ready for `readOutput`.
    #[wasm_bindgen(getter, js_name = outputReady)]
    pub fn output_ready(&self) -> u32 {
        self.inner.output_ready() as u32
    }

    /// Output frame before which audio is already rendered; controls
    /// scheduled earlier are late.
    #[wasm_bindgen(getter, js_name = commitHorizon)]
    pub fn commit_horizon(&self) -> f64 {
        self.inner.commit_horizon() as f64
    }

    /// Conservative maximum output block between control boundaries.
    #[wasm_bindgen(getter, js_name = controlQuantumFrames)]
    pub fn control_quantum_frames(&self) -> u32 {
        self.inner.timing().control_quantum_frames as u32
    }

    /// All output has been rendered (not necessarily collected or played).
    #[wasm_bindgen(getter, js_name = renderedFinished)]
    pub fn rendered_finished(&self) -> bool {
        self.inner.rendered_finished()
    }

    /// All output has been rendered, collected and acknowledged.
    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.inner.finished()
    }

    /// The error behind a `Fault` status, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn fault(&self) -> Option<String> {
        self.inner.fault().map(|e| format!("{e:?}"))
    }
}
