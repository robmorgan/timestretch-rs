use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::audio_engine::AudioEngine;
use crate::decoder;
use crate::processor;
use crate::state::*;
use crate::waveform::{self, WaveformPeaks};

const MIN_STRETCH_RATIO: f64 = 0.25;
/// Slowest playback the streaming engine sustains cleanly on every profile
/// (Live handles ~12x; 10x leaves margin). Sets the tempo fader's floor at
/// `detected_bpm / 10` (~-90%, near-stop).
const MAX_STRETCH_RATIO: f64 = 10.0;
/// Ratio ceiling on the varispeed-first tempo path (the library bounds the
/// varispeed resampler's step range to `[0.25, 4.0]`).
const MAX_VARISPEED_RATIO: f64 = 4.0;
/// Tempo fader reaches double the track BPM (+100%).
const MAX_TEMPO_FACTOR: f64 = 2.0;
/// Tempo fader width in points (3x egui's ~100pt default) for fine control.
const TEMPO_SLIDER_WIDTH: f32 = 300.0;

pub struct TimeStretchApp {
    state: SharedStateHandle,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,

    // Audio engine (lives for the lifetime of the app)
    audio_engine: Option<AudioEngine>,
    output_sample_rate: u32,

    // Source audio data
    source_audio: Option<Arc<Vec<f32>>>,

    // Processing thread
    processing_handle: Option<JoinHandle<()>>,
    stop_flag: Option<Arc<StopFlag>>,

    // File info
    file_name: String,
    file_path: Option<PathBuf>,

    // Waveform
    waveform_peaks: Option<WaveformPeaks>,

    // UI state
    stretch_ratio: f64,
    /// Target playback BPM the tempo fader binds to (0.0 until a BPM is
    /// detected). Derived view of `stretch_ratio`, kept in sync.
    target_bpm: f64,
    pitch_semitones: f32,
    volume: f32,
    preset: PresetChoice,
    stream_profile: StreamProfile,
    streaming_engine: StreamingEngine,
    control_path: ControlPath,
    target_bpm_text: String,

    // Error messages
    error_message: Option<String>,
}

impl TimeStretchApp {
    /// Widest stretch ratio the active tempo path supports.
    #[inline]
    fn max_stretch_ratio(&self) -> f64 {
        match self.control_path {
            ControlPath::VarispeedFirst => MAX_VARISPEED_RATIO,
            ControlPath::VocoderTempo => MAX_STRETCH_RATIO,
        }
    }

    #[inline]
    fn clamp_stretch_ratio(&self, ratio: f64) -> f64 {
        ratio.clamp(MIN_STRETCH_RATIO, self.max_stretch_ratio())
    }

    /// Sets the playback tempo to `target_bpm` (for a track at `detected_bpm`)
    /// and syncs every derived view — stretch ratio, the fader's `target_bpm`,
    /// and the BPM text box — to the value that actually plays after the
    /// engine's ratio clamp. Single write point for the tempo control.
    fn apply_target_bpm(&mut self, detected_bpm: f64, target_bpm: f64) {
        let ratio = self.clamp_stretch_ratio(detected_bpm / target_bpm.max(1e-6));
        let effective_bpm = detected_bpm / ratio;
        self.stretch_ratio = ratio;
        self.target_bpm = effective_bpm;
        self.target_bpm_text = format!("{effective_bpm:.1}");
        let mut st = self.state.lock().unwrap();
        st.stretch_ratio = ratio;
        st.target_bpm = effective_bpm;
    }

    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let state = Arc::new(Mutex::new(SharedState::new()));
        let stream_active = Arc::new(AtomicBool::new(false));
        let position = Arc::new(AtomicPosition::new());

        // Try to create audio engine to detect default sample rate
        let dummy_flush = Arc::new(AtomicBool::new(false));
        let (audio_engine, output_sample_rate) =
            match AudioEngine::new(state.clone(), stream_active.clone(), None, dummy_flush) {
                Ok((engine, _producer)) => {
                    let sr = engine.output_sample_rate;
                    // We'll create a new engine when loading a file
                    // since we need the producer for the processing thread
                    (None, sr)
                }
                Err(e) => {
                    log::error!("Failed to create audio engine: {e}");
                    (None, 44100)
                }
            };

        Self {
            state,
            position,
            stream_active,
            audio_engine,
            output_sample_rate,
            source_audio: None,
            processing_handle: None,
            stop_flag: None,
            file_name: String::new(),
            file_path: None,
            waveform_peaks: None,
            stretch_ratio: 1.0,
            target_bpm: 0.0,
            pitch_semitones: 0.0,
            volume: 0.8,
            preset: PresetChoice::DjBeatmatch,
            stream_profile: StreamProfile::Live,
            streaming_engine: StreamingEngine::Deterministic,
            control_path: ControlPath::VarispeedFirst,
            target_bpm_text: String::new(),
            error_message: None,
        }
    }

    fn load_file(&mut self, path: PathBuf) {
        // Stop any existing playback
        self.stop_playback();

        self.error_message = None;
        self.file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        log::info!("Loading file: {}", path.display());

        match decoder::decode_file(&path) {
            Ok(decoded) => {
                log::info!(
                    "Decoded: {} frames, {} Hz, {} ch",
                    decoded.num_frames,
                    decoded.sample_rate,
                    decoded.channels
                );

                let sample_rate = decoded.sample_rate;
                let num_frames = decoded.num_frames;
                let channel_layout = timestretch::Channels::from_count(decoded.channels as usize)
                    .unwrap_or(timestretch::Channels::Stereo);
                let bpm_buffer =
                    timestretch::AudioBuffer::new(decoded.samples, sample_rate, channel_layout);

                // Compute waveform peaks
                self.waveform_peaks = Some(WaveformPeaks::compute(&bpm_buffer.data, 2, 800));

                // Detect BPM from channel-aware buffer (stereo-safe).
                let bpm = timestretch::detect_bpm_buffer(&bpm_buffer);
                log::info!("Detected BPM: {bpm:.1}");
                self.target_bpm_text = if bpm > 0.0 {
                    format!("{bpm:.1}")
                } else {
                    String::new()
                };
                // A fresh track starts at its own tempo (fader centered) and
                // unity ratio, regardless of any prior track's settings.
                self.stretch_ratio = 1.0;
                self.target_bpm = bpm.max(0.0);
                let num_channels = bpm_buffer.channels.count();
                let samples = Arc::new(bpm_buffer.into_data());

                // Update shared state
                let analysis_generation = {
                    let mut st = self.state.lock().unwrap();
                    st.sample_rate = sample_rate;
                    st.total_frames = num_frames;
                    st.position_frames = 0;
                    st.detected_bpm = bpm;
                    st.target_bpm = bpm.max(0.0);
                    st.transport = Transport::Stopped;
                    st.stretch_ratio = self.stretch_ratio;
                    st.pitch_semitones = self.pitch_semitones;
                    st.volume = self.volume;
                    st.preset = self.preset;
                    st.stream_profile = self.stream_profile;
                    st.streaming_engine = self.streaming_engine;
                    st.pre_analysis = None;
                    st.loop_region = None;
                    st.loop_in = None;
                    st.analysis_generation += 1;
                    st.analysis_generation
                };

                // Analyze-on-load: a matching sidecar is used immediately;
                // otherwise a background thread analyzes once and caches the
                // result next to the file. `detect_bpm_buffer` above keeps
                // the UI BPM instant either way — the artifact upgrades
                // subsequent processor rebuilds when it lands.
                spawn_pre_analysis(
                    self.state.clone(),
                    samples.clone(),
                    num_channels,
                    sample_rate,
                    sidecar_path(&path),
                    analysis_generation,
                );

                self.source_audio = Some(samples);
                self.file_path = Some(path);
                self.output_sample_rate = sample_rate;
                self.position.store(0);
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load: {e}"));
                log::error!("Failed to load file: {e}");
            }
        }
    }

    fn start_playback(&mut self) {
        let source = match &self.source_audio {
            Some(s) => s.clone(),
            None => return,
        };

        // Stop any existing processing thread
        self.stop_processing_thread();

        let sample_rate = {
            let st = self.state.lock().unwrap();
            st.sample_rate
        };

        // Create audio engine with ring buffer, matching the source file's sample rate
        // so playback speed is correct regardless of the device's native rate.
        let flush_ring = Arc::new(AtomicBool::new(false));
        let (engine, producer) = match AudioEngine::new(
            self.state.clone(),
            self.stream_active.clone(),
            Some(sample_rate),
            flush_ring.clone(),
        ) {
            Ok((e, p)) => (e, p),
            Err(e) => {
                self.error_message = Some(format!("Audio error: {e}"));
                return;
            }
        };
        self.audio_engine = Some(engine);

        // Pitch is applied live by the stream processor; no pre-render pass.
        {
            let mut st = self.state.lock().unwrap();
            st.transport = Transport::Playing;
            st.total_frames = source.len() / 2;
        }

        let stop_flag = Arc::new(StopFlag::new());
        self.stop_flag = Some(stop_flag.clone());

        let handle = processor::start_processing_thread(
            self.state.clone(),
            source,
            producer,
            sample_rate,
            self.position.clone(),
            self.stream_active.clone(),
            stop_flag,
            flush_ring,
        );

        self.processing_handle = Some(handle);
    }

    fn stop_playback(&mut self) {
        {
            let mut st = self.state.lock().unwrap();
            st.transport = Transport::Stopped;
            st.position_frames = 0;
        }
        self.position.store(0);
        self.stop_processing_thread();
    }

    fn toggle_pause(&mut self) {
        let mut st = self.state.lock().unwrap();
        match st.transport {
            Transport::Playing => st.transport = Transport::Paused,
            Transport::Paused => st.transport = Transport::Playing,
            Transport::Stopped => {
                drop(st);
                self.start_playback();
            }
        }
    }

    fn stop_processing_thread(&mut self) {
        if let Some(flag) = self.stop_flag.take() {
            flag.set();
        }
        if let Some(handle) = self.processing_handle.take() {
            let _ = handle.join();
        }
        self.stream_active.store(false, Ordering::Relaxed);
        self.audio_engine = None;
    }

    fn format_time(secs: f64) -> String {
        let mins = (secs / 60.0) as u32;
        let s = secs % 60.0;
        format!("{mins}:{s:05.2}")
    }
}

impl eframe::App for TimeStretchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Sync position from atomic counter
        let pos_frames = self.position.load();
        {
            let mut st = self.state.lock().unwrap();
            st.position_frames = pos_frames;
        }

        // Request repaint for continuous UI updates during playback
        let transport = self.state.lock().unwrap().transport;
        if transport == Transport::Playing {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Timestretch Desktop");
            ui.add_space(8.0);

            // Error message
            if let Some(ref err) = self.error_message {
                ui.colored_label(egui::Color32::RED, err);
                ui.add_space(4.0);
            }

            // File panel
            self.file_panel(ui);
            ui.add_space(8.0);

            // Waveform
            self.waveform_panel(ui);
            ui.add_space(8.0);

            // Transport
            self.transport_panel(ui);
            ui.add_space(12.0);

            // Controls
            self.controls_panel(ui);
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_processing_thread();
    }
}

impl TimeStretchApp {
    fn file_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Load Audio File").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
                    .pick_file()
                {
                    self.load_file(path);
                }
            }

            if !self.file_name.is_empty() {
                ui.separator();
                ui.label(&self.file_name);

                let st = self.state.lock().unwrap();
                ui.separator();
                ui.label(format!("{} Hz", st.sample_rate));
                ui.separator();
                ui.label(format!("{:.1}s", st.duration_secs()));
                ui.separator();
                if st.detected_bpm > 0.0 {
                    ui.label(format!("{:.1} BPM", st.detected_bpm));
                } else {
                    ui.label("BPM: --");
                }
            }
        });
    }

    fn waveform_panel(&mut self, ui: &mut egui::Ui) {
        let (total_frames, pos_frames) = {
            let st = self.state.lock().unwrap();
            (st.total_frames, st.position_frames)
        };

        let progress = if total_frames > 0 {
            pos_frames as f32 / total_frames as f32
        } else {
            0.0
        };

        let empty_peaks = WaveformPeaks {
            pos: vec![],
            neg: vec![],
        };
        let peaks = self.waveform_peaks.as_ref().unwrap_or(&empty_peaks);
        let (_response, seek_pos) = waveform::paint_waveform(ui, peaks, progress);

        // Handle click-to-seek
        if let Some(frac) = seek_pos {
            if total_frames > 0 {
                let seek_frame = (frac * total_frames as f32) as usize;
                let mut st = self.state.lock().unwrap();
                st.seek_request = Some(seek_frame);
            }
        }
    }

    fn transport_panel(&mut self, ui: &mut egui::Ui) {
        let (transport, pos_secs, duration_secs) = {
            let st = self.state.lock().unwrap();
            (st.transport, st.position_secs(), st.duration_secs())
        };

        ui.horizontal(|ui| {
            let play_label = match transport {
                Transport::Playing => "Pause",
                Transport::Paused => "Resume",
                Transport::Stopped => "Play",
            };

            if ui
                .add_enabled(self.source_audio.is_some(), egui::Button::new(play_label))
                .clicked()
            {
                self.toggle_pause();
            }

            if ui
                .add_enabled(transport != Transport::Stopped, egui::Button::new("Stop"))
                .clicked()
            {
                self.stop_playback();
            }

            ui.separator();

            ui.monospace(format!(
                "{} / {}",
                Self::format_time(pos_secs),
                Self::format_time(duration_secs)
            ));
        });

        self.loop_and_jump_panel(ui);
    }

    /// Beat-jump buttons and loop in/out/exit controls. Jumps and loop wraps
    /// go through the processing thread's warm-start machinery.
    fn loop_and_jump_panel(&mut self, ui: &mut egui::Ui) {
        let has_audio = self.source_audio.is_some();
        let (pos_frames, total_frames, bpm, sample_rate, loop_region, loop_in) = {
            let st = self.state.lock().unwrap();
            (
                st.position_frames,
                st.total_frames,
                st.detected_bpm,
                st.sample_rate,
                st.loop_region,
                st.loop_in,
            )
        };

        ui.horizontal(|ui| {
            // Beat jumps: relative seeks by whole beats using the detected
            // grid. Disabled until a tempo is known.
            let beat_frames = if bpm > 0.0 {
                (sample_rate as f64 * 60.0 / bpm).round() as i64
            } else {
                0
            };
            let can_jump = has_audio && beat_frames > 0 && total_frames > 0;
            ui.label("Jump:");
            for beats in [-16i64, -4, 4, 16] {
                let label = if beats > 0 {
                    format!("+{beats}")
                } else {
                    format!("{beats}")
                };
                if ui.add_enabled(can_jump, egui::Button::new(label)).clicked() {
                    let delta = beats * beat_frames;
                    let target =
                        (pos_frames as i64 + delta).clamp(0, total_frames as i64 - 1) as usize;
                    self.state.lock().unwrap().seek_request = Some(target);
                }
            }

            ui.separator();

            // Loop: set in point, then out point to arm; exit to clear.
            ui.label("Loop:");
            if ui.add_enabled(has_audio, egui::Button::new("In")).clicked() {
                self.state.lock().unwrap().loop_in = Some(pos_frames);
            }
            let can_close = has_audio && loop_in.is_some_and(|i| pos_frames > i);
            if ui
                .add_enabled(can_close, egui::Button::new("Out"))
                .clicked()
            {
                if let Some(start) = loop_in {
                    let mut st = self.state.lock().unwrap();
                    st.loop_region = Some((start, pos_frames));
                    st.loop_in = None;
                }
            }
            if ui
                .add_enabled(loop_region.is_some(), egui::Button::new("Exit"))
                .clicked()
            {
                self.state.lock().unwrap().loop_region = None;
            }

            match loop_region {
                Some((s, e)) => {
                    let secs = (e - s) as f64 / sample_rate.max(1) as f64;
                    ui.monospace(format!("looping {secs:.2}s"));
                }
                None if loop_in.is_some() => {
                    ui.monospace("in set…");
                }
                None => {}
            }
        });
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("controls_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                // Tempo control. With a detected BPM it is a CDJ-style tempo
                // fader in BPM (0.1-BPM steps, centered on the track tempo);
                // otherwise it falls back to a raw stretch-ratio slider.
                let detected = self.state.lock().unwrap().detected_bpm;
                ui.label("Tempo:");
                ui.horizontal(|ui| {
                    if detected > 0.0 {
                        if self.target_bpm <= 0.0 {
                            self.target_bpm = detected;
                        }
                        let min_bpm = detected / self.max_stretch_ratio();
                        let max_bpm = detected * MAX_TEMPO_FACTOR;
                        let old_bpm = self.target_bpm;
                        // Widen the fader (3x the default) so 0.1-BPM steps
                        // are easy to hit by drag.
                        ui.spacing_mut().slider_width = TEMPO_SLIDER_WIDTH;
                        ui.add(
                            egui::Slider::new(&mut self.target_bpm, min_bpm..=max_bpm)
                                .step_by(0.1)
                                .fixed_decimals(1)
                                .suffix(" BPM"),
                        );
                        if (self.target_bpm - old_bpm).abs() > 0.001 {
                            self.apply_target_bpm(detected, self.target_bpm);
                        }
                        let pct = (1.0 / self.stretch_ratio - 1.0) * 100.0;
                        ui.label(egui::RichText::new(format!("{pct:+.1}%")).weak());
                        if ui.button("Reset").clicked() {
                            self.apply_target_bpm(detected, detected);
                        }
                    } else {
                        let old_ratio = self.stretch_ratio;
                        let max_ratio = self.max_stretch_ratio();
                        ui.add(
                            egui::Slider::new(
                                &mut self.stretch_ratio,
                                MIN_STRETCH_RATIO..=max_ratio,
                            )
                            .text("x")
                            .fixed_decimals(2),
                        );
                        if (self.stretch_ratio - old_ratio).abs() > 0.001 {
                            self.state.lock().unwrap().stretch_ratio = self.stretch_ratio;
                        }
                        if ui.button("Reset").clicked() {
                            self.stretch_ratio = 1.0;
                            self.state.lock().unwrap().stretch_ratio = 1.0;
                        }
                    }
                });
                ui.end_row();

                // BPM panel
                let detected_bpm = self.state.lock().unwrap().detected_bpm;
                ui.label("BPM:");
                ui.horizontal(|ui| {
                    if detected_bpm > 0.0 {
                        ui.label(format!("Detected: {detected_bpm:.1}"));
                        ui.separator();
                        ui.label("Target:");
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.target_bpm_text)
                                .desired_width(60.0),
                        );
                        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            if let Ok(target) = self.target_bpm_text.parse::<f64>() {
                                if target > 0.0 {
                                    // Route through the shared sync point so the
                                    // tempo fader tracks a typed BPM too.
                                    self.apply_target_bpm(detected_bpm, target);
                                }
                            }
                        }
                    } else {
                        ui.label("Load a file to detect BPM");
                    }
                });
                ui.end_row();

                // EDM Preset
                ui.label("Preset:");
                ui.horizontal(|ui| {
                    let old_preset = self.preset;
                    egui::ComboBox::from_id_salt("preset_combo")
                        .selected_text(self.preset.label())
                        .show_ui(ui, |ui| {
                            for &p in PresetChoice::ALL {
                                ui.selectable_value(&mut self.preset, p, p.label());
                            }
                        });
                    if self.preset != old_preset {
                        let mut st = self.state.lock().unwrap();
                        st.preset = self.preset;
                        st.preset_changed = true;
                    }
                });
                ui.end_row();

                ui.label("Playback:");
                ui.horizontal(|ui| {
                    let old_profile = self.stream_profile;
                    egui::ComboBox::from_id_salt("stream_profile_combo")
                        .selected_text(self.stream_profile.label())
                        .show_ui(ui, |ui| {
                            for &profile in StreamProfile::ALL {
                                ui.selectable_value(
                                    &mut self.stream_profile,
                                    profile,
                                    profile.label(),
                                );
                            }
                        });
                    if self.stream_profile != old_profile {
                        let mut st = self.state.lock().unwrap();
                        st.stream_profile = self.stream_profile;
                        st.preset_changed = true;
                    }
                    // Effective latency split reported by the active
                    // processor (published by the processing thread at
                    // every build): constant pipeline delay vs tempo
                    // control-to-audio.
                    let (latency_secs, control_secs) = {
                        let st = self.state.lock().unwrap();
                        (st.reported_latency_secs, st.reported_control_latency_secs)
                    };
                    if latency_secs > 0.0 {
                        let text = if control_secs + 0.0005 < latency_secs {
                            format!(
                                "~{:.0} ms delay · tempo ~{:.1} ms",
                                latency_secs * 1000.0,
                                control_secs * 1000.0
                            )
                        } else {
                            format!("~{:.0} ms", latency_secs * 1000.0)
                        };
                        ui.label(egui::RichText::new(text).weak()).on_hover_text(
                            "Constant pipeline (content) delay for the selected \
                             profile/engine, and how quickly a tempo change \
                             reaches the output on the selected tempo path",
                        );
                    }
                });
                ui.end_row();

                ui.label("Engine:");
                ui.horizontal(|ui| {
                    // The multi-res engine's buffering gate is set by its
                    // sub-bass FFT, which needs the Club profile or larger;
                    // the library rejects it on Live.
                    let multi_res_available = self.stream_profile != StreamProfile::Live;
                    let old_engine = self.streaming_engine;
                    ui.add_enabled_ui(multi_res_available, |ui| {
                        egui::ComboBox::from_id_salt("streaming_engine_combo")
                            .selected_text(engine_label(self.streaming_engine))
                            .show_ui(ui, |ui| {
                                for engine in [
                                    StreamingEngine::Deterministic,
                                    StreamingEngine::MultiResolution,
                                ] {
                                    ui.selectable_value(
                                        &mut self.streaming_engine,
                                        engine,
                                        engine_label(engine),
                                    );
                                }
                            });
                    })
                    .response
                    .on_disabled_hover_text(
                        "Multi-resolution needs the Club or Quality playback \
                         profile; Live stays on the standard engine",
                    );
                    if !multi_res_available
                        && self.streaming_engine == StreamingEngine::MultiResolution
                    {
                        // Profile dropped to Live while multi-res was selected:
                        // revert the control so the UI matches what plays.
                        self.streaming_engine = StreamingEngine::Deterministic;
                    }
                    if self.streaming_engine != old_engine {
                        let mut st = self.state.lock().unwrap();
                        st.streaming_engine = self.streaming_engine;
                        st.preset_changed = true;
                    }
                    if self.streaming_engine == StreamingEngine::MultiResolution {
                        ui.label(egui::RichText::new("3-band").weak())
                            .on_hover_text(
                                "Three-band filterbank: tighter sub-bass phase \
                             coherence, higher buffering latency",
                            );
                    }
                });
                ui.end_row();

                ui.label("Tempo Path:");
                ui.horizontal(|ui| {
                    let old_path = self.control_path;
                    egui::ComboBox::from_id_salt("control_path_combo")
                        .selected_text(control_path_label(self.control_path))
                        .show_ui(ui, |ui| {
                            for path in [ControlPath::VarispeedFirst, ControlPath::VocoderTempo] {
                                ui.selectable_value(
                                    &mut self.control_path,
                                    path,
                                    control_path_label(path),
                                );
                            }
                        });
                    if self.control_path != old_path {
                        // The varispeed path bounds the tempo ratio; pull the
                        // fader back into range before the rebuild applies it.
                        let clamped = self.clamp_stretch_ratio(self.stretch_ratio);
                        if (clamped - self.stretch_ratio).abs() > f64::EPSILON {
                            let detected = self.state.lock().unwrap().detected_bpm;
                            if detected > 0.0 {
                                self.apply_target_bpm(detected, detected / clamped);
                            } else {
                                self.stretch_ratio = clamped;
                                self.state.lock().unwrap().stretch_ratio = clamped;
                            }
                        }
                        let mut st = self.state.lock().unwrap();
                        st.control_path = self.control_path;
                        st.preset_changed = true;
                    }
                    if self.control_path == ControlPath::VarispeedFirst {
                        ui.label(egui::RichText::new("instant tempo").weak())
                            .on_hover_text(
                                "Varispeed-first keylock: the tempo fader drives \
                                 an input resampler sample-accurately; the \
                                 vocoder's buffering becomes a constant delay \
                                 instead of tempo control latency",
                            );
                    }
                });
                ui.end_row();

                // Pitch shift (realtime: applied live by the stream processor)
                ui.label("Pitch Shift:");
                ui.horizontal(|ui| {
                    let old_pitch = self.pitch_semitones;
                    ui.add(
                        egui::Slider::new(&mut self.pitch_semitones, -12.0..=12.0)
                            .text("st")
                            .fixed_decimals(1),
                    );
                    if (self.pitch_semitones - old_pitch).abs() > 0.001 {
                        let mut st = self.state.lock().unwrap();
                        st.pitch_semitones = self.pitch_semitones;
                    }
                    if ui.button("Reset").clicked() && self.pitch_semitones.abs() > 0.001 {
                        self.pitch_semitones = 0.0;
                        let mut st = self.state.lock().unwrap();
                        st.pitch_semitones = 0.0;
                    }
                });
                ui.end_row();

                // Volume
                ui.label("Volume:");
                ui.horizontal(|ui| {
                    let old_vol = self.volume;
                    ui.add(
                        egui::Slider::new(&mut self.volume, 0.0..=1.0)
                            .text("")
                            .fixed_decimals(0)
                            .custom_formatter(|v, _| format!("{}%", (v * 100.0) as u32)),
                    );
                    if (self.volume - old_vol).abs() > 0.001 {
                        self.state.lock().unwrap().volume = self.volume;
                    }
                });
                ui.end_row();
            });
    }
}

/// UI label for a stream-mode rendering engine.
fn engine_label(engine: StreamingEngine) -> &'static str {
    match engine {
        StreamingEngine::Deterministic => "Standard",
        StreamingEngine::MultiResolution => "Multi-resolution",
    }
}

/// UI label for a tempo control path.
fn control_path_label(path: ControlPath) -> &'static str {
    match path {
        ControlPath::VarispeedFirst => "Varispeed (instant)",
        ControlPath::VocoderTempo => "Vocoder (glide)",
    }
}

/// Sidecar artifact path for a loaded audio file: `<file>.tsanalysis.json`.
fn sidecar_path(audio_path: &std::path::Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tsanalysis.json");
    PathBuf::from(os)
}

/// Loads a matching sidecar artifact or analyzes the track on a background
/// thread, storing the result in shared state for the next processor rebuild.
///
/// The result is discarded if another file was loaded in the meantime
/// (`generation` mismatch). Sidecar writes are best-effort: a read-only
/// volume must not break analysis.
fn spawn_pre_analysis(
    state: SharedStateHandle,
    samples: Arc<Vec<f32>>,
    num_channels: usize,
    sample_rate: u32,
    sidecar: PathBuf,
    generation: u64,
) {
    std::thread::spawn(move || {
        let analysis_signal = timestretch::downmix_to_mid(&samples, num_channels);

        let artifact = match timestretch::read_preanalysis_json(&sidecar) {
            Ok(cached) if cached.matches_source(&analysis_signal, sample_rate) => {
                log::info!("Pre-analysis: using cached sidecar {}", sidecar.display());
                cached
            }
            _ => {
                let start = std::time::Instant::now();
                let fresh = timestretch::analyze_for_dj(&analysis_signal, sample_rate);
                log::info!(
                    "Pre-analysis: {:.1} BPM, confidence {:.2}, {} beats, {} onsets ({:.2}s)",
                    fresh.bpm,
                    fresh.confidence,
                    fresh.beat_positions.len(),
                    fresh.transient_onsets.len(),
                    start.elapsed().as_secs_f64()
                );
                if let Err(e) = timestretch::write_preanalysis_json(&sidecar, &fresh) {
                    log::warn!(
                        "Pre-analysis: could not cache sidecar {}: {e}",
                        sidecar.display()
                    );
                }
                fresh
            }
        };

        let mut st = state.lock().unwrap();
        if st.analysis_generation == generation {
            st.pre_analysis = Some(Arc::new(artifact));
        } else {
            log::info!("Pre-analysis: discarding stale result (newer file loaded)");
        }
    });
}
