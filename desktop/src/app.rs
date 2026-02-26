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
    pitch_semitones: f32,
    volume: f32,
    preset: PresetChoice,
    target_bpm_text: String,

    // Error messages
    error_message: Option<String>,
}

impl TimeStretchApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let state = Arc::new(Mutex::new(SharedState::new()));
        let stream_active = Arc::new(AtomicBool::new(false));
        let position = Arc::new(AtomicPosition::new());

        // Try to create audio engine
        let (audio_engine, output_sample_rate) =
            match AudioEngine::new(state.clone(), stream_active.clone()) {
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
            pitch_semitones: 0.0,
            volume: 0.8,
            preset: PresetChoice::None,
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
                let samples = Arc::new(bpm_buffer.into_data());

                // Update shared state
                {
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
                }

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

        // Create audio engine with ring buffer
        let (engine, producer) =
            match AudioEngine::new(self.state.clone(), self.stream_active.clone()) {
                Ok((e, p)) => (e, p),
                Err(e) => {
                    self.error_message = Some(format!("Audio error: {e}"));
                    return;
                }
            };
        self.audio_engine = Some(engine);

        // Prepare working audio (with pitch shift if needed)
        let working_audio = if self.pitch_semitones.abs() < 0.01 {
            source.as_ref().clone()
        } else {
            let factor = 2.0_f64.powf(self.pitch_semitones as f64 / 12.0);
            let params = timestretch::StretchParams::new(1.0)
                .with_sample_rate(sample_rate)
                .with_channels(2);
            match timestretch::pitch_shift(&source, &params, factor) {
                Ok(shifted) => shifted,
                Err(e) => {
                    log::error!("Pitch shift failed: {e}");
                    source.as_ref().clone()
                }
            }
        };

        // Update state
        {
            let mut st = self.state.lock().unwrap();
            st.transport = Transport::Playing;
            st.total_frames = working_audio.len() / 2;
        }

        let stop_flag = Arc::new(StopFlag::new());
        self.stop_flag = Some(stop_flag.clone());

        let handle = processor::start_processing_thread(
            self.state.clone(),
            source,
            working_audio,
            producer,
            sample_rate,
            self.position.clone(),
            self.stream_active.clone(),
            stop_flag,
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
                .add_enabled(
                    transport != Transport::Stopped,
                    egui::Button::new("Stop"),
                )
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
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("controls_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                // Stretch ratio
                ui.label("Stretch Ratio:");
                ui.horizontal(|ui| {
                    let old_ratio = self.stretch_ratio;
                    ui.add(
                        egui::Slider::new(&mut self.stretch_ratio, 0.5..=2.0)
                            .text("x")
                            .fixed_decimals(2),
                    );
                    if (self.stretch_ratio - old_ratio).abs() > 0.001 {
                        let mut st = self.state.lock().unwrap();
                        st.stretch_ratio = self.stretch_ratio;
                        // Update target BPM text if we have detected BPM
                        if st.detected_bpm > 0.0 {
                            let target = st.detected_bpm / self.stretch_ratio;
                            self.target_bpm_text = format!("{target:.1}");
                            st.target_bpm = target;
                        }
                    }
                    if ui.button("Reset").clicked() {
                        self.stretch_ratio = 1.0;
                        let mut st = self.state.lock().unwrap();
                        st.stretch_ratio = 1.0;
                        if st.detected_bpm > 0.0 {
                            self.target_bpm_text = format!("{:.1}", st.detected_bpm);
                            st.target_bpm = st.detected_bpm;
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
                        if response.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        {
                            if let Ok(target) = self.target_bpm_text.parse::<f64>() {
                                if target > 0.0 && detected_bpm > 0.0 {
                                    self.stretch_ratio = detected_bpm / target;
                                    self.stretch_ratio =
                                        self.stretch_ratio.clamp(0.5, 2.0);
                                    let mut st = self.state.lock().unwrap();
                                    st.stretch_ratio = self.stretch_ratio;
                                    st.target_bpm = target;
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

                // Pitch shift
                ui.label("Pitch Shift:");
                ui.horizontal(|ui| {
                    let old_pitch = self.pitch_semitones;
                    ui.add(
                        egui::Slider::new(&mut self.pitch_semitones, -12.0..=12.0)
                            .text("st")
                            .fixed_decimals(1),
                    );
                    if (self.pitch_semitones - old_pitch).abs() > 0.01 {
                        let mut st = self.state.lock().unwrap();
                        st.pitch_semitones = self.pitch_semitones;
                        st.pitch_changed = true;
                    }
                    if ui.button("Reset").clicked() && self.pitch_semitones.abs() > 0.01 {
                        self.pitch_semitones = 0.0;
                        let mut st = self.state.lock().unwrap();
                        st.pitch_semitones = 0.0;
                        st.pitch_changed = true;
                    }
                });
                ui.end_row();

                // Pitch processing indicator
                let pitch_processing = self.state.lock().unwrap().pitch_processing;
                if pitch_processing {
                    ui.label("");
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "Processing pitch shift...",
                    );
                    ui.end_row();
                    ctx_request_repaint(ui);
                }

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

fn ctx_request_repaint(ui: &egui::Ui) {
    ui.ctx().request_repaint();
}
