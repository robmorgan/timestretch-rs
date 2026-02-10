use eframe::egui;

/// Pre-computed waveform peaks for efficient rendering.
pub struct WaveformPeaks {
    /// Positive peaks per bucket.
    pub pos: Vec<f32>,
    /// Negative peaks per bucket.
    pub neg: Vec<f32>,
}

impl WaveformPeaks {
    /// Compute waveform peaks from interleaved stereo samples.
    /// Mixes to mono for display. `num_buckets` controls resolution.
    pub fn compute(samples: &[f32], channels: u32, num_buckets: usize) -> Self {
        let num_frames = samples.len() / channels as usize;
        if num_frames == 0 || num_buckets == 0 {
            return WaveformPeaks {
                pos: vec![0.0; num_buckets],
                neg: vec![0.0; num_buckets],
            };
        }

        let mut pos = Vec::with_capacity(num_buckets);
        let mut neg = Vec::with_capacity(num_buckets);

        let frames_per_bucket = num_frames as f64 / num_buckets as f64;

        for i in 0..num_buckets {
            let start_frame = (i as f64 * frames_per_bucket) as usize;
            let end_frame = (((i + 1) as f64 * frames_per_bucket) as usize).min(num_frames);

            let mut max_val: f32 = 0.0;
            let mut min_val: f32 = 0.0;

            for f in start_frame..end_frame {
                // Mix channels to mono
                let mut mono = 0.0f32;
                for c in 0..channels as usize {
                    mono += samples[f * channels as usize + c];
                }
                mono /= channels as f32;

                max_val = max_val.max(mono);
                min_val = min_val.min(mono);
            }

            pos.push(max_val);
            neg.push(min_val);
        }

        WaveformPeaks { pos, neg }
    }
}

/// Paint a waveform display with playback cursor and click-to-seek.
pub fn paint_waveform(
    ui: &mut egui::Ui,
    peaks: &WaveformPeaks,
    progress: f32,
) -> (egui::Response, Option<f32>) {
    let desired_size = egui::vec2(ui.available_width(), 120.0);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::click());
    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(30, 30, 40));

    if peaks.pos.is_empty() {
        // Draw placeholder text
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Load an audio file to see waveform",
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgb(100, 100, 120),
        );
        return (response, None);
    }

    let num_buckets = peaks.pos.len();
    let center_y = rect.center().y;
    let half_height = rect.height() * 0.45;

    let played_color = egui::Color32::from_rgb(100, 180, 255);
    let unplayed_color = egui::Color32::from_rgb(60, 80, 100);

    let cursor_x = rect.left() + rect.width() * progress;

    for i in 0..num_buckets {
        let x = rect.left() + (i as f32 / num_buckets as f32) * rect.width();
        let bar_width = (rect.width() / num_buckets as f32).max(1.0);

        let top = center_y - peaks.pos[i] * half_height;
        let bottom = center_y - peaks.neg[i] * half_height;

        let color = if x < cursor_x {
            played_color
        } else {
            unplayed_color
        };

        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x, top),
                egui::pos2(x + bar_width, bottom),
            ),
            0.0,
            color,
        );
    }

    // Draw cursor line
    if progress > 0.0 && progress < 1.0 {
        painter.line_segment(
            [
                egui::pos2(cursor_x, rect.top()),
                egui::pos2(cursor_x, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::WHITE),
        );
    }

    // Center line
    painter.line_segment(
        [
            egui::pos2(rect.left(), center_y),
            egui::pos2(rect.right(), center_y),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 60, 80)),
    );

    // Handle click-to-seek
    let seek_pos = if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let frac = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            Some(frac)
        } else {
            None
        }
    } else {
        None
    };

    (response, seek_pos)
}
