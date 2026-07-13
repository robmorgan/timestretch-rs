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

/// A beat marker for the waveform overlay: horizontal position as a
/// fraction of the track (0..1) and whether it is a downbeat (bar start).
#[derive(Debug, Clone, Copy)]
pub struct BeatMark {
    /// Position as a fraction of the total track length.
    pub frac: f32,
    /// True for downbeats (drawn emphasized).
    pub is_downbeat: bool,
}

/// Minimum pixel spacing between adjacent grid lines before a marker tier
/// is drawn at full density.
const MIN_GRID_SPACING_PX: f32 = 6.0;

/// How the overlay adapts the grid to the available width: whether
/// individual beats fit, and how many bars each drawn downbeat line spans
/// (1 = every downbeat; 2/4/8… = phrase markers when bars are too dense).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OverlayPlan {
    draw_beats: bool,
    downbeat_stride: usize,
}

/// Chooses the densest tier that keeps adjacent lines at least
/// [`MIN_GRID_SPACING_PX`] apart. Downbeats never disappear entirely —
/// they thin to every 2^k bars instead, so a full-length track still shows
/// its phrase structure.
fn overlay_plan(width_px: f32, beat_count: usize, downbeat_count: usize) -> OverlayPlan {
    let draw_beats = beat_count >= 2 && width_px / beat_count as f32 >= MIN_GRID_SPACING_PX;
    let mut downbeat_stride = 1usize;
    if downbeat_count > 0 {
        let mut spacing = width_px / downbeat_count as f32;
        while spacing < MIN_GRID_SPACING_PX && downbeat_stride < (1 << 16) {
            downbeat_stride *= 2;
            spacing *= 2.0;
        }
    }
    OverlayPlan {
        draw_beats,
        downbeat_stride,
    }
}

/// Paint a waveform display with playback cursor, beat-grid overlay, and
/// click-to-seek. `beat_marks` must be sorted by position; the overlay is
/// density-adaptive — all beats when they have room, downbeats only when
/// beats would smear, thinning to every 2^k bars (phrase markers) on
/// full-length tracks.
pub fn paint_waveform(
    ui: &mut egui::Ui,
    peaks: &WaveformPeaks,
    progress: f32,
    beat_marks: &[BeatMark],
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
            egui::Rect::from_min_max(egui::pos2(x, top), egui::pos2(x + bar_width, bottom)),
            0.0,
            color,
        );
    }

    // Beat-grid overlay: density-adaptive (see overlay_plan). Spacing is
    // estimated from the marker counts — marks are evenly spread in
    // musical time, so the mean is representative.
    if beat_marks.len() >= 2 {
        let downbeat_count = beat_marks.iter().filter(|m| m.is_downbeat).count();
        let plan = overlay_plan(rect.width(), beat_marks.len(), downbeat_count);

        let beat_stroke = egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 42),
        );
        let downbeat_stroke = egui::Stroke::new(
            2.0,
            egui::Color32::from_rgba_unmultiplied(255, 220, 130, 120),
        );

        // Countdown over downbeats: draw one, skip stride - 1.
        let mut downbeats_until_draw = 0usize;
        for mark in beat_marks {
            let (draw, stroke) = if mark.is_downbeat {
                let draw = downbeats_until_draw == 0;
                downbeats_until_draw = if draw {
                    plan.downbeat_stride - 1
                } else {
                    downbeats_until_draw - 1
                };
                (draw, downbeat_stroke)
            } else {
                (plan.draw_beats, beat_stroke)
            };
            if !draw {
                continue;
            }
            let x = rect.left() + mark.frac.clamp(0.0, 1.0) * rect.width();
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                stroke,
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_never_empty_on_long_tracks() {
        // A 6-minute 123 BPM extended mix in a default-width window: 707
        // beats, 177 downbeats at ~800 px. Beats and per-bar downbeats are
        // both too dense (1.1 px / 4.5 px), but phrase markers must remain.
        let plan = overlay_plan(800.0, 707, 177);
        assert!(!plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 2, "expected 2-bar phrase markers");
        // Drawn lines: ceil(177 / 2) = 89 -> ~9 px apart.
        assert!(800.0 / (177.0 / plan.downbeat_stride as f32) >= MIN_GRID_SPACING_PX);
    }

    #[test]
    fn overlay_full_density_on_short_loops() {
        // A 16-beat loop at any reasonable width: draw everything.
        let plan = overlay_plan(800.0, 16, 4);
        assert!(plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 1);
    }

    #[test]
    fn overlay_downbeats_only_at_medium_density() {
        // ~200 beats at 800 px: beats smear (4 px), bars fit (16 px).
        let plan = overlay_plan(800.0, 200, 50);
        assert!(!plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 1);
    }

    #[test]
    fn overlay_stride_grows_for_very_long_tracks() {
        // A 2-hour DJ mix: 17k beats, 4.3k downbeats at 800 px needs
        // 32-bar phrase markers (4300/32 = 134 lines -> ~6 px).
        let plan = overlay_plan(800.0, 17_000, 4_300);
        assert!(!plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 64);
    }

    #[test]
    fn overlay_handles_no_downbeats() {
        let plan = overlay_plan(800.0, 100, 0);
        assert_eq!(plan.downbeat_stride, 1);
    }
}
