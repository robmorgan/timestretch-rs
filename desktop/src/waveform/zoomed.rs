//! Zoomed scrolling waveform: playhead fixed at horizontal center, the
//! track scrolls underneath. 3-band bars are tessellated per frame from
//! the pyramid level closest to one bucket per pixel (the content moves
//! every frame, so a texture would need constant re-upload; a few thousand
//! rects at the 30 fps repaint cap is cheaper). Beat/downbeat edge ticks,
//! loop overlay, and drag-to-scrub.

use eframe::egui;

use super::peaks::BandPeaks;
use super::{overlay_plan, paint_placeholder, palette, GridMarks};

/// View height in points.
const VIEW_HEIGHT: f32 = 160.0;
/// Edge tick heights in points.
const TICK_BEAT_PX: f32 = 8.0;
const TICK_DOWNBEAT_PX: f32 = 14.0;
/// Scroll distance (points) per zoom step on wheel/trackpad zoom.
const SCROLL_PER_ZOOM_STEP: f32 = 40.0;

/// Zoom presets: bars when a grid exists, seconds otherwise. Same index
/// into both tables so toggling grids keeps a comparable span.
const BAR_PRESETS: [f64; 5] = [1.0, 2.0, 4.0, 8.0, 16.0];
const SEC_PRESETS: [f64; 5] = [2.0, 4.0, 8.0, 16.0, 32.0];
const DEFAULT_PRESET: usize = 2;

/// Visible-span state for the zoomed view.
pub struct ZoomSpan {
    idx: usize,
    /// Accumulated scroll distance toward the next wheel-zoom step.
    scroll_accum: f32,
}

impl Default for ZoomSpan {
    fn default() -> Self {
        Self {
            idx: DEFAULT_PRESET,
            scroll_accum: 0.0,
        }
    }
}

impl ZoomSpan {
    pub fn zoom_in(&mut self) {
        self.idx = self.idx.saturating_sub(1);
    }

    pub fn zoom_out(&mut self) {
        self.idx = (self.idx + 1).min(BAR_PRESETS.len() - 1);
    }

    /// Label for the zoom control, e.g. "4 BARS" or "8 s".
    pub fn label(&self, has_grid: bool) -> String {
        if has_grid {
            let bars = BAR_PRESETS[self.idx];
            if bars == 1.0 {
                "1 BAR".to_string()
            } else {
                format!("{bars:.0} BARS")
            }
        } else {
            format!("{:.0} s", SEC_PRESETS[self.idx])
        }
    }

    /// Visible span in source frames.
    fn span_frames(&self, marks: &GridMarks, sample_rate: u32) -> f64 {
        let beat = marks.median_beat_frames();
        if marks.is_usable() && beat > 0.0 {
            BAR_PRESETS[self.idx] * 4.0 * beat
        } else {
            SEC_PRESETS[self.idx] * sample_rate.max(1) as f64
        }
    }

    /// Step the zoom from accumulated scroll input; scrolling up zooms in.
    fn apply_scroll(&mut self, delta_y: f32) {
        self.scroll_accum += delta_y;
        while self.scroll_accum >= SCROLL_PER_ZOOM_STEP {
            self.zoom_in();
            self.scroll_accum -= SCROLL_PER_ZOOM_STEP;
        }
        while self.scroll_accum <= -SCROLL_PER_ZOOM_STEP {
            self.zoom_out();
            self.scroll_accum += SCROLL_PER_ZOOM_STEP;
        }
    }
}

pub struct ZoomedParams<'a> {
    pub peaks: Option<&'a BandPeaks>,
    pub marks: &'a GridMarks,
    pub position_frames: f64,
    pub total_frames: usize,
    pub sample_rate: u32,
    pub loop_region: Option<(usize, usize)>,
    pub loop_in: Option<usize>,
}

/// Paint the zoomed view. Returns a relative scrub distance in source
/// frames while the user drags (content follows the pointer, so dragging
/// right moves the position backward).
pub fn paint_zoomed(
    ui: &mut egui::Ui,
    params: ZoomedParams<'_>,
    span: &mut ZoomSpan,
) -> Option<f64> {
    let desired_size = egui::vec2(ui.available_width(), VIEW_HEIGHT);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::drag());
    let rect = response.rect;

    painter.rect_filled(rect, 4.0, palette::BACKGROUND);

    let (Some(peaks), true) = (params.peaks, params.total_frames > 0) else {
        paint_placeholder(&painter, rect);
        return None;
    };

    if response.hovered() {
        span.apply_scroll(ui.input(|i| i.smooth_scroll_delta.y));
    }

    let span_frames = span.span_frames(params.marks, params.sample_rate).max(1.0);
    let px_per_frame = rect.width() as f64 / span_frames;
    let start_frame = params.position_frames - span_frames / 2.0;
    let end_frame = start_frame + span_frames;
    let frame_to_x = |frame: f64| rect.left() + ((frame - start_frame) * px_per_frame) as f32;

    // 3-band bars from the pyramid level nearest one bucket per pixel.
    let level = peaks.level_for((px_per_frame * params.sample_rate as f64) as f32);
    let frames_per_bucket = params.sample_rate as f64 / level.buckets_per_sec;
    let first_bucket = (start_frame / frames_per_bucket).floor().max(0.0) as usize;
    let last_bucket = ((end_frame / frames_per_bucket).ceil() as usize).min(level.num_buckets());
    let center_y = rect.center().y;
    let half_height = rect.height() * 0.45;
    let band_colors = [palette::BAND_LOW, palette::BAND_MID, palette::BAND_HIGH];
    for b in first_bucket..last_bucket {
        let x0 = frame_to_x(b as f64 * frames_per_bucket).max(rect.left());
        let x1 = frame_to_x((b + 1) as f64 * frames_per_bucket).min(rect.right());
        if x1 <= x0 {
            continue;
        }
        // Low paints last (on top): see overview::render_level.
        for (band, &color) in band_colors.iter().enumerate().rev() {
            let pos = level.pos[band][b].clamp(0.0, 1.0);
            let neg = level.neg[band][b].clamp(-1.0, 0.0);
            if pos == 0.0 && neg == 0.0 {
                continue;
            }
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0, center_y - pos * half_height),
                    egui::pos2(x1, center_y - neg * half_height),
                ),
                0.0,
                color,
            );
        }
    }

    // Loop overlay: fill plus full-height boundary lines where in view.
    if let Some((start, end)) = params.loop_region {
        let x0 = frame_to_x(start as f64);
        let x1 = frame_to_x(end as f64);
        if x1 > rect.left() && x0 < rect.right() {
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0.max(rect.left()), rect.top()),
                    egui::pos2(x1.min(rect.right()), rect.bottom()),
                ),
                0.0,
                palette::LOOP_FILL,
            );
        }
        for x in [x0, x1] {
            if x >= rect.left() && x <= rect.right() {
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(1.5, palette::LOOP_EDGE),
                );
            }
        }
    } else if let Some(start) = params.loop_in {
        let x = frame_to_x(start as f64);
        if x >= rect.left() && x <= rect.right() {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.5, palette::LOOP_EDGE),
            );
        }
    }

    // Beat/downbeat edge ticks (top and bottom), density-adaptive.
    if params.marks.is_usable() {
        let visible = params.marks.visible_range(start_frame, end_frame);
        let downbeats = visible
            .clone()
            .filter(|&i| params.marks.is_downbeat(i))
            .count();
        let plan = overlay_plan(rect.width(), visible.len(), downbeats);
        let stride = plan.downbeat_stride as u32;
        for i in visible {
            let is_downbeat = params.marks.is_downbeat(i);
            let (height, stroke) = if is_downbeat {
                let bar = params.marks.bar_number(i);
                if bar == 0 || !(bar - 1).is_multiple_of(stride) {
                    continue;
                }
                (
                    TICK_DOWNBEAT_PX,
                    egui::Stroke::new(2.0, palette::TICK_DOWNBEAT),
                )
            } else {
                if !plan.draw_beats {
                    continue;
                }
                (TICK_BEAT_PX, egui::Stroke::new(1.0, palette::TICK_BEAT))
            };
            let x = frame_to_x(params.marks.frame(i));
            painter.line_segment(
                [
                    egui::pos2(x, rect.top()),
                    egui::pos2(x, rect.top() + height),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(x, rect.bottom() - height),
                    egui::pos2(x, rect.bottom()),
                ],
                stroke,
            );
        }
    }

    // Fixed centered playhead — the one full-height line in this view.
    let center_x = rect.center().x;
    painter.line_segment(
        [
            egui::pos2(center_x, rect.top()),
            egui::pos2(center_x, rect.bottom()),
        ],
        egui::Stroke::new(2.0, palette::PLAYHEAD),
    );

    // Drag-to-scrub: content follows the pointer.
    if response.dragged() {
        let dx = response.drag_delta().x;
        if dx != 0.0 {
            return Some(-(dx as f64) / px_per_frame);
        }
    }
    None
}
