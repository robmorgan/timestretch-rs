//! Full-track overview strip: 3-band waveform texture with the played
//! portion dimmed, bar/phrase edge ticks, loop region, position cursor,
//! and click-to-seek.

use eframe::egui;

use super::peaks::BandPeaks;
use super::{GridMarks, overlay_plan, paint_placeholder, palette, render_envelope};

/// Strip height in points.
const STRIP_HEIGHT: f32 = 48.0;
/// Texture height in pixels (2x the strip for retina crispness).
const TEX_HEIGHT: usize = 96;
/// Texture columns per coarsest-level bucket, so the interpolated
/// envelope — not per-bucket columns — defines the strip's shape.
const TEX_SUPERSAMPLE: usize = 2;
/// Bottom-edge tick heights in points.
const TICK_BAR_PX: f32 = 6.0;
const TICK_PHRASE_PX: f32 = 12.0;

/// The full track pre-rendered once per load from the coarsest pyramid
/// level, bands overlaid per column (low widest, high narrowest). Drawn
/// twice per frame: full-width with a white tint, then UV-clipped to the
/// playhead with a grey tint that dims the played part (CDJ-style).
pub struct OverviewTexture {
    tex: egui::TextureHandle,
}

impl OverviewTexture {
    pub fn from_peaks(ctx: &egui::Context, peaks: &BandPeaks) -> Self {
        let level = peaks.coarsest();
        Self {
            tex: ctx.load_texture(
                "waveform_overview",
                render_envelope(level, 0..level.num_buckets(), TEX_SUPERSAMPLE, TEX_HEIGHT),
                egui::TextureOptions::LINEAR,
            ),
        }
    }
}

pub struct OverviewParams<'a> {
    pub texture: Option<&'a OverviewTexture>,
    pub marks: &'a GridMarks,
    /// Playback position as a fraction of the track (0..1).
    pub progress: f32,
    pub total_frames: usize,
    pub loop_region: Option<(usize, usize)>,
    pub loop_in: Option<usize>,
}

/// Paint the overview strip. Returns the click-to-seek target as a track
/// fraction, if clicked.
pub fn paint_overview(ui: &mut egui::Ui, params: OverviewParams<'_>) -> Option<f32> {
    let desired_size = egui::vec2(ui.available_width(), STRIP_HEIGHT);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::click());
    let rect = response.rect;

    painter.rect_filled(rect, 4.0, palette::BACKGROUND);

    let Some(texture) = params.texture else {
        paint_placeholder(&painter, rect);
        return None;
    };

    let progress = params.progress.clamp(0.0, 1.0);
    let cursor_x = rect.left() + rect.width() * progress;

    // Unplayed across the full width, played dimmed on top (UV-clipped).
    painter.image(
        texture.tex.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    if progress > 0.0 {
        painter.image(
            texture.tex.id(),
            egui::Rect::from_min_max(rect.left_top(), egui::pos2(cursor_x, rect.bottom())),
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(progress, 1.0)),
            palette::PLAYED_TINT,
        );
    }

    let frac_x = |frame: usize| {
        rect.left() + rect.width() * (frame as f32 / params.total_frames.max(1) as f32)
    };

    // Loop region / staged loop-in.
    if let Some((start, end)) = params.loop_region {
        let (x0, x1) = (frac_x(start), frac_x(end));
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
            0.0,
            palette::LOOP_FILL,
        );
        for x in [x0, x1] {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0_f32, palette::LOOP_EDGE),
            );
        }
    } else if let Some(start) = params.loop_in {
        let x = frac_x(start);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.0_f32, palette::LOOP_EDGE),
        );
    }

    // Bar ticks along the bottom edge, phrase starts emphasized. Density
    // thins by bar-number stride so surviving ticks stay phrase-aligned.
    // Low-confidence grids draw dimmed, matching the zoomed view.
    if params.marks.is_usable() && params.total_frames > 0 {
        let (bar_color, phrase_color) = if params.marks.low_confidence() {
            (
                palette::TICK_BEAT.gamma_multiply(super::LOW_CONFIDENCE_TICK_DIM),
                palette::TICK_PHRASE.gamma_multiply(super::LOW_CONFIDENCE_TICK_DIM),
            )
        } else {
            (palette::TICK_BEAT, palette::TICK_PHRASE)
        };
        let plan = overlay_plan(
            rect.width(),
            params.marks.len(),
            params.marks.downbeat_count(),
        );
        let stride = plan.downbeat_stride as u32;
        let inv_total = 1.0 / params.total_frames as f64;
        for i in 0..params.marks.len() {
            if !params.marks.is_downbeat(i) {
                continue;
            }
            // Bars are 1-based; align the stride so bar 1 always draws.
            let bar = params.marks.bar_number(i);
            if bar == 0 || !(bar - 1).is_multiple_of(stride) {
                continue;
            }
            let phrase = params.marks.is_phrase_start(i);
            let x = rect.left()
                + rect.width() * ((params.marks.frame(i) * inv_total) as f32).clamp(0.0, 1.0);
            let (height, stroke) = if phrase {
                (TICK_PHRASE_PX, egui::Stroke::new(2.0_f32, phrase_color))
            } else {
                (TICK_BAR_PX, egui::Stroke::new(1.0_f32, bar_color))
            };
            painter.line_segment(
                [
                    egui::pos2(x, rect.bottom() - height),
                    egui::pos2(x, rect.bottom()),
                ],
                stroke,
            );
        }
    }

    // Position cursor.
    painter.line_segment(
        [
            egui::pos2(cursor_x, rect.top()),
            egui::pos2(cursor_x, rect.bottom()),
        ],
        egui::Stroke::new(1.0_f32, palette::CURSOR),
    );

    // Click-to-seek.
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        return Some(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0));
    }
    None
}
