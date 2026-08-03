//! Zoomed scrolling waveform: playhead fixed at horizontal center, the
//! track scrolls underneath. 3-band bars are rasterized into track-space
//! texture tiles from the pyramid level closest to one bucket per pixel,
//! cached per (level, tile), so a frame of scrolling is just a few textured
//! quads at subpixel offsets — cheap enough to repaint at full display
//! rate. Beat/downbeat edge ticks, loop overlay, and drag-to-scrub stay
//! immediate-mode on top.

use std::collections::HashMap;

use eframe::egui;

use super::peaks::{BandPeaks, PeakLevel};
use super::{GridMarks, overlay_plan, paint_placeholder, palette, render_columns};

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

/// Buckets per cached tile.
const TILE_BUCKETS: usize = 512;
/// Tile texture height in pixels (2x the 160pt view for retina crispness).
const TILE_TEX_HEIGHT: usize = 320;
/// Tile cache cap. A viewport spans ~4 tiles, so this comfortably holds
/// the active zoom level plus a few recently visited ones.
const TILE_CACHE_CAP: usize = 32;

/// Cache of rasterized waveform tiles for the zoomed view, keyed by
/// (pyramid level, tile index). Tiles live in track space, so scrolling
/// never invalidates them — they rasterize once when first visible and
/// evict least-recently-used. Must be cleared when the track (peaks)
/// changes.
#[derive(Default)]
pub struct ZoomedTiles {
    tiles: HashMap<(usize, usize), CachedTile>,
    /// Monotonic paint counter stamping tile use, for LRU eviction.
    clock: u64,
}

struct CachedTile {
    tex: egui::TextureHandle,
    last_used: u64,
}

impl ZoomedTiles {
    /// Drop every tile (the peaks they rasterized are gone).
    pub fn clear(&mut self) {
        self.tiles.clear();
    }

    /// The texture for tile `tile_idx` of pyramid level `level_idx`,
    /// rasterizing it on first use.
    fn get_or_render(
        &mut self,
        ctx: &egui::Context,
        level: &PeakLevel,
        level_idx: usize,
        tile_idx: usize,
    ) -> egui::TextureId {
        let clock = self.clock;
        let entry = self.tiles.entry((level_idx, tile_idx)).or_insert_with(|| {
            let b0 = tile_idx * TILE_BUCKETS;
            let b1 = ((tile_idx + 1) * TILE_BUCKETS).min(level.num_buckets());
            CachedTile {
                tex: ctx.load_texture(
                    format!("wave_tile_{level_idx}_{tile_idx}"),
                    render_columns(level, b0..b1, TILE_TEX_HEIGHT),
                    egui::TextureOptions::LINEAR,
                ),
                last_used: clock,
            }
        });
        entry.last_used = clock;
        let id = entry.tex.id();
        if self.tiles.len() > TILE_CACHE_CAP {
            self.evict_lru();
        }
        id
    }

    /// Drop the least-recently-used tile; never one stamped this paint.
    fn evict_lru(&mut self) {
        if let Some((&key, _)) = self
            .tiles
            .iter()
            .filter(|(_, t)| t.last_used < self.clock)
            .min_by_key(|(_, t)| t.last_used)
        {
            self.tiles.remove(&key);
        }
    }
}

/// Drag lifecycle of the zoomed view, for audible scrubbing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScrubGesture {
    /// Pointer moved while dragging: relative scrub distance in source
    /// frames (content follows the pointer, so dragging right moves the
    /// position backward).
    Drag(f64),
    /// The drag ended this frame.
    Release,
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

/// Paint the zoomed view. Reports the drag lifecycle while the user
/// scrubs: pointer deltas as [`ScrubGesture::Drag`] and the drop as
/// [`ScrubGesture::Release`].
pub fn paint_zoomed(
    ui: &mut egui::Ui,
    params: ZoomedParams<'_>,
    span: &mut ZoomSpan,
    tiles: &mut ZoomedTiles,
) -> Option<ScrubGesture> {
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

    // 3-band bars from the pyramid level nearest one bucket per pixel,
    // via cached track-space tiles: each visible tile is one textured quad
    // at a subpixel offset, scissored to the panel rect.
    let level_idx = peaks.level_index_for((px_per_frame * params.sample_rate as f64) as f32);
    let level = peaks.level(level_idx);
    let frames_per_bucket = params.sample_rate as f64 / level.buckets_per_sec;
    let first_bucket = (start_frame / frames_per_bucket).floor().max(0.0) as usize;
    let last_bucket = ((end_frame / frames_per_bucket).ceil() as usize).min(level.num_buckets());
    tiles.clock += 1;
    if first_bucket < last_bucket {
        let clipped = painter.with_clip_rect(rect);
        let full_uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        for t in first_bucket / TILE_BUCKETS..=(last_bucket - 1) / TILE_BUCKETS {
            let b0 = t * TILE_BUCKETS;
            let b1 = ((t + 1) * TILE_BUCKETS).min(level.num_buckets());
            let tex = tiles.get_or_render(ui.ctx(), level, level_idx, t);
            clipped.image(
                tex,
                egui::Rect::from_min_max(
                    egui::pos2(frame_to_x(b0 as f64 * frames_per_bucket), rect.top()),
                    egui::pos2(frame_to_x(b1 as f64 * frames_per_bucket), rect.bottom()),
                ),
                full_uv,
                egui::Color32::WHITE,
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
                    egui::Stroke::new(1.5_f32, palette::LOOP_EDGE),
                );
            }
        }
    } else if let Some(start) = params.loop_in {
        let x = frame_to_x(start as f64);
        if x >= rect.left() && x <= rect.right() {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.5_f32, palette::LOOP_EDGE),
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
                    egui::Stroke::new(2.0_f32, palette::TICK_DOWNBEAT),
                )
            } else {
                if !plan.draw_beats {
                    continue;
                }
                (TICK_BEAT_PX, egui::Stroke::new(1.0_f32, palette::TICK_BEAT))
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
        egui::Stroke::new(2.0_f32, palette::PLAYHEAD),
    );

    // Drag-to-scrub: content follows the pointer.
    if response.drag_stopped() {
        return Some(ScrubGesture::Release);
    }
    if response.dragged() {
        let dx = response.drag_delta().x;
        return Some(ScrubGesture::Drag(-(dx as f64) / px_per_frame));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A level with per-bucket variation so rendered tiles are non-trivial.
    fn synthetic_level(buckets: usize) -> PeakLevel {
        PeakLevel {
            buckets_per_sec: 150.0,
            pos: std::array::from_fn(|band| {
                (0..buckets)
                    .map(|b| ((b + band) % 7) as f32 / 7.0)
                    .collect()
            }),
            neg: std::array::from_fn(|band| {
                (0..buckets)
                    .map(|b| -(((b + band) % 5) as f32) / 5.0)
                    .collect()
            }),
        }
    }

    #[test]
    fn tile_cache_reuses_rendered_tiles() {
        let ctx = egui::Context::default();
        let level = synthetic_level(2 * TILE_BUCKETS);
        let mut tiles = ZoomedTiles::default();
        tiles.clock += 1;
        let first = tiles.get_or_render(&ctx, &level, 0, 1);
        tiles.clock += 1;
        let again = tiles.get_or_render(&ctx, &level, 0, 1);
        assert_eq!(first, again);
        assert_eq!(tiles.tiles.len(), 1);
    }

    #[test]
    fn tile_cache_evicts_lru_at_cap() {
        let ctx = egui::Context::default();
        let total = TILE_CACHE_CAP + 8;
        let level = synthetic_level(total * TILE_BUCKETS);
        let mut tiles = ZoomedTiles::default();
        for t in 0..total {
            tiles.clock += 1;
            tiles.get_or_render(&ctx, &level, 0, t);
            assert!(tiles.tiles.len() <= TILE_CACHE_CAP);
        }
        // The most recent tile survives; the oldest was evicted.
        assert!(tiles.tiles.contains_key(&(0, total - 1)));
        assert!(!tiles.tiles.contains_key(&(0, 0)));
    }
}
