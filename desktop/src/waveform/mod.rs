//! CDJ-style deck waveforms: a zoomed scrolling view with a centered
//! playhead, a full-track overview strip, 3-band RGB peak coloring, and a
//! bar.beat counter. Shared grid/palette machinery lives here; the painters
//! live in the submodules.

mod counter;
mod overview;
mod zoomed;

/// The peaks pyramid lives in the library now (`.tsa` PEAK chunk); the
/// alias keeps the painters' `peaks::` paths reading naturally.
use timestretch::analysis::waveform as peaks;

pub use counter::paint_beat_counter;
pub use overview::{OverviewParams, OverviewTexture, paint_overview};
pub use peaks::BandPeaks;
#[cfg(test)]
pub use peaks::NUM_BANDS;
pub use zoomed::{ScrubGesture, ZoomSpan, ZoomedParams, ZoomedTiles, paint_zoomed};

use eframe::egui;

/// CDJ-3000-flavored palette shared by the deck painters.
pub(crate) mod palette {
    use eframe::egui::Color32;

    /// Panel background, near-black.
    pub const BACKGROUND: Color32 = Color32::from_rgb(10, 10, 15);
    /// Low band (kick/bass): blue.
    pub const BAND_LOW: Color32 = Color32::from_rgb(40, 90, 220);
    /// Mid band: amber.
    pub const BAND_MID: Color32 = Color32::from_rgb(230, 150, 40);
    /// High band (hats/air): near-white.
    pub const BAND_HIGH: Color32 = Color32::from_rgb(235, 235, 240);
    /// Zoomed-view playhead.
    pub const PLAYHEAD: Color32 = Color32::from_rgb(230, 40, 40);
    /// Overview position cursor.
    pub const CURSOR: Color32 = Color32::WHITE;
    /// Regular beat tick.
    pub const TICK_BEAT: Color32 = Color32::from_rgba_premultiplied(200, 200, 205, 200);
    /// Downbeat (bar start) tick.
    pub const TICK_DOWNBEAT: Color32 = Color32::from_rgb(230, 40, 40);
    /// Phrase-start tick on the overview (every 16 bars).
    pub const TICK_PHRASE: Color32 = Color32::WHITE;
    /// Loop region fill.
    pub const LOOP_FILL: Color32 = Color32::from_rgba_premultiplied(60, 40, 8, 60);
    /// Loop in/out boundary lines and staged loop-in marker.
    pub const LOOP_EDGE: Color32 = Color32::from_rgb(235, 160, 40);
    /// Placeholder / secondary text.
    pub const TEXT_DIM: Color32 = Color32::from_rgb(100, 100, 120);
    /// Grey multiply tint that dims the played part of the overview.
    pub const PLAYED_TINT: Color32 = Color32::from_rgb(110, 110, 118);
}

/// Grid-confidence threshold below which the deck presents the grid as
/// tentative: beat/downbeat ticks draw dimmed and the counter row shows a
/// "grid: low confidence" hint.
///
/// What `timestretch::BeatGrid::confidence` means:
/// - Tracked (DP) grids score 0.4 * tempogram path salience + 0.3 *
///   beat-level onset support + 0.3 * interval regularity
///   (`grid_confidence`, src/analysis/beat.rs:600-631).
/// - Rigid-adopted grids keep at least that: confidence =
///   `grid.confidence.max(fit.phase_lock)` (src/analysis/rigid_grid.rs:298),
///   so rigid adoption never lowers the reading.
///
/// Corpus evidence (benchmarks/baselines/bpm_accuracy_baseline_latest.json):
/// every real-music grid across the 16-track corpus — rigid-adopted or
/// tracked — reports confidence 0.79-0.94. 0.6 sits with clear margin below
/// that cluster, so a healthy grid never dims, while grids whose own
/// evidence collapses (weak periodicity or poor beat-level onset support:
/// ambient, rubato, speech-heavy material) fall through the salience and
/// support terms of the formula and flag.
///
/// Honest limit: this flag cannot catch a wandering DP grid on quantized
/// material — somebody-to-love-extended-mix measures confidence 0.845 with
/// beat F 0.31 in the same baseline, because the metric scores internal
/// consistency, not ground truth. That failure class is handled upstream by
/// rigid-grid adoption and tracked-beat corroboration
/// (src/analysis/rigid_grid.rs:45-61), not by this display threshold.
pub(crate) const LOW_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// Gamma-space multiplier applied to tick colors on a low-confidence grid
/// (~40% alpha versions of the palette colors).
pub(crate) const LOW_CONFIDENCE_TICK_DIM: f32 = 0.4;

/// Beats in a bar for the counter/phrase math. The Stage 10 grid carries a
/// 4/4 prior; bars with other beat counts wrap modulo 4 for display.
const BEATS_PER_BAR: usize = 4;
/// Bars per phrase for the overview's emphasized ticks.
const BARS_PER_PHRASE: u32 = 16;

/// Frame-based beat-grid cache for the painters, built once per track load
/// from the detected [`timestretch::BeatGrid`]. Everything a painter needs
/// per frame is a binary search plus indexed lookups — no per-frame
/// allocation.
pub struct GridMarks {
    /// Fractional-sample beat positions, ascending.
    frames: Vec<f64>,
    /// Downbeat flag per beat.
    downbeat: Vec<bool>,
    /// 1-based bar number per beat; 0 for beats before the first downbeat.
    bar_of: Vec<u32>,
    /// 1-based beat-within-bar per beat (1..=4).
    beat_in_bar: Vec<u8>,
    /// Median beat interval in frames (0.0 when fewer than 2 beats).
    median_beat_frames: f64,
    /// Whether the detector's grid confidence fell below
    /// [`LOW_CONFIDENCE_THRESHOLD`]; painters dim their ticks and the
    /// counter row shows a hint.
    low_confidence: bool,
}

impl GridMarks {
    pub fn empty() -> Self {
        Self {
            frames: Vec::new(),
            downbeat: Vec::new(),
            bar_of: Vec::new(),
            beat_in_bar: Vec::new(),
            median_beat_frames: 0.0,
            low_confidence: false,
        }
    }

    pub fn from_grid(grid: &timestretch::BeatGrid) -> Self {
        let frames = grid.beats.clone();
        let mut downbeat = vec![false; frames.len()];
        for &idx in &grid.downbeats {
            if let Some(flag) = downbeat.get_mut(idx) {
                *flag = true;
            }
        }

        let first_downbeat = downbeat.iter().position(|&d| d);
        let mut bar_of = vec![0u32; frames.len()];
        let mut beat_in_bar = vec![0u8; frames.len()];
        let mut bar = 0u32;
        let mut last_downbeat: Option<usize> = None;
        for i in 0..frames.len() {
            if downbeat[i] {
                bar += 1;
                last_downbeat = Some(i);
            }
            bar_of[i] = bar;
            beat_in_bar[i] = match (last_downbeat, first_downbeat) {
                // At or after a downbeat: count forward from it.
                (Some(d), _) => ((i - d) % BEATS_PER_BAR + 1) as u8,
                // Before the first downbeat: count backward from it, so the
                // beat right before a bar start reads as beat 4.
                (None, Some(d0)) => {
                    ((BEATS_PER_BAR - (d0 - i) % BEATS_PER_BAR) % BEATS_PER_BAR + 1) as u8
                }
                // No downbeats detected at all: free-running 1..=4.
                (None, None) => (i % BEATS_PER_BAR + 1) as u8,
            };
        }

        let median_beat_frames = if frames.len() >= 2 {
            let mut intervals: Vec<f64> = frames.windows(2).map(|w| w[1] - w[0]).collect();
            intervals.sort_by(f64::total_cmp);
            intervals[intervals.len() / 2]
        } else {
            0.0
        };

        Self {
            frames,
            downbeat,
            bar_of,
            beat_in_bar,
            median_beat_frames,
            low_confidence: grid.confidence < LOW_CONFIDENCE_THRESHOLD,
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether there is enough grid to draw/count against.
    pub fn is_usable(&self) -> bool {
        self.frames.len() >= 2
    }

    /// Whether the detector reported this grid below
    /// [`LOW_CONFIDENCE_THRESHOLD`] (ticks dim, counter hints).
    pub fn low_confidence(&self) -> bool {
        self.low_confidence
    }

    pub fn frame(&self, i: usize) -> f64 {
        self.frames[i]
    }

    pub fn is_downbeat(&self, i: usize) -> bool {
        self.downbeat[i]
    }

    /// Whether beat `i` starts a 16-bar phrase (bars 1, 17, 33, …).
    pub fn is_phrase_start(&self, i: usize) -> bool {
        self.downbeat[i] && self.bar_of[i] % BARS_PER_PHRASE == 1
    }

    /// 1-based bar number of beat `i` (0 before the first downbeat). Tick
    /// thinning anchors on this so strided ticks stay phrase-aligned.
    pub fn bar_number(&self, i: usize) -> u32 {
        self.bar_of[i]
    }

    pub fn downbeat_count(&self) -> usize {
        self.downbeat.iter().filter(|&&d| d).count()
    }

    /// Median beat interval in frames; 0.0 without a usable grid.
    pub fn median_beat_frames(&self) -> f64 {
        self.median_beat_frames
    }

    /// Indices of beats within `[start_frame, end_frame)`.
    pub fn visible_range(&self, start_frame: f64, end_frame: f64) -> std::ops::Range<usize> {
        let lo = self.frames.partition_point(|&f| f < start_frame);
        let hi = self.frames.partition_point(|&f| f < end_frame);
        lo..hi
    }

    /// Index of the last beat at or before `frame`.
    pub fn beat_at_or_before(&self, frame: f64) -> Option<usize> {
        self.frames.partition_point(|&f| f <= frame).checked_sub(1)
    }

    /// `(bar, beat_in_bar)` at a playback position: bar is 1-based (0 while
    /// before the first downbeat), beat is 1..=4. `None` before the first
    /// beat or without a usable grid.
    pub fn bar_beat(&self, frame: f64) -> Option<(u32, u8)> {
        if !self.is_usable() {
            return None;
        }
        let i = self.beat_at_or_before(frame)?;
        Some((self.bar_of[i], self.beat_in_bar[i]))
    }
}

/// Minimum pixel spacing between adjacent grid lines before a marker tier
/// is drawn at full density.
pub(crate) const MIN_GRID_SPACING_PX: f32 = 6.0;

/// How an overlay adapts the grid to the available width: whether
/// individual beats fit, and how many bars each drawn downbeat tick spans
/// (1 = every downbeat; 2/4/8… = thinned when bars are too dense).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OverlayPlan {
    pub draw_beats: bool,
    pub downbeat_stride: usize,
}

/// Chooses the densest tier that keeps adjacent ticks at least
/// [`MIN_GRID_SPACING_PX`] apart. Downbeats never disappear entirely —
/// they thin to every 2^k bars instead, so a full-length track still shows
/// its phrase structure.
pub(crate) fn overlay_plan(width_px: f32, beat_count: usize, downbeat_count: usize) -> OverlayPlan {
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

/// Rasterizes a range of a peak level's buckets into a transparent-
/// background image, one column per bucket. Bands paint in high → mid →
/// low order — low on top — so kick-heavy passages read blue and highs
/// surface only where the lows drop out (CDJ RGB semantics; in a dense
/// master the high band's *peak* is near full scale everywhere and would
/// bury the image if on top).
pub(crate) fn render_columns(
    level: &peaks::PeakLevel,
    buckets: std::ops::Range<usize>,
    height: usize,
) -> egui::ColorImage {
    let width = buckets.len().max(1);
    let mut image = egui::ColorImage::new([width, height], egui::Color32::TRANSPARENT);
    let center = height as f32 / 2.0;
    let half_height = height as f32 * 0.45;
    let band_colors = [palette::BAND_LOW, palette::BAND_MID, palette::BAND_HIGH];
    for (x, b) in buckets.enumerate() {
        for (band, &color) in band_colors.iter().enumerate().rev() {
            let pos = level.pos[band][b].clamp(0.0, 1.0);
            let neg = level.neg[band][b].clamp(-1.0, 0.0);
            let top = (center - pos * half_height).floor().max(0.0) as usize;
            let bottom = ((center - neg * half_height).ceil() as usize).min(height);
            for y in top..bottom {
                image.pixels[y * width + x] = color;
            }
        }
    }
    image
}

/// Paint the "load a file" placeholder shared by both waveform panels.
pub(crate) fn paint_placeholder(painter: &egui::Painter, rect: egui::Rect) {
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "Load an audio file to see waveform",
        egui::FontId::proportional(14.0),
        palette::TEXT_DIM,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_columns_subrange_matches_full_render() {
        let buckets = 64;
        let level = peaks::PeakLevel {
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
        };
        let height = 32;
        let full = render_columns(&level, 0..buckets, height);
        let (lo, hi) = (10, 20);
        let sub = render_columns(&level, lo..hi, height);
        for y in 0..height {
            for x in 0..(hi - lo) {
                assert_eq!(
                    sub.pixels[y * (hi - lo) + x],
                    full.pixels[y * buckets + lo + x],
                    "pixel mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn overlay_never_empty_on_long_tracks() {
        // A 6-minute 123 BPM extended mix in a default-width window: 707
        // beats, 177 downbeats at ~800 px. Beats and per-bar downbeats are
        // both too dense (1.1 px / 4.5 px), but thinned bar ticks remain.
        let plan = overlay_plan(800.0, 707, 177);
        assert!(!plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 2, "expected 2-bar tick thinning");
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
        // 64-bar tick thinning (4300/64 = 67 ticks -> ~12 px).
        let plan = overlay_plan(800.0, 17_000, 4_300);
        assert!(!plan.draw_beats);
        assert_eq!(plan.downbeat_stride, 64);
    }

    #[test]
    fn overlay_handles_no_downbeats() {
        let plan = overlay_plan(800.0, 100, 0);
        assert_eq!(plan.downbeat_stride, 1);
    }

    /// A grid of 16 beats at one-second intervals (sr 100 for readable
    /// numbers), downbeats every 4 beats starting at beat index 2.
    fn test_grid() -> GridMarks {
        let mut grid = timestretch::BeatGrid::empty(100);
        grid.beats = (0..16).map(|i| i as f64 * 100.0).collect();
        grid.downbeats = vec![2, 6, 10, 14];
        grid.bpm = 60.0;
        GridMarks::from_grid(&grid)
    }

    #[test]
    fn bar_beat_counts_within_bars() {
        let marks = test_grid();
        // Beat idx 2 is the first downbeat -> bar 1, beat 1.
        assert_eq!(marks.bar_beat(200.0), Some((1, 1)));
        assert_eq!(marks.bar_beat(300.0), Some((1, 2)));
        assert_eq!(marks.bar_beat(599.0), Some((1, 4)));
        assert_eq!(marks.bar_beat(600.0), Some((2, 1)));
        // Position between beats belongs to the last passed beat.
        assert_eq!(marks.bar_beat(250.0), Some((1, 1)));
    }

    #[test]
    fn bar_beat_before_first_downbeat_counts_backward() {
        let marks = test_grid();
        // Beats 0 and 1 precede the first downbeat (bar 0); the beat right
        // before a bar start reads as beat 4.
        assert_eq!(marks.bar_beat(0.0), Some((0, 3)));
        assert_eq!(marks.bar_beat(100.0), Some((0, 4)));
        // Before the first beat entirely: no reading.
        assert_eq!(marks.bar_beat(-1.0), None);
    }

    #[test]
    fn bar_beat_without_downbeats_free_runs() {
        let mut grid = timestretch::BeatGrid::empty(100);
        grid.beats = (0..8).map(|i| i as f64 * 100.0).collect();
        let marks = GridMarks::from_grid(&grid);
        assert_eq!(marks.bar_beat(0.0), Some((0, 1)));
        assert_eq!(marks.bar_beat(400.0), Some((0, 1)));
        assert_eq!(marks.bar_beat(700.0), Some((0, 4)));
    }

    #[test]
    fn visible_range_is_half_open() {
        let marks = test_grid();
        assert_eq!(marks.visible_range(200.0, 600.0), 2..6);
        assert_eq!(marks.visible_range(-50.0, 50.0), 0..1);
        assert_eq!(marks.visible_range(2000.0, 3000.0), 16..16);
    }

    #[test]
    fn phrase_starts_on_bar_1_17_33() {
        let mut grid = timestretch::BeatGrid::empty(100);
        grid.beats = (0..80).map(|i| i as f64 * 100.0).collect();
        grid.downbeats = (0..20).map(|b| b * 4).collect();
        let marks = GridMarks::from_grid(&grid);
        let phrase_beats: Vec<usize> = (0..marks.len())
            .filter(|&i| marks.is_phrase_start(i))
            .collect();
        // Bars 1 and 17 -> beat indices 0 and 64.
        assert_eq!(phrase_beats, vec![0, 64]);
    }

    #[test]
    fn low_confidence_flag_tracks_grid_confidence() {
        let mut grid = timestretch::BeatGrid::empty(100);
        grid.beats = (0..16).map(|i| i as f64 * 100.0).collect();
        grid.downbeats = vec![0, 4, 8, 12];
        grid.bpm = 60.0;

        // Corpus-healthy reading (tracked or rigid-adopted): not flagged.
        grid.confidence = 0.85;
        assert!(!GridMarks::from_grid(&grid).low_confidence());

        // Collapsed evidence (weak periodicity / onset support): flagged.
        grid.confidence = 0.3;
        assert!(GridMarks::from_grid(&grid).low_confidence());

        // Exactly at the threshold: not low — the gate is strict-less-than.
        grid.confidence = LOW_CONFIDENCE_THRESHOLD;
        assert!(!GridMarks::from_grid(&grid).low_confidence());
    }

    #[test]
    fn empty_marks_are_not_flagged_low_confidence() {
        assert!(!GridMarks::empty().low_confidence());
    }

    #[test]
    fn median_interval_ignores_outliers() {
        let mut grid = timestretch::BeatGrid::empty(100);
        // Regular 100-frame intervals with one 500-frame gap.
        grid.beats = vec![0.0, 100.0, 200.0, 300.0, 800.0, 900.0, 1000.0];
        let marks = GridMarks::from_grid(&grid);
        assert_eq!(marks.median_beat_frames(), 100.0);
    }
}
