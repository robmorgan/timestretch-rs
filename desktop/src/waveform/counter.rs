//! CDJ-style beat counter: `BAR n.b` readout plus a 4-segment indicator
//! with the current beat-in-bar lit. Doubles as a live check of the
//! detected grid — wrong downbeats are immediately visible here.

use eframe::egui;

use super::{GridMarks, palette};

/// Segment size and gap in points.
const SEG_SIZE: egui::Vec2 = egui::vec2(16.0, 10.0);
const SEG_GAP: f32 = 4.0;
const SEGMENTS: u8 = 4;

/// Paint the readout + segments. Shows placeholders before the first beat;
/// paints nothing without a usable grid.
pub fn paint_beat_counter(ui: &mut egui::Ui, marks: &GridMarks, position_frames: f64) {
    if !marks.is_usable() {
        return;
    }
    let bar_beat = marks.bar_beat(position_frames);

    let text = match bar_beat {
        Some((bar, beat)) => format!("BAR {bar:>3}.{beat}"),
        None => "BAR   -.-".to_string(),
    };
    ui.label(egui::RichText::new(text).monospace().strong());

    let width = SEGMENTS as f32 * SEG_SIZE.x + (SEGMENTS - 1) as f32 * SEG_GAP;
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(width, SEG_SIZE.y.max(ui.spacing().interact_size.y)),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    let top = rect.center().y - SEG_SIZE.y / 2.0;
    for seg in 1..=SEGMENTS {
        let x = rect.left() + (seg - 1) as f32 * (SEG_SIZE.x + SEG_GAP);
        let seg_rect = egui::Rect::from_min_size(egui::pos2(x, top), SEG_SIZE);
        if bar_beat.is_some_and(|(_, beat)| beat == seg) {
            painter.rect_filled(seg_rect, 2.0, palette::PLAYHEAD);
        } else {
            painter.rect_stroke(
                seg_rect,
                2.0,
                egui::Stroke::new(1.0_f32, palette::TEXT_DIM),
                egui::StrokeKind::Inside,
            );
        }
    }
}
