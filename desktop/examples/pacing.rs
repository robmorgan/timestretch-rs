//! Minimal repaint-pacing probe: an empty egui window repainting at full
//! display rate, counting frames that arrive more than 1.5 vsync slots
//! (12.5 ms at 120 Hz) after their predecessor. Establishes the eframe/
//! winit/compositor baseline miss rate on this machine, independent of the
//! deck app — anything the app adds shows up as a higher rate than this.
//!
//! Measured 2026-08: ~2 misses/s on a ProMotion MBP under normal desktop
//! load — the same rate as the full deck app (whose frame CPU is ~200 µs),
//! and unchanged on eframe 0.35. The residual scroll twitch is this
//! eframe/macOS pacing floor, not app work.

use std::time::Instant;

struct Probe {
    start: Instant,
    last: Instant,
    dts_ms: Vec<f64>,
}

impl eframe::App for Probe {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt_ms = now.duration_since(self.last).as_secs_f64() * 1000.0;
        self.last = now;
        // Skip the first 2 s of window/session setup noise.
        if self.start.elapsed().as_secs_f64() > 2.0 {
            self.dts_ms.push(dt_ms);
        }
        if self.start.elapsed().as_secs_f64() > 14.0 {
            let secs = 12.0;
            let misses: Vec<&f64> = self.dts_ms.iter().filter(|d| **d > 12.5).collect();
            println!(
                "frames {} over {secs:.0}s, misses {} ({:.2}/s): {:?}",
                self.dts_ms.len(),
                misses.len(),
                misses.len() as f64 / secs,
                misses
                    .iter()
                    .map(|d| (**d * 100.0).round() / 100.0)
                    .collect::<Vec<_>>(),
            );
            std::process::exit(0);
        }
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!("{dt_ms:.2} ms"));
        });
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "pacing probe",
        Default::default(),
        Box::new(|_| {
            Ok(Box::new(Probe {
                start: Instant::now(),
                last: Instant::now(),
                dts_ms: Vec::new(),
            }))
        }),
    )
}
