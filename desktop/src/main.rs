mod app;
mod audio_engine;
mod decoder;
mod processor;
mod state;
mod waveform;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 500.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Timestretch Desktop",
        options,
        Box::new(|cc| Ok(Box::new(app::TimeStretchApp::new(cc)))),
    )
}
