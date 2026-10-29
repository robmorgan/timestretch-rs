mod app;
mod audio_engine;
mod backend;
mod deck;
mod decoder;
mod delivery;
mod output_queue;
mod render_worker;
mod scrub;
mod state;
mod waveform;
mod worker_priority;

fn main() -> eframe::Result<()> {
    env_logger::init();

    // Optional audio file to load on startup (skips the file dialog).
    let initial_file = std::env::args().nth(1).map(std::path::PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 600.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Timestretch Desktop",
        options,
        Box::new(|cc| Ok(Box::new(app::TimeStretchApp::new(cc, initial_file)))),
    )
}
