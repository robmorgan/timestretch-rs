mod app;
#[cfg(not(target_arch = "wasm32"))]
mod audio_engine;
mod backend;
mod deck;
mod decoder;
mod delivery;
mod output_queue;
mod playback;
mod render_worker;
mod scrub;
mod state;
mod waveform;
#[cfg(target_arch = "wasm32")]
mod web_audio;
mod worker_priority;

#[cfg(not(target_arch = "wasm32"))]
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

/// Browser entry point: run the same app on the page's `<canvas id="app">`.
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .and_then(|window| window.document())
            .expect("no document");
        let canvas = document
            .get_element_by_id("app")
            .and_then(|element| element.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("missing <canvas id=\"app\">");
        let started = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(app::TimeStretchApp::new(cc, None)))),
            )
            .await;
        // Replace the loading note with the outcome.
        if let Some(note) = document.get_element_by_id("loading") {
            match started {
                Ok(()) => note.remove(),
                Err(error) => {
                    note.set_text_content(Some(&format!("The app failed to start: {error:?}")));
                }
            }
        }
    });
}
