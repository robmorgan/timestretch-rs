use timestretch::{Parameters, SessionConfig, TempoRate, render};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<_> = (0..44100)
        .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let audio = render(
        &input,
        SessionConfig {
            channels: 1,
            parameters: Parameters::keylock(TempoRate::new(0.75)?),
            ..Default::default()
        },
    )?;
    println!(
        "{} frames, {:?}, achieved tempo {}",
        audio.report.output_frames,
        audio.report.backend,
        audio.report.parameters.tempo()
    );
    Ok(())
}
