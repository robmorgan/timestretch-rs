//! Phase 0 spike for the self-paced Metal waveform layer.
//!
//! Attaches a child NSView with a CAMetalLayer over an eframe window and
//! renders a moving bar from a dedicated thread. Pacing does not rely on
//! wakeup timing at all: the loop blocks on drawable back-pressure
//! (`nextDrawable`, 3-deep queue) and pins every frame to an explicit
//! future vsync with `presentDrawable(_:atTime:)`, so wakeup jitter up to
//! a full queue depth never reaches the glass. The animation is rendered
//! for its scheduled presentation instant, not for "now".
//!
//! Stats print to stdout every ~5 s, measured from the compositor's
//! actual presentation timestamps (`addPresentedHandler`), not thread
//! wakeups — ground truth for missed slots. The egui window above draws
//! the same bar at eframe pacing (~2 misses/s floor, see
//! `examples/pacing.rs`) as the visual control.
//!
//! ## Phase 0 verdict (2026-08, ProMotion MBP under desktop load): NO-GO
//!
//! Missed 120 Hz slots per second, by strategy (glass-time measured):
//!
//! | strategy                                            | misses/s |
//! |-----------------------------------------------------|----------|
//! | eframe empty window (`examples/pacing.rs` baseline) | ~2.0     |
//! | full deck app                                       | ~1.4–2.0 |
//! | CADisplayLink callbacks (default and UI QoS)        | ~1.5     |
//! | scheduled presents, in-window, egui churning        | ~1.5–2   |
//! | scheduled presents, in-window, egui frozen          | ~10      |
//! | + exact-120 `preferredFrameRateRange` pin           | ~10      |
//! | scheduled presents, own borderless child window     | ~5       |
//! | fullscreen                                          | ~4      |
//!
//! No app-side pacing strategy beats the empty-eframe floor: the misses
//! originate in WindowServer's frame delivery under system load, upstream
//! of how (or from which window/layer) frames are presented. Notably, a
//! *quiet* window fares much worse — the system adaptively downshifts its
//! cadence — so the deck app's continuous repaint while playing is
//! already the best-case regime. Env levers for re-testing:
//! `METALWAVE_STATIC_EGUI=1`, `METALWAVE_OWN_WINDOW=1`,
//! `METALWAVE_NO_PIN=1`.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metalwave_spike is macOS-only");
}

#[cfg(target_os = "macos")]
fn main() -> eframe::Result<()> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1000.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "metalwave spike",
        options,
        Box::new(|cc| {
            let handle = cc
                .window_handle()
                .expect("eframe exposes the native window handle");
            let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
                panic!("not an AppKit window");
            };
            metalwave::install(appkit.ns_view.as_ptr());
            Ok(Box::new(SpikeApp {
                start: std::time::Instant::now(),
            }))
        }),
    )
}

/// The egui side: the same moving bar, painted by eframe's repaint loop,
/// as the visual control for the Metal layer below it.
#[cfg(target_os = "macos")]
struct SpikeApp {
    start: std::time::Instant,
}

#[cfg(target_os = "macos")]
impl eframe::App for SpikeApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        use eframe::egui;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("egui-painted bar (eframe pacing):");
            let (response, painter) =
                ui.allocate_painter(egui::vec2(940.0, 120.0), egui::Sense::hover());
            let rect = response.rect;
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(10, 10, 15));
            let x =
                rect.left() + metalwave::bar_x(self.start.elapsed().as_secs_f64(), rect.width());
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x, rect.top()),
                    egui::pos2(x + metalwave::BAR_WIDTH_PT, rect.bottom()),
                ),
                0.0,
                egui::Color32::from_rgb(235, 40, 40),
            );
            ui.add_space(8.0);
            ui.label("Metal bar renders in the panel below (pacing stats on stdout):");
        });
        // A/B lever: METALWAVE_STATIC_EGUI=1 freezes the egui layer to
        // isolate whether its churn in the same window gates the Metal
        // sublayer's presents.
        if std::env::var_os("METALWAVE_STATIC_EGUI").is_none() {
            ctx.request_repaint();
        }
    }
}

#[cfg(target_os = "macos")]
mod metalwave {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::mpsc;

    use block2::RcBlock;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2::{AnyThread, MainThreadMarker, define_class, msg_send, sel};
    use objc2_app_kit::{
        NSBackingStoreType, NSView, NSWindow, NSWindowOrderingMode, NSWindowStyleMask,
    };
    use objc2_foundation::{
        NSDefaultRunLoopMode, NSObject, NSPoint, NSRect, NSRunLoop, NSSize, NSString,
    };
    use objc2_metal::{
        MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLLoadAction, MTLPixelFormat,
        MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
        MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    };
    use objc2_quartz_core::{
        CACurrentMediaTime, CADisplayLink, CAFrameRateRange, CAMetalDrawable, CAMetalLayer,
    };

    define_class!(
        // SAFETY: NSObject has no subclassing requirements; RatePin does
        // not implement Drop. The link fires `tick:` on the main runloop;
        // the method touches nothing.
        #[unsafe(super(NSObject))]
        #[name = "MetalwaveRatePin"]
        #[ivars = ()]
        struct RatePin;

        impl RatePin {
            #[unsafe(method(tick:))]
            fn tick(&self, _link: &CADisplayLink) {}
        }
    );

    impl RatePin {
        fn new() -> Retained<Self> {
            let this = Self::alloc().set_ivars(());
            unsafe { msg_send![super(this), init] }
        }
    }

    /// Bar sweep speed in points per second and width in points — shared
    /// with the egui control bar so both animations are identical.
    const BAR_SPEED_PT_PER_SEC: f64 = 300.0;
    pub const BAR_WIDTH_PT: f32 = 6.0;
    /// Where the Metal panel sits inside the window's content view
    /// (AppKit coordinates) and its size in points.
    const PANEL_ORIGIN_PT: (f64, f64) = (30.0, 40.0);
    const PANEL_SIZE_PT: (f64, f64) = (940.0, 160.0);
    /// How many refresh intervals ahead a frame is scheduled when the
    /// pipeline (re)fills. With a 3-deep drawable queue this leaves one
    /// interval of slack for wakeup jitter.
    const SCHEDULE_AHEAD_SLOTS: f64 = 2.0;
    /// Stats window between stdout reports.
    const REPORT_EVERY_FRAMES: usize = 600;

    /// Bar x-position at time `t` for a panel `width` wide.
    pub fn bar_x(t: f64, width: f32) -> f32 {
        ((t * BAR_SPEED_PT_PER_SEC) % width as f64) as f32
    }

    const SHADER_SRC: &str = "
        #include <metal_stdlib>
        using namespace metal;
        struct U { float x0; float x1; };
        vertex float4 vmain(uint vid [[vertex_id]], constant U& u [[buffer(0)]]) {
            float x = (vid & 1) ? u.x1 : u.x0;
            float y = (vid & 2) ? 1.0 : -1.0;
            return float4(x, y, 0.0, 1.0);
        }
        fragment float4 fmain() { return float4(0.92, 0.16, 0.16, 1.0); }
    ";

    #[repr(C)]
    struct BarUniforms {
        x0: f32,
        x1: f32,
    }

    /// `CAMetalLayer` is created on the main thread but rendered from the
    /// dedicated thread only (the documented background-rendering
    /// pattern); Metal queue/pipeline objects are thread-safe.
    struct SendState {
        layer: Retained<CAMetalLayer>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        refresh_interval: f64,
    }
    unsafe impl Send for SendState {}

    /// Install the Metal panel as a child view of the winit content view
    /// and start the render thread. Must be called on the main thread with
    /// a valid NSView pointer; everything created here lives for the
    /// process (spike has no teardown).
    pub fn install(ns_view: *mut c_void) {
        let mtm = MainThreadMarker::new().expect("install must run on the main thread");
        let parent: &NSView = unsafe { &*ns_view.cast() };
        let window = parent.window();
        let scale = window
            .as_ref()
            .map(|w| w.backingScaleFactor())
            .unwrap_or(2.0);
        let max_fps = window
            .as_ref()
            .and_then(|w| w.screen())
            .map(|s| s.maximumFramesPerSecond() as f64)
            .filter(|f| *f > 0.0)
            .unwrap_or(120.0);

        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setContentsScale(scale);
        layer.setDrawableSize(NSSize::new(
            PANEL_SIZE_PT.0 * scale,
            PANEL_SIZE_PT.1 * scale,
        ));

        let frame = NSRect::new(
            NSPoint::new(PANEL_ORIGIN_PT.0, PANEL_ORIGIN_PT.1),
            NSSize::new(PANEL_SIZE_PT.0, PANEL_SIZE_PT.1),
        );
        let child = NSView::initWithFrame(mtm.alloc(), frame);
        child.setLayer(Some(&layer));
        child.setWantsLayer(true);

        if std::env::var_os("METALWAVE_OWN_WINDOW").is_some() {
            // Overlay child-window variant: a borderless NSWindow attached
            // above the app window gets its own WindowServer compositing
            // stream, instead of inheriting the app window's update
            // cadence. It moves with the parent automatically.
            let parent_window = window.as_ref().expect("parent view must be in a window");
            let base = parent_window.frame();
            let screen_rect = NSRect::new(
                NSPoint::new(
                    base.origin.x + PANEL_ORIGIN_PT.0,
                    base.origin.y + PANEL_ORIGIN_PT.1,
                ),
                NSSize::new(PANEL_SIZE_PT.0, PANEL_SIZE_PT.1),
            );
            let overlay = unsafe {
                NSWindow::initWithContentRect_styleMask_backing_defer(
                    mtm.alloc(),
                    screen_rect,
                    NSWindowStyleMask::Borderless,
                    NSBackingStoreType::Buffered,
                    false,
                )
            };
            overlay.setOpaque(true);
            overlay.setContentView(Some(&child));
            unsafe {
                parent_window.addChildWindow_ordered(&overlay, NSWindowOrderingMode::Above);
            }
            overlay.orderFront(None);
            std::mem::forget(overlay);
        } else {
            parent.addSubview(&child);
        }

        // ProMotion rate pin: without an explicit frame-rate-range demand
        // the system adaptively downshifts the display cadence and our
        // 120 Hz scheduled presents land on slower glass (measured ~10
        // misses/s with a quiet window). The link's only job is to hold
        // that demand; its tick does nothing. METALWAVE_NO_PIN=1 disables
        // it for A/B runs.
        if std::env::var_os("METALWAVE_NO_PIN").is_none() {
            let pin = RatePin::new();
            let link = unsafe { child.displayLinkWithTarget_selector(&pin, sel!(tick:)) };
            link.setPreferredFrameRateRange(CAFrameRateRange {
                minimum: 120.0,
                maximum: 120.0,
                preferred: 120.0,
            });
            unsafe { link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSDefaultRunLoopMode) };
            // The runloop and link keep each other alive; the link retains
            // the pin. Leak both deliberately — spike has no teardown.
            std::mem::forget(link);
            std::mem::forget(pin);
        }

        let queue = device.newCommandQueue().expect("no command queue");
        let source = NSString::from_str(SHADER_SRC);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("shader compile failed");
        let vmain = library
            .newFunctionWithName(&NSString::from_str("vmain"))
            .expect("vmain missing");
        let fmain = library
            .newFunctionWithName(&NSString::from_str("fmain"))
            .expect("fmain missing");
        let pipeline_desc = MTLRenderPipelineDescriptor::new();
        pipeline_desc.setVertexFunction(Some(&vmain));
        pipeline_desc.setFragmentFunction(Some(&fmain));
        unsafe {
            pipeline_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0)
                .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        }
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
            .expect("pipeline build failed");

        let state = SendState {
            layer,
            queue,
            pipeline,
            refresh_interval: 1.0 / max_fps,
        };
        std::thread::Builder::new()
            .name("metalwave-render".into())
            .spawn(move || {
                unsafe {
                    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                }
                render_loop(state);
            })
            .expect("failed to spawn render thread");
    }

    fn render_loop(st: SendState) {
        let interval = st.refresh_interval;
        let started = CACurrentMediaTime();
        // Presented-time feedback from the compositor, via
        // addPresentedHandler blocks (fired on a CoreAnimation thread).
        let (tx, rx) = mpsc::channel::<f64>();
        let mut presented: Vec<f64> = Vec::new();
        let mut frames_total = 0usize;
        let mut misses_total = 0usize;
        let mut resyncs = 0usize;
        let mut window_dts: Vec<f64> = Vec::new();
        let mut last_presented = 0.0f64;
        let mut next_t = CACurrentMediaTime() + SCHEDULE_AHEAD_SLOTS * interval;

        loop {
            // Back-pressure pacing: blocks while the 3-deep queue is full.
            let Some(drawable) = st.layer.nextDrawable() else {
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            };

            // If the schedule fell behind the wall clock (long stall, app
            // occluded), re-anchor ahead of now instead of burning frames
            // catching up.
            let now = CACurrentMediaTime();
            if next_t < now + 0.5 * interval {
                next_t = now + SCHEDULE_AHEAD_SLOTS * interval;
                resyncs += 1;
            }

            // Render the bar where it belongs at the scheduled instant.
            let width = PANEL_SIZE_PT.0 as f32;
            let x = bar_x(next_t - started, width);
            let uniforms = BarUniforms {
                x0: x / width * 2.0 - 1.0,
                x1: (x + BAR_WIDTH_PT) / width * 2.0 - 1.0,
            };

            let desc = MTLRenderPassDescriptor::renderPassDescriptor();
            let attachment = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&drawable.texture()));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setClearColor(MTLClearColor {
                red: 0.04,
                green: 0.04,
                blue: 0.06,
                alpha: 1.0,
            });
            attachment.setStoreAction(MTLStoreAction::Store);
            let Some(cmd) = st.queue.commandBuffer() else {
                continue;
            };
            let Some(encoder) = cmd.renderCommandEncoderWithDescriptor(&desc) else {
                continue;
            };
            encoder.setRenderPipelineState(&st.pipeline);
            unsafe {
                encoder.setVertexBytes_length_atIndex(
                    NonNull::from(&uniforms).cast(),
                    std::mem::size_of::<BarUniforms>(),
                    0,
                );
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                );
            }
            encoder.endEncoding();

            // Ground-truth presentation timestamp, reported by the
            // compositor once the frame actually reaches the glass.
            // (`addPresentedHandler` isn't in the generated bindings.)
            let tx_frame = tx.clone();
            let handler = RcBlock::new(move |d: NonNull<AnyObject>| {
                let t: f64 = unsafe { msg_send![d.as_ref(), presentedTime] };
                let _ = tx_frame.send(t);
            });
            let _: () = unsafe { msg_send![&*drawable, addPresentedHandler: &*handler] };

            cmd.presentDrawable_atTime(ProtocolObject::from_ref(&*drawable), next_t);
            cmd.commit();
            next_t += interval;

            // Fold in compositor feedback (2 s warmup) and report.
            while let Ok(t) = rx.try_recv() {
                if t <= 0.0 {
                    continue;
                }
                if last_presented > 0.0 && t - started > 2.0 {
                    let dt = t - last_presented;
                    window_dts.push(dt);
                    frames_total += 1;
                    if dt > 1.5 * interval {
                        misses_total += 1;
                    }
                }
                last_presented = t;
            }
            if window_dts.len() >= REPORT_EVERY_FRAMES {
                let mut win = std::mem::take(&mut window_dts);
                win.sort_by(f64::total_cmp);
                let p99 = win[win.len() * 99 / 100] * 1000.0;
                let max = win.last().unwrap() * 1000.0;
                let misses = win.iter().filter(|d| **d > 1.5 * interval).count();
                presented.clear();
                println!(
                    "metal(glass): {} frames, window misses {misses}, p99 {p99:.2}ms, \
                     max {max:.2}ms (total {frames_total} frames, {misses_total} misses, \
                     {resyncs} resyncs)",
                    win.len(),
                );
            }
        }
    }

    const QOS_CLASS_USER_INTERACTIVE: std::ffi::c_uint = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(
            qos_class: std::ffi::c_uint,
            relative_priority: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
}
