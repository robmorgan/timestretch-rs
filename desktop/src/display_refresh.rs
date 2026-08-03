//! Fixed-refresh display pin.
//!
//! ProMotion's adaptive refresh causes ~2 missed vsync slots/s in any
//! windowed app on macOS, which the eye reads as scroll twitches in the
//! deck waveform (measured in `examples/metalwave_spike.rs`; a fixed
//! 60 Hz mode collapses the misses to ~0.1/s). This module pins the main
//! display to a fixed rate while the user wants smooth scrolling and
//! restores the previous mode when dropped.
//!
//! The switch is session-scoped (`kCGConfigureForSession`): even if the
//! app dies without restoring, the mode reverts at logout — it can never
//! stick permanently.

/// A pinned display mode; dropping it restores the previous mode.
pub struct RefreshPin {
    /// Held for its `Drop` impl, which restores the previous mode.
    #[cfg(target_os = "macos")]
    _inner: imp::Pin,
    /// The fixed rate now active, for the UI label.
    pinned_hz: f64,
}

impl RefreshPin {
    /// Pin the main display to a fixed refresh rate. Errors if the
    /// platform is unsupported, no suitable fixed mode exists (e.g. the
    /// display is already at one), or the mode switch fails.
    pub fn pin() -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let (inner, pinned_hz) = imp::pin()?;
            Ok(Self {
                _inner: inner,
                pinned_hz,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("fixed-refresh pinning is macOS-only".to_string())
        }
    }

    pub fn pinned_hz(&self) -> f64 {
        self.pinned_hz
    }
}

/// Pick the mode to pin to from `(width, height, refresh)` candidates:
/// same pixel dimensions as `current`, preferring exactly 60 Hz (clean,
/// universally supported), else the highest refresh at least 1 Hz below
/// the current rate. `None` when the display is already at or below
/// 60 Hz — pinning only makes sense to escape an adaptive high-rate
/// panel, never to push an already-slow display lower.
fn choose_fixed(modes: &[(usize, usize, f64)], current: (usize, usize, f64)) -> Option<usize> {
    if current.2 <= 61.0 {
        return None;
    }
    let same_res = |m: &(usize, usize, f64)| m.0 == current.0 && m.1 == current.1;
    if let Some(idx) = modes
        .iter()
        .position(|m| same_res(m) && (m.2 - 60.0).abs() < 0.1)
    {
        return Some(idx);
    }
    modes
        .iter()
        .enumerate()
        .filter(|(_, m)| same_res(m) && m.2 > 0.0 && m.2 < current.2 - 1.0)
        .max_by(|(_, a), (_, b)| a.2.total_cmp(&b.2))
        .map(|(idx, _)| idx)
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2_core_foundation::{CFBoolean, CFDictionary, CFRetained, CFString, kCFBooleanTrue};
    use objc2_core_graphics::{
        CGBeginDisplayConfiguration, CGCompleteDisplayConfiguration,
        CGConfigureDisplayWithDisplayMode, CGConfigureOption, CGDirectDisplayID,
        CGDisplayConfigRef, CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode, CGDisplayMode,
        CGError, CGMainDisplayID, kCGDisplayShowDuplicateLowResolutionModes,
    };

    pub struct Pin {
        display: CGDirectDisplayID,
        /// The mode active before pinning, restored on drop.
        previous: CFRetained<CGDisplayMode>,
    }

    pub fn pin() -> Result<(Pin, f64), String> {
        let display = CGMainDisplayID();
        let previous = CGDisplayCopyDisplayMode(display)
            .ok_or_else(|| "no current display mode".to_string())?;
        let current = describe(&previous);

        // Without this option CoreGraphics hides the same-resolution
        // lower-rate variants — exactly the modes we want to pin to.
        let options = CFDictionary::<CFString, CFBoolean>::from_slices(
            &[unsafe { kCGDisplayShowDuplicateLowResolutionModes }],
            &[unsafe { kCFBooleanTrue }.ok_or("no kCFBooleanTrue")?],
        );
        // SAFETY: the options dictionary holds the documented key/value
        // types for CGDisplayCopyAllDisplayModes.
        let array = unsafe { CGDisplayCopyAllDisplayModes(display, Some(options.as_ref())) }
            .ok_or_else(|| "no display modes".to_string())?;
        let count = array.count();
        let mut modes = Vec::with_capacity(count as usize);
        let mut infos = Vec::with_capacity(count as usize);
        for i in 0..count {
            // SAFETY: the array from CGDisplayCopyAllDisplayModes holds
            // CGDisplayMode objects; `array` outlives the borrow.
            let mode: &CGDisplayMode = unsafe { &*array.value_at_index(i).cast::<CGDisplayMode>() };
            infos.push(describe(mode));
            modes.push(mode);
        }

        let idx = super::choose_fixed(&infos, current).ok_or_else(|| {
            format!(
                "display is already at a fixed-friendly rate ({:.0} Hz)",
                current.2
            )
        })?;
        apply(display, modes[idx])?;
        let pinned_hz = infos[idx].2;
        Ok((Pin { display, previous }, pinned_hz))
    }

    impl Drop for Pin {
        fn drop(&mut self) {
            if let Err(e) = apply(self.display, &self.previous) {
                log::error!("failed to restore display mode: {e}");
            }
        }
    }

    fn describe(mode: &CGDisplayMode) -> (usize, usize, f64) {
        (
            CGDisplayMode::width(Some(mode)),
            CGDisplayMode::height(Some(mode)),
            CGDisplayMode::refresh_rate(Some(mode)),
        )
    }

    fn apply(display: CGDirectDisplayID, mode: &CGDisplayMode) -> Result<(), String> {
        let mut config: CGDisplayConfigRef = std::ptr::null_mut();
        // SAFETY: standard begin/configure/complete sequence; config is
        // consumed by CGCompleteDisplayConfiguration.
        unsafe {
            let err = CGBeginDisplayConfiguration(&mut config);
            if err != CGError::Success {
                return Err(format!("begin configuration failed ({})", err.0));
            }
            let err = CGConfigureDisplayWithDisplayMode(config, display, Some(mode), None);
            if err != CGError::Success {
                return Err(format!("configure failed ({})", err.0));
            }
            let err = CGCompleteDisplayConfiguration(config, CGConfigureOption::ForSession);
            if err != CGError::Success {
                return Err(format!("complete configuration failed ({})", err.0));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::choose_fixed;

    /// The built-in ProMotion panel: adaptive 120 plus fixed alternates.
    const PROMOTION: &[(usize, usize, f64)] = &[
        (1512, 982, 120.0),
        (1512, 982, 60.0),
        (1512, 982, 59.94),
        (1512, 982, 50.0),
        (1512, 982, 48.0),
        (1512, 982, 47.95),
        // Another resolution's modes must never be picked.
        (1280, 800, 60.0),
    ];

    #[test]
    fn promotion_pins_to_60() {
        let idx = choose_fixed(PROMOTION, (1512, 982, 120.0)).unwrap();
        assert_eq!(PROMOTION[idx], (1512, 982, 60.0));
    }

    #[test]
    fn already_at_60_is_a_no_op() {
        assert_eq!(choose_fixed(PROMOTION, (1512, 982, 60.0)), None);
    }

    #[test]
    fn external_144_prefers_60() {
        let modes = &[(2560, 1440, 144.0), (2560, 1440, 120.0), (2560, 1440, 60.0)];
        let idx = choose_fixed(modes, (2560, 1440, 144.0)).unwrap();
        assert_eq!(modes[idx], (2560, 1440, 60.0));
    }

    #[test]
    fn without_a_60_mode_takes_next_below_current() {
        let modes = &[(1512, 982, 100.0), (1512, 982, 80.0), (1512, 982, 50.0)];
        let idx = choose_fixed(modes, (1512, 982, 100.0)).unwrap();
        assert_eq!(modes[idx], (1512, 982, 80.0));
    }

    #[test]
    fn no_alternative_yields_none() {
        let modes = &[(1512, 982, 60.0)];
        assert_eq!(choose_fixed(modes, (1512, 982, 60.0)), None);
    }
}
