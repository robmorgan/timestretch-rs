//! Application-owned scheduling. The DSP/session libraries never choose a
//! thread or its priority. This guard stays on the thread it promotes.
use std::marker::PhantomData;
use std::rc::Rc;

pub struct AudioPriority {
    #[cfg(target_os = "macos")]
    handle: Option<audio_thread_priority::RtPriorityHandle>,
    // Releasing a policy must happen on the same thread, including unwinding.
    _owner: PhantomData<Rc<()>>,
}
impl AudioPriority {
    pub fn new(sample_rate: u32, budget_frames: usize) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        let handle = audio_thread_priority::promote_current_thread_to_real_time(
            u32::try_from(budget_frames).map_err(|_| "invalid scheduling budget")?,
            sample_rate,
        )
        .map_err(|e| e.to_string())?;
        #[cfg(not(target_os = "macos"))]
        let _ = (sample_rate, budget_frames);
        Ok(Self {
            #[cfg(target_os = "macos")]
            handle: Some(handle),
            _owner: PhantomData,
        })
    }
    pub const fn label() -> &'static str {
        #[cfg(target_os = "macos")]
        return "Mach audio time constraint";
        #[cfg(not(target_os = "macos"))]
        return "platform default";
    }
    pub fn release(&mut self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        if let Some(handle) = self.handle.take() {
            audio_thread_priority::demote_current_thread_from_real_time(handle)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
impl Drop for AudioPriority {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
