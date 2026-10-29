//! Opt-in diagnostic profiling, absent from ordinary builds.
use serde_json::{Value, json};
use std::cell::Cell;

// Opt-in, allocation-free inclusive profiling. Kept separate from timing runs;
// nested regions overlap and must not be summed as exclusive costs.
const REGIONS: [&str; 14] = [
    "core",
    "splitter",
    "marks",
    "synthesis",
    "vocoder",
    "resampler",
    "adapter",
    "collect",
    "emit",
    "pv_input",
    "pv_phase_aux",
    "pv_selector",
    "pv_synthesis",
    "pv_output",
];
#[derive(Clone, Copy)]
struct Profile {
    enabled: bool,
    ns: [u64; 14],
    calls: [u64; 14],
}
thread_local! { static PROFILE: Cell<Profile> = const { Cell::new(Profile { enabled: false, ns: [0;14], calls: [0;14] }) }; }
pub fn profile_begin() {
    PROFILE.set(Profile {
        enabled: true,
        ns: [0; 14],
        calls: [0; 14],
    });
}
pub fn profile_end() -> Value {
    let p = PROFILE.replace(Profile {
        enabled: false,
        ns: [0; 14],
        calls: [0; 14],
    });
    json!(
        REGIONS
            .iter()
            .enumerate()
            .map(|(i, name)| json!({"region":name,"ns":p.ns[i],"calls":p.calls[i]}))
            .collect::<Vec<_>>()
    )
}
pub(crate) struct ProfileRegion(usize, Option<std::time::Instant>);
pub(crate) fn region(index: usize) -> ProfileRegion {
    ProfileRegion(index, PROFILE.get().enabled.then(std::time::Instant::now))
}
impl Drop for ProfileRegion {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            let elapsed = start.elapsed().as_nanos() as u64;
            let mut p = PROFILE.get();
            p.ns[self.0] += elapsed;
            p.calls[self.0] += 1;
            PROFILE.set(p);
        }
    }
}

pub fn microbench(backend: crate::Numerics) -> Result<Value, &'static str> {
    if !backend.available() {
        return Err("unavailable numerical provider");
    }
    Ok({
        use std::{hint::black_box, time::Instant};
        let mut rows = Vec::new();
        for n in [1024, 2048, 4096, 8192] {
            let start = Instant::now();
            let mut fft = super::fft::PackedFft::new(n, backend);
            let init = start.elapsed().as_nanos() as u64;
            let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.079).sin() * 0.4).collect();
            let mut data = input.clone();
            fft.forward(&mut data);
            let spectrum = data.clone();
            for inverse in [false, true] {
                let mut times = Vec::with_capacity(10000);
                for i in 0..10064 {
                    data.copy_from_slice(if inverse { &spectrum } else { &input });
                    let start = Instant::now();
                    if inverse {
                        fft.inverse(&mut data)
                    } else {
                        fft.forward(&mut data)
                    };
                    let ns = start.elapsed().as_nanos() as u64;
                    black_box(&data);
                    if i >= 64 {
                        times.push(ns);
                    }
                }
                times.sort_unstable();
                rows.push(json!({"kernel":if inverse{"inverse_including_packing"}else{"forward_including_packing_scaling"},"size":n,"init_ns":init,"median_ns":times[5000],"p99_ns":times[9900],"iterations":10000}));
            }
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            if backend.apple_fft() {
                let mut f = timestretch_accelerate::RealFft::new(n).unwrap();
                for op in 0..6 {
                    let mut times = Vec::with_capacity(10000);
                    for i in 0..10064 {
                        if op == 1 || op == 3 {
                            f.pack_spectrum(&spectrum);
                        }
                        let start = Instant::now();
                        match op {
                            0 => f.pack(&input),
                            1 => f.transform(false),
                            2 => f.unpack_spectrum(&mut data),
                            3 => f.transform(true),
                            4 => f.unpack(&mut data),
                            _ => f.pack_spectrum(&spectrum),
                        };
                        let ns = start.elapsed().as_nanos() as u64;
                        black_box(&data);
                        if i >= 64 {
                            times.push(ns)
                        }
                    }
                    times.sort_unstable();
                    rows.push(json!({"kernel":(["pack_time","forward_core","unpack_scale","inverse_core","unpack_time","pack_spectrum"][op]),"size":n,"median_ns":times[5000],"p99_ns":times[9900],"iterations":10000}));
                }
            }
            if !backend.apple_fft() {
                let mut f = super::fft::RustFft::new(n);
                for op in 0..6 {
                    let mut times = Vec::with_capacity(10000);
                    for i in 0..10064 {
                        f.pack_spectrum(&spectrum);
                        let start = Instant::now();
                        match op {
                            0 => f.pack(&input),
                            1 => f.forward_core(),
                            2 => f.unpack_spectrum(&mut data),
                            3 => f.inverse_core(),
                            4 => f.unpack(&mut data),
                            _ => f.pack_spectrum(&spectrum),
                        };
                        let ns = start.elapsed().as_nanos() as u64;
                        black_box(&data);
                        if i >= 64 {
                            times.push(ns)
                        }
                    }
                    times.sort_unstable();
                    rows.push(json!({"kernel":(["pack_time","forward_core","unpack_spectrum","inverse_core","unpack_time","pack_spectrum"][op]),"size":n,"median_ns":times[5000],"p99_ns":times[9900],"iterations":10000}));
                }
            }
            let math = super::math::Math::new(backend);
            let mut phase = vec![0.; n / 2];
            let mut mag = vec![0.; n / 2];
            let mut packed = spectrum.clone();
            let mut s = vec![0.; n / 2];
            let mut c = s.clone();
            for len in [1, 8, 32, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
                .into_iter()
                .filter(|&len| len <= n / 2)
            {
                for op in [0, 1, 2] {
                    for _ in 0..64 {
                        math.polar(&mut packed[..len * 2], &mut phase[..len], &mut mag[..len]);
                        math.sin_cos(&input[..len], &mut s[..len], &mut c[..len]);
                    }
                    let start = Instant::now();
                    for _ in 0..10000 {
                        if op == 0 {
                            math.polar(&mut packed[..len * 2], &mut phase[..len], &mut mag[..len]);
                            black_box((&phase, &mag));
                        } else if op == 1 {
                            math.sin_cos(&input[..len], &mut s[..len], &mut c[..len]);
                            black_box((&s, &c));
                        } else {
                            math.reciprocal(&spectrum[..len], &mut mag[..len]);
                            black_box(&mag);
                        }
                    }
                    rows.push(json!({"kernel":(["polar", "sin_cos", "reciprocal"][op]),"size":len,"iterations":10000,"mean_ns":start.elapsed().as_nanos() as f64/10000.}));
                }
            }
        }
        json!(rows)
    })
}
