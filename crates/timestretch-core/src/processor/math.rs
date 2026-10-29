//! Safe scalar equivalents of the numerical adapters. Immutable vector seeds
//! live in tables.rs; dynamic phase/trigonometry retain documented provider drift.
pub(super) fn magnitude(re: f32, im: f32) -> f32 {
    // The vector provider rounds the real square before fusing the imaginary
    // square. Reversing these operations changes low-energy selector decisions.
    im.mul_add(im, re * re).sqrt()
}
/// Selects the numerical provider for polar conversion, reciprocals, and sine/cosine.
/// Portable operations use scalar libm and fused arithmetic; immutable constructor
/// seeds are supplied separately.
pub(super) struct Math {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    apple: bool,
}
impl Math {
    pub fn new(_numerics: crate::Numerics) -> Self {
        Self {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            apple: _numerics.apple_math(),
        }
    }
    pub fn is_apple(&self) -> bool {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            self.apple
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            false
        }
    }
    pub fn polar(&self, packed: &mut [f32], phase: &mut [f32], mag: &mut [f32]) {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        if self.apple {
            timestretch_accelerate::polar(packed, phase, mag);
            return;
        }
        let h = packed.len() / 2;
        for k in 0..h {
            let re = packed[k];
            let im = if k == 0 { 0. } else { packed[h + k] };
            phase[k] = im.atan2(re);
            mag[k] = magnitude(re, im);
        }
    }
    pub fn reciprocal(&self, denominator: &[f32], out: &mut [f32]) {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        if self.apple {
            timestretch_accelerate::reciprocal(denominator, out);
            return;
        }
        for (&d, x) in denominator.iter().zip(out) {
            *x = 1. / d;
        }
    }
    pub fn sin_cos(&self, input: &[f32], sin: &mut [f32], cos: &mut [f32]) {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        if self.apple {
            timestretch_accelerate::sin_cos(input, sin, cos);
            return;
        }
        for ((&x, s), c) in input.iter().zip(sin).zip(cos) {
            (*s, *c) = x.sin_cos();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complex_magnitude_matches_captured_vector_adapter() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/native/numerics.json"))
                .unwrap();
        let floats = |name: &str| {
            data[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                .collect::<Vec<_>>()
        };
        let input = floats("input");
        let phase = floats("phase");
        let expected = floats("magnitude");
        let half = expected.len();
        for k in 1..half {
            assert_eq!(
                magnitude(input[k], input[half + k]).to_bits(),
                expected[k].to_bits(),
                "bin {k}"
            );
            assert!(
                (input[half + k].atan2(input[k]) - phase[k]).abs() <= 3e-7,
                "phase bin {k}"
            );
        }
    }
}
