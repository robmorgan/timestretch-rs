use super::*;
pub(super) fn score(a: &[String]) -> Result<Value> {
    use timestretch::analysis::{comparison as m, loudness::measure_loudness};
    let reference = read_wav_file(&a[0])?;
    let test = read_wav_file(&a[1])?;
    if reference.channels != test.channels || reference.sample_rate != test.sample_rate {
        return Err("scoring format mismatch".into());
    }
    let ch = test.channels.count();
    let sr = test.sample_rate;
    let n = test.data.len().min(reference.data.len());
    let mut peak = 0f64;
    let mut sq = 0.;
    let mut energy = 0.;
    let mut exact = 0;
    for (&x, &y) in test.data.iter().zip(&reference.data) {
        let e = f64::from(x) - f64::from(y);
        peak = peak.max(e.abs());
        sq += e * e;
        energy += f64::from(y).powi(2);
        exact += usize::from(x.to_bits() == y.to_bits());
    }
    let left = |x: &AudioBuffer| x.data.chunks_exact(ch).map(|f| f[0]).collect::<Vec<_>>();
    let t = left(&test);
    let r = left(&reference);
    let corr = m::cross_correlation(&r, &t);
    let onset = m::onset_timing_analysis(&r, &t, sr);
    let stereo = |x: &AudioBuffer| {
        let mut lr = 0.;
        let mut ll = 0.;
        let mut rr = 0.;
        for f in x.data.chunks_exact(ch) {
            let l = f64::from(f[0]);
            let r = f64::from(f[ch - 1]);
            lr += l * r;
            ll += l * l;
            rr += r * r;
        }
        if ll * rr > 0. {
            Some(lr / (ll * rr).sqrt())
        } else {
            None
        }
    };
    let discontinuity = t
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0f32, f32::max);
    let crossings: Vec<_> = t
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] <= 0. && w[1] > 0.)
        .map(|(i, w)| i as f64 - f64::from(w[0]) / f64::from(w[1] - w[0]))
        .collect();
    let mut cents: Vec<_> = crossings
        .windows(45)
        .filter_map(|w| {
            let hz = 44. * sr as f64 / (w[44] - w[0]);
            let c = 1200. * (hz / 440.).log2();
            c.is_finite().then_some(c.abs())
        })
        .collect();
    cents.sort_by(f64::total_cmp);
    Ok(
        json!({"reference_frames":r.len(),"output_frames":t.len(),"length_error_frames":t.len() as i64-r.len() as i64,"exact_samples":exact,"compared_samples":n,"peak_error":peak,"rms_error":(sq/n.max(1) as f64).sqrt(),"error_snr_db":if sq>0.&&energy>0. {Some(10.*(energy/sq).log10())} else {None},"bit_exact":test.data.len()==reference.data.len()&&exact==n,"alignment_error_frames":corr.peak_offset,"cross_correlation":corr.peak_value,"spectral_similarity":m::spectral_similarity(&t,&r,2048,512),"mean_spectral_similarity":m::mean_spectral_similarity(&t,&r,2048,512),"transient_sharpness_similarity":m::spectral_flux_similarity(&t,&r,2048,512),"onset_mean_ms":onset.mean_error_ms,"onset_max_ms":onset.max_error_ms,"stereo_correlation":stereo(&test),"reference_stereo_correlation":stereo(&reference),"max_adjacent_sample_delta":discontinuity,"tone_440_p95_abs_cents":if cents.is_empty(){None}else{Some(cents[(cents.len()-1)*95/100])},"loudness":measure_loudness(&test.data,ch,sr),"peak":test.data.iter().map(|x|x.abs()).fold(0f32,f32::max),"scorer":"fixed Rust analysis::comparison + ebur128; left channel for spectral/onset; no alignment correction"}),
    )
}
pub(super) fn trace(a: &[String]) -> Result<Value> {
    use std::io::Write;
    let input = read_wav_file(&a[1])?;
    let ch = input.channels.count();
    let frames = input.data.len() / ch;
    let mut p = NativeProcessor::new_with_numerics(input.sample_rate, ch, numerics(&a[0])?)?;
    let (_, actual, _) = p.set_ratio(a[3].parse()?, 1., RatioAdjustment::Stretch, true)?;
    let target = (frames as f64 * f64::from(actual)).round() as usize;
    let mut inp = [vec![0.; p.input_capacity()], vec![0.; p.input_capacity()]];
    let mut out = [vec![0.; p.output_capacity()], vec![0.; p.output_capacity()]];
    let mut file = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    let mut consumed = 0;
    let mut written = 0;
    let mut discard = p.initial_unused_frames();
    let mut step = 0;
    while written < target {
        let n = if step == 0 {
            p.prefill_frames()
        } else {
            p.frames_needed(0)?
        };
        for (c, plane) in inp.iter_mut().enumerate().take(ch) {
            for (i, sample) in plane.iter_mut().enumerate().take(n) {
                *sample = if consumed + i < frames {
                    input.data[(consumed + i) * ch + c]
                } else {
                    0.
                };
            }
        }
        let iv = [&inp[0][..n], &inp[1][..n]];
        let [l, r] = &mut out;
        let mut ov = [l.as_mut_slice(), r.as_mut_slice()];
        let count = if step == 0 {
            p.prefill(&iv[..ch], Some(&mut ov[..ch]))?
        } else {
            p.process(&iv[..ch], &mut ov[..ch])?
        };
        consumed += n;
        let mut snapshot = p.qa_snapshot();
        snapshot["step"] = json!(step);
        snapshot["consumed"] = json!(consumed);
        snapshot["output"] = json!(count);
        snapshot["output_hash"] = json!(
            out[..ch]
                .iter()
                .map(|plane| plane[..count]
                    .iter()
                    .fold(14695981039346656037u64, |h, x| (h ^ u64::from(x.to_bits()))
                        .wrapping_mul(1099511628211))
                    .to_string())
                .collect::<Vec<_>>()
        );
        writeln!(file, "{snapshot}")?;
        let skip = discard.min(count);
        discard -= skip;
        written += count - skip;
        step += 1;
    }
    Ok(json!({"steps":step,"actual_stretch":actual}))
}

/// QA-only dependency substitutions use the native oracle host. Production
/// providers above go through StreamSession so adapter fidelity is measured.
pub(super) fn kernel_render(
    input: &AudioBuffer,
    arm: &str,
    duration: f32,
) -> Result<(Vec<f32>, f32)> {
    let ch = input.channels.count();
    let frames = input.num_frames();
    let mut p = NativeProcessor::new_with_numerics(input.sample_rate, ch, numerics(arm)?)?;
    let (status, actual, _) = p.set_ratio(duration, 1.0, RatioAdjustment::Stretch, true)?;
    if status != 0 {
        return Err("ratio rejected".into());
    }
    let target = (frames as f64 * actual as f64).round() as usize;
    let mut audio = Vec::with_capacity(target * ch);
    let mut inp = [vec![0.0; p.input_capacity()], vec![0.0; p.input_capacity()]];
    let mut out = [
        vec![0.0; p.output_capacity()],
        vec![0.0; p.output_capacity()],
    ];
    let mut consumed = 0;
    let mut first = true;
    let mut discard = p.initial_unused_frames();
    while audio.len() < target * ch {
        let n = if first {
            p.prefill_frames()
        } else {
            p.frames_needed(0)?
        };
        for (c, plane) in inp.iter_mut().enumerate().take(ch) {
            for (i, sample) in plane.iter_mut().enumerate().take(n) {
                *sample = input
                    .data
                    .get((consumed + i) * ch + c)
                    .copied()
                    .unwrap_or(0.0);
            }
        }
        let iv = [&inp[0][..n], &inp[1][..n]];
        let [l, r] = &mut out;
        let mut ov = [l.as_mut_slice(), r.as_mut_slice()];
        let count = if first {
            p.prefill(&iv[..ch], Some(&mut ov[..ch]))?
        } else {
            p.process(&iv[..ch], &mut ov[..ch])?
        };
        first = false;
        consumed += n;
        let skip = discard.min(count);
        discard -= skip;
        for i in skip..count {
            if audio.len() == target * ch {
                break;
            }
            for plane in &out[..ch] {
                audio.push(plane[i]);
            }
        }
    }
    Ok((audio, actual))
}
