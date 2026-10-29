use super::*;
use serde_json::Value;
fn i(v: &Value) -> i32 {
    v.as_i64().unwrap() as i32
}
fn ints(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(i).collect()
}
fn f(v: &Value) -> f32 {
    f32::from_bits(v.as_u64().unwrap() as u32)
}
fn control_state(b: &Base) -> Vec<i32> {
    vec![
        b.hi,
        b.initial_hi,
        b.ho,
        b.span_in,
        b.span_out,
        b.n,
        b.next,
        b.calls,
        b.first,
    ]
}
fn direct_state(d: &Direct) -> Vec<i32> {
    vec![
        d.processed,
        d.flush_remaining,
        d.fade_frames,
        0,
        d.output_time,
        d.pending_offset,
        d.time_offset,
        d.initial_hop,
        d.delivered.try_into().unwrap(),
        d.deferred,
        i32::from(d.buffered)
            | i32::from(d.queued) << 8
            | i32::from(d.flush_fresh) << 16
            | i32::from(d.advance_pending) << 24,
        i32::from(d.have_aux),
        d.hop_scale.to_bits() as i32,
    ]
}
fn check_state(d: &Direct, s: &Value) {
    let Core::Hybrid(p) = &d.core else {
        unreachable!()
    };
    let e = d.core.spectral();
    assert_eq!(direct_state(d), ints(&s["direct"]), "Direct");
    assert_eq!(direct_state(&p.inner), ints(&s["inner"]), "inner Direct");
    assert_eq!(control_state(&p.b), ints(&s["hybrid_int"]), "HybridCore");
    assert_eq!(
        control_state(&e.b),
        ints(&s["spectral_int"]),
        "SpectralCore"
    );
    for (name, b) in [("hybrid_times", &p.b), ("spectral_times", &e.b)] {
        for (a, v) in [b.input_time, b.reference, b.real, b.target]
            .iter()
            .zip(s[name].as_array().unwrap())
        {
            assert_eq!(
                a.to_bits(),
                v.as_u64().unwrap(),
                "{name}: {a} vs {}",
                f64::from_bits(v.as_u64().unwrap())
            );
        }
    }
    for (name, b) in [("hybrid_float", &p.b), ("spectral_float", &e.b)] {
        for (a, v) in [b.combined, b.pitch, b.real_pitch, b.inv_pitch, 0.25, b.rate]
            .iter()
            .zip(s[name].as_array().unwrap())
        {
            assert_eq!(a.to_bits(), f(v).to_bits(), "{name}");
        }
    }
    assert_eq!(vec![e.b.modulus, e.b.toggle], ints(&s["toggle"]));
    assert_eq!(p.marks.records().len() as i32, i(&s["marks"]), "mark count");
    assert_eq!(
        vec![d.time(true), d.time(false)],
        s["time"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>(),
        "time"
    );
    let sy = p.synth.state();
    assert_eq!(
        vec![
            sy.buffered_input,
            sy.next_center,
            sy.ready_boundary,
            sy.total_input.try_into().unwrap(),
            sy.previous_output_period,
            sy.input_period,
            0,
            sy.selected_mark_position.try_into().unwrap(),
            sy.selected_mark_index,
            sy.last_mark_index as i32,
            sy.repeat_count
        ],
        ints(&s["synth"]),
        "synthesis state"
    );
    let pv = &e.pv.state;
    assert_eq!(
        vec![
            pv.input_hop as i32,
            pv.output_hop as i32,
            pv.frame_counter as i32,
            pv.transient_counter,
            pv.transient_target,
            pv.alignment_offset
        ],
        ints(&s["pv"]),
        "PV state"
    );
}
#[test]
fn native_reference_complete_scheduler_sequences() {
    check_scheduler_sequences(crate::Numerics::Portable);
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[test]
fn native_reference_complete_scheduler_sequences_accelerate() {
    check_scheduler_sequences(crate::Numerics::Accelerate);
}

fn check_scheduler_sequences(numerics: crate::Numerics) {
    let mut audio_within_bounds = true;
    for fixture in [
        include_str!("../../tests/fixtures/native/scheduler-44100.json"),
        include_str!("../../tests/fixtures/native/scheduler.json"),
        include_str!("../../tests/fixtures/native/scheduler-96000.json"),
    ] {
        let fixture: Value = serde_json::from_str(fixture).unwrap();
        for (case_no, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let ch = i(&case["channels"]) as usize;
            let sample_rate = i(&case["sample_rate"]) as u32;
            let mut processor =
                NativeProcessor::new_with_numerics(sample_rate, ch, numerics).unwrap();
            assert_eq!(
                processor.output_capacity(),
                i(&case["geometry"]["direct_capacity"]) as usize
            );
            assert_eq!(
                processor.prefill_frames(),
                i(&case["geometry"]["prefill"]) as usize
            );
            assert_eq!(
                processor.initial_unused_frames(),
                i(&case["geometry"]["unused"]) as usize
            );
            assert_eq!(processor.direct.core.spectral().pv.history.len(), 32);
            check_state(&processor.direct, &case["initial"]);
            let d = &mut processor.direct;
            let trace = &mut processor.trace;
            let mut out = planes(processor.frame_input[0].len());
            for a in &mut out {
                a.fill(-123.);
            }
            // The 44.1/96 kHz hold/toggle stress sequences have separately measured
            // provider drift and peak/RMS limits. Scheduler, timing, mark, and
            // state comparisons are exact at every rate.
            let (peak_limit, rms_limit) = match (sample_rate, case_no, numerics) {
                // At operation 1826, portable FFT rounding can reverse two
                // nearly equal peaks (bins 964 and 969) and change the phase
                // reference. Measured x86_64 drift is 0.010658 peak / 0.001266
                // RMS. Keep the original limits for the Accelerate provider.
                (44100, 1, crate::Numerics::Portable) => (1.2e-2, 1.5e-3),
                (44100, 0 | 1, _) => (1e-3, 6e-5),
                (96000, 0 | 1, _) => (5e-3, 1e-3),
                _ => (2e-4, 1.1e-5),
            };
            let mut max_error = 0.0f32;
            let mut squared = 0.0f64;
            let mut samples = 0;
            for (step, action) in case["actions"].as_array().unwrap().iter().enumerate() {
                trace.events.clear();
                let data: Vec<Vec<f32>> = action["input"]
                    .as_array()
                    .map(|v| {
                        v.iter()
                            .map(|v| v.as_array().unwrap().iter().map(f).collect())
                            .collect()
                    })
                    .unwrap_or_else(|| vec![vec![]; ch]);
                let input: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
                let mut output = outputs(&mut out);
                let output = &mut output[..ch];
                let result: Result<i32> = (|| {
                    if let Some(op) = action["op"].as_str() {
                        return match op {
                            "ratio" => {
                                let (mut s, mut p) = (
                                    action["stretch"].as_f64().unwrap() as f32,
                                    action["pitch"].as_f64().unwrap() as f32,
                                );
                                let code = d.set_ratio(
                                    &mut s,
                                    &mut p,
                                    action["adjust"].as_bool().unwrap(),
                                    action["continuity"].as_bool().unwrap(),
                                )?;
                                assert_eq!(
                                    [s, p],
                                    [
                                        action["actual"][0].as_f64().unwrap() as f32,
                                        action["actual"][1].as_f64().unwrap() as f32
                                    ]
                                );
                                Ok(code)
                            }
                            "sync" => {
                                let mut p = action["pitch"].as_f64().unwrap() as f32;
                                let code = d.sync(
                                    i(&action["source"]),
                                    i(&action["output_target"]),
                                    &mut p,
                                    action["align"].as_bool().unwrap(),
                                )?;
                                assert_eq!(p, action["actual"].as_f64().unwrap() as f32);
                                Ok(code)
                            }
                            _ => unreachable!(),
                        };
                    }
                    let n = i(&action["count"]) as usize;
                    let option = i(&action["option"]);
                    Ok(match i(&action["op"]) {
                        0 => {
                            d.queue(&input, n)?;
                            0
                        }
                        1 => {
                            d.step(trace)?;
                            0
                        }
                        2 => d.get(output)? as i32,
                        3 => d.needed(option),
                        4 => d.calls(),
                        5 => d.frames_processed(),
                        6 => d.pre_needed(),
                        7 => d.prefill(
                            &input,
                            n,
                            if action["no_output"].as_bool().unwrap() {
                                None
                            } else {
                                Some(output)
                            },
                            trace,
                        )? as i32,
                        8 => d.preprocess(&input, n, output, option, trace)? as i32,
                        9 => {
                            d.reset()?;
                            0
                        }
                        10 => d.core.hold(n != 0, option != 0)?,
                        11 => d.block(&input, n, Some(output), trace)? as i32,
                        12 => d.flush(output, trace)?.map_or(-1, |n| n as i32),
                        13 => {
                            d.reset_local();
                            0
                        }
                        _ => unreachable!(),
                    })
                })();
                let context = format!(
                    "{numerics:?}, {sample_rate} Hz case {case_no}, operation {step} {} {:?}",
                    action["op"],
                    action.get("label")
                );
                assert_eq!(
                    result.unwrap_or_else(|e| panic!("{context}: {e:?}")),
                    i(&action["ret"]),
                    "{context}"
                );
                let expected_trace: Vec<(i32, i32)> = action["trace"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| (i(&v[0]), i(&v[1])))
                    .collect();
                assert_eq!(trace.events, expected_trace, "{context}");
                // Include the operation context on state failures.
                if let Err(err) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    check_state(d, &action["state"])
                })) {
                    panic!("{context}: {err:?}");
                }
                if let Some(detail) = action.get("pv_detail") {
                    assert_eq!(
                        d.core
                            .spectral()
                            .pv
                            .references()
                            .iter()
                            .map(|&x| x as i32)
                            .collect::<Vec<_>>(),
                        ints(&detail["references"]),
                        "{context} references"
                    );
                    assert_eq!(
                        d.core.spectral().pv.saved_states(),
                        ints(&detail["saved"]),
                        "{context} saved states"
                    );
                }
                if let Some(expected) = action["output"].as_array() {
                    for (ch, values) in expected.iter().enumerate() {
                        for (sample, v) in values.as_array().unwrap().iter().enumerate() {
                            let error = (out[ch][sample] - f(v)).abs();
                            max_error = max_error.max(error);
                            squared += f64::from(error).powi(2);
                            samples += 1;
                            audio_within_bounds &= error < peak_limit;
                        }
                    }
                }
            }
            let rms = (squared / samples as f64).sqrt();
            eprintln!(
                "{numerics:?} scheduler {sample_rate} Hz case {case_no}: peak {max_error:e} (limit {peak_limit:e}), rms {rms:e} (limit {rms_limit:e})"
            );
            audio_within_bounds &= rms < rms_limit;
        }
    }
    assert!(
        audio_within_bounds,
        "{numerics:?} scheduler audio exceeded the printed bounds"
    );
}
