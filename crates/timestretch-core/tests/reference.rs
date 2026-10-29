//! Native fixtures contain independently generated reference expectations.
//! Integer/state/control and non-FFT arithmetic are compared exactly. FFT
//! output uses a documented tolerance across Accelerate and rustfft.
use serde_json::Value;
use timestretch_core::processor::{
    FrameProcessor, GrainSynthesis, Interpolation, MarkAnalysis, PitchMark, PitchMarks, Resampler,
    Splitter,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/native/native.json")).unwrap()
}
fn number(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}
fn integer(value: &Value) -> i64 {
    value.as_i64().unwrap()
}
fn float(value: &Value) -> f32 {
    f32::from_bits(value.as_u64().unwrap() as u32)
}
fn planes(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|ch| ch.as_array().unwrap().iter().map(float).collect())
        .collect()
}
fn assert_bits(actual: &[f32], expected: &Value, context: &str) {
    let values = expected.as_array().unwrap();
    assert_eq!(actual.len(), values.len(), "{context}");
    for (i, (&a, b)) in actual.iter().zip(values).enumerate() {
        assert_eq!(
            a.to_bits(),
            float(b).to_bits(),
            "{context}, sample {i}: {a} vs {}",
            float(b)
        );
    }
}
fn scalar_bits(actual: f32, expected: &Value, context: &str) {
    assert_eq!(
        actual.to_bits(),
        float(expected).to_bits(),
        "{context}: {actual} vs {}",
        float(expected)
    );
}

#[test]
fn native_resampler_retained_state_and_both_interpolators() {
    let fixture = fixture();
    for trace in fixture["resamplers"].as_array().unwrap() {
        let channels = number(&trace["channels"]);
        let interpolation = if trace["hermite"].as_bool().unwrap() {
            Interpolation::Hermite
        } else {
            Interpolation::Cubic
        };
        let mut resampler =
            Resampler::new(channels, number(&trace["capacity"]), interpolation).unwrap();
        for (index, step) in trace["steps"].as_array().unwrap().iter().enumerate() {
            if step["reset"].as_bool().unwrap() {
                resampler.reset();
            }
            let input = planes(&step["input"]);
            let refs: Vec<_> = input.iter().map(Vec::as_slice).collect();
            let expected = planes(&step["output"]);
            let mut output = vec![vec![0.0; expected[0].len()]; channels];
            let mut views: Vec<_> = output.iter_mut().map(Vec::as_mut_slice).collect();
            assert_eq!(
                resampler
                    .process(&refs, &mut views, float(&step["ratio"]))
                    .unwrap(),
                expected[0].len()
            );
            for (ch, actual) in output.iter().enumerate() {
                assert_bits(
                    actual,
                    &step["output"][ch],
                    &format!("resampler step {index} ch {ch}"),
                );
            }
            let state = resampler.state();
            let mut values = state.coefficients.to_vec();
            values.extend(state.fractions);
            values.extend(state.slopes);
            assert_bits(&values, &step["state"]["values"], "resampler scalars");
            assert_eq!(state.input_index, number(&step["state"]["input_index"]));
            assert_eq!(
                state.source_position.to_bits(),
                step["state"]["source_position"].as_u64().unwrap()
            );
        }
    }
}

#[test]
fn native_ordinary_synthesis_audio_lookahead_and_state() {
    let fixture = fixture();
    for trace in fixture["synthesis"].as_array().unwrap() {
        let channels = number(&trace["channels"]);
        let generated =
            GrainSynthesis::new(channels, number(&trace["capacity"]), number(&trace["pad"]))
                .unwrap();
        let table: Vec<_> = trace["table"]
            .as_array()
            .unwrap()
            .iter()
            .map(float)
            .collect();
        // Compare the provider boundary separately from exact grain arithmetic.
        for (&actual, &expected) in generated.flank_table().iter().zip(&table) {
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 8,
                "flank sine-provider error: {actual} vs {expected}"
            );
        }
        let mut synthesis = GrainSynthesis::with_flank_table(
            channels,
            number(&trace["capacity"]),
            number(&trace["pad"]),
            &table,
        )
        .unwrap();
        let mut marks = PitchMarks::new(2048).unwrap();
        for position in trace["marks"].as_array().unwrap().iter().skip(1) {
            marks
                .push(PitchMark {
                    position: integer(position),
                    ..PitchMark::default()
                })
                .unwrap();
        }
        for (index, step) in trace["steps"].as_array().unwrap().iter().enumerate() {
            if step["reset"].as_bool().unwrap() {
                synthesis.reset();
            }
            let input = planes(&step["input"]);
            let refs: Vec<_> = input.iter().map(Vec::as_slice).collect();
            synthesis.push(&refs).unwrap();
            synthesis.set_stretch(float(&step["stretch"])).unwrap();
            if let Some(time) = step["play_time"].as_f64() {
                synthesis.set_play_time(time).unwrap();
            }
            let request = number(&step["request"]);
            assert_eq!(
                synthesis.frames_needed(request, &marks).unwrap(),
                Some(number(&step["needed"]))
            );
            let mut output = vec![vec![0.0; request]; channels];
            let mut views: Vec<_> = output.iter_mut().map(Vec::as_mut_slice).collect();
            assert!(synthesis.process(&mut views, &marks).unwrap());
            for (ch, actual) in output.iter().enumerate() {
                assert_bits(
                    actual,
                    &step["output"][ch],
                    &format!("synthesis step {index} ch {ch}"),
                );
            }
            let s = synthesis.state();
            let expected = &step["state"];
            assert_eq!(
                s.input_time.to_bits(),
                expected["input_time"].as_u64().unwrap(),
                "synthesis time step {index}"
            );
            for (name, actual) in [
                ("buffered_input", i64::from(s.buffered_input)),
                ("total_input", s.total_input),
                ("next_center", i64::from(s.next_center)),
                ("ready_boundary", i64::from(s.ready_boundary)),
                (
                    "previous_output_period",
                    i64::from(s.previous_output_period),
                ),
                ("input_period", i64::from(s.input_period)),
                ("selected_mark_position", s.selected_mark_position),
                ("repeat_count", i64::from(s.repeat_count)),
            ] {
                assert_eq!(
                    actual,
                    integer(&expected[name]),
                    "synthesis step {index}, {name}"
                );
            }
            assert_eq!(s.last_mark_index, number(&expected["last_mark_index"]));
            scalar_bits(
                s.window_index_scale,
                &expected["window_index_scale"],
                "synthesis lookup scale",
            );
        }
    }
}

#[test]
fn native_correlation_analysis_marks_and_filter_history() {
    for numerics in [
        timestretch_core::Numerics::Portable,
        timestretch_core::Numerics::Accelerate,
        timestretch_core::Numerics::AppleFft,
        timestretch_core::Numerics::AppleMath,
    ] {
        if !numerics.available() {
            continue;
        }

        let fixture = fixture();
        for trace in fixture["analysis"].as_array().unwrap() {
            let base = number(&trace["base"]);
            let channels = number(&trace["channels"]);
            let mut analysis = MarkAnalysis::new_with_numerics(
                base,
                channels,
                number(&trace["capacity"]),
                numerics,
            )
            .unwrap();
            let mut marks = PitchMarks::new(2048).unwrap();
            for (index, step) in trace["steps"].as_array().unwrap().iter().enumerate() {
                if step["reset"].as_bool().unwrap() {
                    analysis.reset();
                    marks.reset();
                }
                let input = planes(&step["input"]);
                let refs: Vec<_> = input.iter().map(Vec::as_slice).collect();
                analysis.process(&refs, &mut marks).unwrap();
                let context = format!("analysis base {base} channels {channels} step {index}");
                let expected = step["marks"].as_array().unwrap();
                assert_eq!(marks.records().len(), expected.len(), "{context}");
                for (mark, reference) in marks.records().iter().zip(expected) {
                    assert_eq!(mark.position, integer(&reference["position"]), "{context}");
                    assert_bits(
                        &[
                            mark.energy,
                            mark.transient,
                            mark.tonality,
                            mark.onset,
                            mark.estimated_period,
                        ],
                        &reference["values"],
                        &context,
                    );
                }
                let state = analysis.state();
                let expected = &step["state"];
                assert_eq!(
                    state.buffered_input,
                    number(&expected["buffered_input"]),
                    "{context}"
                );
                for (name, actual) in [
                    ("last_position", state.last_position),
                    ("candidate_interval", i64::from(state.candidate_interval)),
                    ("previous_interval", i64::from(state.previous_interval)),
                    ("predicted_lag", i64::from(state.predicted_lag)),
                    ("analysis_counter", state.analysis_counter),
                ] {
                    assert_eq!(actual, integer(&expected[name]), "{context}, {name}");
                }
                assert_bits(
                    &state.previous_maxima,
                    &expected["previous_maxima"],
                    &context,
                );
                assert_bits(&state.filter_history, &expected["filter_history"], &context);
                assert_eq!(
                    state.transients,
                    [
                        expected["transients"][0].as_bool().unwrap(),
                        expected["transients"][1].as_bool().unwrap()
                    ]
                );
            }
        }
    }
}

#[test]
fn native_splitter_fragment_counts_alignment_and_samples() {
    for numerics in [
        timestretch_core::Numerics::Portable,
        timestretch_core::Numerics::Accelerate,
        timestretch_core::Numerics::AppleFft,
        timestretch_core::Numerics::AppleMath,
    ] {
        if !numerics.available() {
            continue;
        }

        let fixture = fixture();
        for trace in fixture["splitters"].as_array().unwrap() {
            let channels = number(&trace["channels"]);
            let size = number(&trace["size"]);
            let mut splitter = Splitter::new_with_numerics(
                size,
                number(&trace["rate"]) as u32,
                channels,
                number(&trace["capacity"]),
                numerics,
            )
            .unwrap();
            assert_eq!(splitter.delay_frames(), size / 2);
            for (index, step) in trace["steps"].as_array().unwrap().iter().enumerate() {
                if step["reset"].as_bool().unwrap() {
                    splitter.reset();
                }
                let input = planes(&step["input"]);
                let refs: Vec<_> = input.iter().map(Vec::as_slice).collect();
                let frames = step["a"][0].as_array().unwrap().len();
                let mut a = vec![vec![0.0; frames]; channels];
                let mut b = a.clone();
                let mut av: Vec<_> = a.iter_mut().map(Vec::as_mut_slice).collect();
                let mut bv: Vec<_> = b.iter_mut().map(Vec::as_mut_slice).collect();
                assert_eq!(splitter.process(&refs, &mut av, &mut bv).unwrap(), frames);
                for (name, actual) in [("a", &a), ("b", &b)] {
                    let expected = planes(&step[name]);
                    for ch in 0..channels {
                        for i in 0..frames {
                            assert!(
                                (actual[ch][i] - expected[ch][i]).abs() < 2e-6,
                                "splitter {name}, size {size}, step {index}, ch {ch}, sample {i}: {} vs {}",
                                actual[ch][i],
                                expected[ch][i]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_buffers_return_errors_without_advancing_resampler() {
    let mut resampler = Resampler::new(1, 128, Interpolation::Hermite).unwrap();
    let before = resampler.state().source_position.to_bits();
    assert!(
        resampler
            .process(&[&[0.0; 32]], &mut [&mut [0.0; 1]], 1.0)
            .is_err()
    );
    assert_eq!(before, resampler.state().source_position.to_bits());
    assert!(resampler.output_frames(32, f32::NAN).is_err());
    assert!(MarkAnalysis::new(1, 2, 128).is_err());
    assert!(GrainSynthesis::new(0, 128, 32).is_err());
    assert!(Splitter::new(1024, 0, 2, 2048).is_err());
}

#[test]
fn component_chain_is_independent_of_host_feed_chunks() {
    fn run(pattern: &[usize]) -> (Vec<PitchMark>, Vec<f32>) {
        let signal: Vec<_> = (0..8192)
            .map(|i| {
                let phase = i as f32 * 0.079;
                0.3 * phase.sin() + 0.12 * (i as f32 * 0.53).sin()
            })
            .collect();
        let right: Vec<_> = signal.iter().map(|x| 0.3 * x).collect();
        let mut splitter = Splitter::new(1024, 48000, 2, 16384).unwrap();
        let mut analysis = MarkAnalysis::new(512, 2, 16384).unwrap();
        let mut marks = PitchMarks::new(2048).unwrap();
        let mut synthesis = GrainSynthesis::new(2, 16384, 2048).unwrap();
        let mut offset = 0;
        let mut block = 0;
        let mut a = vec![vec![0.0; 4096]; 2];
        let mut b = a.clone();
        while offset < signal.len() {
            let count = pattern[block % pattern.len()].min(signal.len() - offset);
            let input = [
                &signal[offset..offset + count],
                &right[offset..offset + count],
            ];
            let mut av: Vec<_> = a.iter_mut().map(Vec::as_mut_slice).collect();
            let mut bv: Vec<_> = b.iter_mut().map(Vec::as_mut_slice).collect();
            let written = splitter.process(&input, &mut av, &mut bv).unwrap();
            let band = [&b[0][..written], &b[1][..written]];
            analysis.process(&band, &mut marks).unwrap();
            synthesis.push(&band).unwrap();
            offset += count;
            block += 1;
        }
        synthesis.set_stretch(1.5).unwrap();
        let mut resampler = Resampler::new(2, 2048, Interpolation::Hermite).unwrap();
        let mut result = Vec::new();
        for requested in [1, 17, 128, 257, 512, 1024] {
            assert_eq!(synthesis.frames_needed(requested, &marks).unwrap(), Some(0));
            let mut aux = vec![vec![0.0; requested]; 2];
            let mut av: Vec<_> = aux.iter_mut().map(Vec::as_mut_slice).collect();
            assert!(synthesis.process(&mut av, &marks).unwrap());
            let needed = resampler.output_frames(requested, 1.25).unwrap();
            let mut output = vec![vec![0.0; needed]; 2];
            let mut ov: Vec<_> = output.iter_mut().map(Vec::as_mut_slice).collect();
            resampler
                .process(&[&aux[0], &aux[1]], &mut ov, 1.25)
                .unwrap();
            for (&left, &right) in output[0].iter().zip(&output[1]) {
                result.push(left);
                result.push(right);
            }
        }
        (marks.records().to_vec(), result)
    }
    let small = run(&[1, 17, 127, 511]);
    let large = run(&[1024, 512]);
    assert_eq!(small.0, large.0);
    assert_eq!(
        small.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        large.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert!(small.1.iter().any(|x| x.abs() > 1e-4));
}

#[test]
fn native_complete_controlled_frame_chain() {
    let data: Value = serde_json::from_str(include_str!("fixtures/native/pv.json")).unwrap();
    for trace in data["linked"].as_array().unwrap() {
        let channels = number(&trace["channels"]);
        let source = planes(&trace["source"]);
        let mut processor = FrameProcessor::new(
            number(&trace["rate"]) as u32,
            channels,
            number(&trace["n"]),
            number(&trace["capacity"]),
        )
        .unwrap();
        let mut offset = 0;
        let mut maximum_error = 0.0f32;
        for count in trace["chunks"].as_array().unwrap().iter().map(number) {
            let views: Vec<_> = source
                .iter()
                .map(|ch| &ch[offset..offset + count])
                .collect();
            assert_eq!(processor.push(&views).unwrap(), count);
            offset += count;
        }
        for (step, case) in trace["steps"].as_array().unwrap().iter().enumerate() {
            processor
                .set_hops(
                    number(&case["hi"]),
                    number(&case["ho"]),
                    float(&case["ratio"]),
                )
                .unwrap();
            let expected = planes(&case["output"]);
            let pitch = float(&case["pitch"]);
            let frames = processor.output_frames(pitch).unwrap();
            assert_eq!(frames, expected[0].len());
            let mut output = vec![vec![0.0; frames]; channels];
            let mut views: Vec<_> = output.iter_mut().map(Vec::as_mut_slice).collect();
            assert_eq!(
                processor
                    .process_frame(case["source_time"].as_f64().unwrap(), pitch, &mut views)
                    .unwrap(),
                frames
            );
            let max = output
                .iter()
                .zip(&expected)
                .flat_map(|(a, b)| a.iter().zip(b))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            maximum_error = maximum_error.max(max);
            assert!(
                max < 2e-5,
                "connected chain channels={channels} step={step} max_error={max}"
            );
        }
        assert_eq!(processor.consumed_source_frames(), 5120);
        eprintln!(
            "native connected chain channels={channels}, maximum sample error={maximum_error:e}"
        );
    }
}

#[test]
fn scheduled_block_and_fragmented_stream_hosts_are_identical() {
    for numerics in [
        timestretch_core::Numerics::Portable,
        timestretch_core::Numerics::Accelerate,
        timestretch_core::Numerics::AppleFft,
        timestretch_core::Numerics::AppleMath,
    ] {
        if !numerics.available() {
            continue;
        }

        use timestretch_core::processor::{NativeProcessor, RatioAdjustment, StartupOptimization};
        for ratio in [0.75, 1.5, 8.0] {
            let mut block = NativeProcessor::new_with_numerics(48000, 2, numerics).unwrap();
            let mut fragments = NativeProcessor::new_with_numerics(48000, 2, numerics).unwrap();
            for p in [&mut block, &mut fragments] {
                assert_eq!(
                    p.set_ratio(ratio, 1., RatioAdjustment::Stretch, true)
                        .unwrap()
                        .0,
                    0
                );
            }
            let source: [Vec<f32>; 2] = std::array::from_fn(|ch| {
                (0..32768)
                    .map(|i| {
                        0.3 * (i as f32 * 0.079 + ch as f32 * 0.3).sin()
                            + 0.12 * (i as f32 * 0.53).sin()
                    })
                    .collect()
            });
            let mut a = [
                vec![0.; block.output_capacity()],
                vec![0.; block.output_capacity()],
            ];
            let mut b = [
                vec![0.; block.output_capacity()],
                vec![0.; block.output_capacity()],
            ];
            let pre = block.prefill_frames();
            let input = [&source[0][..pre], &source[1][..pre]];
            let [al, ar] = &mut a;
            let [bl, br] = &mut b;
            let n = block
                .preprocess(&input, &mut [al, ar], StartupOptimization::Moderate)
                .unwrap();
            assert_eq!(
                fragments
                    .preprocess(&input, &mut [bl, br], StartupOptimization::Moderate)
                    .unwrap(),
                n
            );
            for ch in 0..2 {
                assert_eq!(a[ch][..n], b[ch][..n]);
            }
            let mut position = pre;
            for _ in 0..48 {
                let need = block.frames_needed(0).unwrap();
                assert_eq!(need, fragments.frames_needed(0).unwrap());
                let input = [
                    &source[0][position..position + need],
                    &source[1][position..position + need],
                ];
                let [al, ar] = &mut a;
                let n = block.process(&input, &mut [al, ar]).unwrap();
                let mut offset = 0;
                while offset < need {
                    let end = (offset + 17).min(need);
                    fragments
                        .queue(&[&input[0][offset..end], &input[1][offset..end]])
                        .unwrap();
                    offset = end;
                }
                fragments.queue(&[&[], &[]]).unwrap();
                for _ in 0..fragments.process_calls() {
                    fragments.step().unwrap();
                }
                let [bl, br] = &mut b;
                assert_eq!(fragments.read(&mut [bl, br]).unwrap(), n);
                for ch in 0..2 {
                    assert_eq!(a[ch][..n], b[ch][..n], "ratio {ratio}, source {position}");
                }
                position += need;
            }
        }
    }
}

#[test]
fn transport_restart_matches_a_fresh_graph_at_native_rates() {
    for numerics in [
        timestretch_core::Numerics::Portable,
        timestretch_core::Numerics::Accelerate,
        timestretch_core::Numerics::AppleFft,
        timestretch_core::Numerics::AppleMath,
    ] {
        if !numerics.available() {
            continue;
        }

        use timestretch_core::processor::{NativeProcessor, RatioAdjustment};
        for sr in [44100, 48000, 96000] {
            let mut used = NativeProcessor::new_with_numerics(sr, 2, numerics).unwrap();
            let mut fresh = NativeProcessor::new_with_numerics(sr, 2, numerics).unwrap();
            let cap = used.output_capacity();
            let audio: Vec<f32> = (0..cap).map(|i| 0.3 * (i as f32 * 0.17).sin()).collect();
            let zeros = vec![0.; cap];
            let mut a = [vec![0.; cap], vec![0.; cap]];
            let mut b = a.clone();
            used.set_ratio(1.5, 0.8, RatioAdjustment::Stretch, true)
                .unwrap();
            let pre = used.prefill_frames();
            let [l, r] = &mut a;
            used.prefill(&[&audio[..pre], &audio[..pre]], Some(&mut [l, r]))
                .unwrap();
            for _ in 0..12 {
                let n = used.frames_needed(0).unwrap();
                let [l, r] = &mut a;
                used.process(&[&audio[..n], &audio[..n]], &mut [l, r])
                    .unwrap();
            }
            used.restart().unwrap();
            for p in [&mut used, &mut fresh] {
                p.set_ratio(0.75, 1., RatioAdjustment::Stretch, true)
                    .unwrap();
            }
            for block in 0..12 {
                let n = if block == 0 {
                    used.prefill_frames()
                } else {
                    used.frames_needed(0).unwrap()
                };
                let signal = if block == 0 { &zeros } else { &audio };
                let input = [&signal[..n], &signal[..n]];
                let [l, r] = &mut a;
                let [fl, fr] = &mut b;
                let x = if block == 0 {
                    used.prefill(&input, Some(&mut [l, r]))
                } else {
                    used.process(&input, &mut [l, r])
                }
                .unwrap();
                let y = if block == 0 {
                    fresh.prefill(&input, Some(&mut [fl, fr]))
                } else {
                    fresh.process(&input, &mut [fl, fr])
                }
                .unwrap();
                assert_eq!(x, y);
                assert_eq!(a[0][..x], b[0][..y], "restart leaks at {sr}");
                assert_eq!(a[1][..x], b[1][..y]);
            }
        }
    }
}
