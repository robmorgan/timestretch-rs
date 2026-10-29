use super::*;
use serde_json::Value;

fn fixture() -> &'static Value {
    static FIXTURE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        serde_json::from_str(include_str!("../../tests/fixtures/native/pv.json")).unwrap()
    })
}
fn num(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}
fn int(v: &Value) -> i32 {
    v.as_i64().unwrap() as i32
}
fn f(v: &Value) -> f32 {
    f32::from_bits(v.as_u64().unwrap() as u32)
}
fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(f).collect()
}
fn ints(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(int).collect()
}
fn indices(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(num).collect()
}
fn eq_bits(actual: &[f32], expected: &Value, context: &str) {
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{context} at {i}: {a} vs {b}");
    }
}
fn eq_auxiliary_window(actual: &[f32], expected: &Value) {
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        // These coefficients use the platform's scalar sine. The fixture was
        // recorded on macOS; other implementations can round one ULP away.
        // Retain exact checks for macOS and for the zero-filled window regions.
        let allowed_ulps = u32::from(!cfg!(target_os = "macos") && b != 0.0);
        assert!(
            a.to_bits().abs_diff(b.to_bits()) <= allowed_ulps,
            "aux window at {i}: {a} vs {b} (limit {allowed_ulps} ULP)"
        );
    }
}
fn scalars(pv: &PhaseVocoder, expected: &Value, exact_float: bool) {
    let s = &pv.state;
    for (name, value) in [
        ("input_hop", s.input_hop),
        ("output_hop", s.output_hop),
        ("frame_counter", s.frame_counter),
    ] {
        assert_eq!(value, num(&expected[name]), "{name}");
    }
    for (name, value) in [
        ("current_stretch", s.current_stretch),
        ("target_stretch", s.target_stretch),
        ("window_area_gain", s.window_area_gain),
    ] {
        if exact_float {
            assert_eq!(
                value.to_bits(),
                f(&expected[name]).to_bits(),
                "{name}: {value} vs {}",
                f(&expected[name])
            );
        } else {
            assert!((value - f(&expected[name])).abs() < 2e-5, "{name}");
        }
    }
    assert_eq!(
        s.original_spectrum_bypass,
        expected["original_spectrum_bypass"].as_bool().unwrap()
    );
    for (name, value) in [
        ("transient_counter", s.transient_counter),
        ("transient_target", s.transient_target),
        ("alignment_offset", s.alignment_offset),
        ("auxiliary_window_position", s.auxiliary_window_position),
    ] {
        assert_eq!(value, int(&expected[name]), "{name}");
    }
}
#[test]
fn native_reference_selector_exact_arithmetic() {
    let data = fixture();
    for case in data["selectors"].as_array().unwrap() {
        let n = num(&case["n"]);
        let mut selector = Selector::new(n);
        selector.process(
            num(&case["hop"]),
            num(&case["limit"]),
            case["infinite"].as_bool().unwrap(),
            &floats(&case["phase"]),
            &floats(&case["magnitude"]),
            &ints(&case["saved"]),
        );
        assert_eq!(selector.count, num(&case["count"]));
        assert_eq!(selector.references, indices(&case["references"]));
        assert_eq!(selector.map, indices(&case["map"]));
        assert_eq!(selector.work, ints(&case["work"]));
    }
}
#[test]
fn native_transient_controller_exact_arithmetic_and_history() {
    let data = fixture();
    for case in data["classifiers"].as_array().unwrap() {
        let n = num(&case["n"]);
        let mut pv = PhaseVocoder::new(n, 1).unwrap();
        let initial = &case["initial"];
        pv.state.input_hop = num(&case["hop"]);
        pv.state.frame_counter = num(&initial["frame_counter"]);
        pv.state.current_stretch = f(&initial["current_stretch"]);
        pv.state.target_stretch = f(&initial["target_stretch"]);
        pv.state.original_spectrum_bypass = initial["original_spectrum_bypass"].as_bool().unwrap();
        pv.state.transient_counter = int(&initial["transient_counter"]);
        pv.state.transient_target = int(&initial["transient_target"]);
        pv.infinite = case["infinite"].as_bool().unwrap();
        pv.sync_enabled = case["sync"].as_bool().unwrap();
        pv.magnitude[0] = floats(&case["magnitude"]);
        for history in &mut pv.history {
            let n = history.len();
            history.copy_from_slice(&floats(&case["older"])[..n]);
        }
        pv.time_cross = floats(&case["cross"]);
        pv.saved_state = ints(&case["saved"]);
        pv.centered = floats(&case["center"]);
        let references = indices(&case["references"]);
        pv.selector.count = references.len();
        pv.selector.references[..references.len()].copy_from_slice(&references);
        pv.selector.map = indices(&case["map"]);
        pv.current_state.fill(9);
        pv.change.fill(-999.0);
        pv.location.fill(-999.0);
        pv.energy.fill(-999.0);
        pv.classify();
        assert_eq!(pv.current_state, ints(&case["states"]));
        eq_bits(&pv.change, &case["change"], "change");
        eq_bits(&pv.location, &case["locations"], "location");
        eq_bits(&pv.energy, &case["energy"], "energy");
        eq_bits(&pv.centered, &case["center_after"], "center");
        assert_eq!(
            pv.state.alignment_offset,
            int(&case["state"]["alignment_offset"])
        );
        assert_eq!(
            pv.state.original_spectrum_bypass,
            case["state"]["original_spectrum_bypass"].as_bool().unwrap()
        );
    }
}
#[test]
fn native_window_setter_including_unchanged_hops() {
    let data = fixture();
    for trace in data["windows"].as_array().unwrap() {
        let n = num(&trace["n"]);
        let mut pv = PhaseVocoder::new(n, num(&trace["channels"])).unwrap();
        pv.window_cosine = floats(&trace["cosine"]);
        pv.set_auxiliary(true, true);
        for (step, case) in trace["steps"].as_array().unwrap().iter().enumerate() {
            pv.set_hops(num(&case["hi"]), num(&case["ho"]), f(&case["ratio"]))
                .unwrap();
            eq_bits(
                &pv.analysis_window,
                &case["analysis"],
                &format!("analysis window step {step}"),
            );
            eq_bits(&pv.synthesis_window, &case["synthesis"], "synthesis window");
            eq_auxiliary_window(&pv.aux_window, &case["auxiliary"]);
            eq_bits(
                &pv.normalization_product,
                &case["normalization"],
                "normalization product",
            );
            scalars(&pv, &case["state"], true);
        }
    }
}
#[test]
fn native_full_pv_frames_with_reset_stereo_and_auxiliary() {
    let data = fixture();
    for trace in data["frames"].as_array().unwrap() {
        let n = num(&trace["n"]);
        let channels = num(&trace["channels"]);
        let mut pv = PhaseVocoder::new(n, channels).unwrap();
        pv.window_cosine = floats(&trace["cosine"]);
        pv.set_auxiliary(true, true);
        pv.set_crossover_bin(n / 8).unwrap();
        for (step, case) in trace["steps"].as_array().unwrap().iter().enumerate() {
            if case["reset"].as_bool().unwrap() {
                pv.reset();
                pv.set_auxiliary(true, true);
                pv.set_crossover_bin(n / 8).unwrap();
            }
            pv.set_hops(num(&case["hi"]), num(&case["ho"]), f(&case["ratio"]))
                .unwrap();
            let main: Vec<_> = case["main"]
                .as_array()
                .unwrap()
                .iter()
                .map(floats)
                .collect();
            let main_views: Vec<_> = main.iter().map(Vec::as_slice).collect();
            let aux: Option<Vec<_>> = case["aux"]
                .as_array()
                .map(|x| x.iter().map(floats).collect());
            let aux_views: Option<Vec<_>> =
                aux.as_ref().map(|x| x.iter().map(Vec::as_slice).collect());
            let ho = num(&case["ho"]);
            let mut output = vec![vec![0.0; ho]; channels];
            let mut views: Vec<_> = output.iter_mut().map(Vec::as_mut_slice).collect();
            assert_eq!(
                pv.process(&main_views, aux_views.as_deref(), &mut views)
                    .unwrap(),
                ho
            );
            for (ch, actual) in output.iter().enumerate() {
                let expected = floats(&case["output"][ch]);
                let max = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(max < 2e-5, "PV n={n} step={step} ch={ch} max error={max}");
            }
            scalars(&pv, &case["state"], false);
            assert_eq!(pv.references(), indices(&case["references"]));
            assert_eq!(pv.saved_states(), ints(&case["saved"]));
        }
    }
}
#[test]
fn original_time_ramp_and_invalid_hops() {
    for n in [512, 1024, 4096] {
        let mut pv = PhaseVocoder::new(n, 1).unwrap();
        assert_eq!(pv.time_ramp[0], -1.0);
        assert_eq!(pv.time_ramp[n / 4], -0.5);
        assert_eq!(pv.time_ramp[n / 2], 0.0);
        assert_eq!(pv.time_ramp[3 * n / 4], 0.5);
        assert!(pv.set_hops(1, n / 8, 1.0).is_err());
        assert!(pv.set_hops(n / 8, n, 1.0).is_err());
    }
}

#[test]
fn native_synthesis_states_and_nonzero_alignment() {
    let data = fixture();
    for case in data["branches"].as_array().unwrap() {
        let n = num(&case["n"]);
        let initial = &case["initial"];
        let mut pv = PhaseVocoder::new(n, 2).unwrap();
        pv.state.frame_counter = 5;
        pv.state.current_stretch = 0.5;
        pv.state.target_stretch = 0.75;
        pv.state.window_area_gain = 1.3;
        pv.state.alignment_offset = int(&case["alignment_offset"]);
        pv.alignment_position = 512;
        pv.original = initial["original"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        pv.input_phase = initial["input_phase"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        pv.output_phase = initial["output_phase"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        pv.stored = initial["stored"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        pv.magnitude = initial["magnitude"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        pv.rotation = floats(&initial["rotation"][0]);
        pv.cos_rotation = floats(&initial["cos_rotation"][0]);
        pv.sin_rotation = floats(&initial["sin_rotation"][0]);
        pv.phase_increment = floats(&initial["phase_increment"][0]);
        pv.expected = floats(&initial["expected"][0]);
        pv.location = floats(&initial["location"][0]);
        pv.normalization_product = floats(&initial["normalization"][0]);
        pv.current_state = ints(&initial["current_state"]);
        pv.saved_state = ints(&initial["saved_state"]);
        let references = indices(&case["references"]);
        pv.selector.count = references.len();
        pv.selector.references[..references.len()].copy_from_slice(&references);
        pv.selector.map = indices(&case["map"]);
        pv.synthesize(num(&case["channel"]), true);
        let expected = &case["result"];
        eq_bits(&pv.rotation, &expected["rotation"][0], "reference rotation");
        for (name, actual) in [
            ("cos_rotation", &pv.cos_rotation),
            ("sin_rotation", &pv.sin_rotation),
        ] {
            for (&a, b) in actual.iter().zip(floats(&expected[name][0])) {
                assert!(
                    a.to_bits().abs_diff(b.to_bits()) <= 4,
                    "{name}: provider difference {a} vs {b}"
                );
            }
        }
        assert_eq!(
            pv.processed.len(),
            expected["processed"].as_array().unwrap().len()
        );
        // Windows scalar sine/cosine can round differently even when reference
        // rotations are supplied. Use the same output bound as computed rotations;
        // phase, magnitude, and synthesis state comparisons below remain exact.
        if num(&case["channel"]) == 1 && !cfg!(target_os = "windows") {
            eq_bits(
                &pv.processed,
                &expected["processed"],
                "processed state spectrum",
            );
        } else {
            for (&a, b) in pv.processed.iter().zip(floats(&expected["processed"])) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "rotation provider propagation: {a} vs {b}"
                );
            }
        }
        for ch in 0..2 {
            eq_bits(
                &pv.output_phase[ch],
                &expected["output_phase"][ch],
                "output phase",
            );
            eq_bits(&pv.stored[ch], &expected["stored"][ch], "stored magnitude");
        }
        assert_eq!(pv.current_state, ints(&expected["current_state"]));
        assert_eq!(pv.saved_state, ints(&expected["saved_state"]));
    }
}
