# Time-Stretch / Pitch-Shift Research Synthesis

Date: 2026-02-28

## Scope Notes
- The patent IDs requested in the prompt appear to include at least one mismatch:
  - `US8805696` is an ExxonMobil process patent (not audio DSP).
  - `US9263026` was not found as a valid/public audio patent record in the queried sources.
- Closest directly relevant patent family members found (Disch/Nagel/Walther lineage): `US8837750`, `US9230557`, `US9240196`, plus earlier phase-lock/partition work (`EP1918911A1` / `US20050010397A1`).
- When a detail below is marked **Inference**, it is reconstructed from multiple sources rather than explicitly stated.

### Requested Source Coverage (Prompt Checklist)
- `US 8,805,696`: checked; not audio DSP (mismatch).
- `US 9,263,026`: requested ID did not resolve to a matching/public audio record in searched sources.
- zplane/Disch/Wolff/Nagel/Walther patent lineage analyzed through: `US8837750B2`, `US9230557B2`, `US9240196B2`, `EP1918911A1`, `US20050010397A1`, `US10283130B2`.
- Laroche & Dolson (1999): covered.
- Dolson (1986): covered.
- Bonada (2000): covered via survey/toolbox lineage due limited direct full text access.
- Röbel (2003): covered from DAFx paper.
- Driedger/Müller/Disch and Driedger/Müller/Ewert (2014): covered via DAFx toolbox + SPL/HPSS hybrid references.
- Driedger & Müller (2016): covered in depth (review synthesis).
- Verhelst & Roelands (1993): covered.
- Grofit & Lavner (2008): covered.
- Magalhães & de Góes (2013): covered via cited selective phase-locking method summary.
- Moinet & Dutoit (2011): covered from DAFx PVSOLA paper.

---

## 1. Patent Analysis

### 1.1 Patent Inventory (Relevant)
- `US8837750B2` — transient-event handling in spectral manipulation with padded-block strategy.
- `US9230557B2` — continuation/refinement of transient-event handling and routing/recombination.
- `US9240196B2` — handling transient events under replay-speed/pitch modification.
- `EP1918911A1` / `US20050010397A1` — phase-locking with Bark-scale spectral partition and transient-focused processing.
- `US10283130B2` — vertical phase correction in subbands for phase-vocoder-style processing.

### 1.2 Per-Patent Architecture Extraction

| Patent | Signal Flow / Block Diagram | Transient Detection | Decomposition | Per-Component Processing | Recombination | Parameters / Values | Phase Coherence Strategy | Real-Time / Latency Claims |
|---|---|---|---|---|---|---|---|---|
| `US8837750B2` | Input -> transient detector -> adaptive pre/post processing around transient -> modified spectral path -> output | High-frequency-energy change around event time; transition-state selection around transient | Transient vs non-transient sub-regions via state machine | At transient windows: no time-stretch or alternate manipulation; non-transient regions stretch normally | Region-wise OLA with special transient insertion | Explicit example: process block length ~716 samples, time-stretch block length ~1024, ratio guidance ~1.4x | Preserve or reinsert transient-local phase/state while allowing spectral manipulation elsewhere | Designed for streaming blocks with bounded decisions per block |
| `US9230557B2` | Similar to `US8837750` with explicit subband routing and adjusted overlap skipping during transient | Same family: detects transient segment and routes around it | Spectral subbands + transient event path | Non-transient: stretch/pitch manipulation; transient bins/segments bypass or reduced manipulation | Re-add skipped overlap for output-length consistency | Window and skip/reinsert logic described at block boundaries | Avoids severe phase inconsistency by preserving original transient segments | Blockwise method suitable for real-time frame processing |
| `US9240196B2` | Audio manipulation pipeline with transient-aware correction during replay-speed/pitch change | Detect transient events, maintain event timing during manipulation | Event vs background separation | Event path preserved/adjusted distinctly from manipulated background | Event-aligned recombination | Event timing and frame-alignment constraints emphasized | Maintains event-local coherence, reduces smearing | Explicitly targets replay speed/pitch use-cases (practical real-time scenario) |
| `EP1918911A1` / `US20050010397A1` | STFT -> adaptive bark-like band partition -> phase-lock groups -> synthesis | Uses “dominant line” and/or abrupt spectral structure change (incl. transients) | Multiple frequency partitions (low/mid/high regions by spectral structure) | Different phase-lock strengths/peak-grouping by band | Weighted overlap-add synthesis | Bark-scale partition + varying lock weights; references 50%/75% overlap trade-offs in related work | Identity/peak-linked phase correction to reduce phasiness | Framed as efficient transform-domain processing |
| `US10283130B2` | Multi-band vocoder path with vertical phase correction in selected bands | Not primarily a detector patent; complements detector-based stacks | Subband decomposition | Per-band phase correction to retain vertical coherence | Standard synthesis recombination | Subband-local correction rules (no fixed FFT-size claim in extracted snippet) | Explicit vertical coherence correction | Focus on reducing artifacts without major extra compute |

### 1.3 zplane / Inventor Search Outcome
- Primary inventor trails for Sascha Disch in public records are heavily linked to Fraunhofer-assigned transient-handling and phase-vocoder patents that match Elastique-like behavior.
- Explicit “zplane development GmbH assigned” records were sparse in this specific subtopic search; most technically aligned records were in the Fraunhofer/Disch/Nagel lineage.
- **Inference:** product architecture likely combines patented transient-event bypass/reinsertion with modern multi-band phase-locking and robust real-time buffer management.

---

## 2. Academic Technique Synthesis

## 2.1 Phase Vocoder Foundations

### Dolson (1986) — Phase Vocoder Tutorial
- Core: STFT analysis/synthesis, phase unwrapping, instantaneous frequency estimation, overlap-add.
- Complexity: `O(T * N log N)` for FFT frames.
- Failure modes: transient smearing, phasiness from weak vertical phase coherence.
- Implementation-critical: consistent analysis/synthesis windows and COLA behavior.

### Laroche & Dolson (1999) — Improved PV / Phase Locking
- Core: preserve vertical coherence by locking neighboring bins to spectral peaks (identity phase locking).
- Complexity: PV + peak detection / bin-group assignment (near-linear in bins per frame).
- Artifacts reduced: metallic/phasey character on harmonic/polyphonic material.
- Tradeoff: peak assignment errors can cause local warble.

## 2.2 Transient Preservation

### Bonada (2000) (as summarized in Driedger/Müller survey lineage)
- Core direction: transient-aware frequency-domain handling to approach near-lossless TSM quality.
- Typical strategy: detect/segment transient regions and avoid naive phase propagation through attacks.
- Known issue addressed: blurred attacks and temporal envelope damage.

### Röbel (2003) — New Transient Processing in PV
- Core: bin-level transient detection/processing using spectral-bin criteria and transient position relative to window center.
- Important result: no strict need to force local ratio=1 if phase reinitialization is aligned near optimal transient position.
- Complexity: PV + bin-level transient logic and local criteria.
- Failure modes: detector mistakes still leak artifacts, but improved over broad-band transient gating.

### Driedger, Müller, Ewert (2014 SPL / 2014 DAFx toolbox context)
- Core: HPSS-based hybrid TSM:
  - Harmonic component -> phase-vocoder + phase locking.
  - Percussive component -> OLA/WSOLA-style path.
- Benefit: preserves both stable harmonics and transient punch better than single-method approaches.

## 2.3 Hybrid / Modern System Patterns

### Driedger & Müller (2016) Review
- Taxonomy: time-domain, frequency-domain, model-based, and hybrid combinations.
- Practical takeaway: no single algorithm dominates all signals; hybrid routing + transient-aware logic is strongest general strategy.
- Recommended evaluation includes objective + listening metrics, content-dependent presets.

### Nagel & Walther transient handling (AES 2009 citation chain)
- Core direction: explicit transient management within time-stretch algorithms to avoid attack corruption.
- **Inference:** similar design objective to transient bypass/reinsert patent family.

## 2.4 Time-Domain Methods

### Verhelst & Roelands (1993) — WSOLA
- Core: overlap-add with waveform similarity search to align local periodic structure.
- Complexity: `O(T * search_window * frame_len)` worst-case without optimization.
- Strengths: strong local transient/time-domain preservation for speech/percussive events.
- Weaknesses: stutter/drift artifacts when search picks poor matches; less robust on dense polyphony.

### Grofit & Lavner (2008) — Enhanced WSOLA
- Core: WSOLA + transient management (PST detection, selective handling of transients vs steady sections).
- Stronger quality on music than uniform WSOLA.

## 2.5 Phase Coherence Extensions

### Magalhães & de Góes (2013) — Selective Phase Locking
- Core idea: selective (not uniform) locking to preserve key spectral relationships while avoiding overconstraint.
- Benefit: better balance between coherence and flexibility.

### Moinet & Dutoit (2011) — PVSOLA
- Core: periodically reset PV phase coherence by inserting synchronization points selected with SOLA-like cross-correlation.
- Hybrid temporal/spectral correction loop.
- Strength: notable phasiness reduction without full-time phase-lock overhead.
- Cost: additional cross-correlation and reset scheduling logic.

---

## 3. Open-Source Implementation Study

## 3.1 Rubber Band (R3/Finer)
- Multi-resolution analysis bands (long/mid/short FFT ranges).
- Harmonic/Percussive/Residual classification using horizontal/vertical median filters.
- Segmentation guides adaptive FFT-band assignment and phase-lock bands.
- Guided phase advance with:
  - peak tracking,
  - channel-aware locking,
  - kick/pre-kick logic,
  - high-frequency unlock at extreme ratios.
- Real-time architecture:
  - fixed-capacity ring buffers,
  - explicit start pad/start delay reporting,
  - bounded hop/ratio update logic.
- Elastique likely differs by:
  - stronger proprietary event handling and psychoacoustic tuning,
  - deeper per-content heuristics/presets,
  - potentially tighter commercial QA across edge cases.

## 3.2 SoundTouch
- WSOLA-like time-domain core with auto sequence/seek window adaptation by tempo.
- Strong simplicity and speed.
- Limited transform-domain coherence controls; more artifact-prone under large/modulated ratios.

## 3.3 Paulstretch
- Extreme-stretch spectral method with randomized phases.
- Not suitable as general DJ-grade low-latency method.

## 3.4 librosa
- Educational/reference PV implementation.
- Explicitly not transient-robust; good baseline, not production quality target.

---

## 4. Taxonomy of Approaches

| Category | Strengths | Weaknesses | Best Use |
|---|---|---|---|
| Time-domain (OLA/WSOLA) | attack sharpness, low conceptual complexity | drift/stutter, weaker harmonic handling | speech/percussive mild ranges |
| Frequency-domain (PV + phase locking) | harmonic consistency, large ratio flexibility | transient smear if unmanaged | polyphonic/tonal material |
| HPSS hybrid | best broad quality trade-off | more modules + tuning complexity | full-mix music, DJ workflows |
| Model-based (sinusoidal+noise etc.) | high controllability | model mismatch risk, complexity | offline high-end processing |

---

## 5. Elastique Architecture Reconstruction (Best-Effort)

**Inference-based reconstruction:**
1. Multi-feature transient detector (flux/energy/phase + band emphasis).
2. Event-aware segmentation with lookahead.
3. Split processing:
   - transient/event path (time-domain or minimally manipulated reinsertion),
   - steady-state path (multi-resolution PV + adaptive phase locking),
   - possible residual/noise path with relaxed locking.
4. Event-aligned recombination with adaptive crossfades.
5. Dynamic quality/latency mode controls and robust realtime buffering.
6. Pitch shift path includes envelope/formant protection and transient-aware event handling.

---

## 6. Critical Implementation Details (What Separates Good vs Great)

- Analysis/synthesis window pair must preserve overlap-add normalization under chosen hop ratios.
- Phase locking should be selective and peak-informed, not globally rigid.
- Transient detection must include adaptive thresholding and local timing refinement.
- Event reinsertion/bypass should be aligned to window-center criteria (Röbel-style insight).
- Multi-resolution bands improve quality but require robust guide logic at ratio extremes.
- Cross-channel coherence handling (stereo mid/side or shared guidance) is essential for DJ use.

---

## 7. Transient Handling Deep Dive

- Detector features used in high-quality systems:
  - weighted spectral flux,
  - frame energy slope,
  - phase deviation / group-delay cues,
  - low/high-band specific checks for kick/hat discrimination.
- Decision mechanics:
  - adaptive thresholds (median/MAD or mean/sigma),
  - minimum onset spacing,
  - lookahead confirmation (avoid false spikes),
  - event strength scoring for path routing.
- Processing:
  - transient path via WSOLA/direct-event insertion,
  - steady path via PV,
  - optional HPSS residual handling.
- Recombination:
  - raised-cosine crossfades at boundaries,
  - overlap compensation to keep exact final timeline.

---

## 8. Phase Coherence Strategies

- Horizontal coherence (frame-to-frame bin continuity): standard PV unwrapped phase propagation.
- Vertical coherence (across bins in one frame): identity/selective phase locking around spectral peaks.
- Channel coherence: shared peak/channel-lock policy at low-mid frequencies.
- Practical pattern: use stronger locking in low bands, relax in highs at high stretch ratios.

---

## 9. Real-Time Constraints and Budgets

Target (DJ): `<10 ms` perceived/control-latency envelope where feasible.

Implementation constraints:
- fixed-capacity ring buffers,
- zero-allocation in steady audio callback path,
- no locks/syscalls in callback,
- bounded per-block loops,
- preplanned FFT/resampler allocations,
- SIMD or vectorized hot paths for OLA/windowing/magnitude/phase loops.

---

## 10. Quality Benchmarking Guidance

Objective metrics to track:
- transient timing MAE (ms) at expected event anchors,
- STFT magnitude distortion vs reference (`L1/L2` normalized),
- phase-coherence stability (pairwise harmonic phase-diff variance),
- unexpected-band energy ratio (artifact leakage proxy),
- loudness drift (LUFS delta).

Subjective checks:
- kick/snare attack sharpness,
- bass solidity (no flanging/cancellation),
- vocal clarity/formant stability,
- phasiness/metallic tails on pads and cymbals.

---

## Sources

### Patents / Patent Records
- [US8805696](https://patents.google.com/patent/US8805696)
- [US8837750B2](https://patents.google.com/patent/US8837750B2/en)
- [US9230557B2](https://patents.google.com/patent/US9230557B2/en)
- [US9240196B2](https://patents.google.com/patent/US9240196B2/en)
- [EP1918911A1](https://patents.google.com/patent/EP1918911A1/en)
- [US20050010397A1](https://patents.google.com/patent/US20050010397A1/en)
- [US10283130B2](https://patents.google.com/patent/US10283130B2/en)

### Papers / Technical Sources
- [Laroche & Dolson 1999 (IEEE record)](https://ieeexplore.ieee.org/document/759041)
- [Dolson 1986 tutorial text](http://www.panix.com/~jens/pvoc-dolson.par)
- [Röbel 2003 DAFx PDF](https://www.dafx.de/paper-archive/2003/pdfs/dafx32.pdf)
- [Driedger & Müller 2016 review (MDPI)](https://www.mdpi.com/2076-3417/6/2/57)
- [TSM Toolbox DAFx 2014](https://www.dafx.de/paper-archive/2014/dafx14_jonathan_driedger_tsm_toolbox_matlab_imple.pdf)
- [WSOLA original IEEE record](https://ieeexplore.ieee.org/document/319366)
- [Enhanced WSOLA w/ transients IEEE record](https://ieeexplore.ieee.org/document/4381234)
- [PVSOLA DAFx 2011 PDF](https://www.dafx.de/paper-archive/2011/Papers/57_e.pdf)

### Open-Source Implementations
- [Rubber Band](https://github.com/breakfastquay/rubberband)
- [SoundTouch](https://codeberg.org/soundtouch/soundtouch)
- [Paulstretch](https://github.com/paulnasca/paulstretch_cpp)
- [librosa](https://github.com/librosa/librosa)
