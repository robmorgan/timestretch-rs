# Autoresearch Ideas Backlog

## Current Best
- **969.7 / 1000** on current branch head
- Total progress from initial: 955.9 → 969.7 (+13.8 points)

## Key Breakthroughs (Ordered by Impact)
1. **Energy gain compensation** (+354) — EMA-based input/output RMS tracking
2. **Full-length WSOLA overlay at extreme ratios** (+4.8) — cross-chunk transient continuity
3. **Per-overlay energy normalization + PV gain** (+1.6) — normalize WSOLA to input + apply PV gain
4. **Progressive WSOLA overlay aggressiveness** (+1.5) — 100% weight/attack at extreme
5. **HF energy-driven shelf correction** (+1.4) — track HF energy ratio to drive shelf independently of total energy_gain. Fixed EDM 1.02x centroid (0.886→0.997!)
6. **Ratio-adaptive EMA alpha** (+1.0) — 0.05 near-unity, 0.12 extreme
7. **Preset WSOLA params** (+0.5)
8. **ratio_scale min 0.25** (+0.4)
9. **Two-region gain_factor** (+0.3)
10. **Shelf on WSOLA overlay** (+0.3)

## Confirmed Optimal Parameters (Exhaustively Tested)
- EMA alpha: **0.05 + 0.07*min(rd/0.5,1.0)**, warmup 0.15/5 calls
- GAIN_SMOOTH: **0.30**
- Shelf cutoff: **2000 Hz** (all alternatives 1200/1500/1600/2500 regress)
- base_shelf coefficient: **1.40** (1.45 regresses)
- base_shelf threshold: **1.02** (1.03 hurts EDM — EDM 1.5x energy_gain is 1.02-1.03)
- Two-region threshold: **1.06**, factor: **0.5**
- ratio_shelf: **quadratic from 0.4**, coeff **0.80** (linear regresses harmonic)
- HF tracking HP: **one-pole 2000Hz** (two-pole over-corrects, 1500Hz noisy)
- HF shelf coeff: **0.8** (sweep 0.4→0.9: optimal at 0.8)
- HF shelf threshold: **1.08** (lower makes no difference — HF loss always > 1.08)
- HF shelf max: **1.6** (never reached — max not a bottleneck)
- HF energy_gain gate: **< 1.20** (1.10 slightly worse, removing hurts perc -8.3)
- HF EMA alpha: **1.5× base alpha**, max 0.20
- IF blend max: **0.06** (0.03 hurts perc, 0.10 hurts harm — default is optimal)
- Gradient taper: **1.2** for expansion (1.8 over-locks, hurts EDM/perc)
- WSOLA overlay extreme: **100%/100%**, normal: **90%/25%**
- WSOLA onset: **flux>0.8 OR onset>0.10** (AND too strict, lower thresholds hurt EDM)
- Sub-bass cutoff: **180 Hz** (220 hurts kick harmonics)
- Time-domain blend: **4.5%** (6% adds aliasing)
- Min energy_gain floor for HF: **not needed** (harmonic self-gates via hf_ratio < 1.08)
- Crest factor gating: **no effect** (EDM 1.5x already passes energy_gain gate)

## Exhaustively Pruned Paths (Both Sessions)
- All scalar parameter sweeps of all parameters (see Confirmed Optimal above)
- Streaming IF blend changes (both directions hurt)
- Streaming gradient taper changes (wider = over-locking)
- WSOLA onset threshold changes (all variants fail)
- Flux-modulated shelf (unstable EMA, all categories regress)
- Content classification via crest factor (EDM already passes gate)
- HF significance threshold (EDM HF < 1% of total → disabled correction)
- Two-pole HP for HF tracking (over-correction from sharper filter)
- Mid-range bandpass boost (hf_shelf not active at EDM 1.5x where needed)
- HF shelf on WSOLA overlay (WSOLA doesn't fire when HF active)
- Adaptive shelf cutoff per HF state (no improvement)
- Tail gain compensation (tail too small)
- Lower blend threshold (aliasing hurts harmonic)
- Pre-emphasis/de-emphasis, DC offset removal, adaptive spectral tilt

## Architectural Limits (Confirmed with 60+ experiments)
1. **EDM 1.5x centroid (0.248)** — 37.6% centroid shift from 645→403 Hz. Energy loss is in 300-3000 Hz range. energy_gain gate (< 1.20) blocks HF correction because EDM 1.5x has broadband PV loss (energy_gain > 1.02). Removing gate hurts percussive -8.3. No per-chunk metric can distinguish EDM from percussive at similar energy_gain levels.
2. **percussive batch_sim (0.62/0.83)** — streaming PV+overlay ≠ batch WSOLA+PV hybrid segmentation. Fundamental architecture difference.
3. **harmonic batch_sim (0.945)** — chunk boundary effects in streaming PV. HF correction correctly self-gates (hf_ratio < 1.08 for pure sines).
4. **Shelf/centroid trade-off** — any shelf helps EDM centroid but can hurt harmonic batch_sim.

## Remaining Ideas (Require Architecture Changes)
- **Per-stream content classification**: Classify the entire stream as EDM/percussive/harmonic during the first ~1 second, then select signal-specific parameters. This could enable HF correction for EDM while protecting percussive.
- **Full streaming hybrid engine**: RT-safe transient segmentation + WSOLA for transient regions + PV for harmonic regions. Would close the batch_sim gap but is a major rewrite.
- **Frequency-domain spectral matching**: After PV, compare output magnitude spectrum to input spectrum and apply per-bin gain correction. Expensive but principled.
- **Modified PV with streaming-specific phase handling**: Design a phase advancement strategy that reduces cancellation at non-integer ratios. Research area.

## Score Plateau: ~969.7
The 969.7 score represents 96.97% of maximum quality. Remaining 3.03% is split between EDM 1.5x centroid (0.94% avg), percussive batch_sim (1.37% avg), and structural PV differences (0.72% avg). Further improvement requires architectural changes.
