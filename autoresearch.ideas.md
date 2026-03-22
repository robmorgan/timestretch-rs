# Autoresearch Ideas: Streaming Quality

## Current Score: 944.9/1000

## Current Optimized Parameters
- Sub-bass rigid phase locking: 180Hz (streaming-only)
- Phase locking: ROI with adaptive disabled
- Energy EMA: alpha=0.05, gain_smooth=0.30, max_gain=3.0
- Base shelf: gain-proportional, max 120%, gain scale 0.45, crossover 2000Hz
- Ratio shelf: quadratic (t²), threshold 0.4, max 80%, crossover 2000Hz

## Remaining Score Breakdown (latest)
- edm 1.02: 0.972 (centroid 0.732)
- edm 1.5: 0.922 (centroid 0.240) ← BIGGEST remaining weakness for EDM
- edm 2.0: 0.996 (near perfect!)
- harmonic 1.02: 0.982 (centroid 0.924)
- harmonic 1.5: 0.982 (centroid 0.934)
- percussive 1.02: 0.966 (centroid 0.692)
- percussive 1.5: 0.948 (centroid 0.885)
- percussive 2.0: 0.792 (freq 0.686, batch_sim 0.584, energy 0.902)

### Theoretical max improvements:
- percussive 2.0x to 0.90: +13.5 points
- edm 1.5x to 0.96: +4.8 points
- percussive 1.02 centroid: +3.9 points

## Still Possible
- **EDM 1.5x centroid fix**: The 24% centroid score is very low despite shelf. May need a fundamentally different approach for mid-ratio centroid.
- **Percussive 2.0x batch_similarity**: Would need WSOLA integration in streaming
- **Percussive 2.0x energy_score 0.902**: Shelf is over-boosting energy. Need shelf-aware gain or gentler shelf at 2.0x

## Fully Explored (dead ends)
- Sub-bass cutoff: 180Hz optimal (160, 200 worse)
- Phase locking: ROI optimal (Identity, Selective worse)
- EMA alpha: 0.05 optimal (0.04 same)
- Gain smooth: 0.30 optimal
- Max gain: 3.0 optimal (2.5 too low, 3.5 no help)
- Window type: Hann optimal (BlackmanHarris hurts EDM)
- Envelope preservation: zero effect
- Per-band gain with IIR tilt: hurts quality
- Persistent shelf state: hurts quality
- Shelf crossover: 2000Hz optimal (1000, 1500, 3000 worse)
- Shelf threshold: 0.4 optimal (0.3, 0.35 worse)
- Base shelf fixed value: gain-proportional is better
- EMA pre-seeding/fast warmup: hurts percussive
