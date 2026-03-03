#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import librosa
import soundfile as sf
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def calculate_spectral_convergence(ref, test):
    """
    Measures how close the spectrogram magnitudes are.
    Value: 0 to 1 (lower is better, but we invert for score).
    Commonly used in phase retrieval evaluation.
    """
    ref_mag = np.abs(librosa.stft(ref))
    test_mag = np.abs(librosa.stft(test))
    
    # Align lengths if necessary
    min_cols = min(ref_mag.shape[1], test_mag.shape[1])
    ref_mag = ref_mag[:, :min_cols]
    test_mag = test_mag[:, :min_cols]
    
    numerator = np.linalg.norm(ref_mag - test_mag, 'fro')
    denominator = np.linalg.norm(ref_mag, 'fro')
    
    if denominator == 0: return 0
    return numerator / denominator

def calculate_log_spectral_distance(ref, test):
    """
    Measures distance in dB scale. Closer to 0 is better.
    """
    ref_stft = np.abs(librosa.stft(ref))
    test_stft = np.abs(librosa.stft(test))
    
    min_cols = min(ref_stft.shape[1], test_stft.shape[1])
    ref_stft = ref_stft[:, :min_cols]
    test_stft = test_stft[:, :min_cols]
    
    # Avoid log(0)
    eps = 1e-10
    lsd = np.mean(np.sqrt(np.mean((20 * np.log10(ref_stft + eps) - 20 * np.log10(test_stft + eps))**2, axis=0)))
    return lsd

def calculate_mfcc_distance(ref, test, sr):
    """
    Measures timbral similarity using Mel-frequency cepstral coefficients.
    """
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr)
    test_mfcc = librosa.feature.mfcc(y=test, sr=sr)
    
    min_cols = min(ref_mfcc.shape[1], test_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_cols]
    test_mfcc = test_mfcc[:, :min_cols]
    
    # Cosine distance
    dot_product = np.sum(ref_mfcc * test_mfcc)
    norm_ref = np.linalg.norm(ref_mfcc)
    norm_test = np.linalg.norm(test_mfcc)
    
    if norm_ref == 0 or norm_test == 0: return 1.0
    cosine_sim = dot_product / (norm_ref * norm_test)
    return 1.0 - cosine_sim

def calculate_transient_preservation(ref, test, sr, tolerance_ms=15):
    """
    Measures how well onsets (transients) are preserved.
    Score 0-1, higher is better.
    Uses spectral flux (onset_strength).
    """
    ref_onset = librosa.onset.onset_strength(y=ref, sr=sr)
    test_onset = librosa.onset.onset_strength(y=test, sr=sr)
    
    ref_peaks = librosa.util.peak_pick(ref_onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    test_peaks = librosa.util.peak_pick(test_onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    
    if len(ref_peaks) == 0:
        return 1.0 if len(test_peaks) == 0 else 0.5
        
    # Convert frames to ms
    hop_length = 512
    ref_times = librosa.frames_to_time(ref_peaks, sr=sr, hop_length=hop_length) * 1000
    test_times = librosa.frames_to_time(test_peaks, sr=sr, hop_length=hop_length) * 1000
    
    matched = 0
    for rt in ref_times:
        diffs = np.abs(test_times - rt)
        if len(diffs) > 0 and np.min(diffs) <= tolerance_ms:
            matched += 1
            
    return matched / len(ref_times)

def calculate_stereo_width_preservation(ref_L, ref_R, test_L, test_R):
    """
    Measures how well the mid/side energy ratio is preserved.
    Compares the side-to-mid energy ratio between reference and test.
    Score 0-1, higher is better (1 = identical stereo width).
    """
    eps = 1e-10
    ref_mid = (ref_L + ref_R) / 2.0
    ref_side = (ref_L - ref_R) / 2.0
    test_mid = (test_L + test_R) / 2.0
    test_side = (test_L - test_R) / 2.0

    ref_mid_e = np.sum(ref_mid ** 2) + eps
    ref_side_e = np.sum(ref_side ** 2) + eps
    test_mid_e = np.sum(test_mid ** 2) + eps
    test_side_e = np.sum(test_side ** 2) + eps

    ref_ratio = ref_side_e / ref_mid_e
    test_ratio = test_side_e / test_mid_e

    # Ratio of ratios, capped. 1.0 = perfect preservation.
    if ref_ratio > test_ratio:
        preservation = test_ratio / ref_ratio
    else:
        preservation = ref_ratio / test_ratio

    return float(np.clip(preservation, 0, 1))


def calculate_interchannel_correlation_preservation(ref_L, ref_R, test_L, test_R):
    """
    Measures how well the L-R phase/correlation relationship is preserved.
    Computes the normalized cross-correlation between L and R for both
    reference and test, then compares them.
    Score 0-1, higher is better.
    """
    eps = 1e-10

    def icc(L, R):
        norm_L = np.linalg.norm(L)
        norm_R = np.linalg.norm(R)
        if norm_L < eps or norm_R < eps:
            return 0.0
        return float(np.dot(L, R) / (norm_L * norm_R))

    ref_icc = icc(ref_L, ref_R)
    test_icc = icc(test_L, test_R)

    # Both ICC values are in [-1, 1]. Compare how close they are.
    # Max possible difference is 2.0.
    diff = abs(ref_icc - test_icc)
    return float(1.0 - diff / 2.0)


def calculate_panning_consistency(ref_L, ref_R, test_L, test_R, sr, hop_length=512):
    """
    Measures frame-by-frame panning preservation.
    Computes per-frame L-R energy balance for ref and test, then
    measures correlation between the two balance curves.
    Score 0-1, higher is better.
    """
    eps = 1e-10
    frame_size = hop_length * 2

    def pan_curve(L, R):
        n_frames = len(L) // frame_size
        if n_frames == 0:
            return np.array([0.0])
        balance = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            l_e = np.sum(L[start:end] ** 2) + eps
            r_e = np.sum(R[start:end] ** 2) + eps
            balance[i] = (l_e - r_e) / (l_e + r_e)
        return balance

    ref_pan = pan_curve(ref_L, ref_R)
    test_pan = pan_curve(test_L, test_R)

    # Align lengths
    min_len = min(len(ref_pan), len(test_pan))
    if min_len == 0:
        return 1.0
    ref_pan = ref_pan[:min_len]
    test_pan = test_pan[:min_len]

    # Correlation of panning curves
    ref_norm = np.linalg.norm(ref_pan)
    test_norm = np.linalg.norm(test_pan)
    if ref_norm < eps or test_norm < eps:
        # Both nearly silent or center-panned — consider it a match
        return 1.0 if (ref_norm < eps and test_norm < eps) else 0.5

    correlation = np.dot(ref_pan, test_pan) / (ref_norm * test_norm)
    # Map from [-1, 1] to [0, 1]
    return float(np.clip((correlation + 1.0) / 2.0, 0, 1))


def score_pair(ref_path, test_path, weights, stereo=False):
    # Load mono-downmixed for standard metrics
    ref, sr_ref = librosa.load(ref_path, sr=None, mono=True)
    test, sr_test = librosa.load(test_path, sr=sr_ref, mono=True)

    # Trim to match
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]

    sc = calculate_spectral_convergence(ref, test)
    lsd = calculate_log_spectral_distance(ref, test)
    mfcc_dist = calculate_mfcc_distance(ref, test, sr_ref)
    tp = calculate_transient_preservation(ref, test, sr_ref)

    # Map raw metrics to 0-100 scores
    sc_score = max(0, 100 * (1.0 - sc))
    lsd_score = 100 * np.exp(-max(0, lsd - 3.0) / 7.0)
    mfcc_score = max(0, 100 * (1.0 - mfcc_dist))
    tp_score = 100 * tp

    metrics = {
        "spectral_convergence": sc_score,
        "log_spectral_distance": lsd_score,
        "mfcc_distance": mfcc_score,
        "transient_preservation": tp_score
    }
    raw = {
        "sc": float(sc),
        "lsd": float(lsd),
        "mfcc_dist": float(mfcc_dist),
        "tp": float(tp)
    }

    # Stereo-specific metrics
    if stereo:
        ref_stereo, _ = librosa.load(ref_path, sr=sr_ref, mono=False)
        test_stereo, _ = librosa.load(test_path, sr=sr_ref, mono=False)

        # Ensure 2-channel; librosa returns (channels, samples)
        if ref_stereo.ndim == 1:
            ref_stereo = np.stack([ref_stereo, ref_stereo])
        if test_stereo.ndim == 1:
            test_stereo = np.stack([test_stereo, test_stereo])

        min_stereo = min(ref_stereo.shape[1], test_stereo.shape[1])
        ref_L, ref_R = ref_stereo[0, :min_stereo], ref_stereo[1, :min_stereo]
        test_L, test_R = test_stereo[0, :min_stereo], test_stereo[1, :min_stereo]

        swp = calculate_stereo_width_preservation(ref_L, ref_R, test_L, test_R)
        icc = calculate_interchannel_correlation_preservation(ref_L, ref_R, test_L, test_R)
        pp = calculate_panning_consistency(ref_L, ref_R, test_L, test_R, sr_ref)

        swp_score = 100 * swp
        icc_score = 100 * icc
        pp_score = 100 * pp

        metrics["stereo_width"] = swp_score
        metrics["interchannel_correlation"] = icc_score
        metrics["panning_consistency"] = pp_score
        raw["swp"] = swp
        raw["icc"] = icc
        raw["pp"] = pp

    # Compute total score with normalized weights (sum to 1.0)
    mono_keys = ['spectral_convergence', 'log_spectral_distance', 'mfcc_distance', 'transient_preservation']
    stereo_keys = ['stereo_width', 'interchannel_correlation', 'panning_consistency']

    if stereo:
        active_keys = mono_keys + stereo_keys
    else:
        active_keys = mono_keys

    raw_sum = sum(weights.get(k, 0) for k in active_keys)
    if raw_sum > 0:
        norm = {k: weights.get(k, 0) / raw_sum for k in active_keys}
    else:
        norm = {k: 1.0 / len(active_keys) for k in active_keys}

    total_score = (
        sc_score * norm.get('spectral_convergence', 0) +
        lsd_score * norm.get('log_spectral_distance', 0) +
        mfcc_score * norm.get('mfcc_distance', 0) +
        tp_score * norm.get('transient_preservation', 0)
    )
    if stereo:
        total_score += (
            swp_score * norm.get('stereo_width', 0) +
            icc_score * norm.get('interchannel_correlation', 0) +
            pp_score * norm.get('panning_consistency', 0)
        )

    return {
        "total_score": total_score,
        "metrics": metrics,
        "raw": raw,
        "stereo": stereo
    }

def generate_spectrogram_diff(ref_path, test_path, out_path):
    if plt is None:
        print("Warning: matplotlib not installed, skipping spectrogram diff.")
        return
        
    ref, sr = librosa.load(ref_path, sr=None)
    test, _ = librosa.load(test_path, sr=sr)
    
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    ref_stft = librosa.amplitude_to_db(np.abs(librosa.stft(ref)), ref=np.max)
    test_stft = librosa.amplitude_to_db(np.abs(librosa.stft(test)), ref=np.max)
    
    diff = np.abs(ref_stft - test_stft)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(ref_stft, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Reference Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 2)
    librosa.display.specshow(test_stft, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Test Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(diff, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Absolute Difference')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perceptual audio quality scoring")
    parser.add_argument("--pair", nargs=2, metavar=('REF', 'TEST'), help="Score a single pair of files")
    parser.add_argument("--batch", metavar='OUTPUT_JSON', help="Batch score based on manifest")
    parser.add_argument("--manifest", default="test_manifest.json", help="Path to manifest")
    parser.add_argument("--spectrogram-diff", nargs=3, metavar=('REF', 'TEST', 'OUT'), help="Generate diff plot")
    parser.add_argument("--baseline", metavar='OUTPUT_JSON', help="Score rubberband HQ vs medium refs (baseline)")
    parser.add_argument("--ref-source", choices=["rubberband", "ableton"], default="rubberband",
                        help="Reference source: rubberband (default) or ableton")
    parser.add_argument("--ableton-manifest", default=None,
                        help="Path to ableton_manifest.json (for --ref-source ableton)")
    parser.add_argument("--config", default="optimize/config.toml", help="Path to config.toml")

    args = parser.parse_args()
    
    # Load weights from config
    weights = {
        "spectral_convergence": 0.30,
        "log_spectral_distance": 0.25,
        "mfcc_distance": 0.20,
        "transient_preservation": 0.25,
        "stereo_width": 0.10,
        "interchannel_correlation": 0.10,
        "panning_consistency": 0.05,
    }
    
    if os.path.exists(args.config):
        try:
            import tomllib # Python 3.11+
            with open(args.config, "rb") as f:
                config = tomllib.load(f)
                if 'scoring' in config:
                    weights.update(config['scoring'])
        except ImportError:
            # Simple fallback parser for weights
            with open(args.config, "r") as f:
                for line in f:
                    if '=' in line and any(k in line for k in weights.keys()):
                        k, v = line.split('=')
                        k = k.strip()
                        if k in weights:
                            weights[k] = float(v.split('#')[0].strip().replace(',', ''))

    if args.pair:
        result = score_pair(args.pair[0], args.pair[1], weights)
        print(json.dumps(result, indent=2))
        
    if args.spectrogram_diff:
        generate_spectrogram_diff(args.spectrogram_diff[0], args.spectrogram_diff[1], args.spectrogram_diff[2])

    if args.batch:
        results = []
        repo_root = os.getcwd()

        if args.ref_source == "ableton":
            # Ableton reference mode: score library outputs against Ableton refs
            ableton_manifest_path = args.ableton_manifest or os.path.join(
                repo_root, "optimize", "ableton", "ableton_manifest.json")
            if not os.path.exists(ableton_manifest_path):
                print(f"Error: Ableton manifest not found: {ableton_manifest_path}", file=sys.stderr)
                sys.exit(1)

            with open(ableton_manifest_path, 'r') as f:
                ableton_manifest = json.load(f)

            ableton_ref_dir = os.path.join(repo_root, "optimize", "ableton", "refs", "ableton")
            library_ref_dir = os.path.join(repo_root, "optimize", "ableton", "refs", "library")

            for entry in ableton_manifest:
                is_stereo = entry.get('stereo', False)
                ratio = entry['ratio']
                track_id = entry['track_id']
                target_bpm = entry['target_bpm']

                ref_path = entry.get('ref_path',
                    os.path.join(ableton_ref_dir, f"{track_id}_{target_bpm}bpm.wav"))

                # Score batch output
                batch_test = os.path.join(library_ref_dir, f"{track_id}_{target_bpm}bpm_batch.wav")
                if os.path.exists(ref_path) and os.path.exists(batch_test):
                    desc = entry.get('description', f"{track_id} @ {target_bpm} BPM")
                    print(f"Scoring [ableton] {desc}...", file=sys.stderr)
                    res = score_pair(ref_path, batch_test, weights, stereo=is_stereo)
                    res['description'] = desc
                    res['ratio'] = ratio
                    res['mode'] = 'batch'
                    res['ref_source'] = 'ableton'
                    results.append(res)
                else:
                    print(f"Warning: Skipping {track_id} batch (missing ref or test)", file=sys.stderr)

                # Score streaming output
                stream_test = os.path.join(library_ref_dir, f"{track_id}_{target_bpm}bpm_stream.wav")
                if os.path.exists(ref_path) and os.path.exists(stream_test):
                    desc = entry.get('description', f"{track_id} @ {target_bpm} BPM")
                    print(f"Scoring [ableton/streaming] {desc}...", file=sys.stderr)
                    res = score_pair(ref_path, stream_test, weights, stereo=is_stereo)
                    res['description'] = f"{desc} [streaming]"
                    res['ratio'] = ratio
                    res['mode'] = 'streaming'
                    res['ref_source'] = 'ableton'
                    results.append(res)
                else:
                    print(f"Warning: Skipping {track_id} streaming (missing ref or test)", file=sys.stderr)

        else:
            # Default rubberband reference mode
            if not os.path.exists(args.manifest):
                print(f"Error: Manifest {args.manifest} not found.")
                sys.exit(1)

            with open(args.manifest, 'r') as f:
                manifest = json.load(f)

            # Score batch outputs
            for item in manifest:
                ratio = item['ratio']
                is_stereo = item.get('stereo', False)
                source_base = os.path.basename(item['source']).replace('.wav', '')
                ref_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}.wav")
                test_path = os.path.join(repo_root, "optimize/outputs", f"{source_base}_test_{ratio}.wav")

                if not os.path.exists(ref_path) or not os.path.exists(test_path):
                    print(f"Warning: Skipping {item['description']}, missing files.")
                    continue

                print(f"Scoring {item['description']}...", file=sys.stderr)
                res = score_pair(ref_path, test_path, weights, stereo=is_stereo)
                res['description'] = item['description']
                res['ratio'] = ratio
                res['mode'] = 'batch'
                results.append(res)

            # Score streaming outputs
            for item in manifest:
                ratio = item['ratio']
                is_stereo = item.get('stereo', False)
                source_base = os.path.basename(item['source']).replace('.wav', '')
                ref_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}.wav")
                test_path = os.path.join(repo_root, "optimize/outputs", f"{source_base}_stream_{ratio}.wav")

                if not os.path.exists(ref_path) or not os.path.exists(test_path):
                    print(f"Warning: Skipping [streaming] {item['description']}, missing files.")
                    continue

                print(f"Scoring [streaming] {item['description']}...", file=sys.stderr)
                res = score_pair(ref_path, test_path, weights, stereo=is_stereo)
                res['description'] = f"{item['description']} [streaming]"
                res['ratio'] = ratio
                res['mode'] = 'streaming'
                results.append(res)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.batch, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        avg_score = np.mean([r['total_score'] for r in results])
        batch_scores = [r['total_score'] for r in results if r.get('mode') == 'batch']
        stream_scores = [r['total_score'] for r in results if r.get('mode') == 'streaming']
        mono_scores = [r['total_score'] for r in results if not r.get('stereo', False)]
        stereo_scores = [r['total_score'] for r in results if r.get('stereo', False)]
        print(f"\nBatch completed. Average Score: {avg_score:.2f}", file=sys.stderr)
        if batch_scores:
            print(f"  Batch avg: {np.mean(batch_scores):.2f}", file=sys.stderr)
        if stream_scores:
            print(f"  Streaming avg: {np.mean(stream_scores):.2f}", file=sys.stderr)
        if mono_scores:
            print(f"  Mono avg: {np.mean(mono_scores):.2f}", file=sys.stderr)
        if stereo_scores:
            print(f"  Stereo avg: {np.mean(stereo_scores):.2f}", file=sys.stderr)
            # Show stereo metric averages
            swp_avg = np.mean([r['metrics'].get('stereo_width', 0) for r in results if r.get('stereo')])
            icc_avg = np.mean([r['metrics'].get('interchannel_correlation', 0) for r in results if r.get('stereo')])
            pp_avg = np.mean([r['metrics'].get('panning_consistency', 0) for r in results if r.get('stereo')])
            print(f"  Stereo metrics: SWP={swp_avg:.1f}, ICC={icc_avg:.1f}, PP={pp_avg:.1f}", file=sys.stderr)

    if args.baseline:
        if not os.path.exists(args.manifest):
            print(f"Error: Manifest {args.manifest} not found.", file=sys.stderr)
            sys.exit(1)

        with open(args.manifest, 'r') as f:
            manifest = json.load(f)

        repo_root = os.getcwd()
        results = []

        for item in manifest:
            ratio = item['ratio']
            is_stereo = item.get('stereo', False)
            source_base = os.path.basename(item['source']).replace('.wav', '')
            hq_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}.wav")
            med_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}_med.wav")

            if not os.path.exists(hq_path) or not os.path.exists(med_path):
                continue

            print(f"Baseline scoring {item['description']}...", file=sys.stderr)
            res = score_pair(hq_path, med_path, weights, stereo=is_stereo)
            res['description'] = item['description']
            res['ratio'] = ratio
            results.append(res)

        if not results:
            print("No baseline pairs found (need *_med.wav files from generate_references.sh).", file=sys.stderr)
            sys.exit(1)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.baseline, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        avg = np.mean([r['total_score'] for r in results])
        sc_avg = np.mean([r['metrics']['spectral_convergence'] for r in results])
        lsd_avg = np.mean([r['metrics']['log_spectral_distance'] for r in results])
        raw_lsd_avg = np.mean([r['raw']['lsd'] for r in results])
        print(f"\nBaseline (rubberband HQ vs medium): avg={avg:.2f}, SC={sc_avg:.1f}, LSD={lsd_avg:.1f} (raw {raw_lsd_avg:.1f} dB)", file=sys.stderr)

if __name__ == "__main__":
    main()
