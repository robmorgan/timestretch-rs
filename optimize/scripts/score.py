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

def score_pair(ref_path, test_path, weights):
    ref, sr_ref = librosa.load(ref_path, sr=None)
    test, sr_test = librosa.load(test_path, sr=sr_ref) # Resample test to match ref if needed
    
    # Trim to match
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    sc = calculate_spectral_convergence(ref, test)
    lsd = calculate_log_spectral_distance(ref, test)
    mfcc_dist = calculate_mfcc_distance(ref, test, sr_ref)
    tp = calculate_transient_preservation(ref, test, sr_ref)
    
    # Map raw metrics to 0-100 scores
    # SC: 0 is perfect, >1 is bad. 1 - SC is roughly 0-1.
    sc_score = max(0, 100 * (1.0 - sc))
    # LSD: 0 is perfect, 10-20 is quite different. 
    lsd_score = max(0, 100 * (1.0 - lsd/15.0))
    # MFCC dist: 0 is perfect, 1 is total difference.
    mfcc_score = max(0, 100 * (1.0 - mfcc_dist))
    # TP: 1.0 is perfect.
    tp_score = 100 * tp
    
    total_score = (
        sc_score * weights['spectral_convergence'] +
        lsd_score * weights['log_spectral_distance'] +
        mfcc_score * weights['mfcc_distance'] +
        tp_score * weights['transient_preservation']
    )
    
    return {
        "total_score": total_score,
        "metrics": {
            "spectral_convergence": sc_score,
            "log_spectral_distance": lsd_score,
            "mfcc_distance": mfcc_score,
            "transient_preservation": tp_score
        },
        "raw": {
            "sc": float(sc),
            "lsd": float(lsd),
            "mfcc_dist": float(mfcc_dist),
            "tp": float(tp)
        }
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
    parser.add_argument("--config", default="optimize/config.toml", help="Path to config.toml")
    
    args = parser.parse_args()
    
    # Load weights from config
    weights = {
        "spectral_convergence": 0.30,
        "log_spectral_distance": 0.25,
        "mfcc_distance": 0.20,
        "transient_preservation": 0.25
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
        if not os.path.exists(args.manifest):
            print(f"Error: Manifest {args.manifest} not found.")
            sys.exit(1)

        with open(args.manifest, 'r') as f:
            manifest = json.load(f)

        results = []
        repo_root = os.getcwd()

        # Score batch outputs
        for item in manifest:
            ratio = item['ratio']
            source_base = os.path.basename(item['source']).replace('.wav', '')
            ref_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}.wav")
            test_path = os.path.join(repo_root, "optimize/outputs", f"{source_base}_test_{ratio}.wav")

            if not os.path.exists(ref_path) or not os.path.exists(test_path):
                print(f"Warning: Skipping {item['description']}, missing files.")
                continue

            print(f"Scoring {item['description']}...", file=sys.stderr)
            res = score_pair(ref_path, test_path, weights)
            res['description'] = item['description']
            res['ratio'] = ratio
            res['mode'] = 'batch'
            results.append(res)

        # Score streaming outputs
        for item in manifest:
            ratio = item['ratio']
            source_base = os.path.basename(item['source']).replace('.wav', '')
            ref_path = os.path.join(repo_root, "optimize/references", f"{source_base}_ref_{ratio}.wav")
            test_path = os.path.join(repo_root, "optimize/outputs", f"{source_base}_stream_{ratio}.wav")

            if not os.path.exists(ref_path) or not os.path.exists(test_path):
                print(f"Warning: Skipping [streaming] {item['description']}, missing files.")
                continue

            print(f"Scoring [streaming] {item['description']}...", file=sys.stderr)
            res = score_pair(ref_path, test_path, weights)
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
        print(f"\nBatch completed. Average Score: {avg_score:.2f}", file=sys.stderr)
        if batch_scores:
            print(f"  Batch avg: {np.mean(batch_scores):.2f}", file=sys.stderr)
        if stream_scores:
            print(f"  Streaming avg: {np.mean(stream_scores):.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
