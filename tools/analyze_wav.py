#!/usr/bin/env python3
"""Analyze WAV file quality for TTS debugging."""
import sys
import struct
import numpy as np
from scipy import signal

def read_wav(path):
    with open(path, 'rb') as f:
        riff = f.read(4)
        assert riff == b'RIFF', f"Not a RIFF file: {riff}"
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b'WAVE'

        fmt_found = False
        data = None
        sr = 0
        bits = 0
        channels = 0

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                audio_fmt = struct.unpack('<H', fmt_data[0:2])[0]
                channels = struct.unpack('<H', fmt_data[2:4])[0]
                sr = struct.unpack('<I', fmt_data[4:8])[0]
                bits = struct.unpack('<H', fmt_data[14:16])[0]
                fmt_found = True
            elif chunk_id == b'data':
                raw = f.read(chunk_size)
                if bits == 16:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif bits == 32:
                    samples = np.frombuffer(raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported bit depth: {bits}")
                if channels > 1:
                    samples = samples.reshape(-1, channels)[:, 0]  # mono
                data = samples
            else:
                f.read(chunk_size)

    return data, sr

def analyze(path):
    print(f"=== Analyzing: {path} ===\n")

    samples, sr = read_wav(path)
    duration = len(samples) / sr

    print(f"Format: {sr} Hz, {len(samples)} samples, {duration:.3f} sec")

    # Basic stats
    peak = np.max(np.abs(samples))
    rms = np.sqrt(np.mean(samples**2))
    print(f"Peak: {peak:.4f}, RMS: {rms:.4f}, Crest factor: {peak/rms:.1f} dB" if rms > 0 else f"Peak: {peak:.4f}, RMS: 0 (SILENT)")

    # Zero crossings
    zc = np.sum(np.diff(np.sign(samples)) != 0) / duration
    print(f"Zero crossing rate: {zc:.0f} Hz")

    # Check for silence/clipping
    silent_pct = np.mean(np.abs(samples) < 1e-6) * 100
    clipped_pct = np.mean(np.abs(samples) > 0.99) * 100
    print(f"Silent samples (<-120dB): {silent_pct:.1f}%")
    print(f"Clipped samples (>-0.1dB): {clipped_pct:.1f}%")

    # DC offset
    dc = np.mean(samples)
    print(f"DC offset: {dc:.6f}")

    # Spectral analysis
    print(f"\n--- Spectral Analysis ---")
    freqs, psd = signal.welch(samples, sr, nperseg=min(4096, len(samples)))

    # Spectral centroid
    centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    print(f"Spectral centroid: {centroid:.0f} Hz")

    # Energy in frequency bands
    bands = [
        ("Sub-bass (0-100 Hz)", 0, 100),
        ("Bass (100-300 Hz)", 100, 300),
        ("Low-mid (300-1000 Hz)", 300, 1000),
        ("Mid (1-3 kHz)", 1000, 3000),
        ("High-mid (3-6 kHz)", 3000, 6000),
        ("High (6-12 kHz)", 6000, 12000),
        ("Air (12+ kHz)", 12000, sr/2),
    ]
    total_energy = np.sum(psd)
    print(f"\nFrequency band energy distribution:")
    for name, lo, hi in bands:
        if lo >= sr/2:
            break
        mask = (freqs >= lo) & (freqs < min(hi, sr/2))
        band_energy = np.sum(psd[mask]) / total_energy * 100 if total_energy > 0 else 0
        bar = '#' * int(band_energy)
        print(f"  {name:25s}: {band_energy:5.1f}% {bar}")

    # Check for periodicity (speech has pitch ~80-400 Hz)
    print(f"\n--- Periodicity (Pitch) Analysis ---")
    # Autocorrelation on a segment
    seg_len = min(sr // 2, len(samples))  # 0.5 sec
    seg = samples[:seg_len]
    if len(seg) > 2000:
        acf = np.correlate(seg[:2000], seg[:2000], mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]  # normalize

        # Find first peak after lag corresponding to 400Hz
        min_lag = int(sr / 400)  # 400 Hz max pitch
        max_lag = int(sr / 60)   # 60 Hz min pitch
        if max_lag < len(acf):
            search = acf[min_lag:max_lag]
            if len(search) > 0:
                peak_idx = np.argmax(search) + min_lag
                peak_val = acf[peak_idx]
                f0 = sr / peak_idx
                print(f"Estimated F0: {f0:.1f} Hz (autocorrelation peak: {peak_val:.3f})")
                if peak_val > 0.3:
                    print(f"  → Periodic signal detected (likely voiced speech)")
                elif peak_val > 0.1:
                    print(f"  → Weak periodicity (noisy or unvoiced)")
                else:
                    print(f"  → No clear periodicity (noise-like)")
            else:
                print("  Segment too short for pitch analysis")
        else:
            print("  Segment too short for pitch analysis")

    # Temporal analysis - check for structure (speech has alternating voiced/unvoiced)
    print(f"\n--- Temporal Structure ---")
    frame_len = int(0.02 * sr)  # 20ms frames
    n_frames = len(samples) // frame_len
    if n_frames > 0:
        frame_rms = np.array([
            np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len]**2))
            for i in range(n_frames)
        ])
        rms_std = np.std(frame_rms) / (np.mean(frame_rms) + 1e-10)
        print(f"Frame RMS variation (CoV): {rms_std:.3f}")
        if rms_std > 0.5:
            print(f"  → Good dynamic range (speech-like)")
        elif rms_std > 0.2:
            print(f"  → Moderate variation")
        else:
            print(f"  → Very uniform (noise/drone-like, NOT speech-like)")

        # Check for envelope shape
        voiced_frames = np.sum(frame_rms > np.mean(frame_rms) * 0.3)
        print(f"Active frames (>30% mean): {voiced_frames}/{n_frames} ({voiced_frames/n_frames*100:.0f}%)")

    # Overall assessment
    print(f"\n=== ASSESSMENT ===")
    issues = []
    if rms < 0.01:
        issues.append("Very low RMS - nearly silent")
    if centroid < 200:
        issues.append("Spectral centroid too low - likely noise/rumble, not speech")
    elif centroid > 6000:
        issues.append("Spectral centroid too high - likely noise/hiss, not speech")
    if zc < 500:
        issues.append("Zero crossing rate very low - likely low-freq noise")
    elif zc > 10000:
        issues.append("Zero crossing rate very high - likely noise/distortion")
    if silent_pct > 90:
        issues.append("Mostly silent")
    if clipped_pct > 5:
        issues.append(f"Significant clipping ({clipped_pct:.1f}%)")
    if dc > 0.1:
        issues.append(f"Large DC offset ({dc:.4f})")
    if n_frames > 0 and rms_std < 0.2:
        issues.append("Too uniform - speech should have dynamic variation")

    if not issues:
        print("No obvious issues detected. Check if it sounds like speech.")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    print()

if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyze(path)
