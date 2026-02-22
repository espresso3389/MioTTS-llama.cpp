#!/usr/bin/env python3
"""Generate a voice embedding using the ONNX global encoder model.

This uses miocodec_global_encoder.onnx (no PyTorch/MioCodec package required,
only onnxruntime + soundfile + numpy).

Usage:
    python tools/generate_voice_emb.py \
        --encoder miocodec_global_encoder.onnx \
        --audio reference.wav \
        --output voice.emb.bin

The output .emb.bin file is a raw float32 binary (128 floats = 512 bytes)
that can be loaded by the miotts C++ pipeline via --voice.
"""

import argparse
import os
import struct
import sys

import numpy as np
import onnxruntime as ort

# MioCodec-25Hz-24kHz expects 24kHz mono audio
TARGET_SAMPLE_RATE = 24000


def load_audio_wav(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load audio file as float32 mono at target sample rate.

    Uses soundfile for WAV/FLAC/OGG. Resamples if needed via linear interpolation
    (scipy is used if available, otherwise basic linear interp).
    """
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(sr, target_sr)
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        except ImportError:
            # Fallback: linear interpolation
            ratio = target_sr / sr
            n_out = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        print(f"  Resampled: {sr} Hz -> {target_sr} Hz")

    # Normalize to [-1, 1]
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 128-dim voice embedding using ONNX global encoder"
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="Path to miocodec_global_encoder.onnx",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to reference audio file (WAV/FLAC/OGG)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for .emb.bin (raw float32 binary)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.encoder):
        print(f"Error: encoder not found: {args.encoder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.audio):
        print(f"Error: audio not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Load audio
    print(f"Loading audio: {args.audio}")
    waveform = load_audio_wav(args.audio)
    print(f"  Samples: {len(waveform)}, Duration: {len(waveform) / TARGET_SAMPLE_RATE:.2f}s")

    # Load ONNX model
    print(f"Loading encoder: {args.encoder}")
    sess = ort.InferenceSession(args.encoder)

    # Run inference
    print("Extracting voice embedding...")
    outputs = sess.run(None, {"waveform": waveform})
    embedding = outputs[0].flatten().astype(np.float32)
    print(f"  Embedding: dim={len(embedding)}, range=[{embedding.min():.4f}, {embedding.max():.4f}]")

    # Save as raw binary
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(embedding.tobytes())

    file_size = os.path.getsize(args.output)
    print(f"Saved: {args.output} ({file_size} bytes, {len(embedding)} floats)")
    print("Done!")


if __name__ == "__main__":
    main()
