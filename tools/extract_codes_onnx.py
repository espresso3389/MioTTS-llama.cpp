#!/usr/bin/env python3
"""Extract speech codes from audio using the ONNX content encoder model.

This uses miocodec_content_encoder.onnx (no PyTorch/MioCodec package required,
only onnxruntime + soundfile + numpy).

Usage:
    python tools/extract_codes_onnx.py \
        --encoder miocodec_content_encoder.onnx \
        --audio input.wav \
        --output codes.txt

The output file contains <|s_N|> formatted tokens that can be fed directly
to miotts via -p for decode-only mode, or piped via -p -.
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

# MioCodec-25Hz-24kHz expects 24kHz mono audio
TARGET_SAMPLE_RATE = 24000


def load_audio_wav(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load audio file as float32 mono at target sample rate."""
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
        print(f"  Resampled: {sr} Hz -> {target_sr} Hz", file=sys.stderr)

    # Normalize to [-1, 1]
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Extract speech codes from audio using ONNX content encoder"
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="Path to miocodec_content_encoder.onnx",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio file (WAV/FLAC/OGG)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for codes text file (default: print to stdout)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.encoder):
        print(f"Error: encoder not found: {args.encoder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.audio):
        print(f"Error: audio not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Load audio
    print(f"Loading audio: {args.audio}", file=sys.stderr)
    waveform = load_audio_wav(args.audio)
    print(
        f"  Samples: {len(waveform)}, Duration: {len(waveform) / TARGET_SAMPLE_RATE:.2f}s",
        file=sys.stderr,
    )

    # Load ONNX model
    print(f"Loading encoder: {args.encoder}", file=sys.stderr)
    sess = ort.InferenceSession(args.encoder)

    # Run inference
    print("Extracting speech codes...", file=sys.stderr)
    tokens = sess.run(None, {"waveform": waveform})[0]
    flat = tokens.flatten()
    print(
        f"  Tokens: {len(flat)}, range [{flat.min()}, {flat.max()}]",
        file=sys.stderr,
    )

    # Format as <|s_N|> tokens
    codes_str = "".join(f"<|s_{int(t)}|>" for t in flat)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(codes_str)
        print(f"Saved: {args.output} ({len(flat)} tokens)", file=sys.stderr)
    else:
        print(codes_str)
        print(f"Printed {len(flat)} tokens to stdout", file=sys.stderr)


if __name__ == "__main__":
    main()
