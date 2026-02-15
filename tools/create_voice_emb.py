#!/usr/bin/env python3
"""Create a .emb.gguf voice embedding file from an audio sample.

Usage:
    python3 tools/create_voice_emb.py INPUT_AUDIO OUTPUT.emb.gguf [--model REPO_ID] [--name LABEL]

The output file can be used with the miotts C++ pipeline via --voice.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import gguf
import numpy as np
import soundfile as sf
import torch
import torchaudio
from miocodec import MioCodecModel


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio from any format. Uses soundfile for WAV/FLAC/OGG, ffmpeg for others (webm, mp3, etc.)."""
    # Try soundfile first (handles WAV, FLAC, OGG natively)
    try:
        data, sr = sf.read(path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        else:
            waveform = waveform.T  # (channels, samples)
        return waveform, sr
    except Exception:
        pass
    # Fallback: convert to WAV via ffmpeg (handles webm, mp3, aac, etc.)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-f", "wav", tmp_path],
            capture_output=True, check=True,
        )
        data, sr = sf.read(tmp_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        return waveform, sr
    except FileNotFoundError:
        print("Error: ffmpeg is required for this audio format. Please install ffmpeg.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg failed to convert audio: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract a 128-dim global voice embedding from audio and save as .emb.gguf"
    )
    parser.add_argument("input_audio", help="Path to input audio file (WAV/MP3/FLAC/etc.)")
    parser.add_argument("output_gguf", help="Output path for .emb.gguf file")
    parser.add_argument(
        "--model",
        default="Aratako/MioCodec-25Hz-44.1kHz-v2",
        help="HuggingFace model ID (default: Aratako/MioCodec-25Hz-44.1kHz-v2)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Descriptive name for embedding metadata (default: derived from input filename)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_audio):
        print(f"Error: input file not found: {args.input_audio}", file=sys.stderr)
        sys.exit(1)

    # Derive name from input filename if not provided
    name = args.name or os.path.splitext(os.path.basename(args.input_audio))[0]

    # Load audio
    print(f"Loading audio: {args.input_audio}")
    waveform, sr = load_audio(args.input_audio)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Load model to get its sample rate
    print(f"Loading MioCodec model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MioCodecModel.from_pretrained(args.model).to(device).eval()
    target_sr = model.config.sample_rate

    # Resample if needed
    if sr != target_sr:
        print(f"Resampling: {sr} Hz -> {target_sr} Hz")
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Normalize to [-1, 1]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # Extract global embedding (shape: (128,))
    print("Extracting global voice embedding...")
    waveform = waveform.squeeze(0).to(device)  # (samples,)
    with torch.inference_mode():
        features = model.encode(waveform, return_content=False, return_global=True)
    embedding = features.global_embedding.cpu().numpy().astype(np.float32)

    print(f"Embedding shape: {embedding.shape}, range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Write .emb.gguf
    print(f"Writing: {args.output_gguf}")
    os.makedirs(os.path.dirname(args.output_gguf) or ".", exist_ok=True)

    writer = gguf.GGUFWriter(args.output_gguf, arch="mio-embedding")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("Mio global embedding")
    writer.add_uint32("mio.embedding.dim", 128)
    writer.add_tensor("mio.global_embedding", embedding)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify
    reader = gguf.GGUFReader(args.output_gguf)
    t = reader.tensors[0]
    file_size = os.path.getsize(args.output_gguf)
    print(f"Verified: {t.name}, shape={t.data.shape}, dtype={t.data.dtype}, file={file_size} bytes")
    print("Done!")


if __name__ == "__main__":
    main()
