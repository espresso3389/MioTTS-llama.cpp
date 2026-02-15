#!/usr/bin/env python3
"""Compare Python MioCodec output against C++ implementation.
Runs the same codes through both and compares intermediate results."""

import sys
import struct
import numpy as np
import torch

def read_wav(path):
    with open(path, 'rb') as f:
        riff = f.read(4)
        assert riff == b'RIFF'
        f.read(4)
        wave = f.read(4)
        assert wave == b'WAVE'
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                sr = struct.unpack('<I', fmt_data[4:8])[0]
                bits = struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                raw = f.read(chunk_size)
                if bits == 16:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    samples = np.frombuffer(raw, dtype=np.float32)
                return samples, sr
            else:
                f.read(chunk_size)
    return None, 0

def load_voice_emb_gguf(path):
    """Load voice embedding from .emb.gguf file."""
    import gguf
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        data = tensor.data
        return torch.tensor(data, dtype=torch.float32)
    return None

def main():
    from miocodec import MioCodecModel

    # Test codes (same as our C++ test)
    test_codes = [12287, 11619, 11774, 12223, 2490, 826, 2257, 1668, 1219, 2319,
                  9994, 12683, 12745, 4215, 12478, 8800, 8696, 375, 1406, 12396]

    cpp_wav_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_e2e_fixed.wav"
    voice_emb_path = sys.argv[2] if len(sys.argv) > 2 else "models/jp_female.emb.gguf"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "Aratako/MioCodec-25Hz-44.1kHz-v2"

    print(f"Loading MioCodec model: {model_name}")
    model = MioCodecModel.from_pretrained(model_name)
    model = model.eval()
    print(f"  Sample rate: {model.config.sample_rate}")
    print(f"  n_fft: {model.config.n_fft}")
    print(f"  hop_length: {model.config.hop_length}")
    if hasattr(model.config, 'wave_conv_upsample_factor'):
        print(f"  wave_conv_upsample_factor: {model.config.wave_conv_upsample_factor}")
    if hasattr(model.config, 'wave_upsample_factors'):
        print(f"  wave_upsample_factors: {model.config.wave_upsample_factors}")

    # Load voice embedding
    print(f"\nLoading voice embedding: {voice_emb_path}")
    try:
        global_emb = load_voice_emb_gguf(voice_emb_path)
    except Exception as e:
        print(f"  Failed to load GGUF emb ({e}), trying to use zeros")
        global_emb = torch.zeros(128)

    print(f"  Voice embedding shape: {global_emb.shape}")
    print(f"  Voice embedding first 5: {global_emb[:5].tolist()}")

    # Prepare inputs
    codes_tensor = torch.tensor([test_codes], dtype=torch.long)  # [1, T]
    global_emb_batch = global_emb.unsqueeze(0)  # [1, 128]
    T = len(test_codes)

    print(f"\nTest codes ({T} tokens): {test_codes[:10]}...")

    # === Run full decode ===
    with torch.no_grad():
        # Step 1: FSQ decode (token embedding lookup)
        content_emb = model.local_quantizer.decode(codes_tensor)
        print(f"\n--- Step 1: FSQ decode ---")
        print(f"  content_emb shape: {content_emb.shape}")
        print(f"  content_emb[0,0,:5]: {content_emb[0,0,:5].tolist()}")
        print(f"  content_emb stats: min={content_emb.min():.4f} max={content_emb.max():.4f} mean={content_emb.mean():.4f}")

        # Step 2: Wave prenet
        local_latent = model.wave_prenet(content_emb)
        print(f"\n--- Step 2: wave_prenet ---")
        print(f"  output shape: {local_latent.shape}")
        print(f"  output[0,0,:5]: {local_latent[0,0,:5].tolist()}")
        print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 3: Conv upsample
        if model.wave_conv_upsample is not None:
            local_latent = model.wave_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
            print(f"\n--- Step 3: wave_conv_upsample ---")
            print(f"  output shape: {local_latent.shape}")
            print(f"  output[0,0,:5]: {local_latent[0,0,:5].tolist()}")
            print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 3.5: Interpolation
        target_audio_length = model._calculate_original_audio_length(T)
        stft_length = model._calculate_target_stft_length(target_audio_length)
        print(f"\n--- Step 3.5: interpolation ---")
        print(f"  target_audio_length: {target_audio_length}")
        print(f"  stft_length: {stft_length}")
        print(f"  current length: {local_latent.shape[1]}")

        mode = getattr(model.config, 'wave_interpolation_mode', 'linear')
        local_latent = torch.nn.functional.interpolate(
            local_latent.transpose(1, 2), size=stft_length, mode=mode
        ).transpose(1, 2)
        print(f"  after interpolate shape: {local_latent.shape}")
        print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 4: Prior net
        local_latent_conv = model.wave_prior_net(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = local_latent_conv
        print(f"\n--- Step 4: wave_prior_net ---")
        print(f"  output shape: {local_latent.shape}")
        print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 5: Wave decoder (Transformer with AdaLN-Zero)
        local_latent = model.wave_decoder(local_latent, condition=global_emb_batch.unsqueeze(1))
        print(f"\n--- Step 5: wave_decoder ---")
        print(f"  output shape: {local_latent.shape}")
        print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 6: Post net
        local_latent = model.wave_post_net(local_latent.transpose(1, 2)).transpose(1, 2)
        print(f"\n--- Step 6: wave_post_net ---")
        print(f"  output shape: {local_latent.shape}")
        print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 7: Upsampler
        if model.wave_upsampler is not None:
            local_latent = model.wave_upsampler(local_latent.transpose(1, 2))
            print(f"\n--- Step 7: wave_upsampler ---")
            print(f"  output shape: {local_latent.shape}")
            print(f"  stats: min={local_latent.min():.4f} max={local_latent.max():.4f} mean={local_latent.mean():.4f}")

        # Step 8: iSTFT head
        waveform = model.istft_head(local_latent)
        print(f"\n--- Step 8: istft_head ---")
        print(f"  waveform shape: {waveform.shape}")
        print(f"  stats: min={waveform.min():.4f} max={waveform.max():.4f} mean={waveform.mean():.4f}")
        print(f"  rms: {waveform.pow(2).mean().sqrt():.4f}")

    # Save Python output as WAV
    py_wav_path = "/tmp/test_python_ref.wav"
    audio_np = waveform[0].cpu().numpy()
    import soundfile as sf
    sf.write(py_wav_path, audio_np, model.config.sample_rate)
    print(f"\nSaved Python reference WAV: {py_wav_path}")

    # Compare with C++ output
    if cpp_wav_path:
        try:
            cpp_samples, cpp_sr = read_wav(cpp_wav_path)
            if cpp_samples is not None:
                print(f"\n=== Comparison with C++ output: {cpp_wav_path} ===")
                print(f"  C++ samples: {len(cpp_samples)}, sr={cpp_sr}")
                print(f"  Python samples: {len(audio_np)}, sr={model.config.sample_rate}")
                print(f"  C++ RMS: {np.sqrt(np.mean(cpp_samples**2)):.4f}")
                print(f"  Python RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")

                # Correlation
                min_len = min(len(cpp_samples), len(audio_np))
                if min_len > 0:
                    corr = np.corrcoef(cpp_samples[:min_len], audio_np[:min_len])[0, 1]
                    print(f"  Correlation: {corr:.4f}")
                    mse = np.mean((cpp_samples[:min_len] - audio_np[:min_len])**2)
                    print(f"  MSE: {mse:.6f}")
        except Exception as e:
            print(f"  Failed to read C++ WAV: {e}")

    # Analyze Python output
    print(f"\n=== Python output analysis ===")
    from scipy import signal
    freqs, psd = signal.welch(audio_np, model.config.sample_rate, nperseg=min(4096, len(audio_np)))
    centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    zc = np.sum(np.diff(np.sign(audio_np)) != 0) / (len(audio_np) / model.config.sample_rate)
    print(f"  Spectral centroid: {centroid:.0f} Hz")
    print(f"  Zero crossing rate: {zc:.0f} Hz")
    print(f"  Peak: {np.max(np.abs(audio_np)):.4f}")

if __name__ == "__main__":
    main()
