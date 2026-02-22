#!/usr/bin/env python3
"""Export MioCodec to ONNX (decoder + global encoder + content encoder).

Downloads MioCodec-25Hz-24kHz from HuggingFace and exports three ONNX models:
  1. miocodec_decoder.onnx         - speech codes + voice -> waveform (required)
  2. miocodec_global_encoder.onnx  - audio -> voice embedding (for voice cloning)
  3. miocodec_content_encoder.onnx - audio -> speech codes (for testing)

Usage:
    pip install miocodec torch onnx onnxruntime numpy
    python tools/export_miocodec_onnx.py --output-dir models/

Key ONNX-compatibility changes applied automatically:
- RoPE: complex-number ops replaced with real-valued cos/sin
- ISTFT: torch.fft.irfft replaced with precomputed iDFT matrix multiply
- Overlap-add: F.fold replaced with ConvTranspose1d
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ONNX-compatible RoPE (real-valued, no complex numbers)
# ============================================================================

def apply_rotary_emb_real(x, cos, sin):
    """Apply rotary embeddings using real ops. x: (B, T, H, D)"""
    x_r = x[..., 0::2]
    x_i = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    y_r = x_r * cos - x_i * sin
    y_i = x_r * sin + x_i * cos
    return torch.stack([y_r, y_i], dim=-1).flatten(-2).type_as(x)


def precompute_rope_real(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


# ============================================================================
# ONNX-compatible Transformer components
# ============================================================================

class AttentionONNX(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = orig.wq, orig.wk, orig.wv, orig.wo
        self.n_heads = orig.n_heads
        self.head_dim = orig.head_dim
        self.scale = orig.scale
        self.use_local_attention = orig.use_local_attention
        if self.use_local_attention:
            self.window_per_side = orig.window_per_side
        self.causal = orig.causal

    def forward(self, x, rope_cos, rope_sin):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        if rope_cos is not None:
            xq = apply_rotary_emb_real(xq, rope_cos[:seqlen], rope_sin[:seqlen])
            xk = apply_rotary_emb_real(xk, rope_cos[:seqlen], rope_sin[:seqlen])

        attn_mask = None
        if self.use_local_attention:
            attn_mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=x.device)
            if self.causal:
                attn_mask = torch.tril(attn_mask)
            attn_mask = torch.triu(attn_mask, diagonal=-self.window_per_side)
            attn_mask = torch.tril(attn_mask, diagonal=self.window_per_side)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)

        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0, scale=self.scale,
        ).transpose(1, 2)
        return self.wo(output.contiguous().view(bsz, seqlen, -1))


class TransformerBlockONNX(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.attention = AttentionONNX(orig.attention)
        self.feed_forward = orig.feed_forward
        self.attention_norm = orig.attention_norm
        self.ffn_norm = orig.ffn_norm
        self.use_adaln_zero = orig.use_adaln_zero

    def forward(self, x, rope_cos, rope_sin, condition=None):
        if self.use_adaln_zero:
            normed, gate = self.attention_norm(x, condition=condition)
            h = x + gate * self.attention(normed, rope_cos, rope_sin)
            normed, gate = self.ffn_norm(h, condition=condition)
            return h + gate * self.feed_forward(normed)
        else:
            h = x + self.attention(self.attention_norm(x), rope_cos, rope_sin)
            return h + self.feed_forward(self.ffn_norm(h))


class TransformerONNX(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.input_proj = orig.input_proj
        self.output_proj = orig.output_proj
        self.norm = orig.norm
        self.use_adaln_zero = orig.use_adaln_zero
        self.layers = nn.ModuleList([TransformerBlockONNX(l) for l in orig.layers])

        dim, n_heads = orig.dim, orig.n_heads
        if orig.freqs_cis is not None:
            cos, sin = precompute_rope_real(dim // n_heads, orig.freqs_cis.shape[0], orig.rope_theta)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(self, x, condition=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, condition)
        if self.use_adaln_zero:
            x, _ = self.norm(x, condition=condition)
        else:
            x = self.norm(x)
        return self.output_proj(x)


# ============================================================================
# ONNX-compatible ISTFT head
# ============================================================================

class ISTFTHeadONNX(nn.Module):
    """ONNX-compatible ISTFT head.

    - iDFT via precomputed cos/sin matrix multiply (no torch.fft)
    - Overlap-add via ConvTranspose1d (no F.fold / col2im)
    """

    def __init__(self, orig_head):
        super().__init__()
        self.out = orig_head.out
        n_fft = orig_head.istft.n_fft
        hop_length = orig_head.istft.hop_length
        win_length = orig_head.istft.win_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding_mode = orig_head.istft.padding
        window = orig_head.istft.window.clone()
        self.register_buffer("window", window)

        # iDFT basis matrices
        freq_bins = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(freq_bins, dtype=torch.float32)
        angles = 2 * math.pi * torch.outer(n, k) / n_fft
        cos_basis = torch.cos(angles)
        sin_basis = torch.sin(angles)
        # Conjugate symmetry: double non-DC/Nyquist bins
        scale = torch.ones(freq_bins)
        scale[1:-1] = 2.0
        self.register_buffer("cos_basis", cos_basis * scale / n_fft)
        self.register_buffer("sin_basis", sin_basis * scale / n_fft)

        # Overlap-add kernel
        ola_weight = torch.eye(win_length).unsqueeze(1)
        self.register_buffer("ola_weight", ola_weight)

        # Window envelope kernel
        env_weight = torch.diag(window.square()).unsqueeze(1)
        self.register_buffer("env_weight", env_weight)

    def forward(self, x):
        """x: (B, L, H) -> audio: (B, samples)"""
        x = self.out(x)
        x = x.transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)

        mag = torch.clamp(torch.exp(mag), max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        ifft = self.cos_basis @ real - self.sin_basis @ imag
        ifft = ifft * self.window.unsqueeze(-1)

        y = F.conv_transpose1d(ifft, self.ola_weight, stride=self.hop_length).squeeze(1)

        T = ifft.shape[2]
        ones = torch.ones(1, self.win_length, T, device=x.device, dtype=x.dtype)
        envelope = F.conv_transpose1d(ones, self.env_weight, stride=self.hop_length).squeeze(1)
        y = y / (envelope + 1e-11)

        if self.padding_mode == "same":
            pad = (self.win_length - self.hop_length) // 2
            y = y[:, pad:-pad]
        elif self.padding_mode == "center":
            pad = self.n_fft // 2
            y = y[:, pad:-pad]

        return y


# ============================================================================
# Decoder wrapper
# ============================================================================

class MioCodecDecoderONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.config.use_wave_decoder
        self.local_quantizer = model.local_quantizer
        self.wave_prenet = TransformerONNX(model.wave_prenet)
        self.wave_decoder = TransformerONNX(model.wave_decoder)
        self.wave_conv_upsample = model.wave_conv_upsample
        self.wave_prior_net = model.wave_prior_net
        self.wave_post_net = model.wave_post_net
        self.wave_upsampler = model.wave_upsampler
        self.istft_head = ISTFTHeadONNX(model.istft_head)
        self.wave_interpolation_mode = model.config.wave_interpolation_mode

    def forward(self, content_token_indices, global_embedding, stft_length):
        content = self.local_quantizer.decode(content_token_indices.unsqueeze(0))
        global_emb = global_embedding.unsqueeze(0)
        x = self.wave_prenet(content)
        if self.wave_conv_upsample is not None:
            x = self.wave_conv_upsample(x.transpose(1, 2)).transpose(1, 2)
        x = F.interpolate(
            x.transpose(1, 2), size=stft_length, mode=self.wave_interpolation_mode
        ).transpose(1, 2)
        x = self.wave_prior_net(x.transpose(1, 2)).transpose(1, 2)
        x = self.wave_decoder(x, condition=global_emb.unsqueeze(1))
        x = self.wave_post_net(x.transpose(1, 2)).transpose(1, 2)
        if self.wave_upsampler is not None:
            x = self.wave_upsampler(x.transpose(1, 2))
        waveform = self.istft_head(x)
        return waveform.squeeze(0)


# ============================================================================
# Encoder wrappers
# ============================================================================

class MioCodecGlobalEncoderONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ssl = model.ssl_feature_extractor
        self.global_encoder = model.global_encoder
        self.global_ssl_layers = model.global_ssl_layers

    def forward(self, waveform):
        wav = waveform.unsqueeze(0)
        if self.ssl.resampler is not None:
            wav = self.ssl.resampler(wav)
        features, _ = self.ssl.model.extract_features(wav, num_layers=max(self.global_ssl_layers))
        if len(self.global_ssl_layers) > 1:
            selected = [features[i - 1] for i in self.global_ssl_layers]
            global_feats = torch.stack(selected, dim=0).mean(dim=0)
        else:
            global_feats = features[self.global_ssl_layers[0] - 1]
        return self.global_encoder(global_feats).squeeze(0)


class MioCodecContentEncoderONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ssl = model.ssl_feature_extractor
        self.local_encoder = TransformerONNX(model.local_encoder)
        self.local_quantizer = model.local_quantizer
        self.local_ssl_layers = model.local_ssl_layers
        self.downsample_factor = model.config.downsample_factor
        self.use_conv_downsample = model.config.use_conv_downsample
        self.normalize_ssl = model.config.normalize_ssl_features
        if self.use_conv_downsample:
            self.conv_downsample = model.conv_downsample
        self.sample_rate = model.config.sample_rate

    def forward(self, waveform):
        wav = waveform.unsqueeze(0)
        audio_length = waveform.size(0)
        num_samples_after = audio_length / self.sample_rate * self.ssl.ssl_sample_rate
        expected_len = math.ceil(num_samples_after / self.ssl.hop_size)
        num_required_after = self.ssl.get_minimum_input_length(expected_len)
        num_required = num_required_after / self.ssl.ssl_sample_rate * self.sample_rate
        padding = math.ceil((num_required - audio_length) / 2)
        if padding > 0:
            wav = F.pad(wav, (padding, padding), mode="constant")
        if self.ssl.resampler is not None:
            wav = self.ssl.resampler(wav)
        features, _ = self.ssl.model.extract_features(wav, num_layers=max(self.local_ssl_layers))
        if len(self.local_ssl_layers) > 1:
            selected = [features[i - 1] for i in self.local_ssl_layers]
            local_feats = torch.stack(selected, dim=0).mean(dim=0)
        else:
            local_feats = features[self.local_ssl_layers[0] - 1]
        if self.normalize_ssl:
            mean = torch.mean(local_feats, dim=1, keepdim=True)
            std = torch.std(local_feats, dim=1, keepdim=True)
            local_feats = (local_feats - mean) / (std + 1e-8)
        local_encoded = self.local_encoder(local_feats)
        if self.downsample_factor > 1:
            if self.use_conv_downsample:
                local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
            else:
                local_encoded = F.avg_pool1d(
                    local_encoded.transpose(1, 2),
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor,
                ).transpose(1, 2)
        _, indices = self.local_quantizer.encode(local_encoded)
        return indices.squeeze(0)


# ============================================================================
# Utilities
# ============================================================================

def calculate_stft_length(token_length, downsample_factor, ssl_hop_size,
                          ssl_sample_rate, sample_rate, hop_length, istft_padding):
    feature_length = token_length * downsample_factor
    num_samples_ssl = (feature_length - 1) * ssl_hop_size + 400
    audio_length = math.ceil(num_samples_ssl / ssl_sample_rate * sample_rate)
    if istft_padding == "same":
        return audio_length // hop_length
    else:
        return audio_length // hop_length + 1


def export_and_verify(module, dummy_inputs, output_path, input_names, output_names,
                      dynamic_axes, opset):
    """Export module to ONNX, validate, and run ORT inference test."""
    import onnx
    import onnxruntime as ort

    print(f"  Exporting (opset {opset})...")
    torch.onnx.export(
        module, dummy_inputs, output_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    print("  Validating...")
    onnx.checker.check_model(onnx.load(output_path))
    print("  OK")

    print("  ONNX Runtime test...")
    sess = ort.InferenceSession(output_path)
    feeds = {}
    for name, inp in zip(input_names, dummy_inputs if isinstance(dummy_inputs, tuple) else (dummy_inputs,)):
        if isinstance(inp, torch.Tensor):
            feeds[name] = inp.numpy()
        elif isinstance(inp, int):
            feeds[name] = np.array(inp, dtype=np.int64)
        else:
            feeds[name] = np.array(inp)
    ort_out = sess.run(None, feeds)[0]
    print(f"  ORT output: {ort_out.shape}")
    return ort_out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export MioCodec to ONNX (decoder + global encoder + content encoder)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Output directory for ONNX files (default: current directory)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--skip-encoders", action="store_true",
        help="Only export decoder (skip global/content encoders)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    decoder_path = os.path.join(args.output_dir, "miocodec_decoder.onnx")
    global_path = os.path.join(args.output_dir, "miocodec_global_encoder.onnx")
    content_path = os.path.join(args.output_dir, "miocodec_content_encoder.onnx")

    # ---- Load model once ----
    print("Loading MioCodec-25Hz-24kHz from HuggingFace...")
    from miocodec import MioCodecModel
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz")
    model.eval().cpu()
    print(f"  sample_rate={model.config.sample_rate}, n_fft={model.config.n_fft}, "
          f"hop={model.config.hop_length}, wave_decoder={model.config.use_wave_decoder}")

    # ---- 1. Decoder ----
    print("\n" + "=" * 60)
    print("1/3  Decoder: tokens + voice -> waveform")
    print("=" * 60)

    decoder = MioCodecDecoderONNX(model)
    decoder.eval()

    seq_len = 50
    codebook_size = model.local_quantizer.all_codebook_size
    embed_dim = model.global_encoder.output_dim
    dummy_tokens = torch.randint(0, codebook_size, (seq_len,), dtype=torch.long)
    dummy_global = torch.randn(embed_dim, dtype=torch.float32)
    stft_len = calculate_stft_length(
        seq_len, model.config.downsample_factor,
        model.ssl_feature_extractor.hop_size, model.ssl_feature_extractor.ssl_sample_rate,
        model.config.sample_rate, model.config.hop_length, model.config.istft_padding,
    )

    print(f"  Test: {seq_len} tokens, embed_dim={embed_dim}, stft_len={stft_len}")
    with torch.no_grad():
        pt_out = decoder(dummy_tokens, dummy_global, stft_len)
    print(f"  PyTorch output: {pt_out.shape}")

    ort_out = export_and_verify(
        decoder,
        (dummy_tokens, dummy_global, stft_len),
        decoder_path,
        input_names=["content_token_indices", "global_embedding", "stft_length"],
        output_names=["waveform"],
        dynamic_axes={"content_token_indices": {0: "seq_len"}, "waveform": {0: "samples"}},
        opset=args.opset,
    )
    print(f"  Max diff (PyTorch vs ORT): {np.max(np.abs(pt_out.numpy() - ort_out)):.6e}")

    if args.skip_encoders:
        print(f"\nDone! Decoder exported to {decoder_path}")
        return

    # ---- 2. Global Encoder ----
    print("\n" + "=" * 60)
    print("2/3  Global Encoder: audio -> voice embedding")
    print("=" * 60)

    global_enc = MioCodecGlobalEncoderONNX(model)
    global_enc.eval()

    dummy_wav = torch.randn(48000, dtype=torch.float32)  # 2 sec at 24kHz
    print(f"  Test: {len(dummy_wav)} samples ({len(dummy_wav)/24000:.1f}s)")
    with torch.no_grad():
        pt_global = global_enc(dummy_wav)
    print(f"  PyTorch output: {pt_global.shape}")

    try:
        ort_global = export_and_verify(
            global_enc,
            (dummy_wav,),
            global_path,
            input_names=["waveform"],
            output_names=["global_embedding"],
            dynamic_axes={"waveform": {0: "samples"}},
            opset=args.opset,
        )
        print(f"  Max diff: {np.max(np.abs(pt_global.numpy() - ort_global)):.6e}")
    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- 3. Content Encoder ----
    print("\n" + "=" * 60)
    print("3/3  Content Encoder: audio -> speech codes")
    print("=" * 60)

    content_enc = MioCodecContentEncoderONNX(model)
    content_enc.eval()

    print(f"  Test: {len(dummy_wav)} samples ({len(dummy_wav)/24000:.1f}s)")
    with torch.no_grad():
        pt_tokens = content_enc(dummy_wav)
    print(f"  PyTorch output: {pt_tokens.shape}, range [{pt_tokens.min()}, {pt_tokens.max()}]")

    try:
        ort_tokens = export_and_verify(
            content_enc,
            (dummy_wav,),
            content_path,
            input_names=["waveform"],
            output_names=["content_token_indices"],
            dynamic_axes={"waveform": {0: "samples"}, "content_token_indices": {0: "seq_len"}},
            opset=args.opset,
        )
        print(f"  Tokens match: {np.array_equal(pt_tokens.numpy(), ort_tokens)}")
    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    for path in [decoder_path, global_path, content_path]:
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {path} (FAILED)")


if __name__ == "__main__":
    main()
