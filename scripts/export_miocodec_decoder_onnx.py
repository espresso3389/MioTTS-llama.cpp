#!/usr/bin/env python3
"""Convert MioCodec decoder (wave mode) from safetensors to ONNX.

Exports the decode path: content_token_indices + global_embedding -> waveform.
This is the component needed for MioTTS audio synthesis.

Key ONNX-compatibility changes:
- RoPE: replaced complex-number operations with real-valued cos/sin
- ISTFT: replaced torch.fft.irfft with precomputed iDFT matrix multiply
- Overlap-add: replaced F.fold with ConvTranspose1d
"""

import argparse
import math

import numpy as np
import onnx
import onnxruntime as ort
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

        # Overlap-add kernel: identity matrix as ConvTranspose1d weight
        # (in_channels=win_length, out_channels=1, kernel_size=win_length)
        ola_weight = torch.eye(win_length).unsqueeze(1)
        self.register_buffer("ola_weight", ola_weight)

        # Window envelope kernel
        env_weight = torch.diag(window.square()).unsqueeze(1)
        self.register_buffer("env_weight", env_weight)

    def forward(self, x):
        """x: (B, L, H) -> audio: (B, samples)"""
        x = self.out(x)
        x = x.transpose(1, 2)  # (B, out_dim, T)
        mag, phase = x.chunk(2, dim=1)  # each (B, freq_bins, T)

        mag = torch.clamp(torch.exp(mag), max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        # iDFT: (n_fft, freq_bins) @ (B, freq_bins, T) -> (B, n_fft, T)
        ifft = self.cos_basis @ real - self.sin_basis @ imag

        # Apply window
        ifft = ifft * self.window.unsqueeze(-1)

        # Overlap-add via ConvTranspose1d: (B, win_length, T) -> (B, 1, output_len)
        y = F.conv_transpose1d(ifft, self.ola_weight, stride=self.hop_length).squeeze(1)

        # Window envelope normalization
        T = ifft.shape[2]
        ones = torch.ones(1, self.win_length, T, device=x.device, dtype=x.dtype)
        envelope = F.conv_transpose1d(ones, self.env_weight, stride=self.hop_length).squeeze(1)
        y = y / (envelope + 1e-11)

        # Trim padding
        if self.padding_mode == "same":
            pad = (self.win_length - self.hop_length) // 2
            y = y[:, pad:-pad]
        elif self.padding_mode == "center":
            pad = self.n_fft // 2
            y = y[:, pad:-pad]

        return y


# ============================================================================
# Main decoder wrapper
# ============================================================================

class MioCodecDecoderONNX(nn.Module):
    """ONNX-exportable MioCodec decoder (wave mode).

    Inputs: content_token_indices (seq_len,), global_embedding (128,), stft_length (int)
    Output: waveform (samples,)
    """

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
        # Decode tokens to embeddings
        content = self.local_quantizer.decode(content_token_indices.unsqueeze(0))
        global_emb = global_embedding.unsqueeze(0)

        # Wave prenet
        x = self.wave_prenet(content)

        # Conv upsample
        if self.wave_conv_upsample is not None:
            x = self.wave_conv_upsample(x.transpose(1, 2)).transpose(1, 2)

        # Interpolate to STFT length
        x = F.interpolate(
            x.transpose(1, 2), size=stft_length, mode=self.wave_interpolation_mode
        ).transpose(1, 2)

        # Prior ResNet -> Transformer -> Post ResNet
        x = self.wave_prior_net(x.transpose(1, 2)).transpose(1, 2)
        x = self.wave_decoder(x, condition=global_emb.unsqueeze(1))
        x = self.wave_post_net(x.transpose(1, 2)).transpose(1, 2)

        # Optional upsampler
        if self.wave_upsampler is not None:
            x = self.wave_upsampler(x.transpose(1, 2))

        # ISTFT
        waveform = self.istft_head(x)
        return waveform.squeeze(0)


# ============================================================================
# Utilities
# ============================================================================

def calculate_stft_length(token_length, downsample_factor, ssl_hop_size,
                          ssl_sample_rate, sample_rate, hop_length, istft_padding):
    """Calculate STFT frame count from token sequence length."""
    feature_length = token_length * downsample_factor
    # Approximate get_minimum_input_length for wavlm_base_plus
    num_samples_ssl = (feature_length - 1) * ssl_hop_size + 400
    audio_length = math.ceil(num_samples_ssl / ssl_sample_rate * sample_rate)
    if istft_padding == "same":
        return audio_length // hop_length
    else:
        return audio_length // hop_length + 1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert MioCodec decoder to ONNX")
    parser.add_argument("--output", default="miocodec_decoder.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Compare with original model")
    parser.add_argument("--seq-len", type=int, default=50, help="Example sequence length")
    args = parser.parse_args()

    print("Loading MioCodec model...")
    from miocodec import MioCodecModel
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz")
    model.eval().cpu()
    print(f"  use_wave_decoder={model.config.use_wave_decoder}, n_fft={model.config.n_fft}, hop={model.config.hop_length}")

    print("Creating ONNX wrapper...")
    decoder = MioCodecDecoderONNX(model)
    decoder.eval()

    seq_len = args.seq_len
    embed_dim = model.global_encoder.output_dim
    codebook_size = model.local_quantizer.all_codebook_size
    dummy_tokens = torch.randint(0, codebook_size, (seq_len,), dtype=torch.long)
    dummy_global = torch.randn(embed_dim, dtype=torch.float32)
    stft_len = calculate_stft_length(
        seq_len, model.config.downsample_factor,
        model.ssl_feature_extractor.hop_size, model.ssl_feature_extractor.ssl_sample_rate,
        model.config.sample_rate, model.config.hop_length, model.config.istft_padding,
    )
    print(f"  tokens={dummy_tokens.shape}, global={dummy_global.shape}, stft_len={stft_len}")

    # Test PyTorch forward
    print("PyTorch forward pass...")
    with torch.no_grad():
        pt_out = decoder(dummy_tokens, dummy_global, stft_len)
    print(f"  output: {pt_out.shape}")

    if args.verify:
        print("Original model forward pass...")
        with torch.no_grad():
            emb = model.decode_token_indices(dummy_tokens.unsqueeze(0)).squeeze(0)
            orig_out = model.decode(
                global_embedding=dummy_global, content_embedding=emb,
                target_audio_length=stft_len * model.config.hop_length,
            )
        print(f"  output: {orig_out.shape}")

    # Export
    print(f"Exporting ONNX (opset {args.opset})...")
    torch.onnx.export(
        decoder,
        (dummy_tokens, dummy_global, stft_len),
        args.output,
        input_names=["content_token_indices", "global_embedding", "stft_length"],
        output_names=["waveform"],
        dynamic_axes={
            "content_token_indices": {0: "seq_len"},
            "waveform": {0: "samples"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {args.output}")

    # Validate
    print("Validating...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("  OK")

    # ONNX Runtime test
    print("ONNX Runtime inference...")
    sess = ort.InferenceSession(args.output)
    ort_out = sess.run(None, {
        "content_token_indices": dummy_tokens.numpy(),
        "global_embedding": dummy_global.numpy(),
        "stft_length": np.array(stft_len, dtype=np.int64),
    })[0]
    print(f"  output: {ort_out.shape}")
    print(f"  max diff (wrapper vs ORT): {np.max(np.abs(pt_out.numpy() - ort_out)):.6e}")

    if args.verify:
        min_len = min(len(ort_out), len(orig_out))
        print(f"  max diff (original vs ORT): {np.max(np.abs(orig_out.numpy()[:min_len] - ort_out[:min_len])):.6e}")

    print("Done!")


if __name__ == "__main__":
    main()
