#!/usr/bin/env python3
"""Convert MioCodec encoder to ONNX.

Exports two models:
1. Global encoder: waveform -> global_embedding (voice style)
2. Content encoder: waveform -> content_token_indices

The global encoder is needed for MioTTS to provide speaker identity.
The content encoder is needed for voice conversion.

Note: WavLM (SSL feature extractor) is included in the export.
"""

import argparse
import math

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the ONNX-compatible transformer from the decoder script
from export_miocodec_decoder_onnx import TransformerONNX


class MioCodecGlobalEncoderONNX(nn.Module):
    """ONNX-exportable global encoder: waveform -> global_embedding.

    Extracts speaker/voice style embedding from audio.
    """

    def __init__(self, model):
        super().__init__()
        self.ssl = model.ssl_feature_extractor
        self.global_encoder = model.global_encoder
        self.global_ssl_layers = model.global_ssl_layers
        self.sample_rate = model.config.sample_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (samples,) float32 audio at 24kHz
        Returns:
            global_embedding: (128,) float32 speaker embedding
        """
        # Add batch dim
        wav = waveform.unsqueeze(0)

        # Resample if needed
        if self.ssl.resampler is not None:
            wav = self.ssl.resampler(wav)

        # Extract SSL features
        features, _ = self.ssl.model.extract_features(wav, num_layers=max(self.global_ssl_layers))

        # Average selected layers
        if len(self.global_ssl_layers) > 1:
            selected = [features[i - 1] for i in self.global_ssl_layers]
            global_feats = torch.stack(selected, dim=0).mean(dim=0)
        else:
            global_feats = features[self.global_ssl_layers[0] - 1]

        # Global encoder
        global_embedding = self.global_encoder(global_feats)
        return global_embedding.squeeze(0)


class MioCodecContentEncoderONNX(nn.Module):
    """ONNX-exportable content encoder: waveform -> content_token_indices.

    Extracts discrete content tokens from audio.
    """

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

    def _normalize_features(self, features, eps=1e-8):
        if not self.normalize_ssl:
            return features
        mean = torch.mean(features, dim=1, keepdim=True)
        std = torch.std(features, dim=1, keepdim=True)
        return (features - mean) / (std + eps)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (samples,) float32 audio at 24kHz
        Returns:
            content_token_indices: (seq_len,) int64 token indices
        """
        wav = waveform.unsqueeze(0)

        # Calculate padding
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

        local_feats = self._normalize_features(local_feats)

        # Local encoder
        local_encoded = self.local_encoder(local_feats)

        # Downsample
        if self.downsample_factor > 1:
            if self.use_conv_downsample:
                local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
            else:
                local_encoded = F.avg_pool1d(
                    local_encoded.transpose(1, 2),
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor,
                ).transpose(1, 2)

        # Quantize
        _, indices = self.local_quantizer.encode(local_encoded)
        return indices.squeeze(0)


class MioCodecFullEncoderONNX(nn.Module):
    """Combined encoder: waveform -> (content_token_indices, global_embedding)."""

    def __init__(self, model):
        super().__init__()
        self.ssl = model.ssl_feature_extractor
        self.local_encoder = TransformerONNX(model.local_encoder)
        self.local_quantizer = model.local_quantizer
        self.global_encoder = model.global_encoder
        self.local_ssl_layers = model.local_ssl_layers
        self.global_ssl_layers = model.global_ssl_layers
        self.downsample_factor = model.config.downsample_factor
        self.use_conv_downsample = model.config.use_conv_downsample
        self.normalize_ssl = model.config.normalize_ssl_features
        if self.use_conv_downsample:
            self.conv_downsample = model.conv_downsample
        self.sample_rate = model.config.sample_rate

    def forward(self, waveform: torch.Tensor, padding: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: (samples,) float32 audio at 24kHz
            padding: int, waveform padding in samples
        Returns:
            content_token_indices: (seq_len,) int64
            global_embedding: (128,) float32
        """
        wav = waveform.unsqueeze(0)
        if padding > 0:
            wav = F.pad(wav, (padding, padding), mode="constant")

        if self.ssl.resampler is not None:
            wav = self.ssl.resampler(wav)

        max_layer = max(max(self.local_ssl_layers), max(self.global_ssl_layers))
        features, _ = self.ssl.model.extract_features(wav, num_layers=max_layer)

        # Global path
        if len(self.global_ssl_layers) > 1:
            selected = [features[i - 1] for i in self.global_ssl_layers]
            global_feats = torch.stack(selected, dim=0).mean(dim=0)
        else:
            global_feats = features[self.global_ssl_layers[0] - 1]
        global_embedding = self.global_encoder(global_feats).squeeze(0)

        # Content path
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
        return indices.squeeze(0), global_embedding


def main():
    parser = argparse.ArgumentParser(description="Convert MioCodec encoder to ONNX")
    parser.add_argument("--output-global", default="miocodec_global_encoder.onnx")
    parser.add_argument("--output-content", default="miocodec_content_encoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--audio-len", type=int, default=48000, help="Example audio length (2 sec at 24kHz)")
    args = parser.parse_args()

    print("Loading MioCodec model...")
    from miocodec import MioCodecModel
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-24kHz")
    model.eval().cpu()

    audio_len = args.audio_len
    dummy_wav = torch.randn(audio_len, dtype=torch.float32)

    # ---- Global Encoder ----
    print("\n=== Global Encoder ===")
    global_enc = MioCodecGlobalEncoderONNX(model)
    global_enc.eval()

    print("PyTorch forward...")
    with torch.no_grad():
        pt_global = global_enc(dummy_wav)
    print(f"  output: {pt_global.shape}")

    print(f"Exporting (opset {args.opset})...")
    try:
        torch.onnx.export(
            global_enc,
            (dummy_wav,),
            args.output_global,
            input_names=["waveform"],
            output_names=["global_embedding"],
            dynamic_axes={"waveform": {0: "samples"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"  Saved: {args.output_global}")

        print("Validating...")
        onnx.checker.check_model(onnx.load(args.output_global))
        print("  OK")

        print("ONNX Runtime test...")
        sess = ort.InferenceSession(args.output_global)
        ort_global = sess.run(None, {"waveform": dummy_wav.numpy()})[0]
        print(f"  output: {ort_global.shape}")
        print(f"  max diff: {np.max(np.abs(pt_global.numpy() - ort_global)):.6e}")

    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Content Encoder ----
    print("\n=== Content Encoder ===")
    content_enc = MioCodecContentEncoderONNX(model)
    content_enc.eval()

    print("PyTorch forward...")
    with torch.no_grad():
        pt_tokens = content_enc(dummy_wav)
    print(f"  output: {pt_tokens.shape}")

    if args.verify:
        print("Comparing with original model...")
        with torch.no_grad():
            orig_feats = model.encode(dummy_wav)
        print(f"  orig tokens: {orig_feats.content_token_indices.shape}")
        print(f"  orig global: {orig_feats.global_embedding.shape}")
        print(f"  tokens match: {torch.equal(pt_tokens, orig_feats.content_token_indices)}")
        print(f"  global max diff: {torch.max(torch.abs(pt_global - orig_feats.global_embedding)).item():.6e}")

    print(f"Exporting (opset {args.opset})...")
    try:
        torch.onnx.export(
            content_enc,
            (dummy_wav,),
            args.output_content,
            input_names=["waveform"],
            output_names=["content_token_indices"],
            dynamic_axes={"waveform": {0: "samples"}, "content_token_indices": {0: "seq_len"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"  Saved: {args.output_content}")

        print("Validating...")
        onnx.checker.check_model(onnx.load(args.output_content))
        print("  OK")

        print("ONNX Runtime test...")
        sess = ort.InferenceSession(args.output_content)
        ort_tokens = sess.run(None, {"waveform": dummy_wav.numpy()})[0]
        print(f"  output: {ort_tokens.shape}")
        print(f"  tokens match: {np.array_equal(pt_tokens.numpy(), ort_tokens)}")

    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
