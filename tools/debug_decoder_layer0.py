#!/usr/bin/env python3
"""Step through decoder layer 0 manually to find divergence with C++."""
import torch
import torch.nn.functional as F
import numpy as np
import gguf

def load_gguf_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return torch.tensor(tensor.data.copy(), dtype=torch.float32)
    return None

def main():
    from miocodec import MioCodecModel

    test_codes = [12287, 11619, 11774, 12223, 2490, 826, 2257, 1668, 1219, 2319,
                  9994, 12683, 12745, 4215, 12478, 8800, 8696, 375, 1406, 12396]

    print("Loading models...")
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2")
    model = model.eval()

    reader = gguf.GGUFReader("models/miocodec.gguf")
    voice_emb = load_gguf_tensor(gguf.GGUFReader("models/jp_female.emb.gguf"),
                                  None)  # get first tensor
    # Actually load first tensor properly
    emb_reader = gguf.GGUFReader("models/jp_female.emb.gguf")
    voice_emb = torch.tensor(emb_reader.tensors[0].data.copy(), dtype=torch.float32)

    codes_tensor = torch.tensor([test_codes], dtype=torch.long)
    global_emb = voice_emb.unsqueeze(0)  # [1, 128]
    T = len(test_codes)

    # Run pipeline up to decoder input (use Python model)
    with torch.no_grad():
        content_emb = model.local_quantizer.decode(codes_tensor)  # [1, T, 768]
        local_latent = model.wave_prenet(content_emb)  # [1, T, 512]

        if model.wave_conv_upsample is not None:
            local_latent = model.wave_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)

        target_audio_length = model._calculate_original_audio_length(T)
        stft_length = model._calculate_target_stft_length(target_audio_length)
        mode = getattr(model.config, 'wave_interpolation_mode', 'linear')
        local_latent = F.interpolate(local_latent.transpose(1, 2), size=stft_length, mode=mode).transpose(1, 2)

        # Prior
        local_latent = model.wave_prior_net(local_latent.transpose(1, 2)).transpose(1, 2)

        decoder_input = local_latent.clone()
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"  stats: min={decoder_input.min():.4f} max={decoder_input.max():.4f} mean={decoder_input.mean():.4f}")

        # Get decoder layer 0
        layer0 = model.wave_decoder.layers[0]
        print(f"\nLayer 0 components: {[name for name, _ in layer0.named_children()]}")

        # Step through layer 0 manually
        x = decoder_input  # [1, 40, 512]
        cond = global_emb.unsqueeze(1)  # [1, 1, 128]
        print(f"Conditioning shape: {cond.shape}")
        print(f"  cond stats: min={cond.min():.4f} max={cond.max():.4f}")

        # Step 1: Attention AdaLN conditioning
        attn_norm = layer0.attention_norm
        print(f"\nAttn norm type: {type(attn_norm)}")
        print(f"  condition_proj: {attn_norm.condition_proj}")

        # Get condition projections
        attn_cond = attn_norm.condition_proj(cond)  # [1, 1, 1536]
        print(f"Attn condition shape: {attn_cond.shape}")
        print(f"  first 5: {attn_cond[0,0,:5].tolist()}")
        shift, scale, gate = attn_cond.chunk(3, dim=-1)
        print(f"  shift[0,0,:5]: {shift[0,0,:5].tolist()}")
        print(f"  scale[0,0,:5]: {scale[0,0,:5].tolist()}")
        print(f"  gate[0,0,:5]: {gate[0,0,:5].tolist()}")
        print(f"  shift stats: min={shift.min():.4f} max={shift.max():.4f} mean={shift.mean():.4f}")
        print(f"  scale stats: min={scale.min():.4f} max={scale.max():.4f} mean={scale.mean():.4f}")
        print(f"  gate stats: min={gate.min():.4f} max={gate.max():.4f} mean={gate.mean():.4f}")

        # Step 2: Apply norm with conditioning
        # Check how MioCodec applies AdaLN (returns tuple)
        norm_result = attn_norm(x, condition=cond)
        if isinstance(norm_result, tuple):
            x_normed = norm_result[0]
            gate_from_norm = norm_result[1] if len(norm_result) > 1 else None
            print(f"\nAdaLN norm returns tuple of {len(norm_result)} elements")
            if gate_from_norm is not None:
                print(f"  gate from norm: {gate_from_norm.shape}")
                print(f"  gate stats: min={gate_from_norm.min():.4f} max={gate_from_norm.max():.4f}")
        else:
            x_normed = norm_result
            gate_from_norm = None
        print(f"After AdaLN norm: {x_normed.shape}")
        print(f"  first 5: {x_normed[0,0,:5].tolist()}")
        print(f"  stats: min={x_normed.min():.4f} max={x_normed.max():.4f} mean={x_normed.mean():.4f}")

        # Step 3: Self-attention
        attn = layer0.attention
        print(f"\nAttention type: {type(attn)}")
        print(f"  wq shape: {attn.wq.weight.shape}")
        print(f"  wk shape: {attn.wk.weight.shape}")

        # Run attention
        attn_out = attn(x_normed)
        print(f"Attention output: {attn_out.shape}")
        print(f"  first 5: {attn_out[0,0,:5].tolist()}")
        print(f"  stats: min={attn_out.min():.4f} max={attn_out.max():.4f} mean={attn_out.mean():.4f}")

        # Step 4: Gated residual (use gate from norm if available)
        actual_gate = gate_from_norm if gate_from_norm is not None else gate
        print(f"\nUsing gate source: {'from norm' if gate_from_norm is not None else 'from chunk'}")
        x_after_attn = x + actual_gate * attn_out
        print(f"\nAfter gated attn residual: {x_after_attn.shape}")
        print(f"  first 5: {x_after_attn[0,0,:5].tolist()}")
        print(f"  stats: min={x_after_attn.min():.4f} max={x_after_attn.max():.4f} mean={x_after_attn.mean():.4f}")

        # Step 5: FFN AdaLN conditioning
        ffn_norm_module = layer0.ffn_norm
        ffn_cond = ffn_norm_module.condition_proj(cond)
        ffn_shift, ffn_scale, ffn_gate = ffn_cond.chunk(3, dim=-1)
        print(f"\nFFN conditioning:")
        print(f"  ffn_shift stats: min={ffn_shift.min():.4f} max={ffn_shift.max():.4f}")
        print(f"  ffn_scale stats: min={ffn_scale.min():.4f} max={ffn_scale.max():.4f}")
        print(f"  ffn_gate stats: min={ffn_gate.min():.4f} max={ffn_gate.max():.4f}")

        # Step 6: FFN norm
        ffn_norm_result = ffn_norm_module(x_after_attn, condition=cond)
        if isinstance(ffn_norm_result, tuple):
            x_ffn_normed = ffn_norm_result[0]
            ffn_gate_from_norm = ffn_norm_result[1] if len(ffn_norm_result) > 1 else ffn_gate
        else:
            x_ffn_normed = ffn_norm_result
            ffn_gate_from_norm = ffn_gate
        print(f"\nAfter FFN AdaLN norm:")
        print(f"  stats: min={x_ffn_normed.min():.4f} max={x_ffn_normed.max():.4f}")

        # Step 7: FFN
        ffn = layer0.feed_forward
        ffn_out = ffn(x_ffn_normed)
        print(f"FFN output:")
        print(f"  stats: min={ffn_out.min():.4f} max={ffn_out.max():.4f}")

        # Step 8: Gated FFN residual
        x_after_ffn = x_after_attn + ffn_gate_from_norm * ffn_out
        print(f"\nAfter gated FFN residual:")
        print(f"  stats: min={x_after_ffn.min():.4f} max={x_after_ffn.max():.4f} mean={x_after_ffn.mean():.4f}")

        # Full layer for comparison
        full_layer_out = layer0(x, condition=cond)
        print(f"\nFull layer 0 output:")
        print(f"  stats: min={full_layer_out.min():.4f} max={full_layer_out.max():.4f} mean={full_layer_out.mean():.4f}")

        # Compare manual vs full
        diff = (x_after_ffn - full_layer_out).abs().max()
        print(f"  manual vs full max diff: {diff:.6f}")

        # Now also check: GGUF weight mapping for decoder
        print("\n=== GGUF Weight Cross-check ===")
        gguf_cond_w = load_gguf_tensor(reader, "wave_decoder.blk.0.attn_cond.weight")
        py_cond_w = attn_norm.condition_proj[1].weight.data

        print(f"  GGUF attn_cond.weight shape: {gguf_cond_w.shape}")
        print(f"  Python condition_proj.1.weight shape: {py_cond_w.shape}")
        print(f"  GGUF first 3: {gguf_cond_w.flatten()[:3].tolist()}")
        print(f"  Python first 3: {py_cond_w.flatten()[:3].tolist()}")
        if gguf_cond_w.numel() == py_cond_w.numel():
            max_diff = (gguf_cond_w.flatten() - py_cond_w.flatten()).abs().max()
            max_diff_t = (gguf_cond_w.flatten() - py_cond_w.T.flatten()).abs().max()
            print(f"  direct match: {max_diff:.6f}")
            print(f"  transposed match: {max_diff_t:.6f}")

        # Also check Q weight
        gguf_q_w = load_gguf_tensor(reader, "wave_decoder.blk.0.attn_q.weight")
        py_q_w = attn.wq.weight.data
        print(f"\n  GGUF attn_q.weight shape: {gguf_q_w.shape}")
        print(f"  Python wq.weight shape: {py_q_w.shape}")
        if gguf_q_w.numel() == py_q_w.numel():
            max_diff = (gguf_q_w.flatten() - py_q_w.flatten()).abs().max()
            max_diff_t = (gguf_q_w.flatten() - py_q_w.T.flatten()).abs().max()
            print(f"  direct match: {max_diff:.6f}")
            print(f"  transposed match: {max_diff_t:.6f}")

if __name__ == "__main__":
    main()
