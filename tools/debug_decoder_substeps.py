#!/usr/bin/env python3
"""Step through decoder layer 0 sub-operations and dump all intermediate values.
Compare these against C++ debug output to find exact divergence point."""
import torch
import torch.nn.functional as F
import numpy as np

def main():
    from miocodec import MioCodecModel
    from miocodec.module.transformer import precompute_freqs_cis, apply_rotary_emb
    import gguf

    test_codes = [12287, 11619, 11774, 12223, 2490, 826, 2257, 1668, 1219, 2319,
                  9994, 12683, 12745, 4215, 12478, 8800, 8696, 375, 1406, 12396]

    print("Loading models...")
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2")
    model = model.eval()

    # Load voice embedding
    emb_reader = gguf.GGUFReader("models/jp_female.emb.gguf")
    voice_emb = torch.tensor(emb_reader.tensors[0].data.copy(), dtype=torch.float32)

    codes_tensor = torch.tensor([test_codes], dtype=torch.long)
    global_emb = voice_emb.unsqueeze(0)  # [1, 128]
    T = len(test_codes)

    def ps(name, t):
        """Print stats of a tensor."""
        print(f"  {name}: shape={list(t.shape)} min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}")
        flat = t.flatten()
        print(f"    first5: {flat[:5].tolist()}")

    with torch.no_grad():
        # Run pipeline up to decoder input
        content_emb = model.local_quantizer.decode(codes_tensor)
        local_latent = model.wave_prenet(content_emb)
        if model.wave_conv_upsample is not None:
            local_latent = model.wave_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        target_audio_length = model._calculate_original_audio_length(T)
        stft_length = model._calculate_target_stft_length(target_audio_length)
        mode = getattr(model.config, 'wave_interpolation_mode', 'linear')
        local_latent = F.interpolate(local_latent.transpose(1, 2), size=stft_length, mode=mode).transpose(1, 2)
        local_latent = model.wave_prior_net(local_latent.transpose(1, 2)).transpose(1, 2)

        decoder_input = local_latent.clone()  # [1, S_dec, 512]
        S_dec = decoder_input.shape[1]
        print(f"\n=== Decoder input ===")
        ps("decoder_input", decoder_input)

        # Get decoder components
        wd = model.wave_decoder
        layer0 = wd.layers[0]
        attn = layer0.attention
        ffn = layer0.feed_forward
        attn_norm = layer0.attention_norm
        ffn_norm = layer0.ffn_norm

        # Prepare freqs_cis (RoPE)
        freqs_cis = wd.freqs_cis.to(decoder_input.device)
        print(f"\nfreqs_cis shape: {freqs_cis.shape}")
        print(f"  freqs_cis[:3,:3]: {freqs_cis[:3,:3]}")

        # Prepare condition
        cond = global_emb.unsqueeze(1)  # [1, 1, 128]
        print(f"\n=== Condition (global_emb) ===")
        ps("cond", cond)

        # Apply input_proj (identity for wave_decoder)
        x = wd.input_proj(decoder_input)  # identity

        # ==============================
        # STEP A: Attention AdaLN conditioning
        # ==============================
        print(f"\n=== STEP A: Attention AdaLN conditioning ===")
        attn_cond_proj = attn_norm.condition_proj
        # The condition_proj is Sequential(SiLU(), Linear(128, 1536))
        silu_cond = F.silu(cond)
        ps("silu(cond)", silu_cond)

        linear_layer = attn_cond_proj[1]
        print(f"  attn_cond Linear weight shape: {linear_layer.weight.shape}")
        print(f"  attn_cond Linear bias shape: {linear_layer.bias.shape}")
        print(f"  weight first5: {linear_layer.weight.flatten()[:5].tolist()}")
        print(f"  bias first5: {linear_layer.bias.flatten()[:5].tolist()}")

        attn_cond_out = linear_layer(silu_cond)  # [1, 1, 1536]
        ps("attn_cond_proj_out", attn_cond_out)

        shift, scale, gate = attn_cond_out.chunk(3, dim=-1)
        ps("shift", shift)
        ps("scale", scale)
        ps("gate", gate)

        # ==============================
        # STEP B: AdaLN norm
        # ==============================
        print(f"\n=== STEP B: Attention AdaLN norm ===")
        x_norm = attn_norm.norm(x)  # LayerNorm without affine
        ps("x_norm (layernorm)", x_norm)

        x_modulated = x_norm * (1 + scale) + shift
        ps("x_modulated (after adaln)", x_modulated)

        # Verify against full norm call
        norm_result = attn_norm(x, condition=cond)
        x_normed_full, gate_full = norm_result
        diff = (x_modulated - x_normed_full).abs().max().item()
        print(f"  manual vs full norm max diff: {diff:.8f}")

        # ==============================
        # STEP C: Self-attention (manual)
        # ==============================
        print(f"\n=== STEP C: Self-attention ===")
        bsz, seqlen, dim = x_modulated.shape
        n_heads = attn.n_heads
        head_dim = attn.head_dim
        print(f"  bsz={bsz}, seqlen={seqlen}, dim={dim}, n_heads={n_heads}, head_dim={head_dim}")

        # Q, K, V projections
        xq = attn.wq(x_modulated)  # [1, S, 512]
        xk = attn.wk(x_modulated)  # [1, S, 512]
        xv = attn.wv(x_modulated)  # [1, S, 512]
        ps("Q (after wq)", xq)
        ps("K (after wk)", xk)
        ps("V (after wv)", xv)

        xq = xq.view(bsz, seqlen, n_heads, head_dim)
        xk = xk.view(bsz, seqlen, n_heads, head_dim)
        xv = xv.view(bsz, seqlen, n_heads, head_dim)

        # Apply RoPE
        xq_rope = apply_rotary_emb(xq, freqs_cis=freqs_cis[:seqlen])
        xk_rope = apply_rotary_emb(xk, freqs_cis=freqs_cis[:seqlen])
        ps("Q (after RoPE)", xq_rope)
        ps("K (after RoPE)", xk_rope)

        # Create attention mask
        attn_mask = attn.create_mask(bsz, seqlen, None, x.device)
        if attn_mask is not None:
            print(f"  attn_mask shape: {attn_mask.shape}, dtype: {attn_mask.dtype}")
            print(f"  attn_mask[0,0,0,:10]: {attn_mask[0,0,0,:10].tolist()}")
            print(f"  attn_mask[0,0,seqlen-1,-10:]: {attn_mask[0,0,seqlen-1,-10:].tolist()}")
            n_true = attn_mask[0,0].sum().item()
            print(f"  mask True count per head: {n_true} / {seqlen*seqlen}")

        # Run SDPA
        q_sdpa = xq_rope.transpose(1, 2)  # [1, n_heads, seqlen, head_dim]
        k_sdpa = xk_rope.transpose(1, 2)
        v_sdpa = xv.transpose(1, 2)  # Note: V does NOT have RoPE applied
        ps("Q for SDPA (transposed)", q_sdpa)
        ps("K for SDPA (transposed)", k_sdpa)
        ps("V for SDPA (transposed)", v_sdpa)

        attn_output_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scale,
        ).transpose(1, 2)  # [1, seqlen, n_heads, head_dim]
        attn_output_flat = attn_output_sdpa.contiguous().view(bsz, seqlen, -1)
        ps("attn_output (before wo)", attn_output_flat)

        # wo projection
        attn_out = attn.wo(attn_output_flat)
        ps("attn_out (after wo)", attn_out)

        # Verify against full attention call
        attn_out_full = attn(x_modulated, freqs_cis, None)
        diff = (attn_out - attn_out_full).abs().max().item()
        print(f"  manual vs full attention max diff: {diff:.8f}")

        # ==============================
        # STEP D: Gated attention residual
        # ==============================
        print(f"\n=== STEP D: Gated attention residual ===")
        gated_attn = gate * attn_out
        ps("gate * attn_out", gated_attn)
        h = x + gated_attn
        ps("x + gate*attn_out", h)

        # ==============================
        # STEP E: FFN AdaLN conditioning
        # ==============================
        print(f"\n=== STEP E: FFN AdaLN conditioning ===")
        ffn_cond_out = ffn_norm.condition_proj(cond)
        ps("ffn_cond_proj_out", ffn_cond_out)
        ffn_shift, ffn_scale, ffn_gate = ffn_cond_out.chunk(3, dim=-1)
        ps("ffn_shift", ffn_shift)
        ps("ffn_scale", ffn_scale)
        ps("ffn_gate", ffn_gate)

        # ==============================
        # STEP F: FFN AdaLN norm
        # ==============================
        print(f"\n=== STEP F: FFN AdaLN norm ===")
        h_norm = ffn_norm.norm(h)
        ps("h_norm (layernorm)", h_norm)
        h_modulated = h_norm * (1 + ffn_scale) + ffn_shift
        ps("h_modulated (after adaln)", h_modulated)

        # ==============================
        # STEP G: SwiGLU FFN
        # ==============================
        print(f"\n=== STEP G: SwiGLU FFN ===")
        w1_out = ffn.w1(h_modulated)  # gate proj
        ps("w1(x) (gate proj)", w1_out)
        w3_out = ffn.w3(h_modulated)  # up proj
        ps("w3(x) (up proj)", w3_out)
        silu_w1 = F.silu(w1_out)
        ps("silu(w1(x))", silu_w1)
        gated = silu_w1 * w3_out
        ps("silu(w1(x)) * w3(x)", gated)
        ffn_out = ffn.w2(gated)
        ps("w2(gated) = ffn_out", ffn_out)

        # ==============================
        # STEP H: Gated FFN residual
        # ==============================
        print(f"\n=== STEP H: Gated FFN residual ===")
        gated_ffn = ffn_gate * ffn_out
        ps("ffn_gate * ffn_out", gated_ffn)
        out = h + gated_ffn
        ps("h + ffn_gate*ffn_out = layer0_out", out)

        # Verify against full layer call
        layer0_out_full = layer0(decoder_input, freqs_cis, None, condition=cond)
        diff = (out - layer0_out_full).abs().max().item()
        print(f"\n  manual vs full layer0 max diff: {diff:.8f}")

        # ==============================
        # GGUF Weight Cross-check for FFN
        # ==============================
        print(f"\n=== GGUF Weight Cross-check ===")
        reader = gguf.GGUFReader("models/miocodec.gguf")

        def load_gguf(name):
            for t in reader.tensors:
                if t.name == name:
                    return torch.tensor(t.data.copy(), dtype=torch.float32)
            return None

        pairs = [
            ("wave_decoder.blk.0.attn_cond.weight", attn_norm.condition_proj[1].weight),
            ("wave_decoder.blk.0.attn_cond.bias", attn_norm.condition_proj[1].bias),
            ("wave_decoder.blk.0.ffn_cond.weight", ffn_norm.condition_proj[1].weight),
            ("wave_decoder.blk.0.ffn_cond.bias", ffn_norm.condition_proj[1].bias),
            ("wave_decoder.blk.0.attn_q.weight", attn.wq.weight),
            ("wave_decoder.blk.0.attn_k.weight", attn.wk.weight),
            ("wave_decoder.blk.0.attn_v.weight", attn.wv.weight),
            ("wave_decoder.blk.0.attn_output.weight", attn.wo.weight),
            ("wave_decoder.blk.0.ffn_gate.weight", ffn.w1.weight),
            ("wave_decoder.blk.0.ffn_up.weight", ffn.w3.weight),
            ("wave_decoder.blk.0.ffn_down.weight", ffn.w2.weight),
        ]

        for gguf_name, py_param in pairs:
            g = load_gguf(gguf_name)
            p = py_param.data.cpu()
            if g is None:
                print(f"  {gguf_name:50s}: GGUF MISSING")
                continue
            if g.numel() != p.numel():
                print(f"  {gguf_name:50s}: SIZE MISMATCH gguf={g.shape} py={p.shape}")
                continue
            d1 = (g.flatten() - p.flatten()).abs().max().item()
            d2 = (g.flatten() - p.T.flatten()).abs().max().item()
            match_type = "direct" if d1 < d2 else "transposed"
            d = min(d1, d2)
            status = "MATCH" if d < 1e-4 else "MISMATCH"
            print(f"  {gguf_name:50s}: {status} ({match_type}, max_diff={d:.6f}) gguf={list(g.shape)} py={list(p.shape)}")

if __name__ == "__main__":
    main()
