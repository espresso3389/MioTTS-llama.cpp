#!/usr/bin/env python3
"""Compare GGUF weights with Python MioCodec safetensors weights."""
import sys
import numpy as np
import torch
import gguf

def load_gguf_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor.data.copy()
    return None

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else "models/miocodec.gguf"

    from miocodec import MioCodecModel
    print("Loading Python MioCodec model...")
    model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2")
    model = model.eval()

    print(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # Compare token_embd
    print("\n=== Token Embedding ===")
    gguf_emb = load_gguf_tensor(reader, "token_embd")
    if gguf_emb is not None:
        print(f"  GGUF shape: {gguf_emb.shape}")
        # In Python: model.local_quantizer has proj_out Linear(5, 768) and FSQ
        # The GGUF might have a pre-computed lookup table
        # Let's check what the Python FSQ decode gives for a few indices
        test_indices = torch.tensor([[0, 1, 100, 12799]])
        with torch.no_grad():
            py_emb = model.local_quantizer.decode(test_indices)
        print(f"  Python FSQ decode shape: {py_emb.shape}")
        print(f"  Python index 0, first 5: {py_emb[0,0,:5].tolist()}")
        print(f"  Python index 1, first 5: {py_emb[0,1,:5].tolist()}")
        print(f"  GGUF index 0, first 5: {gguf_emb[0,:5].tolist()}")
        print(f"  GGUF index 1, first 5: {gguf_emb[1,:5].tolist()}")

        # Check if GGUF embedding matches Python FSQ decode
        for idx in [0, 1, 100, 12799]:
            py_val = py_emb[0, list(test_indices[0]).index(idx), :].numpy()
            gguf_val = gguf_emb[idx, :]
            diff = np.max(np.abs(py_val - gguf_val))
            print(f"  Index {idx}: max diff = {diff:.6f}")

    # Compare key weight tensors
    weight_pairs = [
        # (gguf_name, python_path)
        ("wave_prenet.blk.0.attn_norm.weight", "wave_prenet.layers.0.norm1.weight"),
        ("wave_prenet.blk.0.attn_q.weight", "wave_prenet.layers.0.self_attn.q_proj.weight"),
        ("wave_prenet.blk.0.ffn_gate.weight", "wave_prenet.layers.0.mlp.gate_proj.weight"),
        ("wave_prenet.norm.weight", "wave_prenet.norm.weight"),
        ("wave_prenet.output.weight", "wave_prenet.linear.weight"),
        ("wave_upsample.weight", "wave_conv_upsample.weight"),
        ("wave_upsample.bias", "wave_conv_upsample.bias"),
        ("wave_prior.0.norm1.weight", "wave_prior_net.blocks.0.norm1.weight"),
        ("wave_prior.0.conv1.weight", "wave_prior_net.blocks.0.conv1.weight"),
        ("wave_decoder.blk.0.attn_cond.weight", "wave_decoder.layers.0.attn_adaln.linear.weight"),
        ("wave_decoder.blk.0.attn_q.weight", "wave_decoder.layers.0.self_attn.q_proj.weight"),
        ("wave_decoder.norm_cond.weight", "wave_decoder.norm_cond.linear.weight"),
        ("wave_post.0.norm1.weight", "wave_post_net.blocks.0.norm1.weight"),
        ("wave_upsampler.up.0.weight", "wave_upsampler.upsample_layers.0.weight"),
        ("wave_upsampler.out_proj.weight", "wave_upsampler.out_proj.weight"),
        ("wave_upsampler.out_snake.alpha", "wave_upsampler.out_snake.alpha"),
        ("istft_head.out.weight", "istft_head.out.weight"),
        ("istft_head.out.bias", "istft_head.out.bias"),
    ]

    print("\n=== Weight Comparison ===")
    for gguf_name, py_path in weight_pairs:
        gguf_w = load_gguf_tensor(reader, gguf_name)

        # Navigate Python model to get the tensor
        py_w = None
        try:
            parts = py_path.split(".")
            obj = model
            for p in parts:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            py_w = obj.data.cpu().numpy()
        except (AttributeError, IndexError, KeyError) as e:
            pass

        if gguf_w is None:
            print(f"  {gguf_name:50s}: GGUF MISSING")
            continue
        if py_w is None:
            print(f"  {gguf_name:50s}: Python path '{py_path}' NOT FOUND")
            continue

        # Check shape compatibility
        gguf_shape = gguf_w.shape
        py_shape = py_w.shape

        if gguf_w.size == py_w.size:
            # Compare values (may need reshape/transpose)
            gguf_flat = gguf_w.flatten()
            py_flat = py_w.flatten()
            max_diff = np.max(np.abs(gguf_flat - py_flat))
            if max_diff < 1e-4:
                print(f"  {gguf_name:50s}: MATCH (max_diff={max_diff:.6f}) shapes: gguf={gguf_shape} py={py_shape}")
            else:
                # Try transposed comparison
                py_t = py_w.T.flatten()
                max_diff_t = np.max(np.abs(gguf_flat - py_t))
                if max_diff_t < 1e-4:
                    print(f"  {gguf_name:50s}: MATCH (transposed, max_diff={max_diff_t:.6f}) shapes: gguf={gguf_shape} py={py_shape}")
                else:
                    print(f"  {gguf_name:50s}: MISMATCH direct={max_diff:.4f} transposed={max_diff_t:.4f} shapes: gguf={gguf_shape} py={py_shape}")
                    print(f"    gguf[:3]: {gguf_flat[:3].tolist()}")
                    print(f"    py[:3]:   {py_flat[:3].tolist()}")
                    print(f"    py.T[:3]: {py_t[:3].tolist()}")
        else:
            print(f"  {gguf_name:50s}: SIZE MISMATCH gguf={gguf_shape}({gguf_w.size}) py={py_shape}({py_w.size})")

    # List all Python model parameter names (for debugging name mapping)
    print("\n=== Python model parameter names ===")
    for name, param in model.named_parameters():
        print(f"  {name:60s} {list(param.shape)}")

if __name__ == "__main__":
    main()
