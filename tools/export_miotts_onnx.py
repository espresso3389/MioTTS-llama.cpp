#!/usr/bin/env python3
"""Export MioTTS safetensors model to ONNX for miotts-llama.cpp.

Why this script exists:
- MioTTS models (ex: FalconH1/LFM2 variants) currently fail with the default
  torch.export-based ONNX path.
- The legacy TorchScript exporter works, but may emit Transpose perms with
  negative indices that ONNX Runtime rejects.

This script uses:
1) legacy export (`dynamo=False`)
2) post-fix of negative Transpose perm indices
3) optional ONNX Runtime sanity check vs PyTorch logits

Example:
  uv run --with transformers --with accelerate --with onnx --with onnxruntime \
    python tools/export_miotts_onnx.py \
      --model-id Aratako/MioTTS-0.1B \
      --output models/miotts_0.1b.onnx
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM


@dataclass
class ExportStats:
    transpose_patches: int = 0
    mae: float = 0.0
    max_abs: float = 0.0


class MioTtsOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return out.logits


def patch_negative_transpose_perms(model: onnx.ModelProto) -> int:
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name != "perm":
                continue
            perm = list(attr.ints)
            if any(v < 0 for v in perm):
                n = len(perm)
                attr.ints[:] = [v + n if v < 0 else v for v in perm]
                patched += 1
    return patched


def export_onnx(
    model_id: str,
    output_path: str,
    opset: int,
    seq_len: int,
    verify: bool,
) -> ExportStats:
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    wrapped = MioTtsOnnxWrapper(model)

    input_ids = torch.randint(0, 256, (1, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    print(f"Exporting ONNX (opset={opset}, seq_len={seq_len}) -> {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask, position_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "position_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # required for current MioTTS architectures
    )

    print("Patching ONNX graph for ORT compatibility...")
    onnx_model = onnx.load(output_path)
    patched = patch_negative_transpose_perms(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"Patched Transpose nodes: {patched}")

    stats = ExportStats(transpose_patches=patched)

    if verify:
        print("Running ONNX Runtime verification...")
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        test_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
        test_mask = torch.ones_like(test_ids)
        test_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            pt = model(
                input_ids=test_ids,
                attention_mask=test_mask,
                position_ids=test_pos,
                use_cache=False,
            ).logits[:, -1, :].cpu().numpy()
        ort_out = sess.run(
            ["logits"],
            {
                "input_ids": test_ids.numpy(),
                "attention_mask": test_mask.numpy(),
                "position_ids": test_pos.numpy(),
            },
        )[0][:, -1, :]

        stats.mae = float(np.mean(np.abs(pt - ort_out)))
        stats.max_abs = float(np.max(np.abs(pt - ort_out)))
        print(f"Verification MAE={stats.mae:.8f}, MaxAbs={stats.max_abs:.8f}")

    print(f"Done: {output_path} ({os.path.getsize(output_path)} bytes)")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MioTTS safetensors to ONNX")
    parser.add_argument(
        "--model-id",
        default="Aratako/MioTTS-0.1B",
        help="HuggingFace model ID or local model directory",
    )
    parser.add_argument(
        "--output",
        default="models/miotts.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--trace-seq-len",
        type=int,
        default=8,
        help="Sequence length used for tracing/export and optional verification",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ORT numerical verification",
    )
    args = parser.parse_args()

    stats = export_onnx(
        model_id=args.model_id,
        output_path=args.output,
        opset=args.opset,
        seq_len=args.trace_seq_len,
        verify=not args.no_verify,
    )

    if not args.no_verify:
        if not (stats.mae < 1e-3 and stats.max_abs < 5e-3):
            raise SystemExit(
                f"Verification failed: MAE={stats.mae:.6f}, MaxAbs={stats.max_abs:.6f}"
            )


if __name__ == "__main__":
    main()

