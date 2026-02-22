# MioTTS Benchmark Report (2026-02-22)

This document summarizes benchmark and feasibility results gathered in this repository on 2026-02-22.

## Scope
- Compare GGUF and ONNX combinations on Linux host.
- Validate ONNX Runtime execution providers (CPU/CUDA/TensorRT) on Linux host.
- Benchmark Android on-device performance for GGUF path.
- Compare `MioTTS-0.1B-Q8_0.gguf` vs `MioTTS-0.1B-Q4_K_M.gguf` on Android.
- Measure persistent-process startup/warm timing and resident RAM on Android.

## Environment
### Host
- OS: Ubuntu 24.04.3 LTS (WSL2)
- GPU: NVIDIA GeForce RTX 5080
- ONNX Runtime versions tested:
  - CPU: `onnxruntime-linux-x64-1.24.2`
  - GPU: `onnxruntime-linux-x64-gpu-1.24.2`

### Android devices
- `A142`: Android `16`, ABI `arm64-v8a`
- `A001T`: Android `16`, ABI `arm64-v8a`

### Models used
- LLM GGUF: `MioTTS-0.1B-Q8_0.gguf`, `MioTTS-0.1B-Q4_K_M.gguf`
- LLM ONNX: `miotts_0.1b.onnx`
- Codec GGUF: `miocodec.gguf`
- Codec ONNX: `miocodec_decoder.onnx`
- Voice: `jp_female.emb.gguf` / `jp_female.emb.bin`

## Important code fixes/additions made during benchmarking
- Fixed full ONNX one-shot path skip logic in `src/main.cpp`:
  - `skip_llm` now correctly checks both GGUF and ONNX model inputs.
- Added ONNX provider selection support in:
  - `src/llm-backend-onnx.cpp`
  - `src/codec-backend-onnx.cpp`
  - `MIOTTS_ONNX_EP=cpu|cuda|tensorrt`
  - `MIOTTS_ONNX_DEVICE_ID=<int>`
- Added persistent timing mode in `src/main.cpp`:
  - `--serve-stdin`
  - `--timing`

## Linux host results

### 1) Baseline combination benchmark (CPU-oriented)
Source: `/tmp/miotts_bench/results.tsv`

| Combination | Wall (s) | RTF | Max RSS (KB) | Status |
|---|---:|---:|---:|---|
| GGUF LLM + GGUF codec | 1.04 | 0.325 | 907204 | ok |
| GGUF LLM + ONNX codec | 0.93 | 0.291 | 712988 | ok |
| ONNX LLM + GGUF codec | 36.96 | 11.550 | 1803212 | ok |
| ONNX LLM + ONNX codec | 35.15 | 10.984 | 1500636 | ok |

Takeaway: ONNX LLM on CPU was the dominant slowdown.

### 2) ONNX EP benchmark (CPU vs CUDA vs TensorRT)
Source: `/tmp/miotts_ep_bench2/results.tsv`

| EP | Wall (s) | CPU | Max RSS (KB) | Status |
|---|---:|---:|---:|---|
| CPU | 35.89 | 362% | 1515676 | ok |
| CUDA | 8.63 | 97% | 1541092 | ok |
| TensorRT | 36.45 | 365% | 1505108 | fallback to CPU |

Notes:
- CUDA EP became active after installing missing CUDA user-space libraries.
- TensorRT EP failed to initialize due missing `libnvinfer.so.10` at first; after installing TRT libs, model-level TRT initialization still failed for this ONNX export (shape/parsing issues), so TRT is not usable for this model as-is.

### 3) Combination benchmark with ONNX forced to CUDA EP
Source: `/tmp/miotts_combo_bench_cuda/results.tsv`

| Combination | ONNX EP | Wall (s) | CPU | Max RSS (KB) | Status |
|---|---|---:|---:|---:|---|
| GGUF LLM + GGUF codec | n/a | 0.93 | 231% | 907480 | ok |
| GGUF LLM + ONNX codec | CUDA | 3.27 | 122% | 1256520 | ok |
| ONNX LLM + GGUF codec | CUDA | 8.22 | 99% | 2055532 | ok |
| ONNX LLM + ONNX codec | CUDA | 8.24 | 97% | 1557120 | ok |

Takeaway: CUDA dramatically improves ONNX LLM vs ONNX CPU, but GGUF+GGUF remained fastest in this setup.

## Android results

## Build/deploy notes
- Android build: NDK r29 (`arm64-v8a`) produced `build-android/miotts`.
- Required additional runtime library on device:
  - `libomp.so` (from NDK) in app lib path.

### 1) On-device GGUF baseline (Q8)
Source: `/tmp/miotts_android_bench.tsv`

Averages from 3 runs each:
- `A142`
  - full_tts: **3.513 s**
  - tokens_only: **2.043 s**
  - decode_only: **1.243 s**
- `A001T`
  - full_tts: **2.557 s**
  - tokens_only: **1.550 s**
  - decode_only: **1.077 s**

Note: the `status=err` flag in raw TSV for full/tokens was a script filter false-positive (logs contain "cannot be used with preferred buffer type"), not a synthesis failure. WAV/tokens outputs were generated successfully.

### 2) On-device GGUF quant comparison (Q8 vs Q4_K_M)
Sources:
- Q8: `/tmp/miotts_android_bench.tsv`
- Q4: `/tmp/miotts_android_bench_q4.tsv`

Averages and delta (Q4 - Q8):

- `A142`
  - full_tts: 3.513 -> 3.250 s (**-0.263 s**, ~7.5% faster)
  - tokens_only: 2.043 -> 1.970 s (**-0.073 s**, ~3.6% faster)
  - decode_only: 1.243 -> 1.277 s (noise/codec variance)

- `A001T`
  - full_tts: 2.557 -> 2.397 s (**-0.160 s**, ~6.3% faster)
  - tokens_only: 1.550 -> 1.310 s (**-0.240 s**, ~15.5% faster)
  - decode_only: 1.077 -> 1.030 s (small variance)

Takeaway: Q4_K_M gives measurable speed gains, but quality tradeoff may not justify it for all products.

### 3) Android ONNX feasibility
- Built ONNX-enabled Android binary using ONNX Runtime Android AAR (`1.24.2`) extracted into include/lib.
- Device smoke tests passed:
  - ONNX `--tokens-only` generated valid `<|s_N|>` tokens.
  - Full ONNX (`--model-onnx` + `-onnx`) produced WAV.
- Quick timing (`A142`, `max_tokens=20`):
  - tokens_only: **44.92 s**
  - full ONNX: **46.30 s**

Takeaway: ONNX on Android is functionally working but not performance-competitive in this configuration.

### 4) Persistent process startup/warm timing (`--serve-stdin --timing`)
Using the same prompt repeated 3 times in one process:

- `A142` + Q8
  - startup: **415.100 ms**
  - req1/req2/req3: **2653.055 / 2610.794 / 2609.134 ms**
- `A142` + Q4
  - startup: **416.811 ms**
  - req1/req2/req3: **2704.453 / 2636.978 / 2576.086 ms**
- `A001T` + Q8
  - startup: **521.066 ms**
  - req1/req2/req3: **1706.917 / 1711.203 / 1723.968 ms**
- `A001T` + Q4
  - startup: **528.369 ms**
  - req1/req2/req3: **1427.381 / 1474.498 / 1473.982 ms**

Takeaway: persistent mode removes model reload cost from each request and gives stable warm latency after startup.

### 5) Resident memory (persistent mode)
Source: `/tmp/miotts_android_mem.tsv`

| Device | Quant | VmRSS (KB) | PSS (KB) |
|---|---|---:|---:|
| A142 | Q8_0 | 538212 | 535505 |
| A142 | Q4_K_M | 494228 | 491428 |
| A001T | Q8_0 | 538372 | 535929 |
| A001T | Q4_K_M | 494228 | 491761 |

Takeaway:
- Resident model process is roughly ~0.5 GB RAM.
- Q4 saves about ~44 MB vs Q8 resident footprint.
- GPU memory for model execution in this tested Android path is effectively not used (CPU backend path).

## Overall conclusions
1. For current Android deployment, GGUF (`llama.cpp` path) is the best practical option.
2. ONNX is feasible and functional, but currently too slow on Android in this setup.
3. Best near-term latency strategy without quality drop:
   - Keep Q8 model.
   - Use persistent process / warm runtime (`--serve-stdin` equivalent in app service).
4. Q4_K_M yields moderate gains, but product decision depends on acceptable quality tradeoff.

## Repro artifacts
- Linux combo baseline: `/tmp/miotts_bench/results.tsv`
- Linux EP benchmark: `/tmp/miotts_ep_bench2/results.tsv`
- Linux combo (CUDA EP): `/tmp/miotts_combo_bench_cuda/results.tsv`
- Android Q8: `/tmp/miotts_android_bench.tsv`
- Android Q4: `/tmp/miotts_android_bench_q4.tsv`
- Android resident memory: `/tmp/miotts_android_mem.tsv`
