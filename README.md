# MioTTS-llama.cpp

A fast, lightweight text-to-speech tool that runs entirely on your CPU. Give it text, pick a voice, and get a WAV file out.

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [MioTTS](https://huggingface.co/collections/Aratako/miotts) by Aratako.

## What you need

- A C++ compiler (GCC, Clang, or MSVC)
- CMake 3.14+
- ~600 MB disk space for the smallest model set

## Quick start

### 1. Clone and build

```bash
git clone --recursive https://github.com/anthropics/miotts-llama.cpp.git
cd miotts-llama.cpp
cmake -B build
cmake --build build --target miotts
```

To enable the ONNX Runtime codec backend (optional, see [ONNX backend](#onnx-backend) below):
```bash
cmake -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build --target miotts
```

If ONNX Runtime is installed system-wide, it is auto-detected without any extra flags.

### 2. Download models

You need three files in the `models/` directory:

| File | What it is | Where to get it |
|------|-----------|-----------------|
| `MioTTS-0.1B-Q8_0.gguf` | The text-to-speech LLM | [Aratako/MioTTS-GGUF](https://huggingface.co/Aratako/MioTTS-GGUF) |
| `miocodec.gguf` | The audio decoder | [mnga-o/miotts-cpp-gguf](https://huggingface.co/mmnga-o/miotts-cpp-gguf) |
| `jp_female.emb.gguf` (or other voice) | Voice style | [mnga-o/miotts-cpp-gguf](https://huggingface.co/mmnga-o/miotts-cpp-gguf) |

For [mnga-o/miotts-cpp-gguf](https://huggingface.co/mmnga-o/miotts-cpp-gguf) gguf files, the author does not note for the license. But hopefully, it is same to the original models.

Download the required models (codec + all voices + default 0.1B LLM):
```bash
./tools/download-models.sh
```

Windows PowerShell:
```powershell
.\scripts\download-models.ps1
```

Optional: download all MioTTS LLM models from `Aratako/MioTTS-GGUF`:
```bash
./tools/download-models.sh --all-models
```

```powershell
.\scripts\download-models.ps1 -AllModels
```

### 3. Generate speech

Using the GGUF codec (44.1 kHz output):
```bash
./build/miotts \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "ラーメン食べますか?嫌なら食べなくていいですけど、捨てるのもったいないので持って帰ってください。" \
  -o output.wav
```

Or using the ONNX codec (24 kHz output, requires ONNX Runtime at build time):
```bash
./build/miotts \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -onnx models/miocodec_decoder.onnx \
  -v models/voice.emb.bin \
  -p "ラーメン食べますか?" \
  -o output.wav
```

That's it! Open `output.wav` in any audio player.

## Streaming

Build streaming tools:
```bash
cmake --build build --target miotts-stream-device miotts-stream-benchmark miotts-stream-compare
```

### Play directly to audio device

```bash
./build/miotts-stream-device \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。"
```

To debug playback glitches, dump exactly what was fed to the device callback:
```bash
./build/miotts-stream-device \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  --dump-fed-wav fed_audio.wav
```

### Measure realtime ratio (no playback)

```bash
./build/miotts-stream-benchmark \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。"
```

It reports total processing time, generated audio duration, realtime ratio, and stage timings (`llm`, `codec`, `istft`, callback).

### Compare offline vs stream-concat output

```bash
./build/miotts-stream-compare \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  --out-offline offline.wav \
  --out-stream stream_concat.wav
```

### Current streaming defaults

Current defaults are tuned for a quality/speed balance:
- conservative commit holdback to stabilize emitted regions
- chunk-boundary crossfade (~30 ms)
- reduced decode cadence to avoid excessive codec re-decodes

These defaults prioritize clean playback over minimum latency.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model PATH` | LLM model file | |
| `-gguf PATH` | MioCodec GGUF model (44.1 kHz) | |
| `-onnx PATH` | MioCodec ONNX decoder (24 kHz) | |
| `-v, --voice PATH` | Voice embedding (`.emb.gguf` or `.bin`) | |
| `-p, --prompt TEXT` | Text to speak (use `-` to read from stdin) | |
| `-o, --output PATH` | Output WAV file | `output.wav` |
| `-t, --temp FLOAT` | Creativity/variation (higher = more varied) | `0.8` |
| `--max-tokens N` | Maximum length of generated speech | `700` |
| `--threads N` | CPU threads to use | `4` |
| `-ngl N` | GPU layers (if you have CUDA) | `0` |
| `--tokens-only` | Only run LLM, print tokens to stdout | |

### Modes

The tool operates in three modes depending on which flags are provided:

1. **Text-to-speech** (`-m` + `-gguf`/`-onnx` + `-v` + `-p`): Full pipeline, text in, WAV out.
2. **Tokens-only** (`-m` + `--tokens-only` + `-p`): Run only the LLM, print speech codes to stdout. No codec needed.
3. **Decode-only** (`-gguf`/`-onnx` + `-v` + `-p`, no `-m`): Decode pre-generated speech codes to WAV.

## Choosing a voice

Four built-in voices are included:

| File | Voice |
|------|-------|
| `jp_female.emb.gguf` | Japanese female |
| `jp_male.emb.gguf` | Japanese male |
| `en_female.emb.gguf` | English female |
| `en_male.emb.gguf` | English male |

### Create your own voice

You can create a custom voice from any recording (WAV, MP3, FLAC, WebM, etc.).
Use 5-30 seconds of mostly solo speech with low background noise.

**For GGUF backend** (requires PyTorch):
```bash
pip install miocodec torch torchaudio soundfile gguf
python3 tools/create_voice_emb.py my_voice.wav models/my_voice.emb.gguf
```

**For ONNX backend** (lightweight, no PyTorch needed):
```bash
pip install onnxruntime soundfile numpy
python3 tools/create_voice_emb_onnx.py \
  --encoder miocodec_global_encoder.onnx \
  --audio my_voice.wav \
  --output models/my_voice.emb.bin
```

The ONNX global encoder model (`miocodec_global_encoder.onnx`) can be exported from [MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz).

Voice embedding files are tiny (~512-768 bytes), so you can keep multiple voice presets in `models/`.

Note: GGUF voice embeddings (`.emb.gguf`) and ONNX voice embeddings (`.emb.bin`) are **not interchangeable** -- each is specific to its codec model.

## Choosing a model size

Larger models produce higher quality speech but run slower:

| Model | Size (Q8_0) | Notes |
|-------|-------------|-------|
| 0.1B | 125 MB | Fast, decent quality |
| 0.4B | 392 MB | Good balance |
| 0.6B | 653 MB | Better prosody |
| 1.2B | 1.27 GB | High quality |
| 1.7B | 1.86 GB | Higher quality |
| 2.6B | 2.76 GB | Best quality |

Each size is available in multiple quantizations (BF16, Q8_0, Q6_K, Q4_K_M) on [Aratako/MioTTS-GGUF](https://huggingface.co/Aratako/MioTTS-GGUF). Smaller quantizations (Q4_K_M) trade some quality for faster speed and less memory.

## Examples

Japanese (GGUF):
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -gguf models/miocodec.gguf \
  -v models/jp_female.emb.gguf -p "こんにちは、今日はいい天気ですね。" -o hello_jp.wav
```

English (ONNX):
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -onnx models/miocodec_decoder.onnx \
  -v models/voice.emb.bin -p "The quick brown fox jumps over the lazy dog." -o hello_en.wav
```

With GPU acceleration (CUDA):
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -gguf models/miocodec.gguf \
  -v models/en_female.emb.gguf -p "This runs on the GPU." -ngl 99 -o gpu_test.wav
```

Two-step pipeline (generate tokens, then decode separately):
```bash
# Step 1: Text to tokens (no codec needed)
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf --tokens-only -p "こんにちは" > tokens.txt

# Step 2: Tokens to audio (pipe via stdin)
./build/miotts -onnx models/miocodec_decoder.onnx -v models/voice.emb.bin -p - -o output.wav < tokens.txt
```

## Output format

- GGUF backend: 44.1 kHz, 16-bit PCM, mono WAV
- ONNX backend: 24 kHz, 16-bit PCM, mono WAV

## ONNX backend

The ONNX backend uses [ONNX Runtime](https://onnxruntime.ai/) to run the MioCodec-25Hz-24kHz decoder, producing 24 kHz audio directly (ISTFT is baked into the model). It is typically 3-5x faster than the GGML backend for codec decoding.

### Setup

1. Download the ONNX Runtime C++ SDK from [GitHub releases](https://github.com/microsoft/onnxruntime/releases) (e.g., `onnxruntime-linux-x64-1.24.2.tgz`).

2. Build with ONNX support:
```bash
cmake -B build -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.24.2
cmake --build build
```

If ONNX Runtime is installed system-wide (e.g. via a package manager), it is auto-detected and `ONNXRUNTIME_DIR` is not needed.

3. Export the ONNX model files from [MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz):

```bash
pip install miocodec torch onnx onnxruntime numpy gguf
python3 tools/export_miocodec_onnx.py
```

This single command downloads the model from HuggingFace, exports all three ONNX files to `models/`, and converts the built-in voice embeddings (`.emb.gguf` -> `.emb.bin`):

| File | Size | Description |
|------|------|-------------|
| `miocodec_decoder.onnx` | ~354 MB | Speech codes + voice embedding -> waveform (required) |
| `miocodec_global_encoder.onnx` | ~112 MB | Audio -> 128-dim voice embedding (for `create_voice_emb_onnx.py`) |
| `miocodec_content_encoder.onnx` | ~447 MB | Audio -> speech codes (for `extract_codes_onnx.py`, testing) |

Only the decoder is needed at runtime. To export just the decoder: `--skip-encoders`.

The export handles ONNX compatibility issues automatically (RoPE with real-valued ops, iDFT via matrix multiply, overlap-add via ConvTranspose1d).

### Codec benchmark

Compare GGML vs ONNX decode performance:
```bash
cmake --build build --target miotts-codec-benchmark
./build/miotts-codec-benchmark \
  --ggml-codec models/miocodec.gguf --ggml-voice models/jp_female.emb.gguf \
  --onnx-codec models/miocodec_decoder.onnx --onnx-voice models/voice.emb.bin \
  --n-codes 100
```

## Scripts

### `tools/export_miocodec_onnx.py`

Export all MioCodec ONNX models and convert built-in voice embeddings in one command. Downloads MioCodec-25Hz-24kHz from HuggingFace automatically. Requires `miocodec`, `torch`, `onnx`, `onnxruntime`, `gguf`.

```bash
python3 tools/export_miocodec_onnx.py
```

Use `--skip-encoders` to export only the decoder. Use `--output-dir` to change the output directory (default: `models/`).

### `tools/create_voice_emb_onnx.py`

Extract a 128-dim voice embedding from a reference audio file using the ONNX global encoder. Produces a raw `.emb.bin` file for use with the ONNX codec backend.

```bash
pip install onnxruntime soundfile numpy
python3 tools/create_voice_emb_onnx.py \
  --encoder miocodec_global_encoder.onnx \
  --audio reference.wav \
  --output voice.emb.bin
```

### `tools/extract_codes_onnx.py`

Extract speech codes from an audio file using the ONNX content encoder. Outputs `<|s_N|>` formatted tokens for decode-only mode.

```bash
pip install onnxruntime soundfile numpy
python3 tools/extract_codes_onnx.py \
  --encoder miocodec_content_encoder.onnx \
  --audio input.wav \
  --output codes.txt
```

Or pipe directly into the decoder:
```bash
python3 tools/extract_codes_onnx.py \
  --encoder miocodec_content_encoder.onnx \
  --audio input.wav \
  | ./build/miotts -onnx models/miocodec_decoder.onnx -v models/voice.emb.bin -p - -o output.wav
```

## Credits

- [MioTTS](https://huggingface.co/collections/Aratako/miotts) and [MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) by [Aratako](https://huggingface.co/Aratako)
- GGUF packaging used here (`miocodec.gguf` and voice `.emb.gguf` files) from [mnga-o/miotts-cpp-gguf](https://huggingface.co/mmnga-o/miotts-cpp-gguf)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

## License

See [LICENSE](./LICENSE) file for details.
