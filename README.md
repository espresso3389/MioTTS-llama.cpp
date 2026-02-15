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
mkdir build && cd build
cmake ..
cmake --build . --target miotts
```

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
./scripts/download-models.sh
```

Windows PowerShell:
```powershell
.\scripts\download-models.ps1
```

Optional: download all MioTTS LLM models from `Aratako/MioTTS-GGUF`:
```bash
./scripts/download-models.sh --all-models
```

```powershell
.\scripts\download-models.ps1 -AllModels
```

### 3. Generate speech

```bash
./build/miotts \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "ラーメン食べますか?嫌なら食べなくていいですけど、捨てるのもったいないので持って帰ってください。" \
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
  -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。"
```

To debug playback glitches, dump exactly what was fed to the device callback:
```bash
./build/miotts-stream-device \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  --dump-fed-wav fed_audio.wav
```

### Measure realtime ratio (no playback)

```bash
./build/miotts-stream-benchmark \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。"
```

It reports total processing time, generated audio duration, realtime ratio, and stage timings (`llm`, `codec`, `istft`, callback).

### Compare offline vs stream-concat output

```bash
./build/miotts-stream-compare \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -c models/miocodec.gguf \
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
| `-m, --model PATH` | LLM model file | (required) |
| `-c, --codec PATH` | MioCodec model file | (required) |
| `-v, --voice PATH` | Voice embedding file | (required) |
| `-p, --prompt TEXT` | Text to speak | (required) |
| `-o, --output PATH` | Output WAV file | `output.wav` |
| `-t, --temp FLOAT` | Creativity/variation (higher = more varied) | `0.8` |
| `--max-tokens N` | Maximum length of generated speech | `700` |
| `--threads N` | CPU threads to use | `4` |
| `-ngl N` | GPU layers (if you have CUDA) | `0` |

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

1. Install dependencies:
```bash
pip install miocodec torch torchaudio soundfile gguf
```

2. Prepare a clean speech clip.
Use 5-30 seconds of mostly solo speech with low background noise.
If you have a long file, trim it first (example with ffmpeg):
```bash
ffmpeg -i input_audio.wav -ss 00:00:05 -t 20 my_voice.wav
```

3. Generate a voice embedding file:
```bash
python3 tools/create_voice_emb.py my_voice.wav models/my_voice.emb.gguf
```

4. Use your new voice with `miotts`:
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -c models/miocodec.gguf \
  -v models/my_voice.emb.gguf -p "Hello from my custom voice." -o my_voice_test.wav
```

The generated `.emb.gguf` file is tiny (~768 bytes), so you can keep multiple voice presets in `models/`.

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

Japanese:
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf -p "こんにちは、今日はいい天気ですね。" -o hello_jp.wav
```

English:
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -c models/miocodec.gguf \
  -v models/en_male.emb.gguf -p "The quick brown fox jumps over the lazy dog." -o hello_en.wav
```

With GPU acceleration (CUDA):
```bash
./build/miotts -m models/MioTTS-0.1B-Q8_0.gguf -c models/miocodec.gguf \
  -v models/en_female.emb.gguf -p "This runs on the GPU." -ngl 99 -o gpu_test.wav
```

## Output format

- 44.1 kHz sample rate
- 16-bit PCM
- Mono WAV

## Credits

- [MioTTS](https://huggingface.co/collections/Aratako/miotts) and [MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) by [Aratako](https://huggingface.co/Aratako)
- GGUF packaging used here (`miocodec.gguf` and voice `.emb.gguf` files) from [mnga-o/miotts-cpp-gguf](https://huggingface.co/mmnga-o/miotts-cpp-gguf)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

## License

See LICENSE file for details.
