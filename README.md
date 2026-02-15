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
| `miocodec.gguf` | The audio decoder | Included in this repo / see Releases |
| `jp_female.emb.gguf` (or other voice) | Voice style | Included in this repo / see Releases |

Download the LLM model:
```bash
# Pick a size — smaller is faster, larger sounds better
# 0.1B Q8_0 (125 MB) is a good starting point
huggingface-cli download Aratako/MioTTS-GGUF MioTTS-0.1B-Q8_0.gguf --local-dir models/
```

### 3. Generate speech

```bash
./build/miotts \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -c models/miocodec.gguf \
  -v models/jp_female.emb.gguf \
  -p "Hello, this is a test." \
  -o output.wav
```

That's it! Open `output.wav` in any audio player.

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

You can create a custom voice from any audio recording (WAV, MP3, FLAC, WebM, etc.):

```bash
pip install miocodec torch torchaudio soundfile gguf
python3 tools/create_voice_emb.py my_voice.wav models/my_voice.emb.gguf
```

A few seconds of clear speech is enough. The script extracts the speaker characteristics and saves a tiny file (~768 bytes) that you can use with `-v`.

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
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov

## License

See LICENSE file for details.
