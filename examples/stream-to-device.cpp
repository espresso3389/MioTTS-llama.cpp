#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "test-to-speech.h"
#include "wav-writer.h"

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

struct args_t {
    std::string model_path;
    std::string codec_path;
    std::string codec_type = "ggml";
    std::string voice_path;
    std::string prompt;
    std::string dump_fed_wav_path;
    float temperature = 0.8f;
    int max_tokens = 700;
    int n_threads = 4;
    int n_gpu_layers = 0;
    size_t chunk_samples = 4096;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Stream MioTTS audio directly to the system audio device.\n"
        "\n"
        "Codec backend:\n"
        "  -gguf PATH              MioCodec GGUF model (44.1kHz)\n"
        "  -onnx PATH              MioCodec ONNX decoder (24kHz)\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH        MioTTS LLM GGUF model path\n"
        "  -v, --voice PATH        Voice embedding (.emb.gguf or .bin)\n"
        "  -p, --prompt TEXT       Text to synthesize (use '-' for stdin)\n"
        "  -t, --temp FLOAT        Sampling temperature (default: 0.8)\n"
        "  --max-tokens N          Max tokens to generate (default: 700)\n"
        "  --threads N             Number of CPU threads (default: 4)\n"
        "  -ngl N                  Number of GPU layers (default: 0)\n"
        "  --chunk-samples N       Callback chunk size in samples (default: 4096)\n"
        "  --dump-fed-wav PATH     Save exactly what audio callback sent to device\n"
        "  -h, --help              Show this help\n"
        "\n",
        prog);
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) return false;
            args.model_path = argv[i];
        } else if (arg == "-gguf") {
            if (++i >= argc) return false;
            args.codec_path = argv[i];
            args.codec_type = "ggml";
        } else if (arg == "-onnx") {
            if (++i >= argc) return false;
            args.codec_path = argv[i];
            args.codec_type = "onnx";
        } else if (arg == "-c" || arg == "--codec") {
            if (++i >= argc) return false;
            args.codec_path = argv[i];
        } else if (arg == "-v" || arg == "--voice") {
            if (++i >= argc) return false;
            args.voice_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) return false;
            args.prompt = argv[i];
        } else if (arg == "-t" || arg == "--temp") {
            if (++i >= argc) return false;
            args.temperature = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) return false;
            args.max_tokens = std::stoi(argv[i]);
        } else if (arg == "--threads") {
            if (++i >= argc) return false;
            args.n_threads = std::stoi(argv[i]);
        } else if (arg == "-ngl") {
            if (++i >= argc) return false;
            args.n_gpu_layers = std::stoi(argv[i]);
        } else if (arg == "--chunk-samples") {
            if (++i >= argc) return false;
            args.chunk_samples = static_cast<size_t>(std::stoul(argv[i]));
        } else if (arg == "--dump-fed-wav") {
            if (++i >= argc) return false;
            args.dump_fed_wav_path = argv[i];
        } else if (arg == "--skip-llm") {
            // kept for backward compat, no-op
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

class scoped_stderr_silencer {
public:
    explicit scoped_stderr_silencer(bool enable) : enabled_(enable) {
        if (!enabled_) {
            return;
        }
#if defined(_WIN32)
        saved_fd_ = _dup(_fileno(stderr));
        if (saved_fd_ >= 0) {
            int null_fd = _open("NUL", _O_WRONLY);
            if (null_fd >= 0) {
                _dup2(null_fd, _fileno(stderr));
                _close(null_fd);
            }
        }
#else
        saved_fd_ = dup(fileno(stderr));
        if (saved_fd_ >= 0) {
            int null_fd = open("/dev/null", O_WRONLY);
            if (null_fd >= 0) {
                dup2(null_fd, fileno(stderr));
                close(null_fd);
            }
        }
#endif
    }

    ~scoped_stderr_silencer() {
        if (!enabled_ || saved_fd_ < 0) {
            return;
        }
#if defined(_WIN32)
        _dup2(saved_fd_, _fileno(stderr));
        _close(saved_fd_);
#else
        dup2(saved_fd_, fileno(stderr));
        close(saved_fd_);
#endif
    }

private:
    bool enabled_ = false;
    int saved_fd_ = -1;
};

struct playback_state {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<float> queue;
    size_t queued_samples = 0;
    size_t max_queued_samples = 0;
    bool finished = false;
    bool aborted = false;
    bool capture_fed_audio = false;
    std::vector<float> fed_audio;
};

static void audio_callback(ma_device * device, void * output, const void *, ma_uint32 frame_count) {
    auto * st = reinterpret_cast<playback_state *>(device->pUserData);
    float * out = reinterpret_cast<float *>(output);

    std::lock_guard<std::mutex> lock(st->mu);
    for (ma_uint32 i = 0; i < frame_count; ++i) {
        if (!st->queue.empty()) {
            out[i] = st->queue.front();
            st->queue.pop_front();
            st->queued_samples--;
        } else {
            out[i] = 0.0f;
        }
    }
    if (st->capture_fed_audio) {
        st->fed_audio.insert(st->fed_audio.end(), out, out + frame_count);
    }
    st->cv.notify_all();
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    // Read prompt from stdin if "-p -"
    if (args.prompt == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        args.prompt = ss.str();
        while (!args.prompt.empty() && (args.prompt.back() == '\n' || args.prompt.back() == '\r')) {
            args.prompt.pop_back();
        }
    }

    if (args.prompt.empty()) {
        std::fprintf(stderr, "Error: -p is required\n");
        return 1;
    }
    if (args.codec_path.empty()) {
        std::fprintf(stderr, "Error: codec path is required (-gguf or -onnx)\n");
        return 1;
    }
    if (args.voice_path.empty()) {
        std::fprintf(stderr, "Error: -v is required\n");
        return 1;
    }

    bool skip_llm = args.model_path.empty();

    // This sample is intended to play audio, not print backend/model debug logs.
    scoped_stderr_silencer silencer(true);

    TestToSpeech::Config cfg;
    cfg.model_path = args.model_path;
    cfg.codec_path = args.codec_path;
    cfg.codec_type = args.codec_type;
    cfg.n_threads = args.n_threads;
    cfg.n_gpu_layers = args.n_gpu_layers;
    cfg.temperature = args.temperature;
    cfg.max_tokens = args.max_tokens;

    TestToSpeech tts(cfg);
    if (!tts.is_ready()) {
        std::fprintf(stderr, "Error: failed to initialize TestToSpeech\n");
        return 1;
    }

    VoiceModel voice;
    if (!voice.load_from_file(args.voice_path)) {
        std::fprintf(stderr, "Error: failed to load voice model: %s\n", args.voice_path.c_str());
        return 1;
    }

    playback_state st;
    st.max_queued_samples = static_cast<size_t>(tts.sample_rate()) * 10; // ~10s buffer cap
    st.capture_fed_audio = !args.dump_fed_wav_path.empty();
    if (st.capture_fed_audio) {
        st.fed_audio.reserve(static_cast<size_t>(tts.sample_rate()) * 30); // initial 30s
    }

    ma_device_config dev_cfg = ma_device_config_init(ma_device_type_playback);
    dev_cfg.playback.format = ma_format_f32;
    dev_cfg.playback.channels = 1;
    dev_cfg.sampleRate = static_cast<ma_uint32>(tts.sample_rate());
    dev_cfg.dataCallback = audio_callback;
    dev_cfg.pUserData = &st;

    ma_device device;
    if (ma_device_init(nullptr, &dev_cfg, &device) != MA_SUCCESS) {
        std::fprintf(stderr, "Error: failed to initialize audio device\n");
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::fprintf(stderr, "Error: failed to start audio device\n");
        ma_device_uninit(&device);
        return 1;
    }

    TestToSpeech::Options opt;
    opt.temperature = args.temperature;
    opt.max_tokens = args.max_tokens;
    opt.skip_llm = skip_llm;

    const bool ok = tts.synthesize_stream(
        voice,
        args.prompt,
        [&](const float * samples, size_t n_samples, int, bool is_last_chunk) -> bool {
            std::unique_lock<std::mutex> lock(st.mu);
            st.cv.wait(lock, [&]() {
                return st.aborted || (st.queued_samples + n_samples <= st.max_queued_samples);
            });
            if (st.aborted) {
                return false;
            }

            if (samples && n_samples > 0) {
                st.queue.insert(st.queue.end(), samples, samples + n_samples);
                st.queued_samples += n_samples;
            }
            if (is_last_chunk) {
                st.finished = true;
            }
            lock.unlock();
            st.cv.notify_all();
            return true;
        },
        args.chunk_samples,
        opt);

    {
        std::unique_lock<std::mutex> lock(st.mu);
        if (!ok) {
            st.aborted = true;
        } else {
            st.finished = true;
        }
        st.cv.notify_all();
        st.cv.wait(lock, [&]() { return st.aborted || (st.finished && st.queued_samples == 0); });
    }

    ma_device_stop(&device);
    ma_device_uninit(&device);

    if (st.capture_fed_audio) {
        if (!wav_write(args.dump_fed_wav_path, st.fed_audio, tts.sample_rate())) {
            std::printf("Failed to write fed-audio WAV: %s\n", args.dump_fed_wav_path.c_str());
        } else {
            std::printf("Saved fed-audio WAV: %s (%zu samples)\n",
                        args.dump_fed_wav_path.c_str(),
                        st.fed_audio.size());
        }
    }

    if (!ok) {
        std::fprintf(stderr, "Error: synthesize_stream failed or playback aborted\n");
        return 1;
    }
    return 0;
}
