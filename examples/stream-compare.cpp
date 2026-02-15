#include "test-to-speech.h"
#include "wav-writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

struct args_t {
    std::string model_path;
    std::string codec_path;
    std::string voice_path;
    std::string prompt;
    std::string out_offline = "offline.wav";
    std::string out_stream = "stream_concat.wav";
    float temperature = 0.8f;
    int max_tokens = 700;
    int n_threads = 4;
    int n_gpu_layers = 0;
    size_t chunk_samples = 4096;
    bool skip_llm = false;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Compare non-streaming output with concatenated streaming chunks.\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH        MioTTS LLM GGUF model path (required unless --skip-llm)\n"
        "  -c, --codec PATH        MioCodec GGUF model path (required)\n"
        "  -v, --voice PATH        Voice embedding .emb.gguf path (required)\n"
        "  -p, --prompt TEXT       Text to synthesize (required)\n"
        "  --out-offline PATH      Output WAV path for non-streaming audio (default: offline.wav)\n"
        "  --out-stream PATH       Output WAV path for stream-concat audio (default: stream_concat.wav)\n"
        "  -t, --temp FLOAT        Sampling temperature (default: 0.8)\n"
        "  --max-tokens N          Max tokens to generate (default: 700)\n"
        "  --threads N             Number of CPU threads (default: 4)\n"
        "  -ngl N                  Number of GPU layers (default: 0)\n"
        "  --chunk-samples N       Streaming chunk size in samples (default: 4096)\n"
        "  --skip-llm              Treat --prompt as raw <|s_N|> token text\n"
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
        } else if (arg == "-c" || arg == "--codec") {
            if (++i >= argc) return false;
            args.codec_path = argv[i];
        } else if (arg == "-v" || arg == "--voice") {
            if (++i >= argc) return false;
            args.voice_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) return false;
            args.prompt = argv[i];
        } else if (arg == "--out-offline") {
            if (++i >= argc) return false;
            args.out_offline = argv[i];
        } else if (arg == "--out-stream") {
            if (++i >= argc) return false;
            args.out_stream = argv[i];
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
        } else if (arg == "--skip-llm") {
            args.skip_llm = true;
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

static void print_error_metrics(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    if (n == 0) {
        std::printf("compare.error: no overlap samples\n");
        return;
    }

    double mae = 0.0;
    double mse = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        const double ad = std::abs(d);
        mae += ad;
        mse += d * d;
        if (ad > max_abs) max_abs = ad;
    }
    mae /= (double) n;
    mse /= (double) n;
    const double rmse = std::sqrt(mse);
    std::printf("compare.samples=%zu\n", n);
    std::printf("compare.mae=%.8f\n", mae);
    std::printf("compare.rmse=%.8f\n", rmse);
    std::printf("compare.max_abs=%.8f\n", max_abs);
}

static int find_best_lag_rmse(const std::vector<float> & a, const std::vector<float> & b, int max_lag) {
    int best_lag = 0;
    double best_rmse = std::numeric_limits<double>::infinity();

    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        size_t a_start = 0;
        size_t b_start = 0;
        if (lag > 0) {
            a_start = (size_t) lag;
        } else if (lag < 0) {
            b_start = (size_t) (-lag);
        }
        if (a_start >= a.size() || b_start >= b.size()) continue;
        const size_t n = std::min(a.size() - a_start, b.size() - b_start);
        if (n < 1024) continue;

        double mse = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) a[a_start + i] - (double) b[b_start + i];
            mse += d * d;
        }
        mse /= (double) n;
        const double rmse = std::sqrt(mse);
        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_lag = lag;
        }
    }

    return best_lag;
}

static std::vector<float> apply_lag(const std::vector<float> & x, int lag, bool pad_front) {
    if (lag <= 0) return x;
    std::vector<float> out((size_t) lag, 0.0f);
    if (!pad_front) out.clear();
    out.insert(out.end(), x.begin(), x.end());
    return out;
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.prompt.empty()) {
        std::fprintf(stderr, "Error: --prompt is required\n");
        return 1;
    }
    if (args.codec_path.empty()) {
        std::fprintf(stderr, "Error: --codec is required\n");
        return 1;
    }
    if (args.voice_path.empty()) {
        std::fprintf(stderr, "Error: --voice is required\n");
        return 1;
    }
    if (!args.skip_llm && args.model_path.empty()) {
        std::fprintf(stderr, "Error: --model is required (or use --skip-llm)\n");
        return 1;
    }

    TestToSpeech::Config cfg;
    cfg.model_path = args.model_path;
    cfg.codec_path = args.codec_path;
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

    TestToSpeech::Options opt;
    opt.temperature = args.temperature;
    opt.max_tokens = args.max_tokens;
    opt.skip_llm = args.skip_llm;
    opt.apply_peak_normalization = false;

    std::string token_text;
    if (!tts.generate_token_text(args.prompt, opt, token_text)) {
        std::fprintf(stderr, "Error: failed to generate token text\n");
        return 1;
    }

    TestToSpeech::Options decode_opt = opt;
    decode_opt.skip_llm = true;

    std::vector<float> offline_audio;
    if (!tts.synthesize_to_vector(voice, token_text, offline_audio, decode_opt)) {
        std::fprintf(stderr, "Error: synthesize_to_vector failed\n");
        return 1;
    }

    std::vector<float> streamed_audio;
    bool ok = tts.synthesize_stream(
        voice,
        token_text,
        [&](const float * samples, size_t n, int, bool) -> bool {
            if (samples && n > 0) {
                streamed_audio.insert(streamed_audio.end(), samples, samples + n);
            }
            return true;
        },
        args.chunk_samples,
        decode_opt);

    if (!ok) {
        std::fprintf(stderr, "Error: synthesize_stream failed\n");
        return 1;
    }

    if (!wav_write(args.out_offline, offline_audio, tts.sample_rate())) {
        std::fprintf(stderr, "Error: failed to write %s\n", args.out_offline.c_str());
        return 1;
    }
    if (!wav_write(args.out_stream, streamed_audio, tts.sample_rate())) {
        std::fprintf(stderr, "Error: failed to write %s\n", args.out_stream.c_str());
        return 1;
    }

    std::printf("offline_samples=%zu\n", offline_audio.size());
    std::printf("stream_samples=%zu\n", streamed_audio.size());
    std::printf("sample_diff=%lld\n", (long long) streamed_audio.size() - (long long) offline_audio.size());
    print_error_metrics(offline_audio, streamed_audio);

    const int best_lag = find_best_lag_rmse(offline_audio, streamed_audio, 4096);
    std::printf("best_lag_samples=%d\n", best_lag);
    if (best_lag != 0) {
        std::vector<float> a = offline_audio;
        std::vector<float> b = streamed_audio;
        if (best_lag > 0) {
            a = apply_lag(a, best_lag, false);
        } else {
            b = apply_lag(b, -best_lag, false);
        }
        std::printf("aligned_metrics:\n");
        print_error_metrics(a, b);
    }

    return 0;
}
