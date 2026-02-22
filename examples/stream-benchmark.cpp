#include "test-to-speech.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

struct args_t {
    std::string model_path;
    std::string model_onnx_path;
    std::string tokenizer_path;
    std::string codec_path;
    std::string codec_type = "ggml";
    std::string voice_path;
    std::string prompt;
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
        "Benchmark streaming TTS processing speed (no playback).\n"
        "\n"
        "Codec backend:\n"
        "  -gguf PATH              MioCodec GGUF model (44.1kHz)\n"
        "  -onnx PATH              MioCodec ONNX decoder (24kHz)\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH        MioTTS LLM GGUF model path\n"
        "  --model-onnx PATH       MioTTS LLM ONNX model path\n"
        "  --tokenizer PATH        GGUF tokenizer model (required with --model-onnx)\n"
        "  -v, --voice PATH        Voice embedding (.emb.gguf or .bin)\n"
        "  -p, --prompt TEXT       Text to synthesize (use '-' for stdin)\n"
        "  -t, --temp FLOAT        Sampling temperature (default: 0.8)\n"
        "  --max-tokens N          Max tokens to generate (default: 700)\n"
        "  --threads N             Number of CPU threads (default: 4)\n"
        "  -ngl N                  Number of GPU layers (default: 0)\n"
        "  --chunk-samples N       Streaming chunk size in samples (default: 4096)\n"
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
        } else if (arg == "--model-onnx") {
            if (++i >= argc) return false;
            args.model_onnx_path = argv[i];
        } else if (arg == "--tokenizer") {
            if (++i >= argc) return false;
            args.tokenizer_path = argv[i];
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
    if (!args.model_path.empty() && !args.model_onnx_path.empty()) {
        std::fprintf(stderr, "Error: use either -m/--model or --model-onnx, not both\n");
        return 1;
    }
    if (!args.model_onnx_path.empty() && args.tokenizer_path.empty()) {
        std::fprintf(stderr, "Error: --tokenizer is required with --model-onnx\n");
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

    bool skip_llm = args.model_path.empty() && args.model_onnx_path.empty();

    TestToSpeech::Config cfg;
    cfg.model_path = args.model_path;
    cfg.model_onnx_path = args.model_onnx_path;
    cfg.tokenizer_path = args.tokenizer_path;
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

    TestToSpeech::Options opt;
    opt.temperature = args.temperature;
    opt.max_tokens = args.max_tokens;
    opt.skip_llm = skip_llm;

    TestToSpeech::StreamProfile prof;
    bool ok = tts.synthesize_stream_profiled(
        voice,
        args.prompt,
        [](const float *, size_t, int, bool) { return true; },
        args.chunk_samples,
        opt,
        prof);

    if (!ok) {
        std::fprintf(stderr, "Error: streaming benchmark failed\n");
        return 1;
    }

    const double audio_sec = prof.emitted_samples > 0
        ? (double) prof.emitted_samples / (double) tts.sample_rate()
        : 0.0;
    const double rtf = audio_sec > 0.0 ? (prof.total_sec / audio_sec) : 0.0;
    const double x_realtime = prof.total_sec > 0.0 ? (audio_sec / prof.total_sec) : 0.0;
    const double total = prof.total_sec > 1e-9 ? prof.total_sec : 1e-9;

    std::printf("stream_bench.total_sec=%.6f\n", prof.total_sec);
    std::printf("stream_bench.audio_sec=%.6f\n", audio_sec);
    std::printf("stream_bench.rtf=%.6f\n", rtf);
    std::printf("stream_bench.x_realtime=%.6f\n", x_realtime);
    std::printf("stream_bench.llm_tokens=%d\n", prof.llm_tokens);
    std::printf("stream_bench.decode_calls=%d\n", prof.decode_calls);
    std::printf("stream_bench.decoded_codes=%zu\n", prof.decoded_codes);
    std::printf("stream_bench.emitted_samples=%zu\n", prof.emitted_samples);
    std::printf("stream_bench.stage.llm_sec=%.6f (%.2f%%)\n", prof.llm_sec, 100.0 * prof.llm_sec / total);
    std::printf("stream_bench.stage.codec_sec=%.6f (%.2f%%)\n", prof.codec_sec, 100.0 * prof.codec_sec / total);
    std::printf("stream_bench.stage.istft_sec=%.6f (%.2f%%)\n", prof.istft_sec, 100.0 * prof.istft_sec / total);
    std::printf("stream_bench.stage.callback_sec=%.6f (%.2f%%)\n", prof.callback_sec, 100.0 * prof.callback_sec / total);
    return 0;
}
