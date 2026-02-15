#include "miocodec.h"
#include "test-to-speech.h"

#include <cstdio>
#include <cstdlib>
#include <string>

struct miotts_params {
    std::string model_path;
    std::string codec_path;
    std::string voice_path;
    std::string output_path = "output.wav";
    std::string text;

    float temperature = 0.8f;
    int max_tokens = 700;
    int n_threads = 4;
    int n_gpu_layers = 0;

    bool dump_tensors = false;
    bool skip_llm = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  -m, --model PATH       MioTTS LLM GGUF model path (required unless --skip-llm)\n"
            "  -c, --codec PATH       MioCodec GGUF model path (required)\n"
            "  -v, --voice PATH       Voice embedding .emb.gguf path (required)\n"
            "  -o, --output PATH      Output WAV file path (default: output.wav)\n"
            "  -p, --prompt TEXT      Text to synthesize (required)\n"
            "  -t, --temp FLOAT       Sampling temperature (default: 0.8)\n"
            "  --max-tokens N         Max tokens to generate (default: 700)\n"
            "  --threads N            Number of CPU threads (default: 4)\n"
            "  -ngl N                 Number of GPU layers (default: 0)\n"
            "  --dump-tensors         Print MioCodec tensor names and exit\n"
            "  --skip-llm             Treat --prompt as raw <|s_N|> token text\n"
            "  -h, --help             Show this help\n"
            "\n",
            prog);
}

static bool parse_args(int argc, char ** argv, miotts_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) return false;
            params.model_path = argv[i];
        } else if (arg == "-c" || arg == "--codec") {
            if (++i >= argc) return false;
            params.codec_path = argv[i];
        } else if (arg == "-v" || arg == "--voice") {
            if (++i >= argc) return false;
            params.voice_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) return false;
            params.output_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) return false;
            params.text = argv[i];
        } else if (arg == "-t" || arg == "--temp") {
            if (++i >= argc) return false;
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) return false;
            params.max_tokens = std::stoi(argv[i]);
        } else if (arg == "--threads") {
            if (++i >= argc) return false;
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-ngl") {
            if (++i >= argc) return false;
            params.n_gpu_layers = std::stoi(argv[i]);
        } else if (arg == "--dump-tensors") {
            params.dump_tensors = true;
        } else if (arg == "--skip-llm") {
            params.skip_llm = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    miotts_params params;

    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    if (params.dump_tensors) {
        if (params.codec_path.empty()) {
            fprintf(stderr, "Error: --codec path required for --dump-tensors\n");
            return 1;
        }
        miocodec_print_tensors(params.codec_path);
        return 0;
    }

    if (params.text.empty()) {
        fprintf(stderr, "Error: --prompt is required\n");
        return 1;
    }
    if (params.codec_path.empty()) {
        fprintf(stderr, "Error: --codec path is required\n");
        return 1;
    }
    if (params.voice_path.empty()) {
        fprintf(stderr, "Error: --voice path is required\n");
        return 1;
    }
    if (!params.skip_llm && params.model_path.empty()) {
        fprintf(stderr, "Error: --model path is required (or use --skip-llm)\n");
        return 1;
    }

    TestToSpeech::Config cfg;
    cfg.model_path = params.model_path;
    cfg.codec_path = params.codec_path;
    cfg.n_threads = params.n_threads;
    cfg.n_gpu_layers = params.n_gpu_layers;
    cfg.temperature = params.temperature;
    cfg.max_tokens = params.max_tokens;

    TestToSpeech tts(cfg);
    if (!tts.is_ready()) {
        fprintf(stderr, "Error: failed to initialize TestToSpeech\n");
        return 1;
    }

    VoiceModel voice;
    if (!voice.load_from_file(params.voice_path)) {
        fprintf(stderr, "Error: failed to load voice model: %s\n", params.voice_path.c_str());
        return 1;
    }

    TestToSpeech::Options opt;
    opt.temperature = params.temperature;
    opt.max_tokens = params.max_tokens;
    opt.skip_llm = params.skip_llm;

    if (!tts.synthesize_to_file(voice, params.text, params.output_path, opt)) {
        fprintf(stderr, "Error: synthesis failed\n");
        return 1;
    }

    fprintf(stderr, "Saved: %s\n", params.output_path.c_str());
    return 0;
}
