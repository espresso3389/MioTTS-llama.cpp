#include "miocodec.h"
#include "test-to-speech.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct miotts_params {
    std::string model_path;
    std::string model_onnx_path;
    std::string tokenizer_path;
    std::string codec_path;
    std::string codec_type = "ggml"; // "ggml" or "onnx"
    std::string voice_path;
    std::string output_path = "output.wav";
    std::string text;

    float temperature = 0.8f;
    int max_tokens = 700;
    int n_threads = 4;
    int n_gpu_layers = 0;

    bool dump_tensors = false;
    bool tokens_only = false;
    bool serve_stdin = false;
    bool timing = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Modes:\n"
            "  1) Text-to-speech:  (-m llm.gguf | --model-onnx llm.onnx --tokenizer tok.gguf) -gguf/-onnx codec -v voice -p \"text\"\n"
            "  2) Tokens-only:     (-m llm.gguf | --model-onnx llm.onnx --tokenizer tok.gguf) --tokens-only -p \"text\"\n"
            "  3) Decode-only:     -gguf/-onnx codec -v voice -p \"<|s_N|>...\"\n"
            "\n"
            "Codec backend:\n"
            "  -gguf PATH             MioCodec GGUF model (44.1kHz)\n"
            "  -onnx PATH             MioCodec ONNX decoder (24kHz)\n"
            "\n"
            "Options:\n"
            "  -m, --model PATH       MioTTS LLM GGUF model path\n"
            "  --model-onnx PATH      MioTTS LLM ONNX model path\n"
            "  --tokenizer PATH       GGUF tokenizer model (required with --model-onnx)\n"
            "  -v, --voice PATH       Voice embedding (.emb.gguf or .bin)\n"
            "  -o, --output PATH      Output WAV file path (default: output.wav)\n"
            "  -p, --prompt TEXT      Text to synthesize (use '-' to read from stdin)\n"
            "  -t, --temp FLOAT       Sampling temperature (default: 0.8)\n"
            "  --max-tokens N         Max tokens to generate (default: 700)\n"
            "  --threads N            Number of CPU threads (default: 4)\n"
            "  -ngl N                 Number of GPU layers (default: 0)\n"
            "  --tokens-only          Only run LLM, print tokens to stdout\n"
            "  --serve-stdin          Keep process alive and synthesize one prompt per stdin line\n"
            "  --timing               Print startup and per-request timing stats to stderr\n"
            "  --dump-tensors         Print MioCodec tensor names and exit\n"
            "  -h, --help             Show this help\n"
            "\n"
            "Examples:\n"
            "  %s -m llm.gguf -onnx decoder.onnx -v voice.bin -p \"hello\"\n"
            "  %s --model-onnx llm.onnx --tokenizer tok.gguf -onnx decoder.onnx -v voice.bin -p \"hello\"\n"
            "  %s -m llm.gguf --tokens-only -p \"hello\" > tokens.txt\n"
            "  %s -onnx decoder.onnx -v voice.bin -p - < tokens.txt\n"
            "\n",
            prog, prog, prog, prog, prog);
}

static bool parse_args(int argc, char ** argv, miotts_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) return false;
            params.model_path = argv[i];
        } else if (arg == "--model-onnx") {
            if (++i >= argc) return false;
            params.model_onnx_path = argv[i];
        } else if (arg == "--tokenizer") {
            if (++i >= argc) return false;
            params.tokenizer_path = argv[i];
        } else if (arg == "-gguf") {
            if (++i >= argc) return false;
            params.codec_path = argv[i];
            params.codec_type = "ggml";
        } else if (arg == "-onnx") {
            if (++i >= argc) return false;
            params.codec_path = argv[i];
            params.codec_type = "onnx";
        } else if (arg == "-c" || arg == "--codec") {
            if (++i >= argc) return false;
            params.codec_path = argv[i];
        } else if (arg == "--codec-type") {
            if (++i >= argc) return false;
            params.codec_type = argv[i];
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
        } else if (arg == "--tokens-only") {
            params.tokens_only = true;
        } else if (arg == "--serve-stdin") {
            params.serve_stdin = true;
        } else if (arg == "--timing") {
            params.timing = true;
        } else if (arg == "--skip-llm") {
            // kept for backward compat, no-op (auto-detected from -m absence)
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

    // Read prompt from stdin if "-p -"
    if (params.text == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        params.text = ss.str();
        // Strip trailing newline(s)
        while (!params.text.empty() && (params.text.back() == '\n' || params.text.back() == '\r')) {
            params.text.pop_back();
        }
    }

    if (params.dump_tensors) {
        if (params.codec_path.empty()) {
            fprintf(stderr, "Error: codec path required for --dump-tensors\n");
            return 1;
        }
        miocodec_print_tensors(params.codec_path);
        return 0;
    }

    if (params.text.empty() && !params.serve_stdin) {
        fprintf(stderr, "Error: -p is required\n");
        return 1;
    }

    if (!params.model_path.empty() && !params.model_onnx_path.empty()) {
        fprintf(stderr, "Error: use either -m/--model or --model-onnx, not both\n");
        return 1;
    }
    if (!params.model_onnx_path.empty() && params.tokenizer_path.empty()) {
        fprintf(stderr, "Error: --tokenizer is required with --model-onnx\n");
        return 1;
    }

    // --tokens-only: just run LLM and print tokens to stdout
    if (params.tokens_only) {
        if (params.model_path.empty() && params.model_onnx_path.empty()) {
            fprintf(stderr, "Error: LLM model is required for --tokens-only (-m or --model-onnx)\n");
            return 1;
        }

        TestToSpeech::Config cfg;
        cfg.model_path = params.model_path;
        cfg.model_onnx_path = params.model_onnx_path;
        cfg.tokenizer_path = params.tokenizer_path;
        cfg.n_threads = params.n_threads;
        cfg.n_gpu_layers = params.n_gpu_layers;
        cfg.temperature = params.temperature;
        cfg.max_tokens = params.max_tokens;

        TestToSpeech tts(cfg);
        if (!tts.is_ready()) {
            fprintf(stderr, "Error: failed to initialize TestToSpeech\n");
            return 1;
        }

        TestToSpeech::Options opt;
        opt.temperature = params.temperature;
        opt.max_tokens = params.max_tokens;

        std::string token_text;
        if (!tts.generate_token_text(params.text, opt, token_text)) {
            fprintf(stderr, "Error: token generation failed\n");
            return 1;
        }

        // Print to stdout (all logs go to stderr)
        printf("%s\n", token_text.c_str());
        return 0;
    }

    // Decode mode: codec + voice required
    if (params.codec_path.empty()) {
        fprintf(stderr, "Error: codec path is required (-gguf or -onnx)\n");
        return 1;
    }
    if (params.voice_path.empty()) {
        fprintf(stderr, "Error: -v is required\n");
        return 1;
    }

    bool skip_llm = params.model_path.empty() && params.model_onnx_path.empty();

    TestToSpeech::Config cfg;
    cfg.model_path = params.model_path;
    cfg.model_onnx_path = params.model_onnx_path;
    cfg.tokenizer_path = params.tokenizer_path;
    cfg.codec_path = params.codec_path;
    cfg.codec_type = params.codec_type;
    cfg.n_threads = params.n_threads;
    cfg.n_gpu_layers = params.n_gpu_layers;
    cfg.temperature = params.temperature;
    cfg.max_tokens = params.max_tokens;

    const auto t_startup0 = std::chrono::steady_clock::now();
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
    const auto t_startup1 = std::chrono::steady_clock::now();
    const double startup_ms = std::chrono::duration<double, std::milli>(t_startup1 - t_startup0).count();

    TestToSpeech::Options opt;
    opt.temperature = params.temperature;
    opt.max_tokens = params.max_tokens;
    opt.skip_llm = skip_llm;

    if (params.serve_stdin) {
        if (params.timing) {
            fprintf(stderr, "TIMING startup_ms=%.3f\n", startup_ms);
        }

        int request_idx = 0;
        bool ok = true;
        auto run_one = [&](const std::string & prompt) -> bool {
            std::vector<float> audio;
            const auto t_req0 = std::chrono::steady_clock::now();
            const bool req_ok = tts.synthesize_to_vector(voice, prompt, audio, opt);
            const auto t_req1 = std::chrono::steady_clock::now();
            const double req_ms = std::chrono::duration<double, std::milli>(t_req1 - t_req0).count();

            request_idx++;
            if (params.timing) {
                fprintf(stderr, "TIMING request_%d_ms=%.3f samples=%zu ok=%d\n",
                        request_idx, req_ms, audio.size(), req_ok ? 1 : 0);
            } else {
                fprintf(stderr, "Request %d: %s (%zu samples)\n",
                        request_idx, req_ok ? "ok" : "failed", audio.size());
            }
            return req_ok;
        };

        if (!params.text.empty()) {
            ok = run_one(params.text);
        }

        std::string line;
        while (ok && std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            ok = run_one(line);
        }
        return ok ? 0 : 1;
    }

    if (params.timing) {
        fprintf(stderr, "TIMING startup_ms=%.3f\n", startup_ms);
    }
    const auto t_req0 = std::chrono::steady_clock::now();
    if (!tts.synthesize_to_file(voice, params.text, params.output_path, opt)) {
        fprintf(stderr, "Error: synthesis failed\n");
        return 1;
    }
    const auto t_req1 = std::chrono::steady_clock::now();
    if (params.timing) {
        const double req_ms = std::chrono::duration<double, std::milli>(t_req1 - t_req0).count();
        fprintf(stderr, "TIMING request_ms=%.3f\n", req_ms);
    }
    fprintf(stderr, "Saved: %s\n", params.output_path.c_str());
    return 0;
}
