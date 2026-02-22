// Benchmark codec decode performance: GGML vs ONNX backends.
//
// Usage:
//   ./miotts-codec-benchmark \
//       --ggml-codec codec.gguf --ggml-voice voice.emb.gguf \
//       --onnx-codec miocodec_decoder.onnx --onnx-voice voice.emb.bin \
//       --codes "<|s_0|><|s_1|>..." --iterations 5
//
// Or with random codes:
//   ./miotts-codec-benchmark \
//       --ggml-codec codec.gguf --ggml-voice voice.emb.gguf \
//       --onnx-codec miocodec_decoder.onnx --onnx-voice voice.emb.bin \
//       --n-codes 100 --iterations 5

#include "codec-backend.h"
#include "codec-backend-ggml.h"
#ifdef MIOTTS_HAS_ONNX
#include "codec-backend-onnx.h"
#endif
#include "miocodec.h"
#include "token-parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct args_t {
    std::string ggml_codec_path;
    std::string ggml_voice_path;
    std::string onnx_codec_path;
    std::string onnx_voice_path;
    std::string codes_text;   // raw "<|s_N|>..." token text
    int n_codes = 0;          // generate random codes of this length
    int iterations = 5;
    int warmup = 1;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Benchmark MioCodec decode performance (GGML vs ONNX).\n"
        "Specify at least one backend. Both can be specified for comparison.\n"
        "\n"
        "Options:\n"
        "  --ggml-codec PATH    GGML MioCodec GGUF path\n"
        "  --ggml-voice PATH    Voice embedding for GGML (.emb.gguf)\n"
        "  --onnx-codec PATH    ONNX MioCodec decoder path\n"
        "  --onnx-voice PATH    Voice embedding for ONNX (.bin or .emb.gguf)\n"
        "  --codes TEXT         Speech codes as <|s_N|> token text\n"
        "  --n-codes N          Generate N random codes (0..12799)\n"
        "  --iterations N       Number of timed iterations (default: 5)\n"
        "  --warmup N           Number of warmup iterations (default: 1)\n"
        "  -h, --help           Show this help\n"
        "\n",
        prog);
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ggml-codec") {
            if (++i >= argc) return false;
            args.ggml_codec_path = argv[i];
        } else if (arg == "--ggml-voice") {
            if (++i >= argc) return false;
            args.ggml_voice_path = argv[i];
        } else if (arg == "--onnx-codec") {
            if (++i >= argc) return false;
            args.onnx_codec_path = argv[i];
        } else if (arg == "--onnx-voice") {
            if (++i >= argc) return false;
            args.onnx_voice_path = argv[i];
        } else if (arg == "--codes") {
            if (++i >= argc) return false;
            args.codes_text = argv[i];
        } else if (arg == "--n-codes") {
            if (++i >= argc) return false;
            args.n_codes = std::stoi(argv[i]);
        } else if (arg == "--iterations") {
            if (++i >= argc) return false;
            args.iterations = std::stoi(argv[i]);
        } else if (arg == "--warmup") {
            if (++i >= argc) return false;
            args.warmup = std::stoi(argv[i]);
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

struct bench_result {
    std::string label;
    int n_codes = 0;
    int sample_rate = 0;
    size_t output_samples = 0;
    std::vector<double> times_sec;

    double mean() const {
        if (times_sec.empty()) return 0.0;
        return std::accumulate(times_sec.begin(), times_sec.end(), 0.0) / times_sec.size();
    }
    double min_time() const {
        if (times_sec.empty()) return 0.0;
        return *std::min_element(times_sec.begin(), times_sec.end());
    }
    double max_time() const {
        if (times_sec.empty()) return 0.0;
        return *std::max_element(times_sec.begin(), times_sec.end());
    }
    double median() const {
        if (times_sec.empty()) return 0.0;
        std::vector<double> sorted = times_sec;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        return (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2];
    }
    double audio_sec() const {
        return sample_rate > 0 ? (double)output_samples / (double)sample_rate : 0.0;
    }
    double rtf() const {
        double a = audio_sec();
        double m = median();
        return (a > 0.0 && m > 0.0) ? m / a : 0.0;
    }
    double x_realtime() const {
        double r = rtf();
        return r > 0.0 ? 1.0 / r : 0.0;
    }
};

static bench_result run_benchmark(CodecBackend & backend,
                                  const std::string & label,
                                  const std::vector<float> & voice_emb,
                                  const std::vector<int> & codes,
                                  int warmup, int iterations) {
    bench_result result;
    result.label = label;
    result.n_codes = static_cast<int>(codes.size());
    result.sample_rate = backend.sample_rate();

    std::vector<float> audio;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        std::fprintf(stderr, "  [%s] warmup %d/%d ...\n", label.c_str(), i + 1, warmup);
        if (!backend.decode_to_audio(codes.data(), result.n_codes, voice_emb.data(), audio)) {
            std::fprintf(stderr, "  [%s] ERROR: decode failed during warmup\n", label.c_str());
            return result;
        }
    }
    result.output_samples = audio.size();

    // Timed iterations
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        backend.decode_to_audio(codes.data(), result.n_codes, voice_emb.data(), audio);
        const auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        result.times_sec.push_back(elapsed);
        std::fprintf(stderr, "  [%s] iter %d/%d: %.3f ms\n",
                     label.c_str(), i + 1, iterations, elapsed * 1000.0);
    }

    return result;
}

static void print_result(const bench_result & r) {
    std::printf("\n=== %s ===\n", r.label.c_str());
    std::printf("  codes:          %d\n", r.n_codes);
    std::printf("  output_samples: %zu\n", r.output_samples);
    std::printf("  sample_rate:    %d Hz\n", r.sample_rate);
    std::printf("  audio_duration: %.3f s\n", r.audio_sec());
    std::printf("  iterations:     %zu\n", r.times_sec.size());
    std::printf("  median:         %.3f ms\n", r.median() * 1000.0);
    std::printf("  mean:           %.3f ms\n", r.mean() * 1000.0);
    std::printf("  min:            %.3f ms\n", r.min_time() * 1000.0);
    std::printf("  max:            %.3f ms\n", r.max_time() * 1000.0);
    std::printf("  RTF:            %.4f\n", r.rtf());
    std::printf("  x_realtime:     %.2fx\n", r.x_realtime());
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    const bool have_ggml = !args.ggml_codec_path.empty();
    const bool have_onnx = !args.onnx_codec_path.empty();

    if (!have_ggml && !have_onnx) {
        std::fprintf(stderr, "Error: specify at least one backend "
                     "(--ggml-codec and/or --onnx-codec)\n");
        return 1;
    }
    if (have_ggml && args.ggml_voice_path.empty()) {
        std::fprintf(stderr, "Error: --ggml-voice required with --ggml-codec\n");
        return 1;
    }
    if (have_onnx && args.onnx_voice_path.empty()) {
        std::fprintf(stderr, "Error: --onnx-voice required with --onnx-codec\n");
        return 1;
    }
    if (args.codes_text.empty() && args.n_codes <= 0) {
        std::fprintf(stderr, "Error: specify --codes or --n-codes\n");
        return 1;
    }

    // Build codes array
    std::vector<int> codes;
    if (!args.codes_text.empty()) {
        codes = parse_speech_tokens(args.codes_text);
        if (codes.empty()) {
            std::fprintf(stderr, "Error: no speech codes parsed from --codes text\n");
            return 1;
        }
    } else {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 12799);
        codes.resize(args.n_codes);
        for (int & c : codes) {
            c = dist(rng);
        }
    }
    std::fprintf(stderr, "Codes: %zu tokens\n", codes.size());

    std::vector<bench_result> results;

    // --- GGML benchmark ---
    if (have_ggml) {
        std::fprintf(stderr, "\nLoading GGML backend: %s\n", args.ggml_codec_path.c_str());
        GgmlCodecBackend ggml_backend(args.ggml_codec_path);
        if (!ggml_backend.is_valid()) {
            std::fprintf(stderr, "Error: failed to load GGML codec\n");
            return 1;
        }

        std::vector<float> ggml_emb = load_voice_embedding(args.ggml_voice_path);
        if (ggml_emb.empty()) {
            std::fprintf(stderr, "Error: failed to load GGML voice embedding\n");
            return 1;
        }

        results.push_back(run_benchmark(
            ggml_backend, "GGML", ggml_emb, codes, args.warmup, args.iterations));
    }

    // --- ONNX benchmark ---
    if (have_onnx) {
#ifdef MIOTTS_HAS_ONNX
        std::fprintf(stderr, "\nLoading ONNX backend: %s\n", args.onnx_codec_path.c_str());
        OnnxCodecBackend onnx_backend(args.onnx_codec_path);
        if (!onnx_backend.is_valid()) {
            std::fprintf(stderr, "Error: failed to load ONNX codec\n");
            return 1;
        }

        std::vector<float> onnx_emb = load_voice_embedding(args.onnx_voice_path);
        if (onnx_emb.empty()) {
            std::fprintf(stderr, "Error: failed to load ONNX voice embedding\n");
            return 1;
        }

        results.push_back(run_benchmark(
            onnx_backend, "ONNX", onnx_emb, codes, args.warmup, args.iterations));
#else
        std::fprintf(stderr, "Error: ONNX support not compiled in (build with -DMIOTTS_ONNX=ON)\n");
        return 1;
#endif
    }

    // --- Print results ---
    std::printf("\n========================================\n");
    std::printf("Codec Benchmark Results (%zu codes)\n", codes.size());
    std::printf("========================================\n");

    for (const auto & r : results) {
        print_result(r);
    }

    // --- Comparison ---
    if (results.size() == 2) {
        const auto & a = results[0];
        const auto & b = results[1];
        double speedup = (b.median() > 0.0) ? a.median() / b.median() : 0.0;
        std::printf("\n--- Comparison ---\n");
        std::printf("  %s median: %.3f ms\n", a.label.c_str(), a.median() * 1000.0);
        std::printf("  %s median: %.3f ms\n", b.label.c_str(), b.median() * 1000.0);
        if (speedup > 1.0) {
            std::printf("  %s is %.2fx faster than %s\n",
                        b.label.c_str(), speedup, a.label.c_str());
        } else if (speedup > 0.0) {
            std::printf("  %s is %.2fx faster than %s\n",
                        a.label.c_str(), 1.0 / speedup, b.label.c_str());
        }
    }

    std::printf("\n");
    return 0;
}
