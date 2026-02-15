#include "llama.h"

#include "miocodec.h"
#include "istft.h"
#include "token-parser.h"
#include "wav-writer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct miotts_params {
    std::string model_path;       // MioTTS LLM GGUF
    std::string codec_path;       // MioCodec GGUF
    std::string voice_path;       // Voice embedding .emb.gguf
    std::string output_path = "output.wav";
    std::string text;

    float temperature = 0.8f;
    int   max_tokens  = 700;
    int   n_threads   = 4;
    int   n_gpu_layers = 0;

    bool  dump_tensors = false;   // --dump-tensors: print MioCodec tensor names
    bool  skip_llm     = false;   // --skip-llm: provide raw <|s_N|> tokens directly
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  -m, --model PATH       MioTTS LLM GGUF model path (required)\n"
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
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

// Build the ChatML prompt for MioTTS.
static std::string build_prompt(const std::string & text) {
    return "<|startoftext|><|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

// Run LLM inference to generate speech tokens.
static std::string run_llm(const miotts_params & params) {
    fprintf(stderr, "Loading LLM model: %s\n", params.model_path.c_str());

    // Initialize llama
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", params.model_path.c_str());
        return "";
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 2048;
    ctx_params.n_batch   = 512;
    ctx_params.n_threads = params.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create llama context\n");
        llama_model_free(model);
        return "";
    }

    // Build and tokenize the prompt
    std::string prompt = build_prompt(params.text);
    fprintf(stderr, "Prompt: %s\n", prompt.c_str());

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return "";
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Prompt tokens: %d\n", n_tokens);

    // Create sampler chain: temperature + dist
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    // Initial batch with prompt tokens
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Initial decode failed\n");
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return "";
    }

    // Autoregressive generation loop
    std::string generated;
    int n_gen = 0;
    int cur_pos = n_tokens;

    // Stop token: <|im_end|> — typically token ID 7 for ChatML models
    // We'll check by looking up the token.
    llama_token eos_token = llama_vocab_eos(vocab);

    // Also try to find <|im_end|> token explicitly
    llama_token im_end_token = -1;
    {
        const char * im_end_str = "<|im_end|>";
        llama_token tmp[8];
        int n = llama_tokenize(vocab, im_end_str, strlen(im_end_str), tmp, 8, false, true);
        if (n == 1) {
            im_end_token = tmp[0];
        }
    }

    fprintf(stderr, "Generating speech tokens...\n");

    while (n_gen < params.max_tokens) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        // Check for stop tokens
        if (new_token == eos_token || new_token == im_end_token) {
            fprintf(stderr, "Stop token reached after %d tokens\n", n_gen);
            break;
        }

        // Detokenize
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            generated.append(buf, len);
        }

        // Prepare next batch
        batch.n_tokens = 1;
        batch.token[0]    = new_token;
        batch.pos[0]      = cur_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]   = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode failed at token %d\n", n_gen);
            break;
        }

        cur_pos++;
        n_gen++;
    }

    fprintf(stderr, "Generated %d tokens\n", n_gen);
    fprintf(stderr, "Raw output: %s\n", generated.c_str());

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return generated;
}

int main(int argc, char ** argv) {
    miotts_params params;

    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // --dump-tensors mode
    if (params.dump_tensors) {
        if (params.codec_path.empty()) {
            fprintf(stderr, "Error: --codec path required for --dump-tensors\n");
            return 1;
        }
        miocodec_print_tensors(params.codec_path);
        return 0;
    }

    // Validate required params
    if (params.text.empty()) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_usage(argv[0]);
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

    // ============================================================
    // Step 1: Generate speech tokens via LLM (or use raw input)
    // ============================================================
    std::string token_text;
    if (params.skip_llm) {
        token_text = params.text;
        fprintf(stderr, "Using raw token text (--skip-llm)\n");
    } else {
        if (params.model_path.empty()) {
            fprintf(stderr, "Error: --model path is required (or use --skip-llm)\n");
            return 1;
        }
        token_text = run_llm(params);
        if (token_text.empty()) {
            fprintf(stderr, "Error: LLM generation failed\n");
            return 1;
        }
    }

    // ============================================================
    // Step 2: Parse speech tokens
    // ============================================================
    std::vector<int> codes = parse_speech_tokens(token_text);
    fprintf(stderr, "Parsed %zu speech codes\n", codes.size());

    if (codes.empty()) {
        fprintf(stderr, "Error: no speech codes found in output\n");
        fprintf(stderr, "Token text: %s\n", token_text.c_str());
        return 1;
    }

    // Print first few codes
    fprintf(stderr, "First codes:");
    for (int i = 0; i < std::min((int)codes.size(), 10); i++) {
        fprintf(stderr, " %d", codes[i]);
    }
    fprintf(stderr, "%s\n", codes.size() > 10 ? " ..." : "");

    // ============================================================
    // Step 3: Load voice embedding
    // ============================================================
    std::vector<float> voice_emb = load_voice_embedding(params.voice_path);
    if (voice_emb.empty()) {
        fprintf(stderr, "Error: failed to load voice embedding\n");
        return 1;
    }

    // ============================================================
    // Step 4: MioCodec decode
    // ============================================================
    fprintf(stderr, "Loading MioCodec: %s\n", params.codec_path.c_str());
    miocodec_context * codec = miocodec_load(params.codec_path);
    if (!codec) {
        fprintf(stderr, "Error: failed to load MioCodec\n");
        return 1;
    }

    int sample_rate = miocodec_sample_rate(codec);
    int n_fft       = miocodec_n_fft(codec);
    int hop_length  = miocodec_hop_length(codec);
    int n_freq      = n_fft / 2 + 1;
    int spt         = miocodec_samples_per_token(codec);

    int n_frames = 0;
    int audio_length = (int)codes.size() * spt;

    fprintf(stderr, "Decoding %zu codes → ~%d audio samples (sr=%d, n_fft=%d, hop=%d)...\n",
            codes.size(), audio_length, sample_rate, n_fft, hop_length);

    std::vector<float> spec = miocodec_decode(
        codec, codes.data(), codes.size(),
        voice_emb.data(), audio_length, &n_frames);

    miocodec_free(codec);

    if (spec.empty()) {
        fprintf(stderr, "Error: MioCodec decode failed\n");
        return 1;
    }

    fprintf(stderr, "Spectrogram: %d frames x %d bins\n", n_frames, n_freq);

    // ============================================================
    // Step 5: iSTFT → waveform
    // ============================================================
    fprintf(stderr, "Running iSTFT (n_fft=%d, hop=%d)...\n", n_fft, hop_length);

    std::vector<float> audio = istft(spec.data(), n_frames, n_fft, hop_length, n_fft);

    if (audio.empty()) {
        fprintf(stderr, "Error: iSTFT failed\n");
        return 1;
    }

    fprintf(stderr, "Audio: %zu samples (%.2f seconds at %dHz)\n",
            audio.size(), audio.size() / (float)sample_rate, sample_rate);

    // Peak normalize to 0.95 to avoid clipping
    {
        float peak = 0.0f;
        for (float s : audio) peak = std::max(peak, std::abs(s));
        if (peak > 1e-8f) {
            float gain = 0.95f / peak;
            for (float & s : audio) s *= gain;
            fprintf(stderr, "Peak normalized: gain=%.4f (peak was %.2f)\n", gain, peak);
        }
    }

    // ============================================================
    // Step 6: Write WAV
    // ============================================================
    if (!wav_write(params.output_path, audio, sample_rate)) {
        fprintf(stderr, "Error: failed to write %s\n", params.output_path.c_str());
        return 1;
    }

    fprintf(stderr, "Saved: %s\n", params.output_path.c_str());
    return 0;
}
