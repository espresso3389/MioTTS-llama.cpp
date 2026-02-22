#include "test-to-speech.h"

#include "codec-backend.h"
#include "codec-backend-ggml.h"
#ifdef MIOTTS_HAS_ONNX
#include "codec-backend-onnx.h"
#include "llm-backend-onnx.h"
#endif
#include "text-normalize.h"
#include "token-parser.h"
#include "wav-writer.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

bool VoiceModel::load_from_file(const std::string & path) {
    std::vector<float> emb = load_voice_embedding(path);
    if (emb.empty()) {
        return false;
    }
    path_ = path;
    embedding_ = std::move(emb);
    return true;
}

bool VoiceModel::is_ready() const {
    return !embedding_.empty();
}

const std::vector<float> & VoiceModel::embedding() const {
    return embedding_;
}

const std::string & VoiceModel::path() const {
    return path_;
}

TestToSpeech::TestToSpeech(const Config & config) : config_(config) {
    llama_backend_init();

    if (!config_.model_path.empty()) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.n_gpu_layers;
        model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
        if (!model_) {
            fprintf(stderr, "TestToSpeech: failed to load LLM model: %s\n", config_.model_path.c_str());
            return;
        }
        vocab_ = llama_model_get_vocab(model_);
    } else if (!config_.model_onnx_path.empty()) {
#ifdef MIOTTS_HAS_ONNX
        if (config_.tokenizer_path.empty()) {
            fprintf(stderr, "TestToSpeech: --tokenizer is required with --model-onnx\n");
            return;
        }

        llama_model_params tok_params = llama_model_default_params();
        tok_params.n_gpu_layers = 0;
        tokenizer_model_ = llama_model_load_from_file(config_.tokenizer_path.c_str(), tok_params);
        if (!tokenizer_model_) {
            fprintf(stderr, "TestToSpeech: failed to load tokenizer GGUF model: %s\n", config_.tokenizer_path.c_str());
            return;
        }
        vocab_ = llama_model_get_vocab(tokenizer_model_);

        auto backend = std::make_unique<OnnxLlmBackend>(config_.model_onnx_path, config_.n_threads);
        if (!backend->is_valid()) {
            fprintf(stderr, "TestToSpeech: failed to load ONNX LLM model: %s\n", config_.model_onnx_path.c_str());
            return;
        }
        onnx_llm_ = std::move(backend);
#else
        fprintf(stderr, "TestToSpeech: ONNX support not compiled in\n");
        return;
#endif
    }

    if (!config_.codec_path.empty()) {
        if (config_.codec_type == "onnx") {
#ifdef MIOTTS_HAS_ONNX
            auto backend = std::make_unique<OnnxCodecBackend>(config_.codec_path);
            if (!backend->is_valid()) {
                fprintf(stderr, "TestToSpeech: failed to load ONNX codec: %s\n", config_.codec_path.c_str());
                return;
            }
            codec_ = std::move(backend);
#else
            fprintf(stderr, "TestToSpeech: ONNX support not compiled in (build with -DMIOTTS_ONNX=ON)\n");
            return;
#endif
        } else {
            auto backend = std::make_unique<GgmlCodecBackend>(config_.codec_path);
            if (!backend->is_valid()) {
                fprintf(stderr, "TestToSpeech: failed to load GGML codec: %s\n", config_.codec_path.c_str());
                return;
            }
            codec_ = std::move(backend);
        }

        sample_rate_ = codec_->sample_rate();
        samples_per_token_ = codec_->samples_per_token();
    }
}

TestToSpeech::~TestToSpeech() {
    codec_.reset();
#ifdef MIOTTS_HAS_ONNX
    onnx_llm_.reset();
#endif
    if (tokenizer_model_) {
        llama_model_free(tokenizer_model_);
        tokenizer_model_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
}

bool TestToSpeech::is_ready() const {
#ifdef MIOTTS_HAS_ONNX
    return model_ != nullptr || onnx_llm_ != nullptr || codec_ != nullptr;
#else
    return model_ != nullptr || codec_ != nullptr;
#endif
}

static int sample_token_from_logits(const std::vector<float> & logits, float temperature,
                                    int vocab_limit, std::mt19937 & rng) {
    const int n = std::min(vocab_limit, static_cast<int>(logits.size()));
    if (n <= 0) {
        return -1;
    }

    if (temperature <= 1e-5f) {
        int best = 0;
        for (int i = 1; i < n; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        max_logit = std::max(max_logit, logits[i] / temperature);
    }

    std::vector<float> probs(n, 0.0f);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float x = (logits[i] / temperature) - max_logit;
        probs[i] = std::exp(x);
        sum += probs[i];
    }
    if (sum <= 0.0) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        probs[i] = static_cast<float>(probs[i] / sum);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Helper: decode codes to audio via CodecBackend with optional timing.
static bool decode_codes_via_backend(CodecBackend & backend,
                                     const std::vector<float> & voice_emb,
                                     const int * codes, int n_codes,
                                     std::vector<float> & out_audio,
                                     double * out_decode_sec = nullptr,
                                     bool apply_peak_normalization = true) {
    if (n_codes <= 0) {
        out_audio.clear();
        return true;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bool ok = backend.decode_to_audio(codes, n_codes, voice_emb.data(), out_audio);
    const auto t1 = std::chrono::steady_clock::now();

    if (!ok) {
        fprintf(stderr, "TestToSpeech: codec decode failed\n");
        return false;
    }
    if (out_decode_sec) {
        *out_decode_sec += std::chrono::duration<double>(t1 - t0).count();
    }

    if (apply_peak_normalization) {
        float peak = 0.0f;
        for (float s : out_audio) {
            peak = std::max(peak, std::abs(s));
        }
        if (peak > 1e-8f) {
            float gain = 0.95f / peak;
            for (float & s : out_audio) {
                s *= gain;
            }
        }
    }

    return true;
}

int TestToSpeech::sample_rate() const {
    return sample_rate_;
}

std::string TestToSpeech::build_prompt(const std::string & text) {
    return "<|startoftext|><|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string TestToSpeech::run_llm(const std::string & text, const Options & options) {
#ifdef MIOTTS_HAS_ONNX
    if ((!model_ && !onnx_llm_) || !vocab_) {
#else
    if (!model_ || !vocab_) {
#endif
        fprintf(stderr, "TestToSpeech: LLM model is not loaded\n");
        return "";
    }

    const float temperature = options.temperature >= 0.0f ? options.temperature : config_.temperature;
    const int max_tokens = options.max_tokens > 0 ? options.max_tokens : config_.max_tokens;

    std::string normalized_text = normalize_tts_text(text);
    std::string prompt = build_prompt(normalized_text);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "TestToSpeech: tokenization failed\n");
        return "";
    }
    tokens.resize(n_tokens);

#ifdef MIOTTS_HAS_ONNX
    if (onnx_llm_) {
        llama_token eos_token = llama_vocab_eos(vocab_);
        llama_token im_end_token = -1;
        {
            const char * im_end_str = "<|im_end|>";
            llama_token tmp[8];
            int n = llama_tokenize(vocab_, im_end_str, std::strlen(im_end_str), tmp, 8, false, true);
            if (n == 1) {
                im_end_token = tmp[0];
            }
        }

        std::vector<int32_t> ids(tokens.begin(), tokens.end());
        std::vector<float> logits;
        std::string generated;
        std::mt19937 rng(42);

        const int n_vocab = llama_vocab_n_tokens(vocab_);
        int n_gen = 0;
        while (n_gen < max_tokens) {
            if (!onnx_llm_->forward_logits(ids, logits)) {
                return "";
            }

            int new_token = sample_token_from_logits(logits, temperature, n_vocab, rng);
            if (new_token < 0) {
                return "";
            }

            if (new_token == eos_token || new_token == im_end_token) {
                break;
            }

            char buf[256];
            int len = llama_token_to_piece(vocab_, new_token, buf, sizeof(buf), 0, true);
            if (len > 0) {
                generated.append(buf, len);
            }

            ids.push_back(new_token);
            n_gen++;
        }
        return generated;
    }
#endif

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = config_.n_threads;

    llama_context * ctx = llama_init_from_model(model_, ctx_params);
    if (!ctx) {
        fprintf(stderr, "TestToSpeech: failed to create llama context\n");
        return "";
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "TestToSpeech: initial decode failed\n");
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        llama_free(ctx);
        return "";
    }

    llama_token eos_token = llama_vocab_eos(vocab_);
    llama_token im_end_token = -1;
    {
        const char * im_end_str = "<|im_end|>";
        llama_token tmp[8];
        int n = llama_tokenize(vocab_, im_end_str, std::strlen(im_end_str), tmp, 8, false, true);
        if (n == 1) {
            im_end_token = tmp[0];
        }
    }

    std::string generated;
    int cur_pos = n_tokens;
    int n_gen = 0;
    while (n_gen < max_tokens) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (new_token == eos_token || new_token == im_end_token) {
            break;
        }

        char buf[256];
        int len = llama_token_to_piece(vocab_, new_token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            generated.append(buf, len);
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = cur_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "TestToSpeech: decode failed at token %d\n", n_gen);
            break;
        }

        cur_pos++;
        n_gen++;
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);

    return generated;
}

bool TestToSpeech::decode_tokens_to_audio(const VoiceModel & voice,
                                          const std::string & token_text,
                                          std::vector<float> & out_audio,
                                          bool apply_peak_normalization) {
    if (!voice.is_ready()) {
        fprintf(stderr, "TestToSpeech: voice model is not ready\n");
        return false;
    }

    const std::vector<int> codes = parse_speech_tokens(token_text);
    if (codes.empty()) {
        fprintf(stderr, "TestToSpeech: no speech codes parsed from text\n");
        return false;
    }

    return decode_codes_via_backend(*codec_, voice.embedding(),
                                    codes.data(), static_cast<int>(codes.size()),
                                    out_audio, nullptr, apply_peak_normalization);
}

bool TestToSpeech::synthesize_to_vector(const VoiceModel & voice,
                                        const std::string & text,
                                        std::vector<float> & out_audio,
                                        const Options & options) {
    if (!is_ready()) {
        fprintf(stderr, "TestToSpeech: not ready\n");
        return false;
    }

    std::string token_text;
    if (!generate_token_text(text, options, token_text)) {
        return false;
    }

    return decode_tokens_to_audio(voice, token_text, out_audio, options.apply_peak_normalization);
}

bool TestToSpeech::synthesize_to_file(const VoiceModel & voice,
                                      const std::string & text,
                                      const std::string & output_path,
                                      const Options & options) {
    std::vector<float> audio;
    if (!synthesize_to_vector(voice, text, audio, options)) {
        return false;
    }
    return wav_write(output_path, audio, sample_rate_);
}

bool TestToSpeech::synthesize_to_file(const VoiceModel & voice,
                                      const std::string & text,
                                      const std::string & output_path) {
    return synthesize_to_file(voice, text, output_path, Options{});
}

bool TestToSpeech::synthesize_stream(const VoiceModel & voice,
                                     const std::string & text,
                                     const StreamCallback & callback,
                                     size_t chunk_samples,
                                     const Options & options) {
    StreamProfile unused_profile;
    return synthesize_stream_profiled(voice, text, callback, chunk_samples, options, unused_profile);
}

bool TestToSpeech::synthesize_stream_profiled(const VoiceModel & voice,
                                              const std::string & text,
                                              const StreamCallback & callback,
                                              size_t chunk_samples,
                                              const Options & options,
                                              StreamProfile & profile) {
    profile = StreamProfile{};
    const auto t_total0 = std::chrono::steady_clock::now();

    if (!callback) {
        return false;
    }
    if (chunk_samples == 0) {
        chunk_samples = 4096;
    }

    const size_t crossfade_samples = std::min<size_t>(static_cast<size_t>(sample_rate_ * 3 / 100), 4096); // ~30 ms
    std::vector<float> crossfade_tail;

    auto emit_range = [&](const std::vector<float> & audio,
                          size_t begin,
                          size_t end,
                          bool is_final) -> bool {
        if (begin >= end) {
            if (is_final) {
                const auto t_cb0 = std::chrono::steady_clock::now();
                const bool ok = callback(nullptr, 0, sample_rate_, true);
                const auto t_cb1 = std::chrono::steady_clock::now();
                profile.callback_sec += std::chrono::duration<double>(t_cb1 - t_cb0).count();
                return ok;
            }
            return true;
        }

        size_t i = begin;
        bool first_chunk = true;
        while (i < end) {
            const size_t n = std::min(chunk_samples, end - i);
            std::vector<float> out_chunk(audio.begin() + i, audio.begin() + i + n);

            if (first_chunk && !crossfade_tail.empty()) {
                const size_t xf = std::min(crossfade_tail.size(), out_chunk.size());
                for (size_t j = 0; j < xf; ++j) {
                    const float a = static_cast<float>(j + 1) / static_cast<float>(xf + 1);
                    const float b = 1.0f - a;
                    out_chunk[j] = b * crossfade_tail[j] + a * out_chunk[j];
                }
            }

            if (n >= crossfade_samples) {
                crossfade_tail.assign(out_chunk.end() - crossfade_samples, out_chunk.end());
            } else {
                crossfade_tail = out_chunk;
            }

            const bool is_last = is_final && (i + n >= end);
            const auto t_cb0 = std::chrono::steady_clock::now();
            if (!callback(out_chunk.data(), n, sample_rate_, is_last)) {
                const auto t_cb1 = std::chrono::steady_clock::now();
                profile.callback_sec += std::chrono::duration<double>(t_cb1 - t_cb0).count();
                return false;
            }
            const auto t_cb1 = std::chrono::steady_clock::now();
            profile.callback_sec += std::chrono::duration<double>(t_cb1 - t_cb0).count();
            profile.emitted_samples += n;
            i += n;
            first_chunk = false;
        }
        return true;
    };

    if (options.skip_llm) {
        std::vector<float> audio;
        if (!decode_tokens_to_audio(voice, text, audio, false)) {
            return false;
        }
        const bool ok = emit_range(audio, 0, audio.size(), true);
        const auto t_total1 = std::chrono::steady_clock::now();
        profile.total_sec = std::chrono::duration<double>(t_total1 - t_total0).count();
        return ok;
    }

#ifdef MIOTTS_HAS_ONNX
    if (onnx_llm_) {
        std::string token_text;
        if (!generate_token_text(text, options, token_text)) {
            return false;
        }
        std::vector<float> audio;
        const auto t_codec0 = std::chrono::steady_clock::now();
        if (!decode_tokens_to_audio(voice, token_text, audio, false)) {
            return false;
        }
        const auto t_codec1 = std::chrono::steady_clock::now();
        profile.codec_sec += std::chrono::duration<double>(t_codec1 - t_codec0).count();
        profile.llm_tokens = static_cast<int>(parse_speech_tokens(token_text).size());
        profile.decode_calls = 1;
        profile.decoded_codes = static_cast<size_t>(profile.llm_tokens);

        const bool ok = emit_range(audio, 0, audio.size(), true);
        const auto t_total1 = std::chrono::steady_clock::now();
        profile.total_sec = std::chrono::duration<double>(t_total1 - t_total0).count();
        return ok;
    }
#endif

    if (!model_ || !vocab_) {
        fprintf(stderr, "TestToSpeech: LLM model is not loaded\n");
        return false;
    }

    const float temperature = options.temperature >= 0.0f ? options.temperature : config_.temperature;
    const int max_tokens = options.max_tokens > 0 ? options.max_tokens : config_.max_tokens;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = config_.n_threads;

    llama_context * ctx = llama_init_from_model(model_, ctx_params);
    if (!ctx) {
        fprintf(stderr, "TestToSpeech: failed to create llama context\n");
        return false;
    }

    std::string normalized_text = normalize_tts_text(text);
    std::string prompt = build_prompt(normalized_text);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "TestToSpeech: tokenization failed\n");
        llama_free(ctx);
        return false;
    }
    tokens.resize(n_tokens);

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "TestToSpeech: initial decode failed\n");
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        llama_free(ctx);
        return false;
    }
    llama_token eos_token = llama_vocab_eos(vocab_);
    llama_token im_end_token = -1;
    {
        const char * im_end_str = "<|im_end|>";
        llama_token tmp[8];
        int n = llama_tokenize(vocab_, im_end_str, std::strlen(im_end_str), tmp, 8, false, true);
        if (n == 1) {
            im_end_token = tmp[0];
        }
    }

    // Streaming policy:
    // - keep a small right-side holdback so we do not emit unstable newest region
    // - decode from a lookback window for continuity at chunk boundaries
    const int stream_check_interval = 20;
    const size_t holdback_codes = 32;
    const size_t min_commit_step_codes = 24;
    std::string generated;
    size_t committed_codes = 0;
    int cur_pos = n_tokens;
    int n_gen = 0;

    auto maybe_emit = [&](bool is_final) -> bool {
        const std::vector<int> codes = parse_speech_tokens(generated);
        if (codes.empty()) {
            return !is_final;
        }

        const size_t target_committed_codes = is_final
            ? codes.size()
            : (codes.size() > holdback_codes ? codes.size() - holdback_codes : 0);
        if (target_committed_codes <= committed_codes) {
            if (is_final) {
                return callback(nullptr, 0, sample_rate_, true);
            }
            return true;
        }
        if (!is_final && (target_committed_codes - committed_codes) < min_commit_step_codes) {
            return true;
        }

        // Quality-first path: decode full accumulated codes so already-generated
        // region matches non-streaming context as closely as possible.
        const size_t window_start_code = 0;
        const int window_n_codes = static_cast<int>(codes.size());

        std::vector<float> window_audio;
        double decode_sec = 0.0;
        if (!decode_codes_via_backend(*codec_,
                                      voice.embedding(),
                                      codes.data(),
                                      window_n_codes,
                                      window_audio,
                                      &decode_sec,
                                      false)) {
            return false;
        }
        profile.codec_sec += decode_sec;
        profile.decode_calls++;
        profile.decoded_codes += (size_t) window_n_codes;

        // Map committed-code positions to sample indices using the actual decoded
        // window length to avoid splice drift from strict samples_per_token rounding.
        const double samples_per_code_actual = codes.empty()
            ? 0.0
            : (double) window_audio.size() / (double) codes.size();
        const size_t begin_in_window = (size_t) std::llround(
            (double) (committed_codes - window_start_code) * samples_per_code_actual);
        const size_t end_in_window = (size_t) std::llround(
            (double) (target_committed_codes - window_start_code) * samples_per_code_actual);
        const size_t safe_end = std::min(end_in_window, window_audio.size());
        if (begin_in_window >= safe_end) {
            if (is_final) {
                return callback(nullptr, 0, sample_rate_, true);
            }
            return true;
        }

        committed_codes = target_committed_codes;
        return emit_range(window_audio, begin_in_window, safe_end, is_final);
    };

    bool ok = true;
    while (n_gen < max_tokens) {
        const auto t_llm0 = std::chrono::steady_clock::now();
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (new_token == eos_token || new_token == im_end_token) {
            break;
        }

        char buf[256];
        const int len = llama_token_to_piece(vocab_, new_token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            generated.append(buf, len);
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = cur_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "TestToSpeech: decode failed at token %d\n", n_gen);
            ok = false;
            break;
        }
        const auto t_llm1 = std::chrono::steady_clock::now();
        profile.llm_sec += std::chrono::duration<double>(t_llm1 - t_llm0).count();

        cur_pos++;
        n_gen++;
        profile.llm_tokens = n_gen;

        if ((n_gen % stream_check_interval) == 0) {
            if (!maybe_emit(false)) {
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        ok = maybe_emit(true);
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    const auto t_total1 = std::chrono::steady_clock::now();
    profile.total_sec = std::chrono::duration<double>(t_total1 - t_total0).count();
    return ok;
}

bool TestToSpeech::synthesize_to_vector(const VoiceModel & voice,
                                        const std::string & text,
                                        std::vector<float> & out_audio) {
    return synthesize_to_vector(voice, text, out_audio, Options{});
}

bool TestToSpeech::synthesize_stream(const VoiceModel & voice,
                                     const std::string & text,
                                     const StreamCallback & callback,
                                     size_t chunk_samples) {
    return synthesize_stream(voice, text, callback, chunk_samples, Options{});
}

bool TestToSpeech::generate_token_text(const std::string & text,
                                       const Options & options,
                                       std::string & out_token_text) {
    out_token_text.clear();
    if (options.skip_llm) {
        out_token_text = text;
        return true;
    }
    out_token_text = run_llm(text, options);
    return !out_token_text.empty();
}
