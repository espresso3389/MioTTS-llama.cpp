#include "test-to-speech.h"

#include "istft.h"
#include "miocodec.h"
#include "text-normalize.h"
#include "token-parser.h"
#include "wav-writer.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
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
    }

    codec_ = miocodec_load(config_.codec_path);
    if (!codec_) {
        fprintf(stderr, "TestToSpeech: failed to load codec: %s\n", config_.codec_path.c_str());
        return;
    }

    sample_rate_ = miocodec_sample_rate(codec_);
    n_fft_ = miocodec_n_fft(codec_);
    hop_length_ = miocodec_hop_length(codec_);
    samples_per_token_ = miocodec_samples_per_token(codec_);
    istft_cache_ = std::make_unique<istft_cache>(n_fft_, n_fft_);
}

TestToSpeech::~TestToSpeech() {
    if (codec_) {
        miocodec_free(codec_);
        codec_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
}

bool TestToSpeech::is_ready() const {
    return codec_ != nullptr;
}

int TestToSpeech::sample_rate() const {
    return sample_rate_;
}

std::string TestToSpeech::build_prompt(const std::string & text) {
    return "<|startoftext|><|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string TestToSpeech::run_llm(const std::string & text, const Options & options) {
    if (!model_ || !vocab_) {
        fprintf(stderr, "TestToSpeech: LLM model is not loaded\n");
        return "";
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
        return "";
    }

    std::string normalized_text = normalize_tts_text(text);
    std::string prompt = build_prompt(normalized_text);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "TestToSpeech: tokenization failed\n");
        llama_free(ctx);
        return "";
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
                                          std::vector<float> & out_audio) {
    if (!voice.is_ready()) {
        fprintf(stderr, "TestToSpeech: voice model is not ready\n");
        return false;
    }

    const std::vector<int> codes = parse_speech_tokens(token_text);
    if (codes.empty()) {
        fprintf(stderr, "TestToSpeech: no speech codes parsed from text\n");
        return false;
    }

    int n_frames = 0;
    int audio_length = static_cast<int>(codes.size()) * samples_per_token_;
    std::vector<float> spec = miocodec_decode(
        codec_, codes.data(), static_cast<int>(codes.size()),
        voice.embedding().data(), audio_length, &n_frames);
    if (spec.empty()) {
        fprintf(stderr, "TestToSpeech: codec decode failed\n");
        return false;
    }

    out_audio = istft(spec.data(), n_frames, hop_length_, *istft_cache_);
    if (out_audio.empty()) {
        fprintf(stderr, "TestToSpeech: iSTFT failed\n");
        return false;
    }

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

    return true;
}

static bool decode_codes_to_audio(miocodec_context * codec,
                                  const std::vector<float> & voice_emb,
                                  int samples_per_token,
                                  int hop_length,
                                  const istft_cache & istft_ctx,
                                  const int * codes,
                                  int n_codes,
                                  std::vector<float> & out_audio,
                                  double * out_codec_sec = nullptr,
                                  double * out_istft_sec = nullptr) {
    if (n_codes <= 0) {
        out_audio.clear();
        return true;
    }

    int n_frames = 0;
    int audio_length = n_codes * samples_per_token;
    const auto t_codec0 = std::chrono::steady_clock::now();
    std::vector<float> spec = miocodec_decode(
        codec, codes, n_codes, voice_emb.data(), audio_length, &n_frames);
    const auto t_codec1 = std::chrono::steady_clock::now();
    if (spec.empty()) {
        fprintf(stderr, "TestToSpeech: codec decode failed (stream)\n");
        return false;
    }
    if (out_codec_sec) {
        *out_codec_sec += std::chrono::duration<double>(t_codec1 - t_codec0).count();
    }

    const auto t_istft0 = std::chrono::steady_clock::now();
    out_audio = istft(spec.data(), n_frames, hop_length, istft_ctx);
    const auto t_istft1 = std::chrono::steady_clock::now();
    if (out_audio.empty()) {
        fprintf(stderr, "TestToSpeech: iSTFT failed (stream)\n");
        return false;
    }
    if (out_istft_sec) {
        *out_istft_sec += std::chrono::duration<double>(t_istft1 - t_istft0).count();
    }

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

    return true;
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
    if (options.skip_llm) {
        token_text = text;
    } else {
        token_text = run_llm(text, options);
        if (token_text.empty()) {
            return false;
        }
    }

    return decode_tokens_to_audio(voice, token_text, out_audio);
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
        while (i < end) {
            const size_t n = std::min(chunk_samples, end - i);
            const bool is_last = is_final && (i + n >= end);
            const auto t_cb0 = std::chrono::steady_clock::now();
            if (!callback(audio.data() + i, n, sample_rate_, is_last)) {
                const auto t_cb1 = std::chrono::steady_clock::now();
                profile.callback_sec += std::chrono::duration<double>(t_cb1 - t_cb0).count();
                return false;
            }
            const auto t_cb1 = std::chrono::steady_clock::now();
            profile.callback_sec += std::chrono::duration<double>(t_cb1 - t_cb0).count();
            profile.emitted_samples += n;
            i += n;
        }
        return true;
    };

    if (options.skip_llm) {
        std::vector<float> audio;
        if (!decode_tokens_to_audio(voice, text, audio)) {
            return false;
        }
        const bool ok = emit_range(audio, 0, audio.size(), true);
        const auto t_total1 = std::chrono::steady_clock::now();
        profile.total_sec = std::chrono::duration<double>(t_total1 - t_total0).count();
        return ok;
    }

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
    const int stream_check_interval = 16;
    const size_t lookback_codes = 24;
    const size_t holdback_codes = 10;
    const size_t min_commit_step_codes = 12;
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

        const size_t window_start_code = committed_codes > lookback_codes
            ? committed_codes - lookback_codes
            : 0;
        const int window_n_codes = static_cast<int>(codes.size() - window_start_code);

        std::vector<float> window_audio;
        double codec_sec = 0.0;
        double istft_sec = 0.0;
        if (!decode_codes_to_audio(codec_,
                                   voice.embedding(),
                                   samples_per_token_,
                                   hop_length_,
                                   *istft_cache_,
                                   codes.data() + window_start_code,
                                   window_n_codes,
                                   window_audio,
                                   &codec_sec,
                                   &istft_sec)) {
            return false;
        }
        profile.codec_sec += codec_sec;
        profile.istft_sec += istft_sec;
        profile.decode_calls++;
        profile.decoded_codes += (size_t) window_n_codes;

        const size_t begin_in_window =
            (committed_codes - window_start_code) * static_cast<size_t>(samples_per_token_);
        const size_t end_in_window =
            (target_committed_codes - window_start_code) * static_cast<size_t>(samples_per_token_);
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
