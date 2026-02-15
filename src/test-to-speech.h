#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class istft_cache;

class VoiceModel {
public:
    VoiceModel() = default;

    bool load_from_file(const std::string & path);
    bool is_ready() const;

    const std::vector<float> & embedding() const;
    const std::string & path() const;

private:
    std::string path_;
    std::vector<float> embedding_;
};

class TestToSpeech {
public:
    struct Config {
        std::string model_path;
        std::string codec_path;
        int n_threads = 4;
        int n_gpu_layers = 0;
        float temperature = 0.8f;
        int max_tokens = 700;
    };

    struct Options {
        float temperature = -1.0f; // negative => use Config default
        int max_tokens = -1;       // negative => use Config default
        bool skip_llm = false;     // when true, text is raw <|s_N|> token text
    };

    using StreamCallback = std::function<bool(const float * samples,
                                              size_t n_samples,
                                              int sample_rate,
                                              bool is_last_chunk)>;

    explicit TestToSpeech(const Config & config);
    ~TestToSpeech();

    TestToSpeech(const TestToSpeech &) = delete;
    TestToSpeech & operator=(const TestToSpeech &) = delete;

    bool is_ready() const;
    int sample_rate() const;

    bool synthesize_to_file(const VoiceModel & voice,
                            const std::string & text,
                            const std::string & output_path,
                            const Options & options);
    bool synthesize_to_file(const VoiceModel & voice,
                            const std::string & text,
                            const std::string & output_path);

    bool synthesize_to_vector(const VoiceModel & voice,
                              const std::string & text,
                              std::vector<float> & out_audio,
                              const Options & options);
    bool synthesize_to_vector(const VoiceModel & voice,
                              const std::string & text,
                              std::vector<float> & out_audio);

    bool synthesize_stream(const VoiceModel & voice,
                           const std::string & text,
                           const StreamCallback & callback,
                           size_t chunk_samples,
                           const Options & options);
    bool synthesize_stream(const VoiceModel & voice,
                           const std::string & text,
                           const StreamCallback & callback,
                           size_t chunk_samples = 4096);

private:
    std::string run_llm(const std::string & text, const Options & options);
    bool decode_tokens_to_audio(const VoiceModel & voice,
                                const std::string & token_text,
                                std::vector<float> & out_audio);
    static std::string build_prompt(const std::string & text);

private:
    Config config_;
    struct llama_model * model_ = nullptr;
    const struct llama_vocab * vocab_ = nullptr;
    struct miocodec_context * codec_ = nullptr;
    int sample_rate_ = 0;
    int n_fft_ = 0;
    int hop_length_ = 0;
    int samples_per_token_ = 0;
    std::unique_ptr<istft_cache> istft_cache_;
};
