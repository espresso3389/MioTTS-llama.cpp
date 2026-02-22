#pragma once

#include "codec-backend.h"
#include "istft.h"
#include "miocodec.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class GgmlCodecBackend : public CodecBackend {
public:
    explicit GgmlCodecBackend(const std::string & codec_path) {
        ctx_ = miocodec_load(codec_path);
        if (!ctx_) {
            fprintf(stderr, "GgmlCodecBackend: failed to load %s\n", codec_path.c_str());
            return;
        }
        sample_rate_ = miocodec_sample_rate(ctx_);
        hop_length_ = miocodec_hop_length(ctx_);
        samples_per_token_ = miocodec_samples_per_token(ctx_);
        int n_fft = miocodec_n_fft(ctx_);
        istft_cache_ = std::make_unique<istft_cache>(n_fft, n_fft);
    }

    ~GgmlCodecBackend() override {
        if (ctx_) {
            miocodec_free(ctx_);
        }
    }

    GgmlCodecBackend(const GgmlCodecBackend &) = delete;
    GgmlCodecBackend & operator=(const GgmlCodecBackend &) = delete;

    bool is_valid() const { return ctx_ != nullptr; }

    bool decode_to_audio(const int * codes, int n_codes,
                         const float * global_emb,
                         std::vector<float> & out_audio) override {
        if (!ctx_ || n_codes <= 0) {
            return false;
        }

        int n_frames = 0;
        int audio_length = n_codes * samples_per_token_;
        std::vector<float> spec = miocodec_decode(
            ctx_, codes, n_codes, global_emb, audio_length, &n_frames);
        if (spec.empty()) {
            return false;
        }

        out_audio = istft(spec.data(), n_frames, hop_length_, *istft_cache_);
        return !out_audio.empty();
    }

    int sample_rate() const override { return sample_rate_; }
    int samples_per_token() const override { return samples_per_token_; }

private:
    miocodec_context * ctx_ = nullptr;
    int sample_rate_ = 0;
    int hop_length_ = 0;
    int samples_per_token_ = 0;
    std::unique_ptr<istft_cache> istft_cache_;
};
