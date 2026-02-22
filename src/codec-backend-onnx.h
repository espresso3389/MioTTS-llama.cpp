#pragma once

#include "codec-backend.h"

#include <memory>
#include <string>
#include <vector>

class OnnxCodecBackend : public CodecBackend {
public:
    explicit OnnxCodecBackend(const std::string & model_path);
    ~OnnxCodecBackend() override;

    OnnxCodecBackend(const OnnxCodecBackend &) = delete;
    OnnxCodecBackend & operator=(const OnnxCodecBackend &) = delete;

    bool is_valid() const;

    bool decode_to_audio(const int * codes, int n_codes,
                         const float * global_emb,
                         std::vector<float> & out_audio) override;

    int sample_rate() const override;
    int samples_per_token() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
