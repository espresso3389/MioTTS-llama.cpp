#pragma once

#include <vector>

class CodecBackend {
public:
    virtual ~CodecBackend() = default;

    // Decode speech token codes to audio waveform.
    // codes: array of integer token indices (0..12799)
    // n_codes: number of codes
    // global_emb: 128-dim float32 speaker embedding
    // out_audio: output waveform samples
    virtual bool decode_to_audio(const int * codes, int n_codes,
                                 const float * global_emb,
                                 std::vector<float> & out_audio) = 0;

    virtual int sample_rate() const = 0;
    virtual int samples_per_token() const = 0;
};
