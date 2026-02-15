#pragma once

#include <string>
#include <vector>

struct miocodec_context;

// Load MioCodec model from GGUF file.
// Returns nullptr on failure.
miocodec_context * miocodec_load(const std::string & model_path);

// Free MioCodec context.
void miocodec_free(miocodec_context * ctx);

// Decode speech token codes to spectrogram (interleaved real/imag).
// codes: array of integer token indices (0..12799)
// n_codes: number of codes
// global_emb: 128-dim float32 global embedding (from .emb.gguf)
// audio_length: target audio length in samples (0 = auto: n_codes * samples_per_token)
//
// Returns spectrogram as interleaved [real, imag] per bin per frame.
// Shape: [n_frames, n_freq*2] where n_freq = n_fft/2 + 1
// out_n_frames is set to the number of STFT frames produced.
std::vector<float> miocodec_decode(
    miocodec_context * ctx,
    const int * codes,
    int n_codes,
    const float * global_emb,
    int audio_length,
    int * out_n_frames);

// Model parameter accessors
int   miocodec_sample_rate(const miocodec_context * ctx);
int   miocodec_n_fft(const miocodec_context * ctx);
int   miocodec_hop_length(const miocodec_context * ctx);
int   miocodec_samples_per_token(const miocodec_context * ctx);

// Load a voice embedding from .emb.gguf file.
// Returns 128-dim float32 vector.
std::vector<float> load_voice_embedding(const std::string & path);

// Print all tensor names in a GGUF file (for debugging).
void miocodec_print_tensors(const std::string & path);
