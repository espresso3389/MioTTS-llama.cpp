#pragma once

#include <vector>

// Precomputed iSTFT cache (Hann window + IRFFT trig tables).
class istft_cache {
public:
    istft_cache(int n_fft, int win_length);

    int n_fft() const;
    int win_length() const;
    int n_freq() const;
    int n_mid() const;
    const std::vector<float> & cos_table() const;
    const std::vector<float> & sin_table() const;
    const std::vector<float> & nyquist_sign() const;
    const std::vector<float> & hann_window() const;

private:
    int n_fft_ = 0;
    int win_length_ = 0;
    int n_freq_ = 0;
    int n_mid_ = 0;

    std::vector<float> cos_table_;    // [n_fft, n_mid]
    std::vector<float> sin_table_;    // [n_fft, n_mid]
    std::vector<float> nyquist_sign_; // [n_fft]
    std::vector<float> hann_window_;  // [win_length]

};

// Inverse STFT via IRFFT + overlap-add using precomputed cache.
// spec: interleaved [real, imag] for each of n_freq bins, for n_frames frames.
//       Layout: spec[frame * n_freq * 2 + bin * 2 + 0] = real
//               spec[frame * n_freq * 2 + bin * 2 + 1] = imag
//       n_freq = cache.n_freq()
// Returns reconstructed waveform samples (trimmed by (win_length - hop_length)/2 from each end).
std::vector<float> istft(
    const float * spec,
    int n_frames,
    int hop_length,
    const istft_cache & cache);
