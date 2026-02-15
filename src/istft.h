#pragma once

#include <vector>

// Inverse STFT via direct IRFFT + overlap-add.
// spec: interleaved [real, imag] for each of n_freq bins, for n_frames frames.
//       Layout: spec[frame * n_freq * 2 + bin * 2 + 0] = real
//               spec[frame * n_freq * 2 + bin * 2 + 1] = imag
//       n_freq = n_fft/2 + 1
// Returns reconstructed waveform samples (trimmed by (win_length - hop_length)/2 from each end).
std::vector<float> istft(
    const float * spec,
    int n_frames,
    int n_fft      = 392,
    int hop_length = 98,
    int win_length = 392);
