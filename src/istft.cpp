#include "istft.h"

#include <algorithm>
#include <cmath>
#include <vector>

istft_cache::istft_cache(int n_fft, int win_length) {
    n_fft_ = n_fft;
    win_length_ = win_length;
    n_freq_ = n_fft_ / 2 + 1;
    n_mid_ = std::max(0, n_freq_ - 2);

    cos_table_.resize((size_t) n_fft_ * n_mid_);
    sin_table_.resize((size_t) n_fft_ * n_mid_);
    nyquist_sign_.resize(n_fft_);
    hann_window_.resize(win_length_);

    const float two_pi_over_n = 2.0f * (float) M_PI / (float) n_fft_;
    for (int n = 0; n < n_fft_; n++) {
        nyquist_sign_[n] = (n & 1) ? -1.0f : 1.0f;
        for (int k = 1; k <= n_mid_; k++) {
            const float w = two_pi_over_n * (float) k * (float) n;
            const size_t idx = (size_t) n * n_mid_ + (size_t) (k - 1);
            cos_table_[idx] = cosf(w);
            sin_table_[idx] = sinf(w);
        }
    }

    for (int i = 0; i < win_length_; i++) {
        hann_window_[i] = 0.5f * (1.0f - cosf(2.0f * (float) M_PI * i / win_length_));
    }
}

int istft_cache::n_fft() const { return n_fft_; }
int istft_cache::win_length() const { return win_length_; }
int istft_cache::n_freq() const { return n_freq_; }
int istft_cache::n_mid() const { return n_mid_; }
const std::vector<float> & istft_cache::cos_table() const { return cos_table_; }
const std::vector<float> & istft_cache::sin_table() const { return sin_table_; }
const std::vector<float> & istft_cache::nyquist_sign() const { return nyquist_sign_; }
const std::vector<float> & istft_cache::hann_window() const { return hann_window_; }

static void irfft(const float * spec, const istft_cache & cache, float * out) {
    const int n_freq = cache.n_freq();
    const int n_fft = cache.n_fft();
    const int n_mid = cache.n_mid();
    const auto & nyq = cache.nyquist_sign();
    const auto & cos_tbl = cache.cos_table();
    const auto & sin_tbl = cache.sin_table();
    const float inv_n = 1.0f / (float) n_fft;

    for (int n = 0; n < n_fft; n++) {
        float sum = spec[0];
        sum += spec[(n_freq - 1) * 2] * nyq[n];

        const size_t row = (size_t) n * n_mid;
        for (int k = 1; k <= n_mid; k++) {
            const float re = spec[k * 2 + 0];
            const float im = spec[k * 2 + 1];
            const size_t idx = row + (size_t) (k - 1);
            sum += 2.0f * (re * cos_tbl[idx] - im * sin_tbl[idx]);
        }

        out[n] = sum * inv_n;
    }
}

std::vector<float> istft(
        const float * spec,
        int n_frames,
        int hop_length,
        const istft_cache & cache) {
    const int n_fft = cache.n_fft();
    const int win_length = cache.win_length();
    const int n_freq = cache.n_freq();
    const int n_pad = (win_length - hop_length) / 2;
    const int n_out = (n_frames - 1) * hop_length + win_length;
    const auto & hann = cache.hann_window();

    std::vector<float> audio(n_out, 0.0f);
    std::vector<float> window_sum(n_out, 0.0f);
    std::vector<float> time_buf(n_fft);

    for (int t = 0; t < n_frames; t++) {
        const float * frame_spec = spec + t * n_freq * 2;
        irfft(frame_spec, cache, time_buf.data());

        const int offset = t * hop_length;
        for (int j = 0; j < win_length; j++) {
            audio[offset + j] += time_buf[j] * hann[j];
            window_sum[offset + j] += hann[j] * hann[j];
        }
    }

    for (int i = 0; i < n_out; i++) {
        if (window_sum[i] > 1e-8f) {
            audio[i] /= window_sum[i];
        }
    }

    const int trim_start = n_pad;
    const int trim_end = n_out - n_pad;
    if (trim_end <= trim_start) {
        return {};
    }

    return std::vector<float>(audio.begin() + trim_start, audio.begin() + trim_end);
}
