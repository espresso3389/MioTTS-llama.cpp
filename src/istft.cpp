#include "istft.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

static void fill_hann_window(float * w, int length) {
    for (int i = 0; i < length; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / length));
    }
}

// Direct IRFFT: reconstruct N real samples from N/2+1 complex bins.
// spec: interleaved [real, imag] pairs, n_freq = N/2+1 bins
// out: N real samples
static void irfft(const float * spec, int n_freq, int n_fft, float * out) {
    // x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
    // Using Hermitian symmetry X[N-k] = conj(X[k]):
    // x[n] = (1/N) * (X[0].re + (-1)^n * X[N/2].re
    //         + 2 * sum_{k=1}^{N/2-1} (X[k].re*cos(w) - X[k].im*sin(w)))
    // where w = 2*pi*k*n/N

    const float inv_n = 1.0f / (float)n_fft;
    const float two_pi_over_n = 2.0f * (float)M_PI / (float)n_fft;

    for (int n = 0; n < n_fft; n++) {
        float sum = spec[0]; // DC: X[0].re (X[0].im = 0)

        // Nyquist: X[N/2].re * (-1)^n
        sum += spec[(n_freq - 1) * 2] * ((n % 2 == 0) ? 1.0f : -1.0f);

        // Middle bins
        for (int k = 1; k < n_freq - 1; k++) {
            float re = spec[k * 2 + 0];
            float im = spec[k * 2 + 1];
            float w = two_pi_over_n * (float)k * (float)n;
            sum += 2.0f * (re * cosf(w) - im * sinf(w));
        }

        out[n] = sum * inv_n;
    }
}

std::vector<float> istft(
        const float * spec,
        int n_frames,
        int n_fft,
        int hop_length,
        int win_length) {

    const int n_freq = n_fft / 2 + 1;
    const int n_pad  = (win_length - hop_length) / 2;

    // Output length before trimming
    const int n_out = (n_frames - 1) * hop_length + win_length;

    // Hann window
    std::vector<float> hann(win_length);
    fill_hann_window(hann.data(), win_length);

    // Accumulation buffers
    std::vector<float> audio(n_out, 0.0f);
    std::vector<float> window_sum(n_out, 0.0f);

    // Temporary buffer for each frame
    std::vector<float> time_buf(n_fft);

    for (int t = 0; t < n_frames; t++) {
        const float * frame_spec = spec + t * n_freq * 2;

        // Inverse FFT via direct computation
        irfft(frame_spec, n_freq, n_fft, time_buf.data());

        // Apply window and overlap-add
        int offset = t * hop_length;
        for (int j = 0; j < win_length; j++) {
            audio[offset + j]      += time_buf[j] * hann[j];
            window_sum[offset + j] += hann[j] * hann[j];
        }
    }

    // Normalize by window envelope (COLA)
    for (int i = 0; i < n_out; i++) {
        if (window_sum[i] > 1e-8f) {
            audio[i] /= window_sum[i];
        }
    }

    // Trim padding: n_pad from each end
    int trim_start = n_pad;
    int trim_end   = n_out - n_pad;
    if (trim_end <= trim_start) {
        return {};
    }

    return std::vector<float>(audio.begin() + trim_start, audio.begin() + trim_end);
}
