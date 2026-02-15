#include "wav-writer.h"
#include <algorithm>
#include <cstdint>
#include <fstream>

#pragma pack(push, 1)
struct wav_header {
    char     riff[4]        = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char     wave[4]        = {'W', 'A', 'V', 'E'};
    char     fmt[4]         = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format   = 1;  // PCM
    uint16_t num_channels   = 1;  // mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char     data[4]        = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};
#pragma pack(pop)

bool wav_write(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    wav_header hdr;
    hdr.sample_rate    = sample_rate;
    hdr.byte_rate      = sample_rate * hdr.num_channels * (hdr.bits_per_sample / 8);
    hdr.block_align    = hdr.num_channels * (hdr.bits_per_sample / 8);
    hdr.data_size      = samples.size() * (hdr.bits_per_sample / 8);
    hdr.chunk_size     = 36 + hdr.data_size;

    f.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

    for (float s : samples) {
        int16_t pcm = static_cast<int16_t>(std::clamp(s * 32767.0f, -32768.0f, 32767.0f));
        f.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }

    return f.good();
}
