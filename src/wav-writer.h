#pragma once

#include <string>
#include <vector>

bool wav_write(const std::string & path, const std::vector<float> & samples, int sample_rate);
