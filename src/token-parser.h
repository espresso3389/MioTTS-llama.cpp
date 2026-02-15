#pragma once

#include <string>
#include <vector>

// Extract integer codes from "<|s_N|>" tokens in the text.
// Returns a vector of integer code indices.
std::vector<int> parse_speech_tokens(const std::string & text);
