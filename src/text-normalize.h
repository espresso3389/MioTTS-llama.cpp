#pragma once

#include <string>

// Apply a subset of miotts_server text normalization for Japanese prompts.
// This improves compatibility with punctuation such as "！" and "？".
std::string normalize_tts_text(const std::string & text);
