#include "text-normalize.h"

#include <cstdint>
#include <string>
#include <vector>

// Upstream reference (MioTTS-Inference):
// - miotts_server/text.py: normalize_text()
//   https://github.com/Aratako/MioTTS-Inference/blob/main/miotts_server/text.py
// - miotts_server/api.py: Japanese-only normalization path before LLM input
//   https://github.com/Aratako/MioTTS-Inference/blob/main/miotts_server/api.py
//
// What this file does:
// - Applies normalization only for JP-like input (heuristic based on Japanese char ratio).
// - Replaces punctuation variants that often break pronunciation/tokenization:
//   "！" -> "!", "？" -> "?", "〜/～" -> "ー".
// - Removes whitespace/control artifacts used in some datasets ("[n]", tabs, full-width spaces).
// - Normalizes select symbols (e.g., "●/◯/〇" -> "○", "♥" -> "♡").
// - Trims surrounding quote/bracket wrappers and trailing Japanese punctuation "。/、".
// This keeps behavior aligned with miotts_server preprocessing in a C++-friendly subset.

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static bool utf8_next_codepoint(const std::string & s, size_t & i, uint32_t & cp) {
    if (i >= s.size()) {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0;
        i += 1;
        return true;
    }
    if ((c0 >> 5) == 0x6 && i + 1 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        i += 2;
        return true;
    }
    if ((c0 >> 4) == 0xE && i + 2 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        i += 3;
        return true;
    }
    if ((c0 >> 3) == 0x1E && i + 3 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        i += 4;
        return true;
    }

    cp = c0;
    i += 1;
    return true;
}

static bool is_japanese_char(uint32_t cp) {
    return (cp >= 0x3040 && cp <= 0x309F) || // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) || // Katakana
           (cp >= 0x4E00 && cp <= 0x9FFF) || // CJK Unified Ideographs
           (cp >= 0x3400 && cp <= 0x4DBF);   // CJK Extension A
}

static bool should_normalize_ja(const std::string & text) {
    int total = 0;
    int ja = 0;
    size_t i = 0;
    uint32_t cp = 0;
    while (utf8_next_codepoint(text, i, cp)) {
        if (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D) {
            continue;
        }
        total++;
        if (is_japanese_char(cp)) {
            ja++;
        }
    }

    if (total == 0) {
        return false;
    }
    return static_cast<float>(ja) / static_cast<float>(total) >= 0.1f;
}

static bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string normalize_tts_text(const std::string & text) {
    if (!should_normalize_ja(text)) {
        return text;
    }

    std::string out = text;

    replace_all(out, "\t", "");
    replace_all(out, "[n]", "");
    replace_all(out, " ", "");
    replace_all(out, "　", "");

    // Match miotts_server behavior for punctuation normalization.
    replace_all(out, "？", "?");
    replace_all(out, "！", "!");
    replace_all(out, "〜", "ー");
    replace_all(out, "～", "ー");

    replace_all(out, "♥", "♡");
    replace_all(out, "●", "○");
    replace_all(out, "◯", "○");
    replace_all(out, "〇", "○");

    while (out.find("………") != std::string::npos) {
        replace_all(out, "………", "……");
    }

    const std::vector<std::pair<std::string, std::string>> wrappers = {
        {"「", "」"},
        {"『", "』"},
        {"（", "）"},
        {"【", "】"},
        {"(", ")"},
    };
    for (const auto & w : wrappers) {
        if (starts_with(out, w.first) && ends_with(out, w.second) &&
            out.size() > w.first.size() + w.second.size()) {
            out = out.substr(w.first.size(), out.size() - w.first.size() - w.second.size());
        }
    }

    while (ends_with(out, "。") || ends_with(out, "、")) {
        if (ends_with(out, "。")) {
            out.resize(out.size() - std::string("。").size());
        } else {
            out.resize(out.size() - std::string("、").size());
        }
    }

    return out;
}
