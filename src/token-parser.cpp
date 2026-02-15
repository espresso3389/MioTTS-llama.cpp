#include "token-parser.h"
#include <cstdlib>
#include <cstring>

std::vector<int> parse_speech_tokens(const std::string & text) {
    std::vector<int> codes;
    const char * p = text.c_str();
    const char * end = p + text.size();

    while (p < end) {
        const char * found = strstr(p, "<|s_");
        if (!found) {
            break;
        }
        const char * num_start = found + 4; // skip "<|s_"
        char * num_end = nullptr;
        long val = strtol(num_start, &num_end, 10);
        if (num_end && num_end > num_start && num_end + 1 < end
            && num_end[0] == '|' && num_end[1] == '>') {
            codes.push_back(static_cast<int>(val));
            p = num_end + 2; // skip "|>"
        } else {
            p = found + 1; // skip past '<' and try again
        }
    }

    return codes;
}
