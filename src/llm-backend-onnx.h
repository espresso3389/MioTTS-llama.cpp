#pragma once

#include <memory>
#include <string>
#include <vector>

class OnnxLlmBackend {
public:
    explicit OnnxLlmBackend(const std::string & model_path, int n_threads);
    ~OnnxLlmBackend();

    bool is_valid() const;
    bool forward_logits(const std::vector<int32_t> & token_ids,
                        std::vector<float> & out_logits_last) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
