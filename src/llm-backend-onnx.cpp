#include "llm-backend-onnx.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static bool append_onnx_provider(Ort::SessionOptions & opts, const std::string & ep, int device_id) {
    if (ep == "cuda") {
        try {
            Ort::CUDAProviderOptions cuda_opts;
            cuda_opts.Update({{"device_id", std::to_string(device_id)}});
            opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
            std::fprintf(stderr, "onnx_ep: using CUDAExecutionProvider (device_id=%d)\n", device_id);
            return true;
        } catch (const Ort::Exception & e) {
            std::fprintf(stderr, "onnx_ep: CUDA append failed: %s\n", e.what());
            return false;
        }
    }
    if (ep == "tensorrt" || ep == "trt") {
        try {
            Ort::TensorRTProviderOptions trt_opts;
            trt_opts.Update({{"device_id", std::to_string(device_id)}});
            opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
            std::fprintf(stderr, "onnx_ep: using TensorRTExecutionProvider (device_id=%d)\n", device_id);
            return true;
        } catch (const Ort::Exception & e) {
            std::fprintf(stderr, "onnx_ep: TensorRT append failed: %s\n", e.what());
            return false;
        }
    }
    return false;
}

static void configure_onnx_providers(Ort::SessionOptions & opts) {
    const char * ep_env = std::getenv("MIOTTS_ONNX_EP");
    const char * dev_env = std::getenv("MIOTTS_ONNX_DEVICE_ID");
    const int device_id = dev_env ? std::max(0, std::atoi(dev_env)) : 0;
    const std::string ep = ep_env ? ep_env : "";

    if (ep.empty() || ep == "cpu" || ep == "default") {
        std::fprintf(stderr, "onnx_ep: using default provider chain (CPU)\n");
        return;
    }
    if (!append_onnx_provider(opts, ep, device_id)) {
        std::fprintf(stderr, "onnx_ep: fallback to default provider chain (CPU)\n");
    }
}

struct OnnxLlmBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "miotts_onnx_llm"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;

    std::vector<std::string> input_names_storage;
    std::vector<const char *> input_names;
    std::vector<std::string> output_names_storage;
    std::vector<const char *> output_names;

    int input_ids_index = -1;
    int attention_mask_index = -1;
    int position_ids_index = -1;
    ONNXTensorElementDataType input_ids_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType mask_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType pos_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    int logits_output_index = -1;
};

static std::vector<char> fp16_to_fp32(const uint16_t * data, size_t n) {
    std::vector<char> out(n * sizeof(float));
    float * f = reinterpret_cast<float *>(out.data());
    for (size_t i = 0; i < n; ++i) {
        const uint16_t h = data[i];
        const uint32_t sign = (h & 0x8000u) << 16;
        const uint32_t exp = (h & 0x7C00u) >> 10;
        const uint32_t mant = (h & 0x03FFu);

        uint32_t bits = 0;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                uint32_t e = 127 - 15 + 1;
                uint32_t m = mant;
                while ((m & 0x0400u) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x03FFu;
                bits = sign | (e << 23) | (m << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        std::memcpy(&f[i], &bits, sizeof(uint32_t));
    }
    return out;
}

OnnxLlmBackend::OnnxLlmBackend(const std::string & model_path, int n_threads) : impl_(std::make_unique<Impl>()) {
    try {
        impl_->opts.SetIntraOpNumThreads(std::max(1, n_threads));
        impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        configure_onnx_providers(impl_->opts);
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_path.c_str(), impl_->opts);

        Ort::AllocatorWithDefaultOptions alloc;
        const size_t n_inputs = impl_->session->GetInputCount();
        impl_->input_names_storage.reserve(n_inputs);
        impl_->input_names.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            const auto n = impl_->session->GetInputNameAllocated(i, alloc);
            impl_->input_names_storage.emplace_back(n.get());
            impl_->input_names.push_back(impl_->input_names_storage.back().c_str());

            const auto & name = impl_->input_names_storage.back();
            const auto type = impl_->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
            if (name == "input_ids") {
                impl_->input_ids_index = static_cast<int>(i);
                impl_->input_ids_type = type;
            } else if (name == "attention_mask") {
                impl_->attention_mask_index = static_cast<int>(i);
                impl_->mask_type = type;
            } else if (name == "position_ids") {
                impl_->position_ids_index = static_cast<int>(i);
                impl_->pos_type = type;
            }
        }

        const size_t n_outputs = impl_->session->GetOutputCount();
        impl_->output_names_storage.reserve(n_outputs);
        impl_->output_names.reserve(n_outputs);
        for (size_t i = 0; i < n_outputs; ++i) {
            const auto n = impl_->session->GetOutputNameAllocated(i, alloc);
            impl_->output_names_storage.emplace_back(n.get());
            impl_->output_names.push_back(impl_->output_names_storage.back().c_str());
            if (impl_->output_names_storage.back() == "logits" && impl_->logits_output_index < 0) {
                impl_->logits_output_index = static_cast<int>(i);
            }
        }

        if (impl_->input_ids_index < 0) {
            throw std::runtime_error("input_ids input not found");
        }
        if (impl_->logits_output_index < 0) {
            impl_->logits_output_index = 0;
        }
        std::fprintf(stderr, "onnx_llm: loaded %s\n", model_path.c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "onnx_llm: failed to load %s: %s\n", model_path.c_str(), e.what());
        impl_.reset();
    }
}

OnnxLlmBackend::~OnnxLlmBackend() = default;

bool OnnxLlmBackend::is_valid() const {
    return impl_ && impl_->session != nullptr;
}

bool OnnxLlmBackend::forward_logits(const std::vector<int32_t> & token_ids,
                                    std::vector<float> & out_logits_last) const {
    out_logits_last.clear();
    if (!is_valid() || token_ids.empty()) {
        return false;
    }

    try {
        const int64_t seq_len = static_cast<int64_t>(token_ids.size());
        const std::array<int64_t, 2> shape{1, seq_len};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_values;
        input_values.reserve(3);

        std::vector<int64_t> ids_i64;
        std::vector<int32_t> ids_i32;
        if (impl_->input_ids_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            ids_i64.assign(token_ids.begin(), token_ids.end());
            input_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
                mem, ids_i64.data(), ids_i64.size(), shape.data(), shape.size()));
        } else {
            ids_i32 = token_ids;
            input_values.emplace_back(Ort::Value::CreateTensor<int32_t>(
                mem, ids_i32.data(), ids_i32.size(), shape.data(), shape.size()));
        }

        std::vector<int64_t> mask_i64;
        std::vector<int32_t> mask_i32;
        if (impl_->attention_mask_index >= 0) {
            if (impl_->mask_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                mask_i64.assign(token_ids.size(), 1);
                input_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    mem, mask_i64.data(), mask_i64.size(), shape.data(), shape.size()));
            } else {
                mask_i32.assign(token_ids.size(), 1);
                input_values.emplace_back(Ort::Value::CreateTensor<int32_t>(
                    mem, mask_i32.data(), mask_i32.size(), shape.data(), shape.size()));
            }
        }

        std::vector<int64_t> pos_i64;
        std::vector<int32_t> pos_i32;
        if (impl_->position_ids_index >= 0) {
            if (impl_->pos_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                pos_i64.resize(token_ids.size());
                for (size_t i = 0; i < token_ids.size(); ++i) pos_i64[i] = static_cast<int64_t>(i);
                input_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    mem, pos_i64.data(), pos_i64.size(), shape.data(), shape.size()));
            } else {
                pos_i32.resize(token_ids.size());
                for (size_t i = 0; i < token_ids.size(); ++i) pos_i32[i] = static_cast<int32_t>(i);
                input_values.emplace_back(Ort::Value::CreateTensor<int32_t>(
                    mem, pos_i32.data(), pos_i32.size(), shape.data(), shape.size()));
            }
        }

        std::vector<const char *> run_input_names;
        run_input_names.push_back(impl_->input_names[impl_->input_ids_index]);
        if (impl_->attention_mask_index >= 0) {
            run_input_names.push_back(impl_->input_names[impl_->attention_mask_index]);
        }
        if (impl_->position_ids_index >= 0) {
            run_input_names.push_back(impl_->input_names[impl_->position_ids_index]);
        }

        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            run_input_names.data(), input_values.data(), input_values.size(),
            impl_->output_names.data(), impl_->output_names.size());

        if (outputs.empty() || !outputs[impl_->logits_output_index].IsTensor()) {
            return false;
        }
        auto logits_info = outputs[impl_->logits_output_index].GetTensorTypeAndShapeInfo();
        auto logits_shape = logits_info.GetShape();
        if (logits_shape.size() != 3 || logits_shape[0] != 1) {
            return false;
        }

        const size_t seq = static_cast<size_t>(logits_shape[1]);
        const size_t vocab = static_cast<size_t>(logits_shape[2]);
        if (seq == 0 || vocab == 0) {
            return false;
        }
        const size_t offset = (seq - 1) * vocab;
        out_logits_last.resize(vocab);

        const auto t = logits_info.GetElementType();
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float * p = outputs[impl_->logits_output_index].GetTensorData<float>();
            std::copy(p + offset, p + offset + vocab, out_logits_last.begin());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const uint16_t * p = outputs[impl_->logits_output_index].GetTensorData<uint16_t>();
            std::vector<char> tmp = fp16_to_fp32(p + offset, vocab);
            std::memcpy(out_logits_last.data(), tmp.data(), vocab * sizeof(float));
        } else {
            std::fprintf(stderr, "onnx_llm: unsupported logits type: %d\n", static_cast<int>(t));
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "onnx_llm: inference failed: %s\n", e.what());
        return false;
    }
}
