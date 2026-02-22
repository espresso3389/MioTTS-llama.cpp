#include "codec-backend-onnx.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

// MioCodec-25Hz-24kHz constants
static constexpr int kSampleRate = 24000;
static constexpr int kSamplesPerToken = 960;
static constexpr int kGlobalEmbDim = 128;

// Compute STFT frame count from token count.
// Matches Python: calculate_stft_length(n_codes, downsample_factor=2,
//   ssl_hop_size=320, ssl_sample_rate=16000, sample_rate=24000,
//   hop_length=480, istft_padding="same")
static int64_t calculate_stft_length(int n_codes) {
    int64_t feature_length = static_cast<int64_t>(n_codes) * 2;
    int64_t num_samples_ssl = (feature_length - 1) * 320 + 400;
    int64_t audio_length = static_cast<int64_t>(
        std::ceil(static_cast<double>(num_samples_ssl) / 16000.0 * 24000.0));
    return audio_length / 480; // istft_padding == "same"
}

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

struct OnnxCodecBackend::Impl {
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};
    Ort::MemoryInfo mem_info{nullptr};
    bool valid = false;
};

OnnxCodecBackend::OnnxCodecBackend(const std::string & model_path)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "miocodec_onnx");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        configure_onnx_providers(opts);
        impl_->session = Ort::Session(impl_->env, model_path.c_str(), opts);
        impl_->mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        impl_->valid = true;
        fprintf(stderr, "onnx_codec: loaded %s\n", model_path.c_str());
    } catch (const Ort::Exception & e) {
        fprintf(stderr, "onnx_codec: failed to load %s: %s\n",
                model_path.c_str(), e.what());
    }
}

OnnxCodecBackend::~OnnxCodecBackend() = default;

bool OnnxCodecBackend::is_valid() const {
    return impl_ && impl_->valid;
}

bool OnnxCodecBackend::decode_to_audio(const int * codes, int n_codes,
                                       const float * global_emb,
                                       std::vector<float> & out_audio) {
    if (!is_valid() || n_codes <= 0) {
        return false;
    }

    try {
        // 1. content_token_indices: int64[n_codes]
        std::vector<int64_t> codes_i64(n_codes);
        for (int i = 0; i < n_codes; i++) {
            codes_i64[i] = static_cast<int64_t>(codes[i]);
        }
        int64_t codes_shape[] = {static_cast<int64_t>(n_codes)};

        // 2. global_embedding: float32[128]
        int64_t emb_shape[] = {kGlobalEmbDim};

        // 3. stft_length: int64 scalar (0-d tensor)
        int64_t stft_len = calculate_stft_length(n_codes);

        std::vector<Ort::Value> inputs;
        inputs.reserve(3);

        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            impl_->mem_info, codes_i64.data(), codes_i64.size(),
            codes_shape, 1));

        inputs.push_back(Ort::Value::CreateTensor<float>(
            impl_->mem_info, const_cast<float *>(global_emb), kGlobalEmbDim,
            emb_shape, 1));

        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            impl_->mem_info, &stft_len, 1,
            nullptr, 0));

        const char * input_names[] = {
            "content_token_indices", "global_embedding", "stft_length"};
        const char * output_names[] = {"waveform"};

        auto outputs = impl_->session.Run(
            Ort::RunOptions{nullptr},
            input_names, inputs.data(), 3,
            output_names, 1);

        auto & out_tensor = outputs[0];
        auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();

        size_t n_samples = 1;
        for (auto d : shape) {
            n_samples *= static_cast<size_t>(d);
        }

        const float * out_data = out_tensor.GetTensorData<float>();
        out_audio.assign(out_data, out_data + n_samples);

        fprintf(stderr, "onnx_codec: decoded %d codes (stft_len=%lld) -> %zu samples\n",
                n_codes, (long long)stft_len, n_samples);
        return true;

    } catch (const Ort::Exception & e) {
        fprintf(stderr, "onnx_codec: inference failed: %s\n", e.what());
        return false;
    }
}

int OnnxCodecBackend::sample_rate() const {
    return kSampleRate;
}

int OnnxCodecBackend::samples_per_token() const {
    return kSamplesPerToken;
}
