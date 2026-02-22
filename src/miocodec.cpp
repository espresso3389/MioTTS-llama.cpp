#include "miocodec.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// Weight map: lookup tensors by name from GGUF context
// ============================================================

struct weight_map {
    std::map<std::string, ggml_tensor *> tensors;

    void load(ggml_context * ctx_meta) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
            tensors[std::string(t->name)] = t;
        }
    }

    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        return it != tensors.end() ? it->second : nullptr;
    }

    ggml_tensor * require(const std::string & name) const {
        auto t = get(name);
        if (!t) {
            fprintf(stderr, "miocodec: missing tensor: %s\n", name.c_str());
        }
        return t;
    }
};

// ============================================================
// MioCodec context
// ============================================================

struct miocodec_context {
    gguf_context  * ctx_gguf  = nullptr;
    ggml_context  * ctx_meta  = nullptr;
    weight_map      weights;
    ggml_backend_t  backend = nullptr;
    std::unordered_map<std::string, std::vector<uint8_t>> tensor_data_cache;
    size_t tensor_data_cache_bytes = 0;

    // Model parameters (read from GGUF KV)
    int sample_rate   = 44100;
    int n_fft         = 392;
    int hop_length    = 98;
    int n_freq        = 197; // n_fft/2 + 1
    int samples_per_token = 1764;
    int head_out_dim  = 394;

    int prenet_layers = 6;
    int prenet_dim    = 768;
    int prenet_heads  = 12;
    int prenet_ff     = 2048;
    int prenet_window = 65;

    int decoder_layers = 8;
    int decoder_dim    = 512;
    int decoder_heads  = 8;
    int decoder_ff     = 1536;
    int decoder_window = 65;
    int adaln_dim      = 128;

    int resnet_blocks  = 2;
    int resnet_groups  = 32;
    int upsampler_stages = 2;

    float rope_theta     = 10000.0f;
    float norm_eps       = 1e-5f;
    float group_norm_eps = 1e-6f;

    // Upsampler factors and kernel sizes (read from GGUF tensors)
    std::vector<int> up_factors;
    std::vector<int> up_kernels;
};

static bool preload_all_tensor_data(miocodec_context * ctx, const std::string & model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "miocodec: failed to open %s for tensor preload\n", model_path.c_str());
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx->ctx_gguf);
    ctx->tensor_data_cache.clear();
    ctx->tensor_data_cache_bytes = 0;
    ctx->tensor_data_cache.reserve(ctx->weights.tensors.size());

    for (const auto & kv : ctx->weights.tensors) {
        const std::string & name = kv.first;
        ggml_tensor * meta = kv.second;
        if (!meta) {
            continue;
        }
        const size_t nbytes = ggml_nbytes(meta);
        if (nbytes == 0) {
            continue;
        }

        int64_t idx = gguf_find_tensor(ctx->ctx_gguf, name.c_str());
        if (idx < 0) {
            fprintf(stderr, "miocodec: WARNING: tensor '%s' not found in GGUF index\n", name.c_str());
            continue;
        }

        std::vector<uint8_t> bytes(nbytes);
        const size_t offset = data_offset + gguf_get_tensor_offset(ctx->ctx_gguf, idx);
        file.seekg(offset);
        file.read(reinterpret_cast<char *>(bytes.data()), nbytes);
        if (!file.good()) {
            fprintf(stderr, "miocodec: failed to preload tensor '%s' (%zu bytes)\n", name.c_str(), nbytes);
            return false;
        }

        ctx->tensor_data_cache_bytes += nbytes;
        ctx->tensor_data_cache.emplace(name, std::move(bytes));
    }

    return true;
}

static bool read_tensor_data(miocodec_context * ctx, const std::string & name, void * dst, size_t dst_size) {
    auto it = ctx->tensor_data_cache.find(name);
    if (it == ctx->tensor_data_cache.end()) {
        return false;
    }
    if (it->second.size() < dst_size) {
        return false;
    }
    std::memcpy(dst, it->second.data(), dst_size);
    return true;
}

static int gguf_get_val_u32_or(gguf_context * gctx, const char * key, int def) {
    int64_t id = gguf_find_key(gctx, key);
    return id >= 0 ? (int)gguf_get_val_u32(gctx, id) : def;
}

static float gguf_get_val_f32_or(gguf_context * gctx, const char * key, float def) {
    int64_t id = gguf_find_key(gctx, key);
    return id >= 0 ? gguf_get_val_f32(gctx, id) : def;
}

// ============================================================
// Helpers: load weight tensor into compute graph
// ============================================================

static ggml_tensor * load_weight(miocodec_context * mctx, ggml_context * ctx, const std::string & name) {
    ggml_tensor * meta = mctx->weights.get(name);
    if (!meta) return nullptr;
    ggml_tensor * t = ggml_dup_tensor(ctx, meta);
    ggml_set_name(t, name.c_str());
    return t;
}

// Fill all weight tensors that exist in both the GGUF and the compute graph.
static int fill_all_weights(miocodec_context * mctx, ggml_context * ctx) {
    int count = 0, skipped = 0;
    for (auto & [name, _] : mctx->weights.tensors) {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (!t) { continue; }
        if (!t->buffer) {
            if (skipped < 5) fprintf(stderr, "miocodec: WARNING: tensor '%s' has no buffer\n", name.c_str());
            skipped++;
            continue;
        }
        auto it = mctx->tensor_data_cache.find(name);
        if (it != mctx->tensor_data_cache.end()) {
            const size_t nbytes = ggml_nbytes(t);
            if (it->second.size() < nbytes) {
                fprintf(stderr, "miocodec: WARNING: cached tensor '%s' too small (%zu < %zu)\n",
                        name.c_str(), it->second.size(), nbytes);
                continue;
            }
            ggml_backend_tensor_set(t, it->second.data(), 0, nbytes);
            count++;
        } else {
            fprintf(stderr, "miocodec: WARNING: missing cached tensor data for '%s'\n", name.c_str());
        }
    }
    if (skipped > 0) fprintf(stderr, "miocodec: %d tensors skipped (no buffer)\n", skipped);
    return count;
}

// ============================================================
// Graph building helpers
// ============================================================

// Linear: y = W^T @ x + b. x: [dim_in, N], W: [dim_in, dim_out], result: [dim_out, N]
static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    if (b) out = ggml_add(ctx, out, b);
    return out;
}

// LayerNorm along ne[0]: norm(x) * w + b
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * out = ggml_norm(ctx, x, eps);
    out = ggml_mul(ctx, out, w);
    if (b) out = ggml_add(ctx, out, b);
    return out;
}

// SwiGLU FFN: w2(silu(gate(x)) * up(x))
static ggml_tensor * swiglu_ffn(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * w_gate, ggml_tensor * w_up, ggml_tensor * w_down) {
    ggml_tensor * g = ggml_silu(ctx, ggml_mul_mat(ctx, w_gate, x));
    ggml_tensor * u = ggml_mul_mat(ctx, w_up, x);
    return ggml_mul_mat(ctx, w_down, ggml_mul(ctx, g, u));
}

// Transpose 2D: [A, B] → [B, A] (contiguous)
static ggml_tensor * transpose2d(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// Fill local attention mask: 0 for allowed, -INF for masked
static void fill_local_attn_mask(ggml_tensor * mask, int seq_len, int window_size) {
    int half_w = window_size / 2;
    std::vector<float> data(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            data[i * seq_len + j] = (abs(i - j) <= half_w) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(mask, data.data(), 0, data.size() * sizeof(float));
}

// Multi-head attention with RoPE. x: [dim, seq_len] (transformer fmt)
static ggml_tensor * mha_rope(
        ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * wo,
        ggml_tensor * mask, ggml_tensor * pos,
        int n_head, int head_dim, float rope_theta) {
    int64_t dim = x->ne[0], seq_len = x->ne[1];

    ggml_tensor * q = ggml_mul_mat(ctx, wq, x);
    ggml_tensor * k = ggml_mul_mat(ctx, wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx, wv, x);

    q = ggml_reshape_3d(ctx, q, head_dim, n_head, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, seq_len);

    q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, 0, 0,
                       rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, 0, 0,
                       rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute to [head_dim, seq_len, n_head] for manual attention
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // Manual scaled dot-product attention (instead of flash_attn_ext)
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q); // [seq_k, seq_q, n_head]
    scores = ggml_scale(ctx, scores, scale);
    if (mask) scores = ggml_add(ctx, scores, mask); // mask [seq, seq] broadcasts to n_head
    scores = ggml_soft_max(ctx, scores); // softmax along ne[0] (key dim)

    // V @ softmax(QK^T): need V^T for ggml_mul_mat
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v)); // [seq_k, head_dim, n_head]
    ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores); // [head_dim, seq_q, n_head]

    // Reshape back to [dim, seq_len]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // [head_dim, n_head, seq_len]
    attn = ggml_reshape_2d(ctx, attn, dim, seq_len);

    return ggml_mul_mat(ctx, wo, attn);
}

// Pre-norm Transformer layer (no bias on attn). x: [dim, seq_len]
static ggml_tensor * prenet_layer(
        ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * an_w, ggml_tensor * an_b,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * wo,
        ggml_tensor * fn_w, ggml_tensor * fn_b,
        ggml_tensor * w_gate, ggml_tensor * w_up, ggml_tensor * w_down,
        ggml_tensor * mask, ggml_tensor * pos,
        int n_head, int head_dim, float rope_theta, float eps) {
    ggml_tensor * h = layer_norm(ctx, x, an_w, an_b, eps);
    h = mha_rope(ctx, h, wq, wk, wv, wo, mask, pos, n_head, head_dim, rope_theta);
    x = ggml_add(ctx, x, h);

    h = layer_norm(ctx, x, fn_w, fn_b, eps);
    h = swiglu_ffn(ctx, h, w_gate, w_up, w_down);
    x = ggml_add(ctx, x, h);
    return x;
}

// AdaLN-Zero conditioning: SiLU(cond) → Linear → split 3 (shift, scale, gate)
struct adaln3 { ggml_tensor *shift, *scale, *gate; };

static adaln3 compute_adaln3(ggml_context * ctx, ggml_tensor * cond,
                              ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * h = linear(ctx, ggml_silu(ctx, cond), w, b); // [3*dim]
    int64_t dim = h->ne[0] / 3;
    adaln3 p;
    p.shift = ggml_view_1d(ctx, h, dim, 0 * dim * sizeof(float));
    p.scale = ggml_view_1d(ctx, h, dim, 1 * dim * sizeof(float));
    p.gate  = ggml_view_1d(ctx, h, dim, 2 * dim * sizeof(float));
    return p;
}

// AdaLN-Zero norm: norm(x) * (1+scale) + shift. x: [dim, seq_len], shift/scale: [dim]
static ggml_tensor * adaln_norm(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * shift, ggml_tensor * scale,
                                 ggml_tensor * ones, float eps) {
    ggml_tensor * out = ggml_norm(ctx, x, eps);
    ggml_tensor * s = ggml_add(ctx, ones, scale);
    out = ggml_mul(ctx, out, s);
    out = ggml_add(ctx, out, shift);
    return out;
}

// AdaLN-Zero Transformer layer for wave_decoder. x: [dim, seq_len]
static ggml_tensor * decoder_layer(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * cond,
        ggml_tensor * attn_cond_w, ggml_tensor * attn_cond_b,
        ggml_tensor * ffn_cond_w, ggml_tensor * ffn_cond_b,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * wo,
        ggml_tensor * w_gate, ggml_tensor * w_up, ggml_tensor * w_down,
        ggml_tensor * ones,
        ggml_tensor * mask, ggml_tensor * pos,
        int n_head, int head_dim, float rope_theta, float eps) {

    adaln3 ac = compute_adaln3(ctx, cond, attn_cond_w, attn_cond_b);
    ggml_tensor * h = adaln_norm(ctx, x, ac.shift, ac.scale, ones, eps);
    h = mha_rope(ctx, h, wq, wk, wv, wo, mask, pos, n_head, head_dim, rope_theta);
    h = ggml_mul(ctx, h, ac.gate);
    x = ggml_add(ctx, x, h);

    adaln3 fc = compute_adaln3(ctx, cond, ffn_cond_w, ffn_cond_b);
    h = adaln_norm(ctx, x, fc.shift, fc.scale, ones, eps);
    h = swiglu_ffn(ctx, h, w_gate, w_up, w_down);
    h = ggml_mul(ctx, h, fc.gate);
    x = ggml_add(ctx, x, h);
    return x;
}

// GroupNorm for conv-format data: x [length, channels]
static ggml_tensor * conv_group_norm(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * w, ggml_tensor * b,
                                      int n_groups, float eps) {
    int64_t length = x->ne[0], channels = x->ne[1];
    ggml_tensor * x3d = ggml_reshape_3d(ctx, x, length, 1, channels);
    x3d = ggml_group_norm(ctx, x3d, n_groups, eps);
    x = ggml_reshape_2d(ctx, x3d, length, channels);
    // Broadcast weight/bias [channels] as [1, channels] to match [length, channels]
    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, 1, channels);
    ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, channels);
    x = ggml_mul(ctx, x, w2);
    x = ggml_add(ctx, x, b2);
    return x;
}

// Add bias in conv format: x [length, channels], bias [channels] → reshape to [1, channels]
static ggml_tensor * add_conv_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    if (!bias) return x;
    int64_t channels = x->ne[1];
    ggml_tensor * b2d = ggml_reshape_2d(ctx, bias, 1, channels);
    return ggml_add(ctx, x, b2d);
}

// Conv1d wrapper: ggml_conv_1d requires F16 kernel, so cast if needed
static ggml_tensor * conv_1d(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * x,
                              int stride, int pad, int dilation) {
    ggml_tensor * k = (kernel->type == GGML_TYPE_F16) ? kernel : ggml_cast(ctx, kernel, GGML_TYPE_F16);
    return ggml_conv_1d(ctx, k, x, stride, pad, dilation);
}

// ResNet block in conv format: x [length, channels]
static ggml_tensor * resnet_block(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * gn1_w, ggml_tensor * gn1_b,
                                   ggml_tensor * c1_w, ggml_tensor * c1_b,
                                   ggml_tensor * gn2_w, ggml_tensor * gn2_b,
                                   ggml_tensor * c2_w, ggml_tensor * c2_b,
                                   int n_groups, float eps) {
    ggml_tensor * r = x;
    x = conv_group_norm(ctx, x, gn1_w, gn1_b, n_groups, eps);
    x = ggml_silu(ctx, x);
    x = conv_1d(ctx, c1_w, x, 1, 1, 1);
    x = add_conv_bias(ctx, x, c1_b);
    x = conv_group_norm(ctx, x, gn2_w, gn2_b, n_groups, eps);
    x = ggml_silu(ctx, x);
    x = conv_1d(ctx, c2_w, x, 1, 1, 1);
    x = add_conv_bias(ctx, x, c2_b);
    return ggml_add(ctx, x, r);
}

// Snake activation in conv format: x [length, channels], alpha/beta [channels]
// Parameters are in log-space (alpha_logscale=True in Python):
//   snake(x) = x + sin²(exp(alpha) * x) / exp(beta)
static ggml_tensor * snake_activation(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * log_alpha, ggml_tensor * log_beta) {
    int64_t channels = x->ne[1];
    ggml_tensor * a2d = ggml_reshape_2d(ctx, ggml_exp(ctx, log_alpha), 1, channels);
    ggml_tensor * b2d = ggml_reshape_2d(ctx, ggml_exp(ctx, log_beta), 1, channels);
    ggml_tensor * ax = ggml_mul(ctx, x, a2d);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * sc = ggml_div(ctx, s2, b2d);
    return ggml_add(ctx, x, sc);
}

// ============================================================
// MioCodec API
// ============================================================

miocodec_context * miocodec_load(const std::string & model_path) {
    auto * ctx = new miocodec_context();

    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    ctx->ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx->ctx_gguf) {
        fprintf(stderr, "miocodec: failed to parse GGUF: %s\n", model_path.c_str());
        delete ctx;
        return nullptr;
    }
    ctx->ctx_meta = meta;
    ctx->weights.load(meta);

    if (!preload_all_tensor_data(ctx, model_path)) {
        fprintf(stderr, "miocodec: tensor preload failed\n");
        miocodec_free(ctx);
        return nullptr;
    }

    // Read model parameters from GGUF KV
    auto * g = ctx->ctx_gguf;
    ctx->sample_rate   = gguf_get_val_u32_or(g, "miocodec.sample_rate", 44100);
    ctx->n_fft         = gguf_get_val_u32_or(g, "miocodec.n_fft", 392);
    ctx->hop_length    = gguf_get_val_u32_or(g, "miocodec.hop_length", 98);
    ctx->n_freq        = ctx->n_fft / 2 + 1;
    ctx->samples_per_token = gguf_get_val_u32_or(g, "miocodec.samples_per_token", 1764);
    ctx->head_out_dim  = gguf_get_val_u32_or(g, "embedding_length_out", 394);

    ctx->prenet_layers = gguf_get_val_u32_or(g, "miocodec.prenet_layers", 6);
    ctx->prenet_dim    = gguf_get_val_u32_or(g, "miocodec.prenet_dim", 768);
    ctx->prenet_heads  = gguf_get_val_u32_or(g, "miocodec.prenet_heads", 12);
    ctx->prenet_ff     = gguf_get_val_u32_or(g, "miocodec.prenet_ff", 2048);
    ctx->prenet_window = gguf_get_val_u32_or(g, "miocodec.prenet_window", 65);

    ctx->decoder_layers = gguf_get_val_u32_or(g, "miocodec.decoder_layers", 8);
    ctx->decoder_dim    = gguf_get_val_u32_or(g, "miocodec.decoder_dim", 512);
    ctx->decoder_heads  = gguf_get_val_u32_or(g, "miocodec.decoder_heads", 8);
    ctx->decoder_ff     = gguf_get_val_u32_or(g, "miocodec.decoder_ff", 1536);
    ctx->decoder_window = gguf_get_val_u32_or(g, "miocodec.decoder_window", 65);
    ctx->adaln_dim      = gguf_get_val_u32_or(g, "miocodec.decoder_adanorm_dim", 128);

    ctx->resnet_blocks  = gguf_get_val_u32_or(g, "miocodec.resnet_blocks", 2);
    ctx->resnet_groups  = gguf_get_val_u32_or(g, "miocodec.resnet_groups", 32);
    ctx->upsampler_stages = gguf_get_val_u32_or(g, "miocodec.wave_upsampler_layers", 0);

    ctx->rope_theta     = gguf_get_val_f32_or(g, "miocodec.rope_theta", 10000.0f);
    ctx->norm_eps       = gguf_get_val_f32_or(g, "miocodec.norm_eps", 1e-5f);
    ctx->group_norm_eps = gguf_get_val_f32_or(g, "miocodec.group_norm_eps", 1e-6f);

    // Read upsampler factors and kernel sizes from GGUF tensors
    int n_up = ctx->upsampler_stages;
    ctx->up_factors.resize(n_up);
    ctx->up_kernels.resize(n_up);
    if (n_up > 0) {
        read_tensor_data(ctx, "miocodec.wave_upsampler.factors", ctx->up_factors.data(), n_up * sizeof(int));
        read_tensor_data(ctx, "miocodec.wave_upsampler.kernel_sizes", ctx->up_kernels.data(), n_up * sizeof(int));
    }

    // CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "miocodec: failed to init CPU backend\n");
        miocodec_free(ctx);
        return nullptr;
    }

    int total_up = 2; // initial wave_upsample 2x
    for (int i = 0; i < n_up; i++) total_up *= ctx->up_factors[i];

    fprintf(stderr, "miocodec: %lld tensors, sr=%d, n_fft=%d, hop=%d, head=%d\n",
            (long long)gguf_get_n_tensors(ctx->ctx_gguf),
            ctx->sample_rate, ctx->n_fft, ctx->hop_length, ctx->head_out_dim);
    fprintf(stderr, "miocodec: preloaded %.2f MiB tensor data into memory cache\n",
            (double) ctx->tensor_data_cache_bytes / (1024.0 * 1024.0));
    if (n_up > 0) {
        fprintf(stderr, "miocodec: upsampler stages=%d total=%dx\n", n_up, total_up);
    } else {
        fprintf(stderr, "miocodec: no upsampler (S_final = S_dec)\n");
    }

    return ctx;
}

void miocodec_free(miocodec_context * ctx) {
    if (!ctx) return;
    if (ctx->backend)  ggml_backend_free(ctx->backend);
    if (ctx->ctx_gguf) gguf_free(ctx->ctx_gguf);
    if (ctx->ctx_meta) ggml_free(ctx->ctx_meta);
    delete ctx;
}

int miocodec_sample_rate(const miocodec_context * ctx) { return ctx->sample_rate; }
int miocodec_n_fft(const miocodec_context * ctx) { return ctx->n_fft; }
int miocodec_hop_length(const miocodec_context * ctx) { return ctx->hop_length; }
int miocodec_samples_per_token(const miocodec_context * ctx) { return ctx->samples_per_token; }

std::vector<float> miocodec_decode(
        miocodec_context * mctx,
        const int * codes,
        int n_codes,
        const float * global_emb,
        int audio_length,
        int * out_n_frames) {

    const int n_fft      = mctx->n_fft;
    const int hop_length = mctx->hop_length;
    const int n_freq     = mctx->n_freq;
    const int head_dim_out = mctx->head_out_dim;
    const int dec_dim    = mctx->decoder_dim;
    const int pre_dim    = mctx->prenet_dim;
    const int adaln_dim  = mctx->adaln_dim;
    const float norm_eps = mctx->norm_eps;
    const float gn_eps   = mctx->group_norm_eps;

    // Compute temporal sizes through the pipeline
    int T = n_codes;
    int S_dec = T * 2; // sequence length for prior/decoder/post (after conv_upsample 2x)
    int up_total = 1;
    for (int i = 0; i < mctx->upsampler_stages; i++) {
        up_total *= mctx->up_factors[i];
    }
    int S_final = S_dec * up_total; // STFT frame count (after upsampler)

    if (audio_length == 0) {
        audio_length = T * mctx->samples_per_token;
    }
    if (out_n_frames) *out_n_frames = S_final;

    fprintf(stderr, "miocodec: T=%d codes → S_dec=%d (decoder) → S_final=%d (STFT) → %d audio samples\n",
            T, S_dec, S_final, audio_length);

    // ================================================================
    // Build the full ggml compute graph
    // ================================================================

    size_t ctx_size = 512ull * 1024 * 1024;
    ggml_init_params gparams = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(gparams);
    if (!ctx) {
        fprintf(stderr, "miocodec: failed to init ggml context\n");
        return {};
    }

    auto W = [&](const std::string & name) -> ggml_tensor * {
        return load_weight(mctx, ctx, name);
    };

    // ---- Input tensors ----

    ggml_tensor * code_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(code_indices, "code_indices");

    ggml_tensor * g_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, adaln_dim);
    ggml_set_name(g_emb, "global_emb");

    ggml_tensor * pos_pre = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos_pre, "pos_pre");

    ggml_tensor * pos_dec = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_dec);
    ggml_set_name(pos_dec, "pos_dec");

    // "ones" tensors for AdaLN: (1 + scale)
    ggml_tensor * ones_dec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec_dim);
    ggml_set_name(ones_dec, "ones_dec");

    ggml_tensor * ones_final = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec_dim);
    ggml_set_name(ones_final, "ones_final");

    // Attention masks
    ggml_tensor * pre_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_name(pre_mask, "pre_mask");

    ggml_tensor * dec_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_dec, S_dec);
    ggml_set_name(dec_mask, "dec_mask");

    // ---- 1. Token embedding lookup ----
    ggml_tensor * tok_emb = W("token_embd");
    ggml_tensor * x = ggml_get_rows(ctx, tok_emb, code_indices);
    // x: [768, T] (transformer format)

    // ---- 2. Wave prenet ----
    int head_dim_pre = pre_dim / mctx->prenet_heads;
    for (int i = 0; i < mctx->prenet_layers; i++) {
        std::string p = "wave_prenet.blk." + std::to_string(i) + ".";
        x = prenet_layer(ctx, x,
            W(p+"attn_norm.weight"), W(p+"attn_norm.bias"),
            W(p+"attn_q.weight"), W(p+"attn_k.weight"), W(p+"attn_v.weight"), W(p+"attn_output.weight"),
            W(p+"ffn_norm.weight"), W(p+"ffn_norm.bias"),
            W(p+"ffn_gate.weight"), W(p+"ffn_up.weight"), W(p+"ffn_down.weight"),
            pre_mask, pos_pre,
            mctx->prenet_heads, head_dim_pre, mctx->rope_theta, norm_eps);
    }

    // Prenet final norm + output projection (768 → 512)
    x = layer_norm(ctx, x, W("wave_prenet.norm.weight"), W("wave_prenet.norm.bias"), norm_eps);
    x = linear(ctx, x, W("wave_prenet.output.weight"), W("wave_prenet.output.bias"));

    // x: [512, T]

    // ---- 3. wave_upsample: ConvTranspose1d(512, 512, k=2, s=2) ----
    x = transpose2d(ctx, x); // [T, 512] conv format
    x = ggml_conv_transpose_1d(ctx, W("wave_upsample.weight"), x, 2, 0, 1);
    x = add_conv_bias(ctx, x, W("wave_upsample.bias"));
    // x: [T*2, 512] = [S_dec, 512] conv format

    // ---- 4. wave_prior: 2 ResNet blocks ----
    for (int blk = 0; blk < mctx->resnet_blocks; blk++) {
        std::string p = "wave_prior." + std::to_string(blk) + ".";
        x = resnet_block(ctx, x,
            W(p+"norm1.weight"), W(p+"norm1.bias"),
            W(p+"conv1.weight"), W(p+"conv1.bias"),
            W(p+"norm2.weight"), W(p+"norm2.bias"),
            W(p+"conv2.weight"), W(p+"conv2.bias"),
            mctx->resnet_groups, gn_eps);
    }
    x = transpose2d(ctx, x); // [512, S_dec] transformer format
    // ---- 5. wave_decoder: 8 AdaLN-Zero Transformer layers ----
    int head_dim_dec = dec_dim / mctx->decoder_heads;
    for (int i = 0; i < mctx->decoder_layers; i++) {
        std::string p = "wave_decoder.blk." + std::to_string(i) + ".";
        x = decoder_layer(ctx, x, g_emb,
            W(p+"attn_cond.weight"), W(p+"attn_cond.bias"),
            W(p+"ffn_cond.weight"), W(p+"ffn_cond.bias"),
            W(p+"attn_q.weight"), W(p+"attn_k.weight"), W(p+"attn_v.weight"), W(p+"attn_output.weight"),
            W(p+"ffn_gate.weight"), W(p+"ffn_up.weight"), W(p+"ffn_down.weight"),
            ones_dec, dec_mask, pos_dec,
            mctx->decoder_heads, head_dim_dec, mctx->rope_theta, norm_eps);
    }

    // ---- 6. Final AdaLN norm ----
    {
        ggml_tensor * nc = linear(ctx, ggml_silu(ctx, g_emb),
            W("wave_decoder.norm_cond.weight"), W("wave_decoder.norm_cond.bias"));
        // nc: [1024] → split shift[512], scale[512]
        ggml_tensor * shift = ggml_view_1d(ctx, nc, dec_dim, 0);
        ggml_tensor * scale = ggml_view_1d(ctx, nc, dec_dim, dec_dim * sizeof(float));
        x = adaln_norm(ctx, x, shift, scale, ones_final, norm_eps);
    }

    // ---- 7. wave_post: 2 ResNet blocks ----
    x = transpose2d(ctx, x); // [S_dec, 512] conv format
    for (int blk = 0; blk < mctx->resnet_blocks; blk++) {
        std::string p = "wave_post." + std::to_string(blk) + ".";
        x = resnet_block(ctx, x,
            W(p+"norm1.weight"), W(p+"norm1.bias"),
            W(p+"conv1.weight"), W(p+"conv1.bias"),
            W(p+"norm2.weight"), W(p+"norm2.bias"),
            W(p+"conv2.weight"), W(p+"conv2.bias"),
            mctx->resnet_groups, gn_eps);
    }
    // x: [S_dec, 512] conv format

    // ---- 8. wave_upsampler stages ----
    // Python: wave_upsampler is applied AFTER wave_post, BEFORE istft_head
    if (mctx->upsampler_stages > 0) {
        for (int stage = 0; stage < mctx->upsampler_stages; stage++) {
            std::string p = "wave_upsampler.";
            int factor = mctx->up_factors[stage];
            int kernel = mctx->up_kernels[stage];
            int trim_pad = (kernel - factor) / 2; // manual trim since ggml asserts p0==0

            // ConvTranspose1d (padding=0, then trim)
            std::string up_name = p + "up." + std::to_string(stage);
            x = ggml_conv_transpose_1d(ctx, W(up_name + ".weight"), x, factor, 0, 1);
            x = add_conv_bias(ctx, x, W(up_name + ".bias"));

            // Trim excess from each side: out_raw = in*factor + (kernel-factor)
            if (trim_pad > 0) {
                int64_t raw_len = x->ne[0];
                int64_t out_ch  = x->ne[1];
                int64_t target  = raw_len - 2 * trim_pad;
                x = ggml_view_2d(ctx, x, target, out_ch, x->nb[1],
                                 (size_t)trim_pad * x->nb[0]);
                x = ggml_cont(ctx, x);
            }
            // Snake activation
            std::string sn = p + "snake." + std::to_string(stage);
            x = snake_activation(ctx, x, W(sn + ".alpha"), W(sn + ".beta"));
            // ResNet block
            std::string rb = p + "resblk." + std::to_string(stage) + ".";
            x = resnet_block(ctx, x,
                W(rb+"norm1.weight"), W(rb+"norm1.bias"),
                W(rb+"conv1.weight"), W(rb+"conv1.bias"),
                W(rb+"norm2.weight"), W(rb+"norm2.bias"),
                W(rb+"conv2.weight"), W(rb+"conv2.bias"),
                mctx->resnet_groups, gn_eps);
        }
        // x: [S_final, 128] conv format

        // Upsampler output: transpose → Linear(128→512) → Snake (transformer format)
        x = transpose2d(ctx, x); // [128, S_final]
        x = linear(ctx, x, W("wave_upsampler.out_proj.weight"), W("wave_upsampler.out_proj.bias"));
        // x: [512, S_final] transformer format
        // out_snake: log-scale alpha/beta [512] — in transformer format ne[0]=512 matches alpha dim
        // Formula: x + sin²(exp(alpha) * x) / exp(beta)
        {
            ggml_tensor * a = ggml_exp(ctx, W("wave_upsampler.out_snake.alpha"));
            ggml_tensor * b = ggml_exp(ctx, W("wave_upsampler.out_snake.beta"));
            ggml_tensor * ax = ggml_mul(ctx, x, a);
            ggml_tensor * s  = ggml_sin(ctx, ax);
            ggml_tensor * s2 = ggml_mul(ctx, s, s);
            ggml_tensor * sc = ggml_div(ctx, s2, b);
            x = ggml_add(ctx, x, sc);
        }
    } else {
        // No upsampler: x is [S_dec, 512] conv format → transpose to transformer format
        x = transpose2d(ctx, x); // [512, S_final]
    }

    // ---- 9. iSTFT head: Linear(512, 394) → mag + phase ----
    x = linear(ctx, x, W("istft_head.out.weight"), W("istft_head.out.bias"));
    // x: [394, S_final]

    // Split into log_mag [197, S_final] and phase [197, S_final]
    ggml_tensor * log_mag = ggml_view_2d(ctx, x, n_freq, S_final, x->nb[1], 0);
    ggml_tensor * phase   = ggml_view_2d(ctx, x, n_freq, S_final, x->nb[1], n_freq * sizeof(float));

    ggml_tensor * mag = ggml_clamp(ctx, ggml_exp(ctx, log_mag), 0.0f, 100.0f);
    ggml_tensor * real_part = ggml_mul(ctx, mag, ggml_cos(ctx, phase));
    ggml_tensor * imag_part = ggml_mul(ctx, mag, ggml_sin(ctx, phase));

    ggml_set_name(real_part, "spec_real");
    ggml_set_name(imag_part, "spec_imag");

    // ---- Build and compute graph ----
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, real_part);
    ggml_build_forward_expand(graph, imag_part);

    fprintf(stderr, "miocodec: graph has %d nodes\n", ggml_graph_n_nodes(graph));

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->backend));
    if (!ggml_gallocr_reserve(allocr, graph)) {
        fprintf(stderr, "miocodec: failed to reserve allocator\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return {};
    }
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "miocodec: failed to alloc graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return {};
    }

    // Fill inputs
    ggml_backend_tensor_set(code_indices, codes, 0, T * sizeof(int));
    ggml_backend_tensor_set(g_emb, global_emb, 0, adaln_dim * sizeof(float));

    std::vector<int32_t> pos_pre_data(T), pos_dec_data(S_dec);
    for (int i = 0; i < T; i++) pos_pre_data[i] = i;
    for (int i = 0; i < S_dec; i++) pos_dec_data[i] = i;
    ggml_backend_tensor_set(pos_pre, pos_pre_data.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(pos_dec, pos_dec_data.data(), 0, S_dec * sizeof(int32_t));

    fill_local_attn_mask(pre_mask, T, mctx->prenet_window);
    fill_local_attn_mask(dec_mask, S_dec, mctx->decoder_window);

    // Ones tensors
    std::vector<float> ones_data(dec_dim, 1.0f);
    ggml_backend_tensor_set(ones_dec, ones_data.data(), 0, dec_dim * sizeof(float));
    ggml_backend_tensor_set(ones_final, ones_data.data(), 0, dec_dim * sizeof(float));

    // Load all weight data
    int n_loaded = fill_all_weights(mctx, ctx);

    fprintf(stderr, "miocodec: loaded %d weight tensors\n", n_loaded);
    fprintf(stderr, "miocodec: computing...\n");
    if (ggml_backend_graph_compute(mctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "miocodec: graph compute failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return {};
    }

    // Read back spectrogram: interleave [real, imag] per frame
    std::vector<float> spec_r(n_freq * S_final), spec_i(n_freq * S_final);
    ggml_backend_tensor_get(real_part, spec_r.data(), 0, spec_r.size() * sizeof(float));
    ggml_backend_tensor_get(imag_part, spec_i.data(), 0, spec_i.size() * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_free(ctx);

    // Interleave: [S_final frames, n_freq*2 (real,imag pairs)]
    std::vector<float> spec(S_final * n_freq * 2);
    for (int t = 0; t < S_final; t++) {
        for (int f = 0; f < n_freq; f++) {
            spec[t * n_freq * 2 + f * 2 + 0] = spec_r[f + t * n_freq];
            spec[t * n_freq * 2 + f * 2 + 1] = spec_i[f + t * n_freq];
        }
    }
    return spec;
}

// ============================================================
// Voice embedding loader
// ============================================================

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<float> load_voice_embedding(const std::string & path) {
    // Raw binary format: 128 float32 values (512 bytes)
    if (ends_with(path, ".bin")) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            fprintf(stderr, "voice_emb: failed to open %s\n", path.c_str());
            return {};
        }
        auto file_size = f.tellg();
        f.seekg(0);
        int n = static_cast<int>(file_size / sizeof(float));
        if (n <= 0) {
            fprintf(stderr, "voice_emb: empty file %s\n", path.c_str());
            return {};
        }
        std::vector<float> emb(n);
        f.read(reinterpret_cast<char *>(emb.data()), n * sizeof(float));
        fprintf(stderr, "voice_emb: loaded %d-dim from %s (raw binary)\n", n, path.c_str());
        return emb;
    }

    // GGUF format
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        fprintf(stderr, "voice_emb: failed to open %s\n", path.c_str());
        return {};
    }

    ggml_tensor * t = ggml_get_first_tensor(meta);
    if (!t) {
        fprintf(stderr, "voice_emb: no tensors in %s\n", path.c_str());
        gguf_free(gctx);
        ggml_free(meta);
        return {};
    }

    int n = ggml_nelements(t);
    size_t data_off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, 0);

    std::ifstream f(path, std::ios::binary);
    f.seekg(data_off);

    std::vector<float> emb(n);
    if (t->type == GGML_TYPE_F32) {
        f.read(reinterpret_cast<char *>(emb.data()), n * sizeof(float));
    } else {
        fprintf(stderr, "voice_emb: unsupported type %d\n", t->type);
        gguf_free(gctx);
        ggml_free(meta);
        return {};
    }

    fprintf(stderr, "voice_emb: loaded %d-dim from %s\n", n, path.c_str());
    gguf_free(gctx);
    ggml_free(meta);
    return emb;
}

// ============================================================
// Debug: print tensor names
// ============================================================

void miocodec_print_tensors(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        return;
    }

    printf("Tensors in %s: %lld\n", path.c_str(), (long long)gguf_get_n_tensors(gctx));
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        printf("  %-60s [%5lld, %5lld, %5lld, %5lld] type=%d\n",
               t->name, (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3], t->type);
    }

    gguf_free(gctx);
    ggml_free(meta);
}
