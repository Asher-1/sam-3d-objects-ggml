// 检查 F16 输入 matmul 的数值、尾部 tile 和重复执行；参考值从已编码权重标量累加。
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float sample(uint32_t index, uint32_t salt) {
    uint32_t value = index * 747796405u + salt * 2891336453u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    value = (value >> 22u) ^ value;
    return (static_cast<int>(value % 257u) - 128) / 128.0f;
}

bool check(const char* backend_name, ggml_type weight_type, ggml_type input_type,
           int64_t m, int64_t k, int64_t n, bool cast_input) {
    sam3d::GraphContext context;
    auto backend = sam3d::Backend::create(backend_name, 6);
    if (!backend) return false;
    ggml_tensor* weight = ggml_new_tensor_2d(context.ctx(), weight_type, k, n);
    ggml_set_name(weight, "regression.weight");
    ggml_set_input(weight);
    ggml_tensor* source = ggml_new_tensor_2d(context.ctx(), cast_input ? GGML_TYPE_F32 : input_type, k, m);
    ggml_set_name(source, "regression.input");
    ggml_set_input(source);
    ggml_tensor* input = cast_input ? ggml_cast(context.ctx(), source, input_type) : source;
    ggml_tensor* output = ggml_mul_mat(context.ctx(), weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    ggml_set_output(output);
    auto* graph = ggml_new_graph_custom(context.ctx(), 256, false);
    ggml_build_forward_expand(graph, output);
    if (!backend->alloc(graph)) return false;

    std::vector<float> weights(static_cast<size_t>(k * n));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = sample(i, 13);
    std::vector<uint8_t> packed(ggml_nbytes(weight));
    if (weight_type == GGML_TYPE_F32) {
        std::memcpy(packed.data(), weights.data(), packed.size());
    } else {
        if (ggml_quantize_chunk(weight_type, weights.data(), packed.data(), 0, n, k, nullptr) != packed.size()) return false;
        ggml_get_type_traits(weight_type)->to_float(packed.data(), weights.data(), weights.size());
    }
    ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());

    bool passed = true;
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
        std::vector<float> x(static_cast<size_t>(k * m));
        for (size_t i = 0; i < x.size(); ++i) x[i] = sample(i, 19 + iteration);
        if (source->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(x.size());
            ggml_fp32_to_fp16_row(x.data(), half.data(), x.size());
            ggml_backend_tensor_set(source, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(source, x.data(), 0, x.size() * sizeof(float));
        }
        if (!backend->run(graph)) return false;
        std::vector<float> actual;
        if (!backend->get_tensor_f32(output, actual)) return false;
        size_t nonfinite = 0;
        for (float value : actual) nonfinite += !std::isfinite(value);
        double max_error = 0, error_sum = 0, aa = 0, bb = 0, ab = 0, a = 0, b = 0;
        // 全量有限性检查 O(MN)，独立 FP64 点积仅采样 1024 点，避免回归测试执行 O(MNK) 标量循环。
        constexpr size_t probes = 1024;
        for (size_t probe = 0; probe < probes; ++probe) {
            const size_t offset = probe == 0 ? actual.size() - 1 :
                (probe * 104729u + iteration * 37u) % actual.size();
            const size_t row = offset / n, column = offset % n;
            double expected = 0;
            for (int64_t inner = 0; inner < k; ++inner) {
                expected += static_cast<double>(weights[column * k + inner]) * x[row * k + inner];
            }
            const double got = actual[offset];
            const double error = std::abs(got - expected);
            max_error = std::max(max_error, error);
            error_sum += error;
            a += expected; b += got; aa += expected * expected; bb += got * got; ab += expected * got;
        }
        const double denominator = std::sqrt((probes * aa - a * a) * (probes * bb - b * b));
        const double correlation = denominator > 0 ? (probes * ab - a * b) / denominator : 0;
        const bool ok = nonfinite == 0 && std::isfinite(correlation) && correlation >= 0.9999 && max_error <= 0.05;
        std::printf("%s weights=%s input=%s cast=%d M=%lld K=%lld N=%lld repeat=%u nonfinite=%zu mae=%.9g max=%.9g corr=%.9g %s\n",
                    backend_name, ggml_type_name(weight_type), ggml_type_name(input_type), cast_input,
                    static_cast<long long>(m), static_cast<long long>(k), static_cast<long long>(n),
                    iteration, nonfinite, error_sum / probes, max_error, correlation, ok ? "PASS" : "FAIL");
        std::fflush(stdout);
        passed &= ok;
    }
    return passed;
}

} // namespace

// Production-shape GEMM benchmark: run the SLat flow's dominant matmul
// shapes repeatedly and report the wall time per run. The point is a direct
// Vulkan-vs-CUDA comparison of the exact GEMM kernels the flow graphs
// dispatch, independent of graph scheduling effects.
int bench(const char* backend_name) {
    struct Shape { int64_t m, k, n; const char* label; };
    const Shape shapes[] = {
        {5239, 1024, 3072, "slat qkv/torso"},
        {26281, 128, 128, "input projection"},
        {26281, 1024, 128, "out layer"},
        {7528, 1024, 2048, "ss cond proj"},
        {4096, 4096, 1024, "ss ffn up"},
        {4096, 1024, 4096, "ss ffn down/qkv"},
        {4096, 1024, 1024, "ss torso"},
        {4096, 1024, 3072, "ss qkv"},
    };
    for (const Shape& shape : shapes) {
        for (auto weight : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            sam3d::GraphContext context;
            auto backend = sam3d::Backend::create(backend_name, 6);
            if (!backend) return 1;
            ggml_tensor* w = ggml_new_tensor_2d(context.ctx(), weight, shape.k, shape.n);
            ggml_tensor* in = ggml_new_tensor_2d(context.ctx(), GGML_TYPE_F16, shape.k, shape.m);
            ggml_tensor* out = ggml_mul_mat(context.ctx(), w, in);
            ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            auto* graph = ggml_new_graph_custom(context.ctx(), 16, false);
            ggml_build_forward_expand(graph, out);
            if (!backend->alloc(graph)) return 1;
            // Deterministic fill; values only need to be finite.
            const size_t w_elems = static_cast<size_t>(shape.k * shape.n);
            std::vector<float> wf(w_elems, 0.01f);
            std::vector<uint8_t> packed(ggml_nbytes(w));
            if (weight == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(w_elems);
                ggml_fp32_to_fp16_row(wf.data(), half.data(), w_elems);
                std::memcpy(packed.data(), half.data(), packed.size());
            } else if (weight == GGML_TYPE_F32) {
                std::memcpy(packed.data(), wf.data(), packed.size());
            } else {
                ggml_quantize_chunk(weight, wf.data(), packed.data(), 0, shape.n, shape.k, nullptr);
            }
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
            std::vector<ggml_fp16_t> in_half(static_cast<size_t>(shape.k * shape.m),
                                             ggml_fp32_to_fp16(0.02f));
            ggml_backend_tensor_set(in, in_half.data(), 0, in_half.size() * sizeof(ggml_fp16_t));
            if (!backend->run(graph)) return 1;  // warm up (pipeline/shader compile)
            const int runs = 20;
            const auto started = std::chrono::steady_clock::now();
            for (int i = 0; i < runs; ++i) {
                if (!backend->run(graph)) return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count() / runs;
            const double tflops = 2.0 * shape.m * shape.k * shape.n / (ms * 1e-3) / 1e12;
            std::printf("%s bench M=%lld K=%lld N=%lld w=%s: %.3f ms/run  %.2f TFLOPS  (%s)\n",
                        backend_name, (long long)shape.m, (long long)shape.k,
                        (long long)shape.n, ggml_type_name(weight), ms, tflops, shape.label);
            std::fflush(stdout);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    ggml_backend_load_all();
    const char* backend = argc > 1 ? argv[1] : "cpu";
    if (argc > 2 && std::strcmp(argv[2], "--bench") == 0) return bench(backend);
    const bool large = argc > 2 && std::strcmp(argv[2], "--large") == 0;
    // The CPU matmul path converts src1 to the weight's vec_dot_type and only
    // accepts F32 sources for that conversion (ggml-cpu.c), so on CPU the
    // input type must equal the weight's dot type. GPU backends dequantize in
    // the shader and take F16 inputs directly.
    const bool is_cpu = std::strcmp(backend, "cpu") == 0;
    bool passed = true;
    for (auto weight : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        for (auto input : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (bool cast : {false, true}) {
                if (cast && input == GGML_TYPE_F32) continue;
                if (is_cpu && input != weight) continue;
                for (int64_t m : {31, 32, 33, 65}) passed &= check(backend, weight, input, m, 256, 65, cast);
                if (large) passed &= check(backend, weight, input, 5239, 1024, 3072, cast);
            }
        }
    }
    return passed ? 0 : 1;
}
