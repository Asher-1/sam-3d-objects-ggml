// Regression for the silent Vulkan skip of GGML_OP_SAM3D_SPARSE_SCATTER_MEAN:
// the op had no Vulkan dispatch, so the destination buffer kept allocator
// garbage and poisoned the whole SLat flow. The test requires the Vulkan
// output to be bit-equal to the deterministic CPU fallback (fine-row
// ascending F16 accumulation, F32 opmath division).
#include "backend.hpp"
#include "graph_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    int64_t channels;
    int64_t fine_rows;
    int64_t coarse_rows;
    int32_t seed;
};

bool run_case(const char* backend_name, const Case& c, std::vector<float>& out, int n_threads,
              std::vector<ggml_fp16_t>* features_dump = nullptr,
              std::vector<int32_t>* rows_dump = nullptr,
              std::vector<float>* counts_dump = nullptr) {
    sam3d::GraphContext context;
    auto backend = sam3d::Backend::create(backend_name, n_threads);
    if (!backend) return false;

    // Weights must not be attached to this graph: every tensor is computed.
    ggml_tensor* features = ggml_new_tensor_2d(context.ctx(), GGML_TYPE_F16, c.channels, c.fine_rows);
    ggml_set_input(features);
    ggml_set_name(features, "scatter.features");
    ggml_tensor* rows = ggml_new_tensor_1d(context.ctx(), GGML_TYPE_I32, c.fine_rows);
    ggml_set_input(rows);
    ggml_set_name(rows, "scatter.rows");
    ggml_tensor* counts = ggml_new_tensor_2d(context.ctx(), GGML_TYPE_F32, 1, c.coarse_rows);
    ggml_set_input(counts);
    ggml_set_name(counts, "scatter.counts");

    ggml_tensor* dst = ggml_sam3d_sparse_scatter_mean(context.ctx(), features, rows, counts);
    ggml_set_output(dst);
    ggml_set_name(dst, "scatter.dst");

    auto* graph = ggml_new_graph_custom(context.ctx(), 64, false);
    ggml_build_forward_expand(graph, dst);
    if (!backend->alloc(graph)) return false;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
    std::vector<ggml_fp16_t> features_host(c.channels * c.fine_rows);
    for (auto& v : features_host) v = ggml_fp32_to_fp16(uniform(rng));
    // Two-pass construction: the first coarse_rows fine rows pin one
    // guaranteed hit per coarse row, the rest draw random destinations
    // (including the skip values). Counts are recomputed from the final
    // mapping, so every division has a nonzero count and no 0/0 NaN can
    // appear for reasons unrelated to the operator under test.
    std::vector<int32_t> rows_host(c.fine_rows);
    std::uniform_int_distribution<int32_t> coarse_pick(0, (int32_t)c.coarse_rows - 1);
    const int64_t pinned = std::min(c.coarse_rows, c.fine_rows);
    for (int64_t i = 0; i < pinned; ++i) rows_host[i] = (int32_t)i;
    for (int64_t i = pinned; i < c.fine_rows; ++i) {
        const uint32_t draw = rng() % 16;
        if (draw == 15) rows_host[i] = -1;                            // skipped
        else if (draw == 14) rows_host[i] = (int32_t)c.coarse_rows;   // skipped
        else rows_host[i] = coarse_pick(rng);
    }
    std::vector<float> counts_host(c.coarse_rows, 0.0f);
    for (int64_t i = 0; i < c.fine_rows; ++i) {
        const int32_t r = rows_host[i];
        if (r >= 0 && r < (int32_t)c.coarse_rows) counts_host[r] += 1.0f;
    }

    ggml_backend_tensor_set(features, features_host.data(), 0, features_host.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(rows, rows_host.data(), 0, rows_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(counts, counts_host.data(), 0, counts_host.size() * sizeof(float));
    if (features_dump) *features_dump = features_host;
    if (rows_dump) *rows_dump = rows_host;
    if (counts_dump) *counts_dump = counts_host;

    if (!backend->run(graph)) return false;
    return backend->get_tensor_f32(dst, out) && out.size() == (size_t)(c.channels * c.coarse_rows);
}

} // namespace

int main(int argc, char** argv) {
    ggml_backend_load_all();
    const char* backend = argc > 1 ? argv[1] : "cpu";
    const int n_threads = argc > 2 ? std::atoi(argv[2]) : 4;
    bool passed = true;
    const Case cases[] = {
        // degenerate and tile-boundary shapes first
        {4, 1, 1, 1},
        {4, 7, 3, 2},
        {33, 65, 17, 3},
        {128, 512, 100, 4},
        // production SLat downsample shape (128 channels, 26281 fine, 5239 coarse)
        {128, 26281, 5239, 5},
    };
    for (const Case& c : cases) {
        const bool dump = argc > 3 && std::strcmp(argv[3], "--dump") == 0 &&
            c.channels == 4 && c.fine_rows == 7 && c.coarse_rows == 3;
        std::vector<ggml_fp16_t> feat_host;
        std::vector<int32_t> rows_host_dump;
        std::vector<float> counts_host_dump;
        std::vector<float> reference;
        if (!run_case("cpu", c, reference, n_threads,
                      dump ? &feat_host : nullptr, dump ? &rows_host_dump : nullptr,
                      dump ? &counts_host_dump : nullptr)) {
            std::printf("%s C(%lld,%lld,%lld) CPU RUN FAIL\n", backend,
                        (long long)c.channels, (long long)c.fine_rows, (long long)c.coarse_rows);
            passed = false;
            continue;
        }
        std::vector<float> out;
        if (!run_case(backend, c, out, n_threads)) {
            std::printf("%s C(%lld,%lld,%lld) RUN FAIL\n", backend,
                        (long long)c.channels, (long long)c.fine_rows, (long long)c.coarse_rows);
            passed = false;
            continue;
        }
        if (dump) {
            std::printf("features:");
            for (auto v : feat_host) std::printf(" %g", ggml_fp16_to_fp32(v));
            std::printf("\nrows:");
            for (auto v : rows_host_dump) std::printf(" %d", v);
            std::printf("\ncounts:");
            for (auto v : counts_host_dump) std::printf(" %g", v);
            std::printf("\nref:");
            for (float v : reference) std::printf(" %.9g", v);
            std::printf("\nout:");
            for (float v : out) std::printf(" %.9g", v);
            std::printf("\n");
        }
        // Bit-exact against the CPU fallback: both accumulate in F16 over
        // ascending fine rows and divide in F32 opmath, so any difference is
        // a real defect, not reduction-order noise. F16 -> F32 readback is
        // lossless, so comparing the F32 views compares the F16 payloads.
        size_t mismatches = 0, first_bad = 0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < out.size() && i < reference.size(); ++i) {
            nonfinite += !std::isfinite(out[i]);
            uint32_t got_bits = 0, ref_bits = 0;
            std::memcpy(&got_bits, &out[i], sizeof(got_bits));
            std::memcpy(&ref_bits, &reference[i], sizeof(ref_bits));
            if (got_bits != ref_bits) {
                if (mismatches == 0) first_bad = i;
                ++mismatches;
            }
        }
        const bool ok = out.size() == reference.size() && mismatches == 0 && nonfinite == 0;
        std::printf("%s C(%lld,%lld,%lld) elems=%zu nonfinite=%zu mismatches=%zu%s\n",
                    backend, (long long)c.channels, (long long)c.fine_rows,
                    (long long)c.coarse_rows, out.size(), nonfinite, mismatches,
                    ok ? " PASS" : " FAIL");
        if (!ok && mismatches > 0) {
            std::printf("  first mismatch at %zu: got %.9g want %.9g\n",
                        first_bad, out[first_bad], reference[first_bad]);
        }
        std::fflush(stdout);
        passed &= ok;
    }
    return passed ? 0 : 1;
}
