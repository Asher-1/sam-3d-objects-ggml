#include "backend.hpp"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "common.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace sam3d {

namespace {

using ProfileClock = std::chrono::steady_clock;

uint64_t elapsed_ns(ProfileClock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ProfileClock::now() - start).count());
}

}  // namespace

static bool name_contains_ci(const char* haystack, const char* needle) {
    const size_t nlen = strlen(needle);
    const size_t hlen = strlen(haystack);
    if (nlen > hlen) return false;
    for (size_t i = 0; i + nlen <= hlen; i++) {
        size_t j = 0;
        while (j < nlen &&
               tolower((unsigned char)haystack[i + j]) == tolower((unsigned char)needle[j]))
            j++;
        if (j == nlen) return true;
    }
    return false;
}

std::unique_ptr<Backend> Backend::create(const std::string& backend_name, int n_threads) {
    auto b = std::unique_ptr<Backend>(new Backend());
    if (!b->init(backend_name, n_threads)) return nullptr;
    return b;
}

bool Backend::init(const std::string& backend_name, int n_threads) {
    n_threads_ = n_threads;
    backend_name_ = "cpu";
    if (const char* profile_path = getenv("SAM3D_E2E_PROFILE_JSONL")) {
        profile_path_ = profile_path;
        profiling_enabled_ = !profile_path_.empty();
    }

    // Always have a CPU backend for fallback / readback.
    {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            LOGE("no CPU backend device found");
            return false;
        }
        cpu_backend_ = ggml_backend_dev_init(dev, nullptr);
        if (!cpu_backend_) {
            LOGE("failed to init CPU backend");
            return false;
        }
    }

    ggml_backend_dev_t selected = nullptr;
    if (backend_name != "cpu") {
        const char* want = backend_name == "auto" ? nullptr : backend_name.c_str();
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char* dev_name = ggml_backend_dev_name(dev);
            if (want && !name_contains_ci(dev_name, want)) continue;
            selected = dev;
            break;
        }
        if (want && !selected) {
            LOGE("requested backend '%s' not available", want);
            return false;
        }
        if (!selected) LOGI("no GPU backend registered in this build, using CPU");
    }

    if (selected) {
        gpu_dev_ = selected;
        gpu_backend_ = ggml_backend_dev_init(gpu_dev_, nullptr);
        if (!gpu_backend_) {
            LOGE("failed to init GPU backend");
            return false;
        }
        backend_name_ = ggml_backend_dev_name(gpu_dev_);
    }

    // Primary path: a single compute backend (GPU when present) with a
    // gallocr, since every weight lives in that same buffer type. The sched
    // below is kept as an opt-in experiment (SAM3D_USE_SCHED=1); the current
    // ggml release fails its auto-realloc path on the very large conv3d
    // graphs of the SS decoder.
    compute_backend_ = gpu_backend_ ? gpu_backend_ : cpu_backend_;
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(compute_backend_));
    if (!gallocr_) {
        LOGE("failed to create graph allocator");
        return false;
    }
    if (getenv("SAM3D_USE_SCHED")) {
        if (gpu_backend_) {
            ggml_backend_t backends[2] = {gpu_backend_, cpu_backend_};
            sched_ = ggml_backend_sched_new(backends, nullptr, 2, 65536, false, true);
        } else {
            ggml_backend_t backends[1] = {cpu_backend_};
            sched_ = ggml_backend_sched_new(backends, nullptr, 1, 65536, false, true);
        }
        if (!sched_) {
            LOGE("failed to create backend scheduler");
            return false;
        }
    }
    LOGI("backend: %s (%s)", backend_name_.c_str(), device_name());
    return true;
}

Backend::~Backend() {
    write_profile_jsonl();
    if (sched_) ggml_backend_sched_free(sched_);
    if (gallocr_) ggml_gallocr_free(gallocr_);
    if (gpu_backend_) ggml_backend_free(gpu_backend_);
    if (cpu_backend_) ggml_backend_free(cpu_backend_);
}

void Backend::set_profile_label(std::string label) {
    if (!label.empty()) profile_label_ = std::move(label);
}

void Backend::write_profile_jsonl() const {
    if (!profiling_enabled_) return;

    std::ofstream output(profile_path_, std::ios::app);
    if (!output) {
        LOGE("failed to open backend profile output: %s", profile_path_.c_str());
        return;
    }
    output << "{\"schema\":\"sam3d.e2e.backend_profile.v1\","
           << "\"label\":\"" << json_escape(profile_label_) << "\","
           << "\"backend\":\"" << json_escape(backend_name_) << "\","
           << "\"device\":\"" << json_escape(std::string(device_name())) << "\","
           << "\"h2d\":{\"calls\":" << profile_.h2d_calls
           << ",\"bytes\":" << profile_.h2d_bytes
           << ",\"submit_ns\":" << profile_.h2d_submit_ns << "},"
           << "\"d2h\":{\"calls\":" << profile_.d2h_calls
           << ",\"bytes\":" << profile_.d2h_bytes
           << ",\"ns\":" << profile_.d2h_ns << "},"
           << "\"graph\":{\"runs\":" << profile_.graph_runs
           << ",\"submit_ns\":" << profile_.graph_submit_ns
           << ",\"sync_ns\":" << profile_.graph_sync_ns
           << ",\"total_ns\":" << profile_.graph_total_ns << "}}\n";
}

ggml_backend_buffer_type_t Backend::weights_buffer_type() const {
    return ggml_backend_get_default_buffer_type(compute_backend_);
}

const char* Backend::device_name() const {
    if (gpu_dev_) return ggml_backend_dev_description(gpu_dev_);
    return "CPU";
}

bool Backend::alloc(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    (void)inputs;
    if (sched_) {
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_reserve(sched_, graph)) {
            LOGE("sched reserve failed");
            return false;
        }
        if (ggml_backend_sched_alloc_graph(sched_, graph) != GGML_STATUS_SUCCESS) {
            LOGE("sched alloc failed (out of memory?)");
            return false;
        }
        return true;
    }
    if (ggml_gallocr_reserve(gallocr_, graph) == false) {
        LOGE("graph reserve failed (graph too large for memory?)");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(gallocr_, graph)) {
        LOGE("graph alloc failed");
        return false;
    }
    return true;
}

bool Backend::run(ggml_cgraph* graph) {
    const auto run_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_t be = sched_ ? nullptr : compute_backend_;
    if (be) {
        const auto submit_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        const ggml_status status = ggml_backend_graph_compute_async(be, graph);
        if (profiling_enabled_) profile_.graph_submit_ns += elapsed_ns(submit_started);
        if (status != GGML_STATUS_SUCCESS) {
            LOGE("graph compute failed");
            return false;
        }
        const auto sync_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_synchronize(be);
        if (profiling_enabled_) {
            profile_.graph_runs++;
            profile_.graph_sync_ns += elapsed_ns(sync_started);
            profile_.graph_total_ns += elapsed_ns(run_started);
        }
        return true;
    }
    const auto submit_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    const bool submitted = ggml_backend_sched_graph_compute_async(sched_, graph);
    if (profiling_enabled_) profile_.graph_submit_ns += elapsed_ns(submit_started);
    if (!submitted) {
        LOGE("graph compute failed");
        return false;
    }
    const auto sync_started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_sched_synchronize(sched_);
    if (profiling_enabled_) {
        profile_.graph_runs++;
        profile_.graph_sync_ns += elapsed_ns(sync_started);
        profile_.graph_total_ns += elapsed_ns(run_started);
    }
    return true;
}

bool Backend::compute(ggml_cgraph* graph, std::vector<ggml_tensor*> inputs) {
    if (!alloc(graph, inputs)) return false;
    return run(graph);
}

bool Backend::set_input_f32(ggml_tensor* t, const float* data, size_t n_floats) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    if ((size_t)ggml_nelements(t) != n_floats) {
        LOGE("set_input_f32: expected %lld floats, got %zu",
             (long long)ggml_nelements(t), n_floats);
        return false;
    }
    const size_t bytes = n_floats * sizeof(float);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_set(t, data, 0, bytes);
    if (profiling_enabled_) {
        profile_.h2d_calls++;
        profile_.h2d_bytes += bytes;
        profile_.h2d_submit_ns += elapsed_ns(started);
    }
    return true;
}

bool Backend::set_input_i32(ggml_tensor* t, const int32_t* data, size_t n_ints) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    if ((size_t)ggml_nelements(t) != n_ints) {
        LOGE("set_input_i32: expected %lld ints, got %zu",
             (long long)ggml_nelements(t), n_ints);
        return false;
    }
    const size_t bytes = n_ints * sizeof(int32_t);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_set(t, data, 0, bytes);
    if (profiling_enabled_) {
        profile_.h2d_calls++;
        profile_.h2d_bytes += bytes;
        profile_.h2d_submit_ns += elapsed_ns(started);
    }
    return true;
}

bool Backend::get_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        if (profiling_enabled_) {
            profile_.d2h_calls++;
            profile_.d2h_bytes += n * sizeof(float);
            profile_.d2h_ns += elapsed_ns(started);
        }
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(n);
        const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_backend_tensor_get(t, f16.data(), 0, n * sizeof(ggml_fp16_t));
        if (profiling_enabled_) {
            profile_.d2h_calls++;
            profile_.d2h_bytes += n * sizeof(ggml_fp16_t);
            profile_.d2h_ns += elapsed_ns(started);
        }
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(f16[i]);
        return true;
    }
    LOGE("get_tensor_f32: unsupported tensor type %s", ggml_type_name(t->type));
    out.clear();
    return false;
}

bool Backend::get_tensor_i32(ggml_tensor* t, std::vector<int32_t>& out) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const size_t n = ggml_nelements(t);
    out.resize(n);
    const auto started = profiling_enabled_ ? ProfileClock::now() : ProfileClock::time_point{};
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(int32_t));
    if (profiling_enabled_) {
        profile_.d2h_calls++;
        profile_.d2h_bytes += n * sizeof(int32_t);
        profile_.d2h_ns += elapsed_ns(started);
    }
    return true;
}

}  // namespace sam3d
