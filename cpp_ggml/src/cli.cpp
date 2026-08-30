// sam3d-cli: command line adapter for the SAM 3D ggml runtime.
#include <cstring>
#include <cstdio>
#include <string>

#include "sam3dggml.h"
#include "asset_io.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "backend.hpp"
#include "ss_decoder_graph.hpp"
#include "dino_graph.hpp"
#include "moge_graph.hpp"
#include "pointpatch_graph.hpp"
#include "ss_flow_graph.hpp"
#include "slat_flow_graph.hpp"
#include "gs_decoder_graph.hpp"
#include "sparse_ops.hpp"
#include "graph_builder.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace sam3d;

static void print_usage() {
    fprintf(stderr,
            "usage: sam3d-cli <command> [options]\n"
            "\n"
            "commands:\n"
            "  info    --model <path.gguf>            print GGUF metadata and tensors\n"
            "  decode-ss --model <ss_decoder.gguf> --input <latent.bin> [--out out.bin]\n"
            "            [--backend auto|cpu|cuda|vulkan] [--threads N]\n"
            "            [--warmup N] [--iters N] [--json out.jsonl]\n"
            "  e2e|run --model <models_dir> <condition_dir> --out <out.ply>\n"
            "            [--backend cpu|cuda|vulkan] [--seed N] [--threads N]\n"
            "  moge-smoke --model <moge.gguf> [--backend cpu|cuda|vulkan]\n"
            "             [--width N] [--height N] [--threads N]\n"
            "  mesh-export --vertices <vertices.samt> --faces <faces.samt>\n"
            "              [--attrs <vertex_attrs.samt>] --out <asset.glb>\n"
            "\n"
            "Use scripts/run_image_to_3d.py for the raw-image E2E benchmark.\n");
}

struct MeshExportOpts {
    std::string vertices;
    std::string faces;
    std::string attributes;
    std::string output;
};

static bool tensor_values_f32(const std::string& path, RawTensor& tensor,
                              std::vector<float>& values, const char* label) {
    if (!load_raw_tensor(path, tensor) || tensor.type != GGML_TYPE_F32 ||
        tensor.data.size() % sizeof(float) != 0) {
        LOGE("mesh-export: %s must be a readable F32 SAMT tensor: %s", label, path.c_str());
        return false;
    }
    const size_t count = tensor.data.size() / sizeof(float);
    values.resize(count);
    std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
    return true;
}

static int cmd_mesh_export(const MeshExportOpts& options) {
    if (options.vertices.empty() || options.faces.empty() || options.output.empty()) {
        LOGE("mesh-export requires --vertices, --faces and --out");
        return 1;
    }

    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<float> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "vertices") ||
        !tensor_values_f32(options.faces, faces_tensor, faces, "faces")) {
        return 1;
    }
    if (vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3) {
        LOGE("mesh-export: vertices must have official SAMT shape [3, vertex_count]");
        return 1;
    }
    if (faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-export: faces must have official SAMT shape [3, face_count]");
        return 1;
    }

    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    // ``postprocessing_utils.to_glb`` rotates the decoder's Z-up coordinates
    // into glTF's Y-up convention before creating the final Trimesh asset.
    // Apply the same row-vector transform here so the native non-baked GLB
    // has the official camera orientation: (x, y, z) -> (x, z, -y).
    for (size_t vertex = 0; vertex < mesh.positions.size(); vertex += 3) {
        const float y = mesh.positions[vertex + 1];
        mesh.positions[vertex + 1] = mesh.positions[vertex + 2];
        mesh.positions[vertex + 2] = -y;
    }
    mesh.indices.reserve(faces.size());
    const size_t vertex_count = mesh.positions.size() / 3;
    for (float value : faces) {
        if (!std::isfinite(value) || value < 0.0f ||
            value > static_cast<float>(std::numeric_limits<uint32_t>::max()) ||
            std::floor(value) != value) {
            LOGE("mesh-export: faces contains a non-integer or invalid index");
            return 1;
        }
        const uint32_t index = static_cast<uint32_t>(value);
        if (index >= vertex_count) {
            LOGE("mesh-export: face index %u is outside %zu vertices", index, vertex_count);
            return 1;
        }
        mesh.indices.push_back(index);
    }

    if (!options.attributes.empty()) {
        RawTensor attributes_tensor;
        std::vector<float> attributes;
        if (!tensor_values_f32(options.attributes, attributes_tensor, attributes, "vertex attrs")) return 1;
        // SLatMeshDecoder with use_color exports six attributes per vertex:
        // RGB followed by its learned normal map. The non-baked official path
        // uses the RGB part as vertex colors; the normal map is not a geometry
        // normal and must not overwrite the extracted triangle normals.
        if (attributes_tensor.ne.size() != 2 || attributes_tensor.ne[0] != 6 ||
            attributes.size() != vertex_count * 6) {
            LOGE("mesh-export: vertex attrs must have official SAMT shape [6, vertex_count]");
            return 1;
        }
        mesh.colors.resize(vertex_count * 3);
        for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
            for (size_t channel = 0; channel < 3; ++channel) {
                mesh.colors[vertex * 3 + channel] =
                    std::max(0.0f, std::min(1.0f, attributes[vertex * 6 + channel]));
            }
        }
    }

    std::string error;
    if (!write_pbr_glb(options.output, mesh, error)) {
        LOGE("mesh-export: %s", error.c_str());
        return 1;
    }
    LOGI("mesh-export: wrote %s (%zu vertices, %zu triangles%s)", options.output.c_str(),
         vertex_count, mesh.indices.size() / 3, mesh.colors.empty() ? "" : ", COLOR_0");
    return 0;
}

struct MogeSmokeOpts {
    std::string model;
    std::string output_prefix;
    std::string backend = "cpu";
    int width = 56;
    int height = 56;
    int threads = 8;
};

static int cmd_moge_smoke(const MogeSmokeOpts& options) {
    if (options.width < 14 || options.height < 14) {
        LOGE("moge-smoke dimensions must both be at least 14");
        return 1;
    }
    auto backend = Backend::create(options.backend, options.threads);
    if (!backend) return 1;
    GGUFModel model;
    if (!model.load(options.model, backend->weights_buffer_type())) return 1;
    if (model.str("sam3d.model") != "moge_vitl") {
        LOGE("moge-smoke requires a moge_vitl GGUF, got %s", model.str("sam3d.model").c_str());
        return 1;
    }

    GraphContext context;
    MogeGraph graph_builder;
    graph_builder.g = &context;
    graph_builder.m = &model;
    ggml_tensor* input = context.input_f32("moge_rgb", {3, options.width, options.height});
    MogeOutputs outputs = graph_builder.build(input);
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 32768, false);
    ggml_set_output(outputs.points);
    ggml_set_output(outputs.mask_logits);
    const bool dump_features = getenv("SAM3D_MOGE_DUMP_FEATURES") != nullptr;
    const bool dump_blocks = getenv("SAM3D_MOGE_DUMP_BLOCKS") != nullptr;
    if ((dump_features || dump_blocks) && options.output_prefix.empty()) {
        LOGE("moge-smoke debug dumps require --out <prefix>");
        return 1;
    }
    if (dump_features) {
        ggml_set_output(outputs.backbone_input);
        for (ggml_tensor* feature : outputs.backbone_features) ggml_set_output(feature);
    }
    if (dump_blocks) {
        ggml_set_output(outputs.debug_q);
        ggml_set_output(outputs.debug_k);
        ggml_set_output(outputs.debug_v);
        ggml_set_output(outputs.debug_attention_context);
        for (ggml_tensor* attention : outputs.backbone_attention_outputs) ggml_set_output(attention);
        for (ggml_tensor* fc1 : outputs.backbone_mlp_fc1_outputs) ggml_set_output(fc1);
        for (ggml_tensor* gelu : outputs.backbone_mlp_gelu_outputs) ggml_set_output(gelu);
        for (ggml_tensor* mlp : outputs.backbone_mlp_outputs) ggml_set_output(mlp);
        for (ggml_tensor* block : outputs.backbone_block_outputs) ggml_set_output(block);
    }
    ggml_build_forward_expand(graph, outputs.points);
    ggml_build_forward_expand(graph, outputs.mask_logits);
    if (dump_features) {
        ggml_build_forward_expand(graph, outputs.backbone_input);
        for (ggml_tensor* feature : outputs.backbone_features) {
            ggml_build_forward_expand(graph, feature);
        }
    }
    if (dump_blocks) {
        ggml_build_forward_expand(graph, outputs.debug_q);
        ggml_build_forward_expand(graph, outputs.debug_k);
        ggml_build_forward_expand(graph, outputs.debug_v);
        ggml_build_forward_expand(graph, outputs.debug_attention_context);
        for (ggml_tensor* attention : outputs.backbone_attention_outputs) {
            ggml_build_forward_expand(graph, attention);
        }
        for (ggml_tensor* fc1 : outputs.backbone_mlp_fc1_outputs) {
            ggml_build_forward_expand(graph, fc1);
        }
        for (ggml_tensor* gelu : outputs.backbone_mlp_gelu_outputs) {
            ggml_build_forward_expand(graph, gelu);
        }
        for (ggml_tensor* mlp : outputs.backbone_mlp_outputs) {
            ggml_build_forward_expand(graph, mlp);
        }
        for (ggml_tensor* block : outputs.backbone_block_outputs) {
            ggml_build_forward_expand(graph, block);
        }
    }
    LOGI("moge-smoke: graph nodes=%d, allocating", ggml_graph_n_nodes(graph));
    if (!backend->alloc(graph)) return 1;

    std::vector<float> rgb(static_cast<size_t>(options.width) * options.height * 3);
    for (int y = 0; y < options.height; ++y) {
        for (int x = 0; x < options.width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * options.width + x) * 3;
            rgb[offset] = static_cast<float>(x) / std::max(1, options.width - 1);
            rgb[offset + 1] = static_cast<float>(y) / std::max(1, options.height - 1);
            rgb[offset + 2] = 0.25f;
        }
    }
    if (!backend->set_input_f32(input, rgb.data(), rgb.size())) return 1;
    size_t f32_index = 0;
    size_t i32_index = 0;
    for (ggml_tensor* tensor : graph_builder.inputs) {
        bool uploaded = false;
        if (tensor->type == GGML_TYPE_F32 && f32_index < graph_builder.f32_data.size()) {
            const auto& values = graph_builder.f32_data[f32_index++];
            uploaded = backend->set_input_f32(tensor, values->data(), values->size());
        } else if (tensor->type == GGML_TYPE_I32 && i32_index < graph_builder.i32_data.size()) {
            const auto& values = graph_builder.i32_data[i32_index++];
            uploaded = backend->set_input_i32(tensor, values->data(), values->size());
        }
        if (!uploaded) {
            LOGE("moge-smoke: failed to upload graph input %s", tensor->name);
            return 1;
        }
    }
    if (!backend->run(graph)) return 1;
    std::vector<float> points;
    std::vector<float> mask;
    if (!backend->get_tensor_f32(outputs.points, points) ||
        !backend->get_tensor_f32(outputs.mask_logits, mask)) return 1;
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    };
    if (!finite(points) || !finite(mask)) {
        LOGE("moge-smoke: graph produced non-finite point or mask values");
        return 1;
    }
    if (!options.output_prefix.empty()) {
        if (!save_raw_tensor_f32(options.output_prefix + ".points.samt",
                                 {options.width, options.height, 3, 1}, points.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".mask_logits.samt",
                                 {options.width, options.height, 1, 1}, mask.data())) {
            LOGE("moge-smoke: failed to save output tensors with prefix %s",
                 options.output_prefix.c_str());
            return 1;
        }
        if (dump_features) {
            std::vector<float> backbone_input;
            if (!backend->get_tensor_f32(outputs.backbone_input, backbone_input) ||
                !save_raw_tensor_f32(options.output_prefix + ".backbone_input.samt",
                                     {outputs.backbone_input->ne[0], outputs.backbone_input->ne[1]},
                                     backbone_input.data())) {
                LOGE("moge-smoke: failed to save DINO input tokens");
                return 1;
            }
            for (size_t index = 0; index < outputs.backbone_features.size(); ++index) {
                std::vector<float> feature;
                ggml_tensor* tensor = outputs.backbone_features[index];
                if (!backend->get_tensor_f32(tensor, feature) ||
                    !save_raw_tensor_f32(options.output_prefix + ".feature" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, feature.data())) {
                    LOGE("moge-smoke: failed to save DINO feature %zu", index);
                    return 1;
                }
            }
        }
        if (dump_blocks) {
            const std::pair<const char*, ggml_tensor*> debug_tensors[] = {
                {"q", outputs.debug_q}, {"k", outputs.debug_k}, {"v", outputs.debug_v},
                {"attention_context", outputs.debug_attention_context},
            };
            for (const auto& [name, tensor] : debug_tensors) {
                std::vector<float> values;
                std::vector<int64_t> shape(tensor->ne, tensor->ne + ggml_n_dims(tensor));
                if (!backend->get_tensor_f32(tensor, values) ||
                    !save_raw_tensor_f32(options.output_prefix + "." + name + ".samt",
                                         shape, values.data())) {
                    LOGE("moge-smoke: failed to save DINO debug tensor %s", name);
                    return 1;
                }
            }
            for (size_t index = 0; index < outputs.backbone_attention_outputs.size(); ++index) {
                std::vector<float> attention;
                ggml_tensor* tensor = outputs.backbone_attention_outputs[index];
                if (!backend->get_tensor_f32(tensor, attention) ||
                    !save_raw_tensor_f32(options.output_prefix + ".attention" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, attention.data())) {
                    LOGE("moge-smoke: failed to save DINO attention %zu", index);
                    return 1;
                }
            }
            for (size_t index = 0; index < outputs.backbone_mlp_fc1_outputs.size(); ++index) {
                std::vector<float> values;
                ggml_tensor* tensor = outputs.backbone_mlp_fc1_outputs[index];
                if (!backend->get_tensor_f32(tensor, values) ||
                    !save_raw_tensor_f32(options.output_prefix + ".mlp_fc1" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, values.data())) {
                    LOGE("moge-smoke: failed to save DINO MLP fc1 %zu", index);
                    return 1;
                }
            }
            for (size_t index = 0; index < outputs.backbone_mlp_gelu_outputs.size(); ++index) {
                std::vector<float> values;
                ggml_tensor* tensor = outputs.backbone_mlp_gelu_outputs[index];
                if (!backend->get_tensor_f32(tensor, values) ||
                    !save_raw_tensor_f32(options.output_prefix + ".mlp_gelu" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, values.data())) {
                    LOGE("moge-smoke: failed to save DINO MLP GELU %zu", index);
                    return 1;
                }
            }
            for (size_t index = 0; index < outputs.backbone_mlp_outputs.size(); ++index) {
                std::vector<float> mlp;
                ggml_tensor* tensor = outputs.backbone_mlp_outputs[index];
                if (!backend->get_tensor_f32(tensor, mlp) ||
                    !save_raw_tensor_f32(options.output_prefix + ".mlp" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, mlp.data())) {
                    LOGE("moge-smoke: failed to save DINO MLP %zu", index);
                    return 1;
                }
            }
            for (size_t index = 0; index < outputs.backbone_block_outputs.size(); ++index) {
                std::vector<float> block;
                ggml_tensor* tensor = outputs.backbone_block_outputs[index];
                if (!backend->get_tensor_f32(tensor, block) ||
                    !save_raw_tensor_f32(options.output_prefix + ".block" + std::to_string(index) + ".samt",
                                         {tensor->ne[0], tensor->ne[1]}, block.data())) {
                    LOGE("moge-smoke: failed to save DINO block %zu", index);
                    return 1;
                }
            }
        }
    }
    printf("moge-smoke: backend=%s points=%zu mask=%zu point_range=[%.6f, %.6f]\n",
           backend->backend_name(), points.size(), mask.size(),
           *std::min_element(points.begin(), points.end()), *std::max_element(points.begin(), points.end()));
    return 0;
}

struct DecodeSsOpts {
    std::string model, input, out, backend;
    std::string json;  // append one JSONL measurement row when set
    int warmup = 1;    // untimed graph runs before measurement
    int iters = 1;     // timed graph runs
    int threads = 8;   // CPU worker threads (matches the parity baseline)
};

static int cmd_decode_ss(const DecodeSsOpts& o) {
    auto backend = Backend::create(o.backend, o.threads);
    if (!backend) return 1;

    GGUFModel m;
    double t0 = now_ms();
    if (!m.load(o.model, backend->weights_buffer_type())) return 1;
    const double load_ms = now_ms() - t0;

    RawTensor in;
    if (!load_raw_tensor(o.input, in)) {
        LOGE("failed to read input tensor %s", o.input.c_str());
        return 1;
    }
    // PyTorch's stage dumper retains batch=1 as the final dimension.  The
    // decoder is single-batch, so both [W,H,D,C] and [W,H,D,C,1] encode the
    // same contiguous payload.  Reject every other batched shape explicitly.
    const bool singleton_batch = in.ne.size() == 5 && in.ne[4] == 1;
    if (in.ne.size() != 4 && !singleton_batch) {
        LOGE("expected latent (W,H,D,C) or (W,H,D,C,1), got ndims=%zu", in.ne.size());
        return 1;
    }

    t0 = now_ms();
    GraphContext gctx;
    ggml_tensor* latent =
        gctx.input_f32("ss_latent", {in.ne[0], in.ne[1], in.ne[2], in.ne[3]});
    SsDecoderGraph graph_builder;
    graph_builder.ctx = gctx.ctx();
    graph_builder.m = &m;
    ggml_tensor* out_t = graph_builder.build(latent);
    // conv3d lowering expands to thousands of nodes; size generously
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 16384, false);
    ggml_build_forward_expand(graph, out_t);
    const double build_ms = now_ms() - t0;

    t0 = now_ms();
    if (!backend->alloc(graph)) return 1;
    const double alloc_ms = now_ms() - t0;

    t0 = now_ms();
    if (!backend->set_input_f32(latent, (const float*)in.data.data(),
                                (size_t)ggml_nelements(latent)))
        return 1;
    const double upload_ms = now_ms() - t0;

    // timed runs: `warmup` untimed passes, then `iters` measured runs of the
    // same graph (deterministic output, so one readback at the end is enough)
    const int total_runs = o.warmup + (o.iters > 0 ? o.iters : 1);
    std::vector<double> run_ms;
    run_ms.reserve(total_runs);
    for (int r = 0; r < total_runs; r++) {
        // Some ggml graph plans reuse transient storage across executions.
        // The latent is an external graph input, so restore it before every
        // warmup and measured run to make repeated execution deterministic.
        if (!backend->set_input_f32(latent, (const float*)in.data.data(),
                                    (size_t)ggml_nelements(latent))) {
            return 1;
        }
        const double r0 = now_ms();
        if (!backend->run(graph)) return 1;
        const double rt = now_ms() - r0;
        if (r >= o.warmup) run_ms.push_back(rt);
    }
    if (getenv("SAM3D_DEBUG_CONV") && graph_builder.debug_tensors.size() >= 1) {
        size_t debug_index = 0;
        if (const char* index = getenv("SAM3D_DEBUG_CONV_INDEX")) {
            debug_index = (size_t)strtoul(index, nullptr, 10);
        }
        if (debug_index >= graph_builder.debug_tensors.size()) {
            LOGE("SS debug tensor index %zu is out of range (%zu)", debug_index,
                 graph_builder.debug_tensors.size());
            return 1;
        }
        ggml_tensor* debug_tensor = graph_builder.debug_tensors[debug_index];
        std::vector<float> dbg;
        backend->get_tensor_f32(debug_tensor, dbg);
        fprintf(stderr, "g[0, 0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
        fprintf(stderr, "g[0, 256..259]: %.4f %.4f %.4f %.4f\n",
                dbg[256], dbg[257], dbg[258], dbg[259]);
        fprintf(stderr, "g[1, 256..259]: %.4f %.4f %.4f %.4f\n",
                dbg[8 + 256], dbg[8 + 257], dbg[8 + 258], dbg[8 + 259]);
        if (const char* path = getenv("SAM3D_DEBUG_CONV_OUT")) {
            std::vector<int64_t> ne(debug_tensor->ne,
                                    debug_tensor->ne + ggml_n_dims(debug_tensor));
            if (!save_raw_tensor_f32(path, ne, dbg.data())) {
                LOGE("failed to write SS debug tensor %s", path);
                return 1;
            }
        }
    }

    LOGI("out_t ne = %lld %lld %lld %lld (n_dims %d)", (long long)out_t->ne[0],
         (long long)out_t->ne[1], (long long)out_t->ne[2], (long long)out_t->ne[3],
         ggml_n_dims(out_t));
    std::vector<float> out_f;
    backend->get_tensor_f32(out_t, out_f);
    if (!o.out.empty()) {
        const int nd = ggml_n_dims(out_t);
        std::vector<int64_t> ne(out_t->ne, out_t->ne + nd);
        save_raw_tensor_f32(o.out, ne, out_f.data());
        LOGI("wrote %s", o.out.c_str());
    }
    float mn = 1e30f, mx = -1e30f;
    for (float v : out_f) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    printf("ss_decoder output: %lld elements, min %.6f max %.6f\n",
           (long long)out_f.size(), mn, mx);

    if (!run_ms.empty()) {
        std::vector<double> sorted(run_ms);
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double v : run_ms) sum += v;
        const double mean = sum / (double)run_ms.size();
        const double p50 = sorted[run_ms.size() / 2];
        const double min_v = sorted.front(), max_v = sorted.back();
        printf("bench ss_decoder: backend=%s device=%s dtype=%s threads=%d "
               "warmup=%d iters=%zu graph_ms mean=%.2f min=%.2f p50=%.2f max=%.2f\n",
               backend->backend_name(), backend->device_name(),
               m.str("sam3d.dtype").c_str(), o.threads, o.warmup, run_ms.size(),
               mean, min_v, p50, max_v);
        if (!o.json.empty()) {
            FILE* f = fopen(o.json.c_str(), "a");
            if (f) {
                fprintf(f,
                        "{\"component\":\"ss_decoder\",\"model\":\"%s\","
                        "\"dtype\":\"%s\",\"backend\":\"%s\",\"device\":\"%s\","
                        "\"n_threads\":%d,\"warmup\":%d,\"iters\":%zu,"
                        "\"load_ms\":%.2f,\"build_ms\":%.2f,\"alloc_ms\":%.2f,"
                        "\"upload_ms\":%.2f,\"graph_ms_mean\":%.2f,"
                        "\"graph_ms_min\":%.2f,\"graph_ms_p50\":%.2f,"
                        "\"graph_ms_max\":%.2f}\n",
                        json_escape(o.model).c_str(),
                        m.str("sam3d.dtype").c_str(),
                        json_escape(backend->backend_name()).c_str(),
                        json_escape(backend->device_name()).c_str(),
                        o.threads, o.warmup, run_ms.size(),
                        load_ms, build_ms, alloc_ms, upload_ms,
                        mean, min_v, p50, max_v);
                fclose(f);
            }
        }
    }
    return 0;
}

static int cmd_dino(const std::string& model_path, const std::string& input_path,
                    const std::string& out_path, const std::string& embedder) {
    auto backend = Backend::create("cpu", getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;

    GGUFModel m;
    if (!m.load(model_path, backend->weights_buffer_type())) return 1;

    RawTensor in;
    if (!load_raw_tensor(input_path, in)) {
        LOGE("failed to read input image %s", input_path.c_str());
        return 1;
    }
    GraphContext gctx;
    if (getenv("SAM3D_SOFTMAX_REPLAY")) {
        // minimal-unit probe: input tensor = the on-disk (verified sane)
        // b0_scores dump; graph = soft_max only. Isolates op + allocator +
        // sched from the rest of the ViT graph.
        ggml_tensor* sc = gctx.input_f32("scores", {in.ne[0], in.ne[1], in.ne[2]});
        ggml_tensor* out2 = ggml_soft_max(gctx.ctx(), sc);
        ggml_cgraph* g2 = ggml_new_graph_custom(gctx.ctx(), 64, false);
        ggml_build_forward_expand(g2, out2);
        if (!backend->alloc(g2)) return 1;
        if (!backend->set_input_f32(sc, (const float*)in.data.data(),
                                    (size_t)ggml_nelements(sc))) return 1;
        if (getenv("SAM3D_VERIFY_UPLOAD")) {
            std::vector<float> rb(16);
            ggml_backend_tensor_get(sc, rb.data(), 0, 16 * sizeof(float));
            fprintf(stderr, "upload readback[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7]);
        }
        if (!backend->run(g2)) return 1;
        std::vector<float> of;
        backend->get_tensor_f32(out2, of);
        size_t nans = 0;
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < of.size(); i++) {
            if (std::isnan(of[i])) nans++;
            if (i < 1374) sum += of[i];
            mn = std::min(mn, of[i]); mx = std::max(mx, of[i]);
        }
        printf("softmax_replay: n=%zu nan=%zu rowsum0=%.6f min=%.6f max=%.6f\n",
               of.size(), nans, sum, mn, mx);
        if (!out_path.empty()) {
            std::vector<int64_t> ne2(out2->ne, out2->ne + ggml_n_dims(out2));
            save_raw_tensor_f32(out_path, ne2, of.data());
        }
        return 0;
    }
    ggml_tensor* img = nullptr;
    if (getenv("SAM3D_GEMM_REPLAY")) {
        // input: (kdim, n_patch) taps matrix; graph: patch-embed GEMM only
        ggml_tensor* taps = gctx.input_f32("taps", {in.ne[0], in.ne[1]});
        ggml_tensor* w = m.get(embedder + ".backbone.patch_embed.proj.weight");
        ggml_tensor* wb = m.get(embedder + ".backbone.patch_embed.proj.bias");
        const int64_t kdim = 588, Cc = 1024;
        const size_t pwts = ggml_type_size(w->type);
        ggml_tensor* w2d = ggml_view_2d(gctx.ctx(), w, kdim, Cc, kdim * pwts, 0);
        ggml_tensor* out2 = getenv("SAM3D_DUMP_WEIGHT")
            ? ggml_reshape_2d(gctx.ctx(),
                              ggml_cont(gctx.ctx(), ggml_cast(gctx.ctx(), w, GGML_TYPE_F32)),
                              kdim, Cc)
            : gb_linear(gctx.ctx(), w2d, wb, taps);
        ggml_cgraph* g2 = ggml_new_graph_custom(gctx.ctx(), 64, false);
        ggml_build_forward_expand(g2, out2);
        if (!backend->alloc(g2)) return 1;
        if (taps->buffer &&
            !backend->set_input_f32(taps, (const float*)in.data.data(),
                                    (size_t)ggml_nelements(taps))) return 1;
        {
            std::vector<float> rb(8);
            ggml_backend_tensor_get(taps, rb.data(), 0, 8 * sizeof(float));
            fprintf(stderr, "taps readback[0..7]: %.6f %.6f %.6f %.6f\n",
                    rb[0], rb[1], rb[2], rb[3]);
            ggml_tensor* wchk = ggml_view_1d(gctx.ctx(), w, 4, 0);
            ggml_backend_tensor_get(wchk, rb.data(), 0, 4 * sizeof(float));
            fprintf(stderr, "w2d col via 1D view[0..3]: %.6f %.6f %.6f %.6f\n",
                    rb[0], rb[1], rb[2], rb[3]);
        }
        if (!backend->run(g2)) return 1;
        std::vector<float> of;
        backend->get_tensor_f32(out2, of);
        std::vector<int64_t> ne2(out2->ne, out2->ne + ggml_n_dims(out2));
        save_raw_tensor_f32("/tmp/gemm_replay.samt", ne2, of.data());
        printf("gemm_replay done: %zu elements\n", of.size());
        return 0;
    }
    img = gctx.input_f32("image", {in.ne[2], in.ne[0], in.ne[1]});
    // in stores (H, W, C) shapes; the graph wants ggml ne = [C, W, H]
    // (HWC memory, channel fastest) so per-channel broadcasts line up
    {  // temporary loader sanity check
        ggml_tensor* w = m.get(embedder + ".backbone.cls_token");
        if (w) {
            uint8_t raw[8];
            ggml_backend_tensor_get(w, raw, 0, 8);
            fprintf(stderr, "cls_token type=%d bytes=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                    (int)w->type, raw[0], raw[1], raw[2], raw[3],
                    raw[4], raw[5], raw[6], raw[7]);
        }
    }
    DinoGraph graph_builder;
    graph_builder.g = &gctx;
    graph_builder.m = &m;
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) graph_builder.debug_stage = st;
    graph_builder.prefix = embedder;
    ggml_tensor* out_t = graph_builder.build(img);
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 8192, false);
    ggml_build_forward_expand(graph, out_t);

    if (!backend->alloc(graph)) return 1;
    // upload inputs that the allocator gave a buffer to. Input-flagged LEAF
    // tensors are not graph nodes, so an "is it a node" check would wrongly
    // skip them (that skipped the image upload and produced garbage).
    if (img->buffer &&
        !backend->set_input_f32(img, (const float*)in.data.data(),
                                (size_t)ggml_nelements(img)))
        return 1;
    for (size_t ti = 0; ti < graph_builder.inputs.size(); ti++) {
        ggml_tensor* t = graph_builder.inputs[ti];
        if (!t->buffer) continue;  // pruned out of this (debug) graph
        const auto& host = graph_builder.table_data[ti];
        // f32 constant inputs (e.g. dino mean/istd) share the i32 storage;
        // dispatch on the graph tensor type
        bool ok = t->type == GGML_TYPE_F32
            ? backend->set_input_f32(t, (const float*)host->data(), host->size())
            : backend->set_input_i32(t, host->data(), host->size());
        if (!ok) return 1;
    }
    if (!backend->run(graph)) return 1;

    std::vector<float> out_f;
    backend->get_tensor_f32(out_t, out_f);
    const int nd = ggml_n_dims(out_t);
    std::vector<int64_t> ne(out_t->ne, out_t->ne + nd);
    if (!out_path.empty()) {
        save_raw_tensor_f32(out_path, ne, out_f.data());
        LOGI("wrote %s", out_path.c_str());
    }
    if (const char* dbg = getenv("SAM3D_DEBUG_GNODE")) {
        const int gi = atoi(dbg);
        const int nn = ggml_graph_n_nodes(graph);
        if (gi < nn) {
            ggml_tensor* t = ggml_graph_node(graph, gi);
            if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
                std::vector<float> d(ggml_nelements(t));
                backend->get_tensor_f32(t, d);
                float mn = 1e30f, mx = -1e30f;
                size_t nan_cnt = 0;
                for (float v : d) { if (std::isnan(v)) nan_cnt++; mn = std::min(mn, v); mx = std::max(mx, v); }
                fprintf(stderr, "gnode %d op=%s ne=%lld,%lld,%lld min=%.6f max=%.6f nan=%zu\n",
                        gi, ggml_op_desc(t), (long long)t->ne[0], (long long)t->ne[1],
                        (long long)t->ne[2], mn, mx, nan_cnt);
            } else {
                fprintf(stderr, "gnode %d op=%s ne=%lld,%lld type=%d (skip dump)\n",
                        gi, ggml_op_desc(t), (long long)t->ne[0], (long long)t->ne[1],
                        (int)t->type);
            }
            if (ggml_n_dims(t) >= 1 && t->src[0]) {
                ggml_tensor* s0 = t->src[0];
                uint8_t raw[8];
                ggml_backend_tensor_get(s0, raw, 0, 8);
                fprintf(stderr, "  src0 op=%s type=%d ne=%lld bytes=%02x%02x%02x%02x%02x%02x%02x%02x data=%p buf=%p\n",
                        ggml_op_desc(s0), (int)s0->type, (long long)s0->ne[0],
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                        s0->data, (void*)s0->buffer);
            }
        }
    }
    float mn = 1e30f, mx = -1e30f;
    size_t nan_cnt = 0;
    for (float v : out_f) {
        if (std::isnan(v)) { nan_cnt++; continue; }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    printf("dino %s tokens: %lld elements (ne %lld x %lld), min %.6f max %.6f nan=%zu/%zu\n",
           embedder.c_str(), (long long)out_f.size(), (long long)ne[0],
           nd > 1 ? (long long)ne[1] : 1, mn, mx, nan_cnt, out_f.size());
    return 0;
}

static int cmd_pointpatch(const std::string& model_path,
                          const std::string& input_path,
                          const std::string& out_path) {
    const char* be = getenv("SAM3D_BACKEND");
    auto backend = Backend::create(be ? be : "cpu",
                                   getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;

    GGUFModel m;
    if (!m.load(model_path, backend->weights_buffer_type())) return 1;

    RawTensor in;
    if (!load_raw_tensor(input_path, in)) {
        LOGE("failed to read input pointmap %s", input_path.c_str());
        return 1;
    }
    GraphContext gctx;
    // torch (1, 3, H, W) CHW memory -> ne = [W, H, 3] (W fastest)
    ggml_tensor* pm = gctx.input_f32("pointmap", {in.ne[0], in.ne[1], in.ne[2]});
    PointPatchGraph graph_builder;
    graph_builder.g = &gctx;
    graph_builder.m = &m;
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) graph_builder.debug_stage = st;
    ggml_tensor* out_t = graph_builder.build(pm);
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 8192, false);
    ggml_build_forward_expand(graph, out_t);

    if (!backend->alloc(graph)) return 1;
    if (pm->buffer &&
        !backend->set_input_f32(pm, (const float*)in.data.data(),
                                (size_t)ggml_nelements(pm))) return 1;
    // host-side masks: fully valid pointmap (all-ones / all-zeros)
    {
        std::vector<float> ones, zeros;
        for (ggml_tensor* mk : graph_builder.mask_inputs) {
            if (!mk->buffer) continue;
            const size_t n_mask = (size_t)ggml_nelements(mk);
            ones.assign(n_mask, 1.0f);
            zeros.assign(n_mask, 0.0f);
            const bool is_valid_mask = !strcmp(mk->name, "pp_valid");
            if (!backend->set_input_f32(mk, is_valid_mask ? ones.data() : zeros.data(),
                                        n_mask)) return 1;
        }
    }
    for (size_t ti = 0; ti < graph_builder.inputs.size(); ti++) {
        ggml_tensor* t = graph_builder.inputs[ti];
        if (!t->buffer || !graph_builder.table_data[ti]) continue;
        const auto& host = graph_builder.table_data[ti];
        bool ok = t->type == GGML_TYPE_F32
            ? backend->set_input_f32(t, (const float*)host->data(), host->size())
            : backend->set_input_i32(t, host->data(), host->size());
        if (!ok) return 1;
    }
    if (!backend->run(graph)) return 1;

    std::vector<float> out_f;
    backend->get_tensor_f32(out_t, out_f);
    const int nd = ggml_n_dims(out_t);
    std::vector<int64_t> ne(out_t->ne, out_t->ne + nd);
    if (!out_path.empty()) {
        save_raw_tensor_f32(out_path, ne, out_f.data());
        LOGI("wrote %s", out_path.c_str());
    }
    float mn = 1e30f, mx = -1e30f;
    size_t nan_cnt = 0;
    for (float v : out_f) {
        if (std::isnan(v)) { nan_cnt++; continue; }
        mn = std::min(mn, v); mx = std::max(mx, v);
    }
    printf("pointpatch tokens: %lld elements (ne %lld x %lld), min %.6f max %.6f nan=%zu\n",
           (long long)out_f.size(), (long long)ne[0], nd > 1 ? (long long)ne[1] : 1,
           mn, mx, nan_cnt);
    return 0;
}



static int cmd_gs_decode(const std::string& model_path,
                         const std::string& e2e_dir,
                         const std::string& out_dir) {
    const char* be = getenv("SAM3D_BACKEND");
    auto backend = Backend::create(be ? be : "cpu",
                                   getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;
    GGUFModel m;
    if (!m.load(model_path, backend->weights_buffer_type())) return 1;

    auto load = [](const std::string& p, RawTensor& t) {
        if (!load_raw_tensor(p, t)) { LOGE("missing %s", p.c_str()); return false; }
        return true;
    };
    RawTensor x_lat, x_coord;
    if (!load(e2e_dir + "/slat_feats_final.samt", x_lat)) return 1;
    if (!load(e2e_dir + "/slat_coords.samt", x_coord)) return 1;
    const int64_t nf = x_coord.ne[1];
    GGML_ASSERT(x_coord.type == GGML_TYPE_I32);
    const int32_t* ci32 = (const int32_t*)x_coord.data.data();

    GsTables tb;
    if (!tb.build(ci32, nf)) { LOGE("gs-decode: table build failed"); return 1; }
    LOGI("gs-decode: tokens nf=%lld", (long long)tb.shifts[0].nf);

    GraphContext gctx;
    GsDecoderGraph gb;
    gb.g = &gctx; gb.m = &m; gb.tb = &tb;
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) gb.debug_stage = st;
    ggml_tensor* xin = gctx.input_f32("x", {x_lat.ne[0], x_lat.ne[1]});
    gb.x = xin;
    gb.inputs.push_back(xin);
    gb.table_data.push_back(nullptr);
    std::vector<ggml_tensor*> outs = gb.build();

    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 65536, false);
    for (auto* o : outs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }
    LOGI("gs-decode: graph built (%lld nodes), alloc...",
         (long long)ggml_graph_n_nodes(graph));
    if (!backend->alloc(graph)) return 1;

    auto up = [&](ggml_tensor* t, const RawTensor& r) {
        if (!t->buffer) return true;
        return backend->set_input_f32(t, (const float*)r.data.data(),
                                      (size_t)ggml_nelements(t));
    };
    if (!up(xin, x_lat)) return 1;
    for (size_t ti = 0; ti < gb.inputs.size(); ti++) {
        ggml_tensor* t = gb.inputs[ti];
        if (!t->buffer || !gb.table_data[ti]) continue;
        const auto& host = gb.table_data[ti];
        bool ok = t->type == GGML_TYPE_F32
            ? backend->set_input_f32(t, (const float*)host->data(), host->size())
            : backend->set_input_i32(t, host->data(), host->size());
        if (!ok) return 1;
    }
    LOGI("gs-decode: inputs uploaded, running");
    if (!backend->run(graph)) return 1;
    LOGI("gs-decode: run done");

    std::vector<float> raw;
    backend->get_tensor_f32(outs[0], raw);
    const char* dbg = getenv("SAM3D_DEBUG_STAGE");
    if (dbg) {  // debug dumps: save every output tensor, then stop
        for (size_t i = 0; i < outs.size(); i++) {
            std::vector<float> of;
            backend->get_tensor_f32(outs[i], of);
            std::vector<int64_t> ne(outs[i]->ne,
                                    outs[i]->ne + ggml_n_dims(outs[i]));
            save_raw_tensor_f32(out_dir + "/gs_dbg_" + dbg + "_" +
                                    std::to_string(i) + ".samt",
                                ne, of.data());
        }
        return 0;
    }
    {
        std::vector<int64_t> ne(outs[0]->ne, outs[0]->ne + ggml_n_dims(outs[0]));
        save_raw_tensor_f32(out_dir + "/gs_raw.samt", ne, raw.data());
        float mn = 1e30f, mx = -1e30f;
        for (float v : raw) { mn = std::min(mn, v); mx = std::max(mx, v); }
        printf("gs-decode out %lld elems [%g, %g] -> %s/gs_raw.samt\n",
               (long long)raw.size(), mn, mx, out_dir.c_str());
    }

    // ---- to_representation (host) ---------------------------------------
    // layout: _xyz[0,96) _features_dc[96,192) _scaling[192,288)
    //         _rotation[288,416) _opacity[416,448); 32 gaussians per token
    const int64_t NG = 32;
    ggml_tensor* pert = m.get("gsdec.offset_perturbation");
    GGML_ASSERT(pert && pert->ne[0] == 3 && pert->ne[1] == NG);
    std::vector<float> pbuf;
    backend->get_tensor_f32(pert, pbuf);
    GGML_ASSERT(pbuf.size() == static_cast<size_t>(NG * 3));

    const int64_t nG = nf * NG;
    std::vector<float> xyz((size_t)nG * 3), fdc((size_t)nG * 3),
        scl((size_t)nG * 3), rot((size_t)nG * 4), op((size_t)nG);
    const float* R = raw.data();  // (448, N) memory (n, 448)
    for (int64_t n = 0; n < nf; n++) {
        const float cx = (ci32[n * 4 + 1] + 0.5f) / 64.0f;
        const float cy = (ci32[n * 4 + 2] + 0.5f) / 64.0f;
        const float cz = (ci32[n * 4 + 3] + 0.5f) / 64.0f;
        const float* rn = R + (size_t)n * 448;
        for (int g = 0; g < NG; g++) {
            const size_t gi = (size_t)n * NG + g;
            // _xyz: offset*1.0 + perturb -> tanh -> /64*0.5*1.5
            for (int a = 0; a < 3; a++) {
                const float off = tanhf(rn[g * 3 + a] + pbuf[(size_t)g * 3 + a]);
                xyz[gi * 3 + a] = (a == 0 ? cx : a == 1 ? cy : cz)
                                + off / 64.0f * 0.5f * 1.5f;
                fdc[gi * 3 + a] = rn[96 + g * 3 + a];
                scl[gi * 3 + a] = rn[192 + g * 3 + a];
            }
            for (int a = 0; a < 4; a++) rot[gi * 4 + a] = 0.1f * rn[288 + g * 4 + a];
            op[gi] = rn[416 + g];
        }
    }
    save_raw_tensor_f32(out_dir + "/gs_xyz.samt", {3, nG}, xyz.data());
    save_raw_tensor_f32(out_dir + "/gs_features_dc.samt", {3, nG}, fdc.data());
    save_raw_tensor_f32(out_dir + "/gs_scaling.samt", {3, nG}, scl.data());
    save_raw_tensor_f32(out_dir + "/gs_rotation.samt", {4, nG}, rot.data());
    save_raw_tensor_f32(out_dir + "/gs_opacity.samt", {1, nG}, op.data());
    printf("gs-decode: %lld gaussians -> %s\n", (long long)nG, out_dir.c_str());
    return 0;
}

static int cmd_slat_step(const std::string& model_path,
                         const std::string& e2e_dir,
                         const std::string& out_dir) {
    const char* be = getenv("SAM3D_BACKEND");
    auto backend = Backend::create(be ? be : "cpu",
                                   getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;
    GGUFModel m;
    if (!m.load(model_path, backend->weights_buffer_type())) return 1;

    auto load = [](const std::string& p, RawTensor& t) {
        if (!load_raw_tensor(p, t)) { LOGE("missing %s", p.c_str()); return false; }
        return true;
    };
    RawTensor x_lat, x_coord, cond_t;
    if (!load(e2e_dir + "/slat_x0.samt", x_lat)) return 1;
    if (!load(e2e_dir + "/slat_coords.samt", x_coord)) return 1;
    if (!load(e2e_dir + "/slat_cond_tokens.samt", cond_t)) return 1;

    // host tables from the (fixed) sparse structure; slat_coords is (4, N)
    // torch (N, 4) token-major, stored as GGML_TYPE_I32 by the dump script.
    SlatTables tb;
    const int64_t nf = x_coord.ne[1];
    GGML_ASSERT(x_coord.type == GGML_TYPE_I32 && "slat_coords must be I32");
    const int32_t* ci32 = (const int32_t*)x_coord.data.data();
    if (!tb.build(ci32, nf)) {
        LOGE("slat-step: failed to build sparse tables");
        return 1;
    }
    LOGI("slat-step: tokens nf=%lld nc=%lld", (long long)tb.nf,
         (long long)tb.nc);

    GraphContext gctx;
    SlatFlowGraph gb;
    gb.g = &gctx;
    gb.m = &m;
    gb.tb = &tb;
    gb.n_cond_tokens = cond_t.ne[1];
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) gb.debug_stage = st;
    std::vector<ggml_tensor*> outs = gb.build();

    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto* o : outs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }
    LOGI("slat-step: graph built (%lld nodes), alloc...",
         (long long)ggml_graph_n_nodes(graph));
    if (!backend->alloc(graph)) return 1;

    auto upload_and_run = [&]() {
        auto up = [&](ggml_tensor* t, const RawTensor& r) {
            if (!t->buffer) return true;
            const size_t expected = (size_t)ggml_nelements(t);
            const int target_ndims = ggml_n_dims(t);
            if (r.type != GGML_TYPE_F32 || r.data.size() != expected * sizeof(float) ||
                r.ne.size() < (size_t)target_ndims ||
                !std::equal(r.ne.begin(), r.ne.begin() + target_ndims, t->ne) ||
                !std::all_of(r.ne.begin() + target_ndims, r.ne.end(),
                             [](int64_t dim) { return dim == 1; })) {
                LOGE("slat-step input '%s' shape/type mismatch", t->name);
                return false;
            }
            return backend->set_input_f32(t, (const float*)r.data.data(),
                                          expected);
        };
        if (!up(gb.x, x_lat)) return false;
        if (!up(gb.cond, cond_t)) return false;
        const float tval = getenv("SAM3D_T") ? (float)atof(getenv("SAM3D_T")) : 0.0f;
        if (gb.t->buffer && !backend->set_input_f32(gb.t, &tval, 1)) return false;
        for (size_t ti = 0; ti < gb.inputs.size(); ti++) {
            ggml_tensor* t = gb.inputs[ti];
            if (!t->buffer || !gb.table_data[ti]) continue;
            const auto& host = gb.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                ? backend->set_input_f32(t, (const float*)host->data(), host->size())
                : backend->set_input_i32(t, host->data(), host->size());
            if (!ok) return false;
        }
        return backend->run(graph);
    };
    if (!upload_and_run()) return 1;
    LOGI("slat-step: run done");

    if (getenv("SAM3D_DEBUG_DUMP_BLOCKS") != nullptr) {
        for (size_t i = 0; i < gb.debug_block_outputs.size(); ++i) {
            ggml_tensor* block = gb.debug_block_outputs[i];
            std::vector<float> values;
            backend->get_tensor_f32(block, values);
            std::vector<int64_t> ne(block->ne, block->ne + ggml_n_dims(block));
            save_raw_tensor_f32(out_dir + "/slat_block_" + std::to_string(i) + ".samt",
                                ne, values.data());
        }
    }

    for (size_t i = 0; i < outs.size(); i++) {
        std::vector<float> out_f;
        backend->get_tensor_f32(outs[i], out_f);
        std::vector<int64_t> ne(outs[i]->ne, outs[i]->ne + ggml_n_dims(outs[i]));
        const char* dbg = getenv("SAM3D_DEBUG_STAGE");
        std::string path = dbg
            ? out_dir + "/slat_dbg_" + dbg + "_" + std::to_string(i) + ".samt"
            : out_dir + "/slat_v.samt";
        save_raw_tensor_f32(path, ne, out_f.data());
        float mn = 1e30f, mx = -1e30f;
        for (float v : out_f) { mn = std::min(mn, v); mx = std::max(mx, v); }
        printf("slat-step out %lld elems [%g, %g] -> %s\n",
               (long long)out_f.size(), mn, mx, path.c_str());
    }
    return 0;
}

static int cmd_ss_step(const std::string& model_path, const std::string& e2e_dir,
                       const std::string& out_dir) {
    const char* be = getenv("SAM3D_BACKEND");
    auto backend = Backend::create(be ? be : "cpu",
                                   getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;
    GGUFModel m;
    if (!m.load(model_path, backend->weights_buffer_type())) return 1;

    // inputs from the e2e reference dump
    auto load = [](const std::string& p, RawTensor& t) {
        if (!load_raw_tensor(p, t)) { LOGE("missing %s", p.c_str()); return false; }
        return true;
    };
    RawTensor x_shape, x_6d, x_sc, x_tr, x_ts, cond_t;
    if (!load(e2e_dir + "/ss_x0_shape.samt", x_shape)) return 1;
    if (!load(e2e_dir + "/ss_x0_6drotation_normalized.samt", x_6d)) return 1;
    if (!load(e2e_dir + "/ss_x0_scale.samt", x_sc)) return 1;
    if (!load(e2e_dir + "/ss_x0_translation.samt", x_tr)) return 1;
    if (!load(e2e_dir + "/ss_x0_translation_scale.samt", x_ts)) return 1;
    if (!load(e2e_dir + "/ss_cond_tokens.samt", cond_t)) return 1;

    GraphContext gctx;
    SsFlowGraph gb;
    gb.g = &gctx;
    gb.m = &m;
    gb.n_cond_tokens = cond_t.ne[1];
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) gb.debug_stage = st;
    std::vector<ggml_tensor*> outs = gb.build();

    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto* o : outs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }

    LOGI("ss-step: graph built (%d nodes), alloc...", ggml_graph_n_nodes(graph));
    if (!backend->alloc(graph)) return 1;
    LOGI("ss-step: alloc ok, uploading inputs");
    auto up = [&](ggml_tensor* t, const RawTensor& r) {
        if (!t->buffer) return true;  // pruned by the debug-stage graph
        return backend->set_input_f32(t, (const float*)r.data.data(),
                                      (size_t)ggml_nelements(t));
    };
    if (!up(gb.x_shape, x_shape)) return 1;
    if (!up(gb.x_6drot, x_6d)) return 1;
    if (!up(gb.x_scale, x_sc)) return 1;
    if (!up(gb.x_trans, x_tr)) return 1;
    if (!up(gb.x_ts, x_ts)) return 1;
    if (!up(gb.cond, cond_t)) return 1;
    {   // t = d = 0 by default; SAM3D_SS_T overrides the scaled timestep
        const float zero = 0.0f;
        const float tv = getenv("SAM3D_SS_T") ? (float)atof(getenv("SAM3D_SS_T")) : 0.0f;
        if (gb.t->buffer && !backend->set_input_f32(gb.t, &tv, 1)) return 1;
        if (gb.d->buffer && !backend->set_input_f32(gb.d, &zero, 1)) return 1;
    }
    LOGI("ss-step: inputs uploaded, running");
    // uploads + run; the CFG uncond branch reuses the same graph with the
    // condition zeroed, so every input is re-uploaded before each run (the
    // CUDA backend requires the full set per run).
    auto upload_and_run = [&](bool zero_cond) {
        auto up = [&](ggml_tensor* t, const RawTensor& r) {
            if (!t->buffer) return true;  // pruned by the debug-stage graph
            return backend->set_input_f32(t, (const float*)r.data.data(),
                                          (size_t)ggml_nelements(t));
        };
        if (!up(gb.x_shape, x_shape)) return false;
        if (!up(gb.x_6drot, x_6d)) return false;
        if (!up(gb.x_scale, x_sc)) return false;
        if (!up(gb.x_trans, x_tr)) return false;
        if (!up(gb.x_ts, x_ts)) return false;
        if (zero_cond) {
            if (gb.cond->buffer) {
                std::vector<float> zeros((size_t)ggml_nelements(gb.cond), 0.0f);
                if (!backend->set_input_f32(gb.cond, zeros.data(), zeros.size()))
                    return false;
            }
        } else if (!up(gb.cond, cond_t)) {
            return false;
        }
        {   // t = d = 0 unless SAM3D_SS_T overrides; BOTH branches share t
            const float zero = 0.0f;
            const float tv = getenv("SAM3D_SS_T") ? (float)atof(getenv("SAM3D_SS_T")) : 0.0f;
            if (gb.t->buffer && !backend->set_input_f32(gb.t, &tv, 1)) return false;
            if (gb.d->buffer && !backend->set_input_f32(gb.d, &zero, 1)) return false;
        }
        for (size_t ti = 0; ti < gb.inputs.size(); ti++) {
            ggml_tensor* t = gb.inputs[ti];
            if (!t->buffer || !gb.table_data[ti]) continue;
            const auto& host = gb.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                ? backend->set_input_f32(t, (const float*)host->data(), host->size())
                : backend->set_input_i32(t, host->data(), host->size());
            if (!ok) return false;
        }
        return backend->run(graph);
    };
    if (!upload_and_run(false)) return 1;
    LOGI("ss-step: run done");
    // SAM3D_SS_REPEAT=N: re-run the cond branch N extra times (run-count
    // state isolation); dumps ss_rep<k>_shape.samt
    if (const char* rep = getenv("SAM3D_SS_REPEAT")) {
        const int n_rep = atoi(rep);
        for (int k = 0; k < n_rep; k++) {
            if (!upload_and_run(false)) return 1;
            std::vector<float> rp;
            backend->get_tensor_f32(outs[2], rp);
            save_raw_tensor_f32(out_dir + "/ss_rep" + std::to_string(k) + "_shape.samt",
                                {8, 4096}, rp.data());
        }
    }

    // outputs (dict order): 6drot, scale, shape, translation, translation_scale
    const char* names[] = {"6drotation_normalized", "scale", "shape",
                           "translation", "translation_scale"};
    const char* dbg = getenv("SAM3D_DEBUG_STAGE");
    const size_t n_out = outs.size() < 5 ? outs.size() : 5;
    std::vector<std::vector<float>> cond_f(n_out);
    for (size_t i = 0; i < n_out; i++) {
        backend->get_tensor_f32(outs[i], cond_f[i]);
    }

    // uncond branch: force_zeros_cond zeroes the fused condition tokens;
    // the graph is identical, re-upload everything with cond = 0.
    std::vector<std::vector<float>> uncond_f(n_out);
    {
        if (!upload_and_run(true)) return 1;
        for (size_t i = 0; i < n_out; i++) backend->get_tensor_f32(outs[i], uncond_f[i]);
    }

    for (size_t i = 0; i < n_out; i++) {
        std::vector<int64_t> ne(outs[i]->ne, outs[i]->ne + ggml_n_dims(outs[i]));
        std::string base = dbg
            ? out_dir + "/ss_dbg_" + dbg + "_" + std::to_string(i)
            : out_dir + "/ss_v_" + names[i];
        save_raw_tensor_f32(base + ".samt", ne, cond_f[i].data());
        save_raw_tensor_f32(out_dir + std::string("/ss_vu_") + names[i] + ".samt", ne, uncond_f[i].data());
        // ShortCut CFG: v = (1 + strength) * cond - strength * uncond
        const float strength = getenv("SAM3D_CFG_STRENGTH") ? (float)atof(getenv("SAM3D_CFG_STRENGTH")) : 7.0f;
        std::vector<float> blended(cond_f[i].size());
        for (size_t j = 0; j < blended.size(); j++)
            blended[j] = (1.0f + strength) * cond_f[i][j] - strength * uncond_f[i][j];
        save_raw_tensor_f32(out_dir + "/ss_cfg_" + names[i] + ".samt", ne, blended.data());
        float mn = 1e30f, mx = -1e30f;
        for (float v : cond_f[i]) { mn = std::min(mn, v); mx = std::max(mx, v); }
        printf("ss-step %-22s %lld elems [%g, %g]\n", names[i],
               (long long)cond_f[i].size(), mn, mx);
    }
    return 0;
}

static int cmd_info(const std::string& model_path) {
    GGUFModel m;
    if (!m.load(model_path, ggml_backend_cpu_buffer_type())) return 1;
    printf("model: %s\n", model_path.c_str());
    printf("arch:  %s\n", m.str("sam3d.arch").c_str());
    printf("name:  %s\n", m.str("sam3d.model").c_str());
    printf("dtype: %s\n", m.str("sam3d.dtype").c_str());
    printf("tensors: %zu\n", m.n_tensors());
    return 0;
}

namespace sam3d {
int cmd_e2e(const std::string& models_dir, const std::string& cond_dir,
            const std::string& noise_dir, const std::string& out_ply,
            const std::string& dbg_dir, unsigned seed, int nthreads);
}  // namespace sam3d

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
        return 0;
    }
    std::string model, input, out;
    std::vector<std::string> pos;  // bare positional args
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else pos.emplace_back(argv[i]);
    }
    if (cmd == "info") {
        if (model.empty()) {
            print_usage();
            return 1;
        }
        return cmd_info(model);
    }
    if (cmd == "decode-ss") {
        DecodeSsOpts o;
        o.model = model;
        o.input = input;
        o.out = out;
        o.backend = "auto";
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--backend") && i + 1 < argc) o.backend = argv[++i];
            else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) o.warmup = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--iters") && i + 1 < argc) o.iters = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o.threads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--json") && i + 1 < argc) o.json = argv[++i];
        }
        return cmd_decode_ss(o);
    }
    if (cmd == "mesh-export") {
        MeshExportOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--attrs") && i + 1 < argc) options.attributes = argv[++i];
        }
        return cmd_mesh_export(options);
    }
    if (cmd == "moge-smoke") {
        MogeSmokeOpts options;
        options.model = model;
        options.output_prefix = out;
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--backend") && i + 1 < argc) options.backend = argv[++i];
            else if (!strcmp(argv[i], "--width") && i + 1 < argc) options.width = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--height") && i + 1 < argc) options.height = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) options.threads = atoi(argv[++i]);
        }
        if (options.model.empty()) {
            print_usage();
            return 1;
        }
        return cmd_moge_smoke(options);
    }
    if (cmd == "dino") {
        std::string embedder = "cemb.emb0";
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--embedder") && i + 1 < argc) embedder = argv[++i];
        }
        return cmd_dino(model, input, out, embedder);
    }
    if (cmd == "pointpatch") {
        return cmd_pointpatch(model, input, out);
    }
    if (cmd == "ss-step" && pos.size() >= 2) {
        return cmd_ss_step(model, pos[0], pos[1]);
    }
    if (cmd == "slat-step" && pos.size() >= 2) {
        return cmd_slat_step(model, pos[0], pos[1]);
    }
    if (cmd == "gs-decode" && pos.size() >= 2) {
        return cmd_gs_decode(model, pos[0], pos[1]);
    }
    if (cmd == "e2e" && pos.size() >= 1) {
        std::string noise_dir, dbg_dir;
        unsigned seed = 0;
        int nthreads = getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--noise-dir") && i + 1 < argc) noise_dir = argv[++i];
            else if (!strcmp(argv[i], "--dbg-dir") && i + 1 < argc) dbg_dir = argv[++i];
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        }
        // pos[0] = condition dir (dump_e2e_stages output); out = PLY path
        if (out.empty()) out = "output.ply";
        return cmd_e2e(model.empty() ? "cpp_ggml/models/gguf" : model,
                       pos[0], noise_dir, out, dbg_dir, seed, nthreads);
    }
    if (cmd == "run" && pos.size() >= 1) {
        const char* be = getenv("SAM3D_BACKEND");
        std::string backend = be ? be : "auto";
        std::string noise_dir, dbg_dir;
        unsigned seed = 42;
        int nthreads = getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8;
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
            else if (!strcmp(argv[i], "--noise-dir") && i + 1 < argc) noise_dir = argv[++i];
            else if (!strcmp(argv[i], "--dbg-dir") && i + 1 < argc) dbg_dir = argv[++i];
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        }
        if (model.empty()) model = "cpp_ggml/models/gguf";
        if (out.empty()) out = "output.ply";
        if (backend != "auto") setenv("SAM3D_BACKEND", backend.c_str(), 1);
        return cmd_e2e(model, pos[0], noise_dir, out, dbg_dir, seed, nthreads);
    }
    // remaining commands are implemented in session.cpp via run_pipeline
    CliOptions opts;
    opts.models_dir = model.empty() ? opts.models_dir : model;
    opts.condition_dir = !pos.empty() ? pos.front() : input;
    opts.out_ply = out;
    if (const char* be = getenv("SAM3D_BACKEND")) opts.backend = be;
    if (const char* nt = getenv("SAM3D_NTHREADS")) opts.n_threads = atoi(nt);
    RunResult r = run_pipeline(opts);
    return r.ok ? 0 : 1;
}
