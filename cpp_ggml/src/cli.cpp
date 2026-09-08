// sam3d-cli: command line adapter for the SAM 3D ggml runtime.
#include <cstring>
#include <cstdio>
#include <string>

#include "sam3dggml.h"
#include "asset_io.hpp"
#include "common.hpp"
#include "image_preprocess.hpp"
#include "gguf_loader.hpp"
#include "backend.hpp"
#include "ss_decoder_graph.hpp"
#include "dino_graph.hpp"
#include "moge_graph.hpp"
#include "pointpatch_graph.hpp"
#include "ss_flow_graph.hpp"
#include "slat_flow_graph.hpp"
#include "gs_decoder_graph.hpp"
#include "flexicubes.hpp"
#include "mesh_postprocess.hpp"
#include "mesh_uv.hpp"
#include "mesh_decoder_graph.hpp"
#if defined(SAM3D_NATIVE_PBR_CUDA)
#include "gaussian_renderer.hpp"
#include "texture_inpaint.hpp"
#endif
#include "sparse_ops.hpp"
#include "graph_builder.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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
            "             [--input image.png] [--width N] [--height N] [--threads N]\n"
            "  preprocess-conditions --image <image> --pointmap <pointmap.samt> --out-dir <dir>\n"
            "             [--mask <binary-mask.png>] [--decoded-rgb-out <image.samt>]\n"
            "  mesh-export --vertices <vertices.samt> --faces <faces.samt>\n"
            "              [--attrs <vertex_attrs.samt>] --out <asset.glb>\n"
            "  mesh-decode --model <slat_decoder_mesh.gguf> --input <slat_feats.samt>\n"
            "              --coords <slat_coords.samt> --out <cube_features.samt>\n"
            "              [--coords-out <subdivided_coords.samt>] [--stage input_layer|blockN|upsampleN]\n"
            "              [--backend cpu|cuda|vulkan]\n"
            "  mesh-extract --features <cube_features.samt> --coords <cube_coords.samt>\n"
            "               --out <asset.glb> [--resolution N] [--vertices-out <vertices.samt>]\n"
            "               [--faces-out <faces.samt>] [--attrs-out <vertex_attrs.samt>]\n"
            "               [--xatlas-uv --uv-out <uv.samt>]\n"
            "  mesh-postprocess --vertices <vertices.samt> --faces <faces.samt>\n"
            "                   --vertices-out <vertices.samt> --faces-out <faces.samt>\n"
            "                   [--target-reduction 0.95 --visibility <frequency.samt>]\n"
            "  mesh-filter-visibility --vertices <vertices.samt> --faces <faces.samt>\n"
            "                         --visibility <frequency.samt>\n"
            "                         --vertices-out <vertices.samt> --faces-out <faces.samt>\n"
            "                         [--candidates-out <face_indices.samt>]\n"
            "  mesh-repair-boundaries --vertices <vertices.samt> --faces <faces.samt>\n"
            "                         --vertices-out <vertices.samt> --faces-out <faces.samt>\n"
            "                         [--max-boundary-edges 55 --no-refine]\n"
            "  mesh-parameterize --vertices <vertices.samt> --faces <faces.samt>\n"
            "                    --vertices-out <vertices.samt> --faces-out <faces.samt>\n"
            "                    --uv-out <uv.samt>\n"
            "  gaussian-render --ply <output_gs.ply> --out-dir <images_dir>\n"
            "               [--views 100] [--resolution 1024]\n"
            "               [--extrinsics <cameras.samt> --intrinsics <cameras.samt>]\n"
            "  texture-inpaint --texture <base_color.png> --mask <holes.png> --out <repaired.png>\n"
            "                  [--radius 3]\n"
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

static bool tensor_indices(const std::string& path, RawTensor& tensor,
                           std::vector<uint32_t>& values, const char* label) {
    if (!load_raw_tensor(path, tensor) ||
        (tensor.type != GGML_TYPE_F32 && tensor.type != GGML_TYPE_I32)) {
        LOGE("%s must be a readable F32 or I32 SAMT tensor: %s", label, path.c_str());
        return false;
    }
    const size_t count = tensor.type == GGML_TYPE_F32
        ? tensor.data.size() / sizeof(float)
        : tensor.data.size() / sizeof(int32_t);
    const size_t element_bytes = tensor.type == GGML_TYPE_F32 ? sizeof(float) : sizeof(int32_t);
    if (tensor.data.size() != count * element_bytes) {
        LOGE("%s has a malformed SAMT payload: %s", label, path.c_str());
        return false;
    }
    values.resize(count);
    if (tensor.type == GGML_TYPE_I32) {
        for (size_t index = 0; index < count; ++index) {
            int32_t value = 0;
            std::memcpy(&value, tensor.data.data() + index * sizeof(value), sizeof(value));
            if (value < 0) {
                LOGE("%s contains a negative index", label);
                return false;
            }
            values[index] = static_cast<uint32_t>(value);
        }
        return true;
    }
    for (size_t index = 0; index < count; ++index) {
        float value = 0.0f;
        std::memcpy(&value, tensor.data.data() + index * sizeof(value), sizeof(value));
        if (!std::isfinite(value) || value < 0.0f ||
            value > static_cast<float>(std::numeric_limits<uint32_t>::max()) ||
            std::floor(value) != value) {
            LOGE("%s contains a non-integer or invalid index", label);
            return false;
        }
        values[index] = static_cast<uint32_t>(value);
    }
    return true;
}

static bool save_mesh_indices_i32(const std::string& path, int64_t face_count,
                                  const std::vector<uint32_t>& indices) {
    if (indices.size() != static_cast<size_t>(face_count) * 3) return false;
    std::vector<int32_t> output;
    output.reserve(indices.size());
    for (uint32_t index : indices) {
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) return false;
        output.push_back(static_cast<int32_t>(index));
    }
    return save_raw_tensor_i32(path, {3, face_count}, output.data());
}

static int cmd_mesh_export(const MeshExportOpts& options) {
    if (options.vertices.empty() || options.faces.empty() || options.output.empty()) {
        LOGE("mesh-export requires --vertices, --faces and --out");
        return 1;
    }

    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-export faces")) {
        return 1;
    }
    if (vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3) {
        LOGE("mesh-export: vertices must have official SAMT shape [3, vertex_count]");
        return 1;
    }
    if (faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-export: faces must have official F32/I32 SAMT shape [3, face_count]");
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
    for (uint32_t index : faces) {
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

struct MeshExtractOpts {
    std::string features;
    std::string coordinates;
    std::string output;
    std::string vertices_output;
    std::string faces_output;
    std::string candidates_output;
    std::string attributes_output;
    std::string uv_output;
    bool xatlas_uv = false;
    int resolution = 256;
};

struct MeshPostprocessOpts {
    std::string vertices;
    std::string faces;
    std::string vertices_output;
    std::string faces_output;
    std::string visibility;
    float target_reduction = 0.95f;
};

struct MeshVisibilityFilterOpts {
    std::string vertices;
    std::string faces;
    std::string visibility;
    std::string vertices_output;
    std::string faces_output;
    std::string candidates_output;
};

struct MeshBoundaryRepairOpts {
    std::string vertices;
    std::string faces;
    std::string vertices_output;
    std::string faces_output;
    int max_boundary_edges = 55;
    bool refine = true;
};

struct MeshParameterizeOpts {
    std::string vertices;
    std::string faces;
    std::string vertices_output;
    std::string faces_output;
    std::string uv_output;
};

static int cmd_mesh_postprocess(const MeshPostprocessOpts& options) {
#if !defined(SAM3D_NATIVE_PBR_CUDA)
    (void)options;
    LOGE("mesh-postprocess requires a CUDA native-PBR build with VTK support");
    return 1;
#else
    if (options.vertices.empty() || options.faces.empty() || options.vertices_output.empty() ||
        options.faces_output.empty()) {
        LOGE("mesh-postprocess requires --vertices, --faces, --vertices-out and --faces-out");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "mesh-postprocess vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-postprocess faces") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-postprocess expects vertices [3,V] F32 and faces [3,F] F32/I32 SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    std::string error;
    MeshSimplifyOptions simplify;
    simplify.target_reduction = options.target_reduction;
    if (!simplify_mesh_official_vtk(mesh, simplify, error)) {
        LOGE("mesh-postprocess: %s", error.c_str());
        return 1;
    }
    MeshVisibilityFilterStats visibility_stats;
    if (!options.visibility.empty()) {
        RawTensor visibility_tensor;
        std::vector<float> visibility;
        if (!tensor_values_f32(options.visibility, visibility_tensor, visibility,
                               "mesh-postprocess visibility") ||
            visibility_tensor.ne.size() != 1 ||
            visibility.size() != mesh.indices.size() / 3) {
            LOGE("mesh-postprocess visibility must be F32 [face_count] after VTK reduction");
            return 1;
        }
        if (!filter_invisible_faces_official(mesh, visibility, {}, &visibility_stats, error)) {
            LOGE("mesh-postprocess visibility/mincut: %s", error.c_str());
            return 1;
        }
    }
    const int64_t vertex_count = static_cast<int64_t>(mesh.positions.size() / 3);
    const int64_t face_count = static_cast<int64_t>(mesh.indices.size() / 3);
    if (!save_raw_tensor_f32(options.vertices_output, {3, vertex_count}, mesh.positions.data()) ||
        !save_mesh_indices_i32(options.faces_output, face_count, mesh.indices)) {
        LOGE("mesh-postprocess: failed to write output SAMT tensors");
        return 1;
    }
    LOGI("mesh-postprocess: VTK reduction %.3f: %lld vertices, %lld triangles; "
         "visibility mincut candidates=%zu removed=%zu (MeshFix boundary repair is separate)",
         options.target_reduction, static_cast<long long>(vertex_count), static_cast<long long>(face_count),
         visibility_stats.mincut_face_count, visibility_stats.removed_face_count);
    return 0;
#endif
}

static int cmd_mesh_filter_visibility(const MeshVisibilityFilterOpts& options) {
#if !defined(SAM3D_NATIVE_PBR_CUDA)
    (void)options;
    LOGE("mesh-filter-visibility requires a CUDA native-PBR build with VTK support");
    return 1;
#else
    if (options.vertices.empty() || options.faces.empty() || options.visibility.empty() ||
        options.vertices_output.empty() || options.faces_output.empty()) {
        LOGE("mesh-filter-visibility requires --vertices, --faces, --visibility, --vertices-out and --faces-out");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    RawTensor visibility_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    std::vector<float> visibility;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "mesh-filter-visibility vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-filter-visibility faces") ||
        !tensor_values_f32(options.visibility, visibility_tensor, visibility,
                           "mesh-filter-visibility visibility") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3 ||
        visibility_tensor.ne.size() != 1 || visibility.size() != faces.size() / 3) {
        LOGE("mesh-filter-visibility expects vertices [3,V], faces [3,F], visibility [F] SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    MeshVisibilityFilterStats stats;
    std::vector<uint32_t> candidates;
    std::string error;
    if (!filter_invisible_faces_official(mesh, visibility, {}, &stats, error,
                                         options.candidates_output.empty() ? nullptr : &candidates)) {
        LOGE("mesh-filter-visibility: %s", error.c_str());
        return 1;
    }
    const int64_t vertex_count = static_cast<int64_t>(mesh.positions.size() / 3);
    const int64_t face_count = static_cast<int64_t>(mesh.indices.size() / 3);
    if (!save_raw_tensor_f32(options.vertices_output, {3, vertex_count}, mesh.positions.data()) ||
        !save_mesh_indices_i32(options.faces_output, face_count, mesh.indices)) {
        LOGE("mesh-filter-visibility: failed to write output SAMT tensors");
        return 1;
    }
    if (!options.candidates_output.empty()) {
        std::vector<int32_t> candidate_indices;
        candidate_indices.reserve(candidates.size());
        for (uint32_t face : candidates) candidate_indices.push_back(static_cast<int32_t>(face));
        if (!save_raw_tensor_i32(options.candidates_output,
                                 {static_cast<int64_t>(candidate_indices.size())},
                                 candidate_indices.data())) {
            LOGE("mesh-filter-visibility: failed to write candidate face indices");
            return 1;
        }
    }
    LOGI("mesh-filter-visibility: invisible=%zu outer=%zu mincut=%zu removed=%zu; "
         "%lld vertices, %lld triangles (before MeshFix)",
         stats.invisible_face_count, stats.outer_face_count, stats.mincut_face_count,
         stats.removed_face_count, static_cast<long long>(vertex_count),
         static_cast<long long>(face_count));
    return 0;
#endif
}

static int cmd_mesh_repair_boundaries(const MeshBoundaryRepairOpts& options) {
    if (options.vertices.empty() || options.faces.empty() || options.vertices_output.empty() ||
        options.faces_output.empty()) {
        LOGE("mesh-repair-boundaries requires --vertices, --faces, --vertices-out and --faces-out");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices,
                           "mesh-repair-boundaries vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-repair-boundaries faces") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-repair-boundaries expects vertices [3,V] and faces [3,F] SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    std::string error;
    if (!repair_mesh_boundaries_official_meshfix(mesh, options.max_boundary_edges, options.refine, error)) {
        LOGE("mesh-repair-boundaries: %s", error.c_str());
        return 1;
    }
    const int64_t vertex_count = static_cast<int64_t>(mesh.positions.size() / 3);
    const int64_t face_count = static_cast<int64_t>(mesh.indices.size() / 3);
    if (!save_raw_tensor_f32(options.vertices_output, {3, vertex_count}, mesh.positions.data()) ||
        !save_mesh_indices_i32(options.faces_output, face_count, mesh.indices)) {
        LOGE("mesh-repair-boundaries: failed to write output SAMT tensors");
        return 1;
    }
    LOGI("mesh-repair-boundaries: %lld vertices, %lld triangles",
         static_cast<long long>(vertex_count), static_cast<long long>(face_count));
    return 0;
}

static int cmd_mesh_parameterize(const MeshParameterizeOpts& options) {
    if (options.vertices.empty() || options.faces.empty() || options.vertices_output.empty() ||
        options.faces_output.empty() || options.uv_output.empty()) {
        LOGE("mesh-parameterize requires --vertices, --faces, --vertices-out, --faces-out and --uv-out");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "mesh-parameterize vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-parameterize faces") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-parameterize expects vertices [3,V] F32 and faces [3,F] F32/I32 SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    std::string error;
    if (!parameterize_mesh_xatlas(mesh, error)) {
        LOGE("mesh-parameterize: %s", error.c_str());
        return 1;
    }
    const int64_t vertex_count = static_cast<int64_t>(mesh.positions.size() / 3);
    const int64_t face_count = static_cast<int64_t>(mesh.indices.size() / 3);
    if (!save_raw_tensor_f32(options.vertices_output, {3, vertex_count}, mesh.positions.data()) ||
        !save_mesh_indices_i32(options.faces_output, face_count, mesh.indices) ||
        !save_raw_tensor_f32(options.uv_output, {2, vertex_count}, mesh.texcoords.data())) {
        LOGE("mesh-parameterize: failed to write output SAMT tensors");
        return 1;
    }
    LOGI("mesh-parameterize: xatlas produced %lld vertices, %lld triangles",
         static_cast<long long>(vertex_count), static_cast<long long>(face_count));
    return 0;
}

struct GaussianRenderOpts {
    std::string ply;
    std::string output_dir;
    std::string extrinsics;
    std::string intrinsics;
    int views = 100;
    int resolution = 1024;
};

struct TextureInpaintOpts {
    std::string texture;
    std::string mask;
    std::string output;
    float radius = 3.0f;
};

static int cmd_texture_inpaint(const TextureInpaintOpts& options) {
#if !defined(SAM3D_NATIVE_PBR_CUDA)
    (void)options;
    LOGE("texture-inpaint requires a native-PBR CUDA build with OpenCV Telea support");
    return 1;
#else
    if (options.texture.empty() || options.mask.empty() || options.output.empty()) {
        LOGE("texture-inpaint requires --texture, --mask and --out");
        return 1;
    }
    RgbaImage texture;
    RgbaImage mask;
    std::string error;
    if (!load_rgba_image(options.texture, texture, error)) {
        LOGE("texture-inpaint: %s", error.c_str());
        return 1;
    }
    if (!load_rgba_image(options.mask, mask, error)) {
        LOGE("texture-inpaint: %s", error.c_str());
        return 1;
    }
    if (texture.width != mask.width || texture.height != mask.height) {
        LOGE("texture-inpaint: texture and mask dimensions differ");
        return 1;
    }
    std::vector<uint8_t> holes(static_cast<size_t>(texture.width) * texture.height);
    for (size_t pixel = 0; pixel < holes.size(); ++pixel) {
        // Official masks are single-channel 0/1 images. Using RGB luminance
        // makes the command accept their PNG form after stb's RGBA decode.
        const uint8_t* value = mask.rgba.data() + pixel * 4;
        holes[pixel] = static_cast<uint8_t>(value[0] != 0 || value[1] != 0 || value[2] != 0);
    }
    if (!inpaint_texture_telea(texture, holes, options.radius, error)) {
        LOGE("texture-inpaint: %s", error.c_str());
        return 1;
    }
    if (!write_rgba_png(options.output, texture, error)) {
        LOGE("texture-inpaint: %s", error.c_str());
        return 1;
    }
    LOGI("texture-inpaint: wrote %s", options.output.c_str());
    return 0;
#endif
}

#if defined(SAM3D_NATIVE_PBR_CUDA)
static bool load_gaussian_cameras(const GaussianRenderOpts& options,
                                  std::vector<GaussianCamera>& cameras,
                                  std::string& error) {
    if (options.extrinsics.empty() != options.intrinsics.empty()) {
        error = "--extrinsics and --intrinsics must be supplied together";
        return false;
    }
    if (options.extrinsics.empty()) {
        GaussianRenderConfig config;
        config.width = options.resolution;
        config.height = options.resolution;
        cameras = make_gaussian_hammersley_cameras(options.views, config, error);
        return !cameras.empty();
    }

    RawTensor extrinsics;
    RawTensor intrinsics;
    if (!load_raw_tensor(options.extrinsics, extrinsics) ||
        !load_raw_tensor(options.intrinsics, intrinsics) ||
        extrinsics.type != GGML_TYPE_F32 || intrinsics.type != GGML_TYPE_F32 ||
        extrinsics.ne.size() != 3 || intrinsics.ne.size() != 3 ||
        extrinsics.ne[0] != 4 || extrinsics.ne[1] != 4 ||
        intrinsics.ne[0] != 3 || intrinsics.ne[1] != 3 ||
        extrinsics.ne[2] <= 0 || intrinsics.ne[2] != extrinsics.ne[2] ||
        extrinsics.data.size() != static_cast<size_t>(extrinsics.ne[2]) * 16 * sizeof(float) ||
        intrinsics.data.size() != static_cast<size_t>(intrinsics.ne[2]) * 9 * sizeof(float)) {
        error = "camera SAMT tensors must be F32 [camera_count,4,4] and [camera_count,3,3]";
        return false;
    }
    const size_t count = static_cast<size_t>(extrinsics.ne[2]);
    cameras.resize(count);
    for (size_t index = 0; index < count; ++index) {
        std::memcpy(cameras[index].extrinsics.data(),
                    extrinsics.data.data() + index * 16 * sizeof(float), 16 * sizeof(float));
        std::memcpy(cameras[index].intrinsics.data(),
                    intrinsics.data.data() + index * 9 * sizeof(float), 9 * sizeof(float));
    }
    return true;
}
#endif

static int cmd_gaussian_render(const GaussianRenderOpts& options) {
#if !defined(SAM3D_NATIVE_PBR_CUDA)
    (void)options;
    LOGE("gaussian-render requires a CUDA build configured with SAM3D_GGML_NATIVE_PBR=ON");
    return 1;
#else
    if (options.ply.empty() || options.output_dir.empty() || options.views <= 0 ||
        options.resolution <= 0) {
        LOGE("gaussian-render requires --ply, --out-dir, positive --views and positive --resolution");
        return 1;
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(options.output_dir, filesystem_error);
    if (filesystem_error) {
        LOGE("gaussian-render: cannot create %s: %s", options.output_dir.c_str(),
             filesystem_error.message().c_str());
        return 1;
    }
    GaussianSplatSet splats;
    std::string error;
    if (!load_gaussian_splat_ply(options.ply, splats, error)) {
        LOGE("gaussian-render: %s", error.c_str());
        return 1;
    }
    GaussianRenderConfig config;
    config.width = options.resolution;
    config.height = options.resolution;
    std::vector<GaussianCamera> cameras;
    if (!load_gaussian_cameras(options, cameras, error)) {
        LOGE("gaussian-render: %s", error.c_str());
        return 1;
    }
    std::vector<RgbaImage> images;
    if (!render_gaussian_views_cuda(splats, cameras, config, images, error)) {
        LOGE("gaussian-render: %s", error.c_str());
        return 1;
    }
    for (size_t index = 0; index < images.size(); ++index) {
        char name[64];
        std::snprintf(name, sizeof(name), "view_%03zu.png", index);
        const std::string output = (std::filesystem::path(options.output_dir) / name).string();
        if (!write_rgba_png(output, images[index], error)) {
            LOGE("gaussian-render: %s", error.c_str());
            return 1;
        }
    }
    LOGI("gaussian-render: wrote %zu official Hammersley views (%dx%d) to %s", images.size(),
         config.width, config.height, options.output_dir.c_str());
    return 0;
#endif
}

static int cmd_mesh_extract(const MeshExtractOpts& options) {
    if (options.features.empty() || options.coordinates.empty() || options.output.empty()) {
        LOGE("mesh-extract requires --features, --coords and --out");
        return 1;
    }
    RawTensor features_tensor;
    RawTensor coordinates_tensor;
    std::vector<float> features;
    if (!tensor_values_f32(options.features, features_tensor, features, "cube features") ||
        !load_raw_tensor(options.coordinates, coordinates_tensor) ||
        coordinates_tensor.type != GGML_TYPE_I32 || coordinates_tensor.ne.size() != 2 ||
        coordinates_tensor.ne[0] != 4 ||
        coordinates_tensor.data.size() % sizeof(int32_t) != 0) {
        LOGE("mesh-extract expects F32 [101, cell_count] features and I32 [4, cell_count] coordinates");
        return 1;
    }
    if (features_tensor.ne.size() != 2 || features_tensor.ne[0] != 101 ||
        features_tensor.ne[1] <= 0 || coordinates_tensor.ne[1] != features_tensor.ne[1] ||
        features.size() != static_cast<size_t>(features_tensor.ne[1]) * 101) {
        LOGE("mesh-extract: feature and coordinate dimensions do not match the official mesh layout");
        return 1;
    }
    if (options.resolution <= 0) {
        LOGE("mesh-extract: --resolution must be positive");
        return 1;
    }
    std::vector<int32_t> coordinates(coordinates_tensor.data.size() / sizeof(int32_t));
    std::memcpy(coordinates.data(), coordinates_tensor.data.data(), coordinates_tensor.data.size());

    FlexiCubesResult extracted;
    std::string error;
    if (!decode_flexicubes(features.data(), coordinates.data(), features_tensor.ne[1],
                           options.resolution, extracted, error)) {
        LOGE("mesh-extract: %s", error.c_str());
        return 1;
    }
    if (!extracted.success()) {
        LOGE("mesh-extract: FlexiCubes found no closed surface");
        return 1;
    }
    const int64_t extracted_vertex_count = static_cast<int64_t>(extracted.positions.size() / 3);
    NativeMesh mesh;
    mesh.positions = extracted.positions;
    mesh.indices = extracted.indices;
    mesh.vertex_attributes = extracted.vertex_attributes;
    mesh.colors.resize(static_cast<size_t>(extracted_vertex_count) * 3);
    for (int64_t vertex = 0; vertex < extracted_vertex_count; ++vertex) {
        mesh.colors[vertex * 3 + 0] = extracted.vertex_attributes[vertex * 6 + 0];
        mesh.colors[vertex * 3 + 1] = extracted.vertex_attributes[vertex * 6 + 1];
        mesh.colors[vertex * 3 + 2] = extracted.vertex_attributes[vertex * 6 + 2];
    }
    if (options.xatlas_uv) {
        if (!parameterize_mesh_xatlas(mesh, error)) {
            LOGE("mesh-extract: %s", error.c_str());
            return 1;
        }
        if (!options.uv_output.empty() &&
            !save_raw_tensor_f32(options.uv_output,
                                 {2, static_cast<int64_t>(mesh.texcoords.size() / 2)},
                                 mesh.texcoords.data())) {
            LOGE("mesh-extract: failed to write %s", options.uv_output.c_str());
            return 1;
        }
    } else if (!options.uv_output.empty()) {
        LOGE("mesh-extract: --uv-out requires --xatlas-uv");
        return 1;
    }
    const int64_t output_vertex_count = static_cast<int64_t>(mesh.positions.size() / 3);
    const int64_t output_face_count = static_cast<int64_t>(mesh.indices.size() / 3);
    for (int64_t vertex = 0; vertex < output_vertex_count; ++vertex) {
        const float y = mesh.positions[vertex * 3 + 1];
        mesh.positions[vertex * 3 + 1] = mesh.positions[vertex * 3 + 2];
        mesh.positions[vertex * 3 + 2] = -y;
    }
    // Every exported tensor is in the same final coordinate system and uses
    // the seam-expanded xatlas vertex order when --xatlas-uv is selected.
    if (!options.vertices_output.empty() &&
        !save_raw_tensor_f32(options.vertices_output, {3, output_vertex_count}, mesh.positions.data())) {
        LOGE("mesh-extract: failed to write %s", options.vertices_output.c_str());
        return 1;
    }
    if (!options.attributes_output.empty() &&
        (mesh.vertex_attributes.size() != static_cast<size_t>(output_vertex_count) * 6 ||
         !save_raw_tensor_f32(options.attributes_output, {6, output_vertex_count},
                              mesh.vertex_attributes.data()))) {
        LOGE("mesh-extract: failed to write %s", options.attributes_output.c_str());
        return 1;
    }
    if (!options.faces_output.empty()) {
        if (!save_mesh_indices_i32(options.faces_output, output_face_count, mesh.indices)) {
            LOGE("mesh-extract: failed to write %s", options.faces_output.c_str());
            return 1;
        }
    }
    if (!write_pbr_glb(options.output, mesh, error)) {
        LOGE("mesh-extract: %s", error.c_str());
        return 1;
    }
    LOGI("mesh-extract: wrote %s (%lld vertices, %lld triangles)", options.output.c_str(),
         static_cast<long long>(output_vertex_count), static_cast<long long>(output_face_count));
    return 0;
}

struct MeshDecodeOpts {
    std::string model;
    std::string input;
    std::string coordinates;
    std::string output;
    std::string output_coordinates;
    std::string stage;
    std::string backend = "auto";
    int threads = 8;
};

static int cmd_mesh_decode(const MeshDecodeOpts& options) {
    if (options.model.empty() || options.input.empty() || options.coordinates.empty() ||
        options.output.empty()) {
        LOGE("mesh-decode requires --model, --input, --coords and --out");
        return 1;
    }

    RawTensor features_tensor;
    RawTensor coords_tensor;
    if (!load_raw_tensor(options.input, features_tensor) ||
        features_tensor.type != GGML_TYPE_F32 || features_tensor.ne.size() != 2 ||
        features_tensor.ne[0] != 8 || features_tensor.data.size() % sizeof(float) != 0) {
        LOGE("mesh-decode: --input must be an F32 SAMT tensor with shape [8, token_count]");
        return 1;
    }
    if (!load_raw_tensor(options.coordinates, coords_tensor) ||
        coords_tensor.type != GGML_TYPE_I32 || coords_tensor.ne.size() != 2 ||
        coords_tensor.ne[0] != 4 || coords_tensor.data.size() % sizeof(int32_t) != 0) {
        LOGE("mesh-decode: --coords must be an I32 SAMT tensor with shape [4, token_count]");
        return 1;
    }
    const int64_t token_count = features_tensor.ne[1];
    if (coords_tensor.ne[1] != token_count || token_count <= 0 ||
        features_tensor.data.size() != static_cast<size_t>(token_count) * 8 * sizeof(float) ||
        coords_tensor.data.size() != static_cast<size_t>(token_count) * 4 * sizeof(int32_t)) {
        LOGE("mesh-decode: feature and coordinate token counts do not agree");
        return 1;
    }

    auto backend = Backend::create(options.backend, options.threads);
    if (!backend) return 1;
    GGUFModel model;
    if (!model.load(options.model, backend->weights_buffer_type())) return 1;
    if (model.str("sam3d.model") != "slat_decoder_mesh") {
        LOGE("mesh-decode requires slat_decoder_mesh GGUF, got %s",
             model.str("sam3d.model").c_str());
        return 1;
    }

    const int32_t* input_coords = reinterpret_cast<const int32_t*>(coords_tensor.data.data());
    MeshTables tables;
    if (!tables.build(input_coords, token_count, model.i32("meshdec.resolution", 64))) {
        LOGE("mesh-decode: failed to construct sparse subdivision or convolution tables");
        return 1;
    }
    const int64_t final_output_tokens = token_count * 64;
    if (tables.levels[2].n != final_output_tokens) {
        LOGE("mesh-decode: unexpected two-stage subdivision size");
        return 1;
    }

    GraphContext context;
    MeshDecoderGraph graph_builder;
    graph_builder.g = &context;
    graph_builder.m = &model;
    graph_builder.tb = &tables;
    graph_builder.debug_stage = options.stage;
    ggml_tensor* input = context.input_f32("mesh_decoder_input", {8, token_count});
    graph_builder.x = input;
    graph_builder.inputs.push_back(input);
    graph_builder.table_data.push_back(nullptr);
    const std::vector<ggml_tensor*> outputs = graph_builder.build();
    GGML_ASSERT(outputs.size() == 1);
    const int64_t output_channels = outputs.front()->ne[0];
    const int64_t output_tokens = outputs.front()->ne[1];
    if (options.stage.empty() &&
        (output_channels != 101 || output_tokens != final_output_tokens)) {
        LOGE("mesh-decode: final graph shape is not [101, %lld]",
             static_cast<long long>(final_output_tokens));
        return 1;
    }
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 65536, false);
    ggml_set_output(outputs.front());
    ggml_build_forward_expand(graph, outputs.front());
    if (!backend->alloc(graph) ||
        !backend->set_input_f32(input,
                                reinterpret_cast<const float*>(features_tensor.data.data()),
                                static_cast<size_t>(token_count) * 8)) {
        return 1;
    }
    for (size_t index = 0; index < graph_builder.inputs.size(); ++index) {
        ggml_tensor* graph_input = graph_builder.inputs[index];
        const auto& data = graph_builder.table_data[index];
        if (!graph_input->buffer || !data) continue;
        const bool uploaded = graph_input->type == GGML_TYPE_F32
            ? backend->set_input_f32(graph_input, reinterpret_cast<const float*>(data->data()), data->size())
            : backend->set_input_i32(graph_input, data->data(), data->size());
        if (!uploaded) return 1;
    }
    if (!backend->run(graph)) return 1;

    std::vector<float> output;
    if (!backend->get_tensor_f32(outputs.front(), output) ||
        output.size() != static_cast<size_t>(output_tokens) * output_channels ||
        !std::all_of(output.begin(), output.end(),
                     [](float value) { return std::isfinite(value); })) {
        LOGE("mesh-decode: graph produced an invalid feature tensor");
        return 1;
    }
    if (!save_raw_tensor_f32(options.output, {output_channels, output_tokens}, output.data())) {
        LOGE("mesh-decode: failed to write %s", options.output.c_str());
        return 1;
    }
    if (!options.output_coordinates.empty()) {
        const MeshConvLevel* output_level = &tables.levels[0];
        if (output_tokens == tables.levels[1].n) output_level = &tables.levels[1];
        if (output_tokens == tables.levels[2].n) output_level = &tables.levels[2];
        if (output_tokens != output_level->n) {
            LOGE("mesh-decode: debug stage has no matching sparse support");
            return 1;
        }
        RawTensor output_coords;
        output_coords.ne = {4, output_tokens};
        output_coords.type = GGML_TYPE_I32;
        output_coords.data.resize(output_level->coords.size() * sizeof(int32_t));
        std::memcpy(output_coords.data.data(), output_level->coords.data(), output_coords.data.size());
        if (!save_raw_tensor(options.output_coordinates, output_coords)) {
            LOGE("mesh-decode: failed to write %s", options.output_coordinates.c_str());
            return 1;
        }
    }
    LOGI("mesh-decode: %lld input cells -> [%lld, %lld] %s (%s)",
         static_cast<long long>(token_count), static_cast<long long>(output_channels),
         static_cast<long long>(output_tokens),
         options.stage.empty() ? "raw features" : options.stage.c_str(), backend->backend_name());
    return 0;
}

struct MogeSmokeOpts {
    std::string model;
    std::string output_prefix;
    std::string image_path;
    std::string backend = "cpu";
    int width = 56;
    int height = 56;
    int threads = 8;
};

struct PreprocessConditionsOpts {
    std::string image;
    std::string mask;
    std::string pointmap;
    std::string output_dir;
    std::string decoded_rgb_output;
};

static bool save_decoded_rgb(const std::string& path, const RgbaImage& image) {
    std::vector<float> rgb(static_cast<size_t>(image.width) * image.height * 3);
    for (size_t pixel = 0; pixel < static_cast<size_t>(image.width) * image.height; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            rgb[pixel * 3 + channel] = static_cast<float>(image.rgba[pixel * 4 + channel]);
        }
    }
    // The Python fixture is contiguous HWC, serialized with reversed ggml
    // dimensions. Keep the byte order as HWC so the decoder contract can be
    // checked directly against `input_image_rgba.samt`.
    return save_raw_tensor_f32(path, {3, image.width, image.height}, rgb.data());
}

static int cmd_preprocess_conditions(const PreprocessConditionsOpts& options) {
    if (options.image.empty() || options.pointmap.empty() || options.output_dir.empty()) {
        LOGE("preprocess-conditions requires --image, --pointmap and --out-dir");
        return 1;
    }
    RgbaImage image;
    std::string error;
    if (!load_rgba_image(options.image, image, error)) {
        LOGE("preprocess-conditions: %s", error.c_str());
        return 1;
    }
    if (!options.decoded_rgb_output.empty() &&
        !save_decoded_rgb(options.decoded_rgb_output, image)) {
        LOGE("preprocess-conditions: failed to write decoded RGB tensor %s",
             options.decoded_rgb_output.c_str());
        return 1;
    }
    if (!options.mask.empty()) {
        RgbaImage mask;
        if (!load_rgba_image(options.mask, mask, error)) {
            LOGE("preprocess-conditions: %s", error.c_str());
            return 1;
        }
        if (mask.width != image.width || mask.height != image.height) {
            LOGE("preprocess-conditions: mask dimensions must match the input image");
            return 1;
        }
        for (size_t pixel = 0; pixel < static_cast<size_t>(image.width) * image.height; ++pixel) {
            // notebook/inference.py::load_mask first applies `mask > 0` and
            // then selects the last channel. With stb's RGBA decode, that is
            // the alpha channel; preserve its boolean semantics exactly.
            image.rgba[pixel * 4 + 3] = mask.rgba[pixel * 4 + 3] == 0 ? 0 : 255;
        }
    }
    RawTensor pointmap;
    if (!load_raw_tensor(options.pointmap, pointmap) || pointmap.type != GGML_TYPE_F32 ||
        (pointmap.ne.size() != 3 && pointmap.ne.size() != 4) || pointmap.ne[0] <= 0 ||
        pointmap.ne[1] <= 0 || pointmap.ne[2] != 3 ||
        (pointmap.ne.size() == 4 && pointmap.ne[3] != 1) ||
        pointmap.data.size() % sizeof(float) != 0) {
        LOGE("preprocess-conditions: pointmap must be F32 SAMT [W,H,3] or [W,H,3,1]");
        return 1;
    }
    const int width = static_cast<int>(pointmap.ne[0]);
    const int height = static_cast<int>(pointmap.ne[1]);
    const size_t count = static_cast<size_t>(width) * height * 3;
    if (pointmap.data.size() != count * sizeof(float)) {
        LOGE("preprocess-conditions: pointmap payload size is inconsistent with its shape");
        return 1;
    }
    std::vector<float> values(count);
    std::memcpy(values.data(), pointmap.data.data(), pointmap.data.size());
    NativeConditionInputs output;
    if (!preprocess_ss_conditions(image, values, width, height, {}, output, error) ||
        !write_ss_conditions(options.output_dir, output, error)) {
        LOGE("preprocess-conditions: %s", error.c_str());
        return 1;
    }
    LOGI("preprocess-conditions: wrote native 518x518 SS conditions to %s", options.output_dir.c_str());
    return 0;
}

static int cmd_moge_smoke(const MogeSmokeOpts& options) {
    int width = options.width;
    int height = options.height;
    std::vector<float> rgb;
    if (!options.image_path.empty()) {
        RgbaImage image;
        std::string error;
        if (!load_rgba_image(options.image_path, image, error)) {
            LOGE("moge-smoke: %s", error.c_str());
            return 1;
        }
        width = image.width;
        height = image.height;
        rgb.resize(static_cast<size_t>(width) * height * 3);
        for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel) {
            rgb[pixel * 3 + 0] = static_cast<float>(image.rgba[pixel * 4 + 0]) / 255.0f;
            rgb[pixel * 3 + 1] = static_cast<float>(image.rgba[pixel * 4 + 1]) / 255.0f;
            rgb[pixel * 3 + 2] = static_cast<float>(image.rgba[pixel * 4 + 2]) / 255.0f;
        }
    }
    if (width < 14 || height < 14) {
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
    ggml_tensor* input = context.input_f32("moge_rgb", {3, width, height});
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

    if (rgb.empty()) {
        rgb.resize(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
                rgb[offset] = static_cast<float>(x) / std::max(1, width - 1);
                rgb[offset + 1] = static_cast<float>(y) / std::max(1, height - 1);
                rgb[offset + 2] = 0.25f;
            }
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
                                 {width, height, 3, 1}, points.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".mask_logits.samt",
                                 {width, height, 1, 1}, mask.data())) {
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
    if (cmd == "gaussian-render") {
        GaussianRenderOpts options;
        options.output_dir = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--ply") && i + 1 < argc) options.ply = argv[++i];
            else if (!std::strcmp(argv[i], "--out-dir") && i + 1 < argc) options.output_dir = argv[++i];
            else if (!std::strcmp(argv[i], "--views") && i + 1 < argc) options.views = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--resolution") && i + 1 < argc) {
                options.resolution = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--extrinsics") && i + 1 < argc) {
                options.extrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--intrinsics") && i + 1 < argc) {
                options.intrinsics = argv[++i];
            }
        }
        return cmd_gaussian_render(options);
    }
    if (cmd == "texture-inpaint") {
        TextureInpaintOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--texture") && i + 1 < argc) options.texture = argv[++i];
            else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) options.mask = argv[++i];
            else if (!std::strcmp(argv[i], "--radius") && i + 1 < argc) {
                options.radius = std::strtof(argv[++i], nullptr);
            }
        }
        return cmd_texture_inpaint(options);
    }
    if (cmd == "mesh-decode") {
        MeshDecodeOpts options;
        options.model = model;
        options.input = input;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--coords") && i + 1 < argc) options.coordinates = argv[++i];
            else if (!std::strcmp(argv[i], "--coords-out") && i + 1 < argc) {
                options.output_coordinates = argv[++i];
            } else if (!std::strcmp(argv[i], "--stage") && i + 1 < argc) {
                options.stage = argv[++i];
            } else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
                options.backend = argv[++i];
            } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
                options.threads = std::atoi(argv[++i]);
            }
        }
        return cmd_mesh_decode(options);
    }
    if (cmd == "mesh-extract") {
        MeshExtractOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--features") && i + 1 < argc) options.features = argv[++i];
            else if (!std::strcmp(argv[i], "--coords") && i + 1 < argc) options.coordinates = argv[++i];
            else if (!std::strcmp(argv[i], "--resolution") && i + 1 < argc) {
                options.resolution = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--vertices-out") && i + 1 < argc) {
                options.vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--faces-out") && i + 1 < argc) {
                options.faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--attrs-out") && i + 1 < argc) {
                options.attributes_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--xatlas-uv")) {
                options.xatlas_uv = true;
            } else if (!std::strcmp(argv[i], "--uv-out") && i + 1 < argc) {
                options.uv_output = argv[++i];
            }
        }
        return cmd_mesh_extract(options);
    }
    if (cmd == "mesh-postprocess") {
        MeshPostprocessOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--vertices-out") && i + 1 < argc) {
                options.vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--faces-out") && i + 1 < argc) {
                options.faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--target-reduction") && i + 1 < argc) {
                options.target_reduction = std::strtof(argv[++i], nullptr);
            } else if (!std::strcmp(argv[i], "--visibility") && i + 1 < argc) {
                options.visibility = argv[++i];
            }
        }
        return cmd_mesh_postprocess(options);
    }
    if (cmd == "mesh-filter-visibility") {
        MeshVisibilityFilterOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--visibility") && i + 1 < argc) {
                options.visibility = argv[++i];
            } else if (!std::strcmp(argv[i], "--vertices-out") && i + 1 < argc) {
                options.vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--faces-out") && i + 1 < argc) {
                options.faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--candidates-out") && i + 1 < argc) {
                options.candidates_output = argv[++i];
            }
        }
        return cmd_mesh_filter_visibility(options);
    }
    if (cmd == "mesh-repair-boundaries") {
        MeshBoundaryRepairOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--vertices-out") && i + 1 < argc) {
                options.vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--faces-out") && i + 1 < argc) {
                options.faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--max-boundary-edges") && i + 1 < argc) {
                options.max_boundary_edges = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--no-refine")) {
                options.refine = false;
            }
        }
        return cmd_mesh_repair_boundaries(options);
    }
    if (cmd == "mesh-parameterize") {
        MeshParameterizeOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--vertices-out") && i + 1 < argc) {
                options.vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--faces-out") && i + 1 < argc) {
                options.faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--uv-out") && i + 1 < argc) {
                options.uv_output = argv[++i];
            }
        }
        return cmd_mesh_parameterize(options);
    }
    if (cmd == "moge-smoke") {
        MogeSmokeOpts options;
        options.model = model;
        options.output_prefix = out;
        options.image_path = input;
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
    if (cmd == "preprocess-conditions") {
        PreprocessConditionsOpts options;
        options.image = input;
        options.pointmap = model;
        options.output_dir = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--image") && i + 1 < argc) options.image = argv[++i];
            else if (!std::strcmp(argv[i], "--pointmap") && i + 1 < argc) options.pointmap = argv[++i];
            else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) options.mask = argv[++i];
            else if (!std::strcmp(argv[i], "--out-dir") && i + 1 < argc) options.output_dir = argv[++i];
            else if (!std::strcmp(argv[i], "--decoded-rgb-out") && i + 1 < argc) {
                options.decoded_rgb_output = argv[++i];
            }
        }
        return cmd_preprocess_conditions(options);
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
