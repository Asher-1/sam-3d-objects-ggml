// sam3d-cli: command line adapter for the SAM 3D ggml runtime.
#include <cstring>
#include <cstdio>
#include <string>

#include "sam3dggml.h"
#include "asset_io.hpp"
#include "common.hpp"
#include "image_preprocess.hpp"
#include "gguf_loader.hpp"
#include "e2e_options.hpp"
#include "backend.hpp"
#include "ss_decoder_graph.hpp"
#include "dino_graph.hpp"
#include "moge_graph.hpp"
#include "moge_inference.hpp"
#include "pose_decoder.hpp"
#include "pointpatch_graph.hpp"
#include "ss_flow_graph.hpp"
#include "slat_flow_graph.hpp"
#include "gs_decoder_graph.hpp"
#include "flexicubes.hpp"
#include "mesh_postprocess.hpp"
#include "mesh_rasterizer.hpp"
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
#include "mesh_rasterizer_cuda.hpp"
#endif
#include "mesh_uv.hpp"
#include "mesh_decoder_graph.hpp"
#include "mesh_decoder_runner.hpp"
#include "pytorch_philox_rng.hpp"
#if defined(SAM3D_USE_CUDA)
#include "pytorch_cuda_rng.hpp"
#endif
#if defined(SAM3D_NATIVE_PBR_CUDA)
#include "gaussian_renderer.hpp"
#include "texture_inpaint.hpp"
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
#include "texture_baker.hpp"
#include "native_pbr_pipeline.hpp"
#endif
#endif
#include "sparse_ops.hpp"
#include "graph_builder.hpp"
#include "scene_assemble.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace sam3d;

static void print_usage() {
    fprintf(stderr,
            "usage: sam3d-cli <command> [options]\n"
            "\n"
            "commands:\n"
            "  info    --model <path.gguf>            print GGUF metadata and tensors\n"
            "  tensor-dump --model <model.gguf> --tensor <name> --out <tensor.samt>\n"
            "              [--tensor-dtype f32|f16] [--backend cpu|cuda|vulkan]\n"
            "  decode-ss --model <ss_decoder.gguf> --input <latent.bin> [--out out.bin]\n"
            "            [--backend auto|cpu|cuda|vulkan] [--threads N]\n"
            "            [--warmup N] [--iters N] [--json out.jsonl]\n"
            "  gs-decode --model <slat_decoder_gs.gguf> <e2e_dir> <out_dir>\n"
            "            decode frozen slat_feats_final.samt and slat_coords.samt\n"
            "  e2e|run --model <models_dir> <condition_dir> --out <out.ply> [--pbr-out <asset.glb>]\n"
            "            [--pose-out <pose.json> --dtype-contract-out <contract.json>]\n"
            "            [--backend cpu|cuda|vulkan] [--seed N] [--threads N]\n"
            "  image-to-3d --image <image> --out <out.ply> [--mask <mask.png>]\n"
            "              [--model <models_dir> --moge-model <moge.gguf> --pbr-out <asset.glb>]\n"
            "              [--mesh-vertices-out <vertices.samt> --mesh-faces-out <faces.samt>]\n"
            "              [--pose-out <pose.json> --dtype-contract-out <contract.json>]\n"
            "              [--conditions-out <directory> --noise-dir <official-stage-dir>]\n"
            "              [--backend cpu|cuda|vulkan]\n"
            "              [--dtype f16|q8_0|q4_0 --ss-attention normal|strict --seed N --threads N]\n"
            "  pose-decode --rotation-6d <tensor.samt> --log-scale <tensor.samt>\n"
            "              --translation <tensor.samt> --log-translation-scale <tensor.samt>\n"
            "              --scene-scale <tensor.samt> --scene-shift <tensor.samt> --out <pose.json>\n"
            "  rng-dump --seed N --sizes N,N,... --out-dir <directory>\n"
            "           [--implementation auto|cuda|portable] [--distribution-blocks N]\n"
            "           [--randperm-size N]\n"
            "  coords-downsample --input <coords.samt> --out <coords.samt> --seed N\n"
            "           --distribution-blocks N --normal-draws N,N,...\n"
            "           [--max-coords N --downsample-factor N]\n"
#if defined(SAM3D_NATIVE_PBR_CUDA)
            "  scene-assemble --objects-list <file> --out-dir <dir>\n"
            "             [--num-frames 300] [--radius 1.0] [--fov 60]\n"
            "             [--resolution 512] [--ply-out scene_posed.ply]\n"
            "             [--frames-dir frames] [--manifest-out scene_manifest.json]\n"
#endif
            "  moge-smoke --model <moge.gguf> [--backend cpu|cuda|vulkan]\n"
            "             [--input image.png] [--width N] [--height N] [--threads N]\n"
            "  moge-infer --model <moge.gguf> --input <image> --out <prefix>\n"
            "             [--backend cpu|cuda|vulkan] [--num-tokens N] [--threads N]\n"
            "             [--force-projection] [--no-apply-mask] [--dump-intermediates] [--dump-blocks]\n"
            "  preprocess-conditions --image <image> --pointmap <pointmap.samt> --out-dir <dir>\n"
            "             [--mask <binary-mask.png>] [--decoded-rgb-out <image.samt>]\n"
            "  mesh-export --vertices <vertices.samt> --faces <faces.samt>\n"
            "              [--attrs <vertex_attrs.samt> | --uv <uv.samt> --texture <base_color.png>]\n"
            "              --out <asset.glb>\n"
            "  mesh-decode --model <slat_decoder_mesh.gguf> --input <slat_feats.samt>\n"
            "              --coords <slat_coords.samt> --out <cube_features.samt>\n"
            "              [--coords-out <subdivided_coords.samt>] [--stage input_layer|blockN|upsampleN]\n"
            "              [--backend cpu|cuda|vulkan] [--portable-attention]\n"
            "  mesh-extract --features <cube_features.samt> --coords <cube_coords.samt>\n"
            "               --out <asset.glb> [--resolution N] [--vertices-out <vertices.samt>]\n"
            "               [--faces-out <faces.samt>] [--attrs-out <vertex_attrs.samt>]\n"
            "               [--xatlas-uv --uv-out <uv.samt>]\n"
            "  mesh-postprocess --vertices <vertices.samt> --faces <faces.samt>\n"
            "                   --vertices-out <vertices.samt> --faces-out <faces.samt>\n"
            "                   [--target-reduction 0.95 --visibility <frequency.samt>]\n"
            "                   [--native-visibility --visibility-views 1000]\n"
            "                   [--visibility-resolution 1024]\n"
            "  mesh-visibility --vertices <vertices.samt> --faces <faces.samt> --out <frequency.samt>\n"
            "                  [--views 1000] [--resolution 1024]\n"
            "                  [--extrinsics <cameras.samt> --intrinsics <cameras.samt>]\n"
            "                  [--view <cameras.samt> --projection <cameras.samt>]\n"
            "  mesh-camera-dump --extrinsics-out <extrinsics.samt> --intrinsics-out <intrinsics.samt>\n"
            "                   [--views-out <views.samt> --projections-out <projections.samt>]\n"
            "                   [--views 1000]\n"
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
            "  texture-bake --vertices <vertices.samt> --faces <faces.samt> --uv <uv.samt>\n"
            "               --observations-dir <views_dir> --extrinsics <cameras.samt>\n"
            "               --intrinsics <cameras.samt> --out <base_color.png>\n"
            "               [--texture-size 1024 --steps 2500 --seed 0 --holes-out <holes.png>]\n"
            "               [--no-inpaint --raw-texture-out <texture.samt>]\n"
            "  texture-raster --vertices <vertices.samt> --faces <faces.samt> --uv <uv.samt>\n"
            "                 --extrinsics <cameras.samt> --intrinsics <cameras.samt>\n"
            "                 --view-index N --uv-out <uv.samt> --uv-dr-out <uv_dr.samt>\n"
            "                 --coverage-out <coverage.samt> [--resolution 1024]\n"
            "  pbr-assemble --vertices <vertices.samt> --faces <faces.samt> --ply <output_gs.ply>\n"
            "               --out <asset.glb> [--already-clean] [--views 100 --resolution 1024]\n"
            "               [--texture-size 1024 --steps 2500 --seed 0]\n"
            "\n"
            "Use scripts/run_image_to_3d.py --native-image-input for raw-image E2E.\n");
}

static int cmd_rng_dump(int argc, char** argv) {
    uint64_t seed = 42;
    std::string sizes_text;
    std::string output_dir;
    std::string implementation = "auto";
    uint32_t distribution_blocks = 0;
    size_t randperm_size = 0;
    for (int index = 0; index < argc; ++index) {
        const char* argument = argv[index];
        if (!strcmp(argument, "--seed") && index + 1 < argc) {
            seed = strtoull(argv[++index], nullptr, 10);
        } else if (!strcmp(argument, "--sizes") && index + 1 < argc) {
            sizes_text = argv[++index];
        } else if (!strcmp(argument, "--out-dir") && index + 1 < argc) {
            output_dir = argv[++index];
        } else if (!strcmp(argument, "--implementation") && index + 1 < argc) {
            implementation = argv[++index];
        } else if (!strcmp(argument, "--distribution-blocks") && index + 1 < argc) {
            char* end = nullptr;
            const unsigned long parsed = strtoul(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<uint32_t>::max()) {
                fprintf(stderr, "rng-dump: --distribution-blocks must be a positive uint32\n");
                return 2;
            }
            distribution_blocks = static_cast<uint32_t>(parsed);
        } else if (!strcmp(argument, "--randperm-size") && index + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "rng-dump: --randperm-size must be a positive int32\n");
                return 2;
            }
            randperm_size = static_cast<size_t>(parsed);
        } else {
            fprintf(stderr, "rng-dump: unknown or incomplete argument: %s\n", argument);
            return 2;
        }
    }
    if (sizes_text.empty() || output_dir.empty()) {
        fprintf(stderr, "rng-dump requires --sizes and --out-dir\n");
        return 2;
    }
    if (implementation != "auto" && implementation != "cuda" && implementation != "portable") {
        fprintf(stderr, "rng-dump: --implementation must be auto, cuda, or portable\n");
        return 2;
    }
#if !defined(SAM3D_USE_CUDA)
    if (implementation == "cuda") {
        fprintf(stderr, "rng-dump: --implementation cuda requires a CUDA build\n");
        return 2;
    }
    if (implementation == "auto") implementation = "portable";
#else
    if (implementation == "auto") implementation = "cuda";
#endif
    uint32_t resolved_distribution_blocks = distribution_blocks;
#if defined(SAM3D_USE_CUDA)
    if (implementation == "cuda" && resolved_distribution_blocks == 0) {
        std::string error;
        if (!PytorchCudaNormalRng::distribution_blocks(resolved_distribution_blocks, error)) {
            fprintf(stderr, "rng-dump: %s\n", error.c_str());
            return 1;
        }
    }
#endif
    if (implementation == "portable" && resolved_distribution_blocks == 0) {
        fprintf(stderr, "rng-dump: portable Philox requires --distribution-blocks from the PyTorch CUDA reference\n");
        return 2;
    }
    std::vector<size_t> sizes;
    size_t start = 0;
    while (start < sizes_text.size()) {
        const size_t end = sizes_text.find(',', start);
        const std::string token = sizes_text.substr(start, end == std::string::npos ? end : end - start);
        char* parse_end = nullptr;
        const unsigned long long value = strtoull(token.c_str(), &parse_end, 10);
        if (token.empty() || parse_end == token.c_str() || *parse_end != '\0' || value == 0 ||
            value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "rng-dump: invalid size '%s'\n", token.c_str());
            return 2;
        }
        sizes.push_back(static_cast<size_t>(value));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
        fprintf(stderr, "rng-dump: cannot create %s: %s\n", output_dir.c_str(),
                directory_error.message().c_str());
        return 1;
    }
#if defined(SAM3D_USE_CUDA)
    std::unique_ptr<PytorchCudaNormalRng> cuda_rng;
    if (implementation == "cuda") {
        cuda_rng = std::make_unique<PytorchCudaNormalRng>(seed, resolved_distribution_blocks);
    }
#endif
    std::unique_ptr<PytorchPhiloxNormalRng> portable_rng;
    if (implementation == "portable") {
        portable_rng = std::make_unique<PytorchPhiloxNormalRng>(seed, resolved_distribution_blocks);
    }
    std::unique_ptr<PytorchPhiloxNormalRng> randperm_rng;
    if (randperm_size != 0) {
        randperm_rng = std::make_unique<PytorchPhiloxNormalRng>(seed, resolved_distribution_blocks);
    }
    for (size_t index = 0; index < sizes.size(); ++index) {
        std::vector<float> values(sizes[index]);
        std::string error;
#if defined(SAM3D_USE_CUDA)
        const bool filled = cuda_rng ? cuda_rng->fill(values, error) :
            portable_rng->fill(values, error);
#else
        const bool filled = portable_rng->fill(values, error);
#endif
        if (!filled) {
            fprintf(stderr, "rng-dump: %s\n", error.c_str());
            return 1;
        }
        if (randperm_rng && !randperm_rng->advance_normal(values.size(), error)) {
            fprintf(stderr, "rng-dump: cannot advance randperm Philox state: %s\n", error.c_str());
            return 1;
        }
        char filename[32];
        snprintf(filename, sizeof(filename), "rng_%02zu.samt", index);
        if (!save_raw_tensor_f32(output_dir + "/" + filename,
                                 {static_cast<int64_t>(values.size())}, values.data())) {
            fprintf(stderr, "rng-dump: cannot write %s\n", filename);
            return 1;
        }
    }
    if (randperm_rng) {
        std::vector<uint32_t> permutation;
        std::string error;
        if (!randperm_rng->randperm(randperm_size, permutation, error)) {
            fprintf(stderr, "rng-dump: PyTorch-compatible randperm failed: %s\n", error.c_str());
            return 1;
        }
        std::vector<int32_t> permutation_i32(permutation.begin(), permutation.end());
        if (!save_raw_tensor_i32(output_dir + "/randperm.samt",
                                 {static_cast<int64_t>(permutation_i32.size())},
                                 permutation_i32.data())) {
            fprintf(stderr, "rng-dump: cannot write randperm.samt\n");
            return 1;
        }
    }
    std::ofstream contract(output_dir + "/rng_contract.json", std::ios::trunc);
    if (!contract) {
        fprintf(stderr, "rng-dump: cannot write rng_contract.json\n");
        return 1;
    }
    contract << "{\n"
             << "  \"schema\": \"sam3d.pytorch-philox-contract.v1\",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"implementation\": \"" << implementation << "\",\n"
             << "  \"distribution_blocks\": " << resolved_distribution_blocks << ",\n"
             << "  \"randperm_size\": " << randperm_size << "\n"
             << "}\n";
    return 0;
}

static int cmd_coords_downsample(int argc, char** argv) {
    std::string input;
    std::string output;
    std::string normal_draws;
    uint64_t seed = 42;
    uint32_t distribution_blocks = 0;
    int64_t max_coordinates = 42000;
    int downsample_factor = 2;
    for (int index = 0; index < argc; ++index) {
        const char* argument = argv[index];
        if (!strcmp(argument, "--input") && index + 1 < argc) {
            input = argv[++index];
        } else if (!strcmp(argument, "--out") && index + 1 < argc) {
            output = argv[++index];
        } else if (!strcmp(argument, "--seed") && index + 1 < argc) {
            seed = strtoull(argv[++index], nullptr, 10);
        } else if (!strcmp(argument, "--distribution-blocks") && index + 1 < argc) {
            char* end = nullptr;
            const unsigned long parsed = strtoul(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<uint32_t>::max()) {
                fprintf(stderr, "coords-downsample: --distribution-blocks must be a positive uint32\n");
                return 2;
            }
            distribution_blocks = static_cast<uint32_t>(parsed);
        } else if (!strcmp(argument, "--normal-draws") && index + 1 < argc) {
            normal_draws = argv[++index];
        } else if (!strcmp(argument, "--max-coords") && index + 1 < argc) {
            max_coordinates = strtoll(argv[++index], nullptr, 10);
        } else if (!strcmp(argument, "--downsample-factor") && index + 1 < argc) {
            downsample_factor = atoi(argv[++index]);
        } else {
            fprintf(stderr, "coords-downsample: unknown or incomplete argument: %s\n", argument);
            return 2;
        }
    }
    if (input.empty() || output.empty() || normal_draws.empty() || distribution_blocks == 0 ||
        max_coordinates <= 0 || downsample_factor <= 0) {
        fprintf(stderr, "coords-downsample requires input, output, positive distribution-blocks, "
                        "normal-draws, max-coords, and downsample-factor\n");
        return 2;
    }
    RawTensor coordinates_tensor;
    if (!load_raw_tensor(input, coordinates_tensor) || coordinates_tensor.type != GGML_TYPE_I32 ||
        coordinates_tensor.ne.size() != 2 || coordinates_tensor.ne[0] != 4 ||
        coordinates_tensor.ne[1] <= 0 ||
        coordinates_tensor.data.size() != static_cast<size_t>(coordinates_tensor.ne[1]) *
            4 * sizeof(int32_t)) {
        fprintf(stderr, "coords-downsample: input must be a non-empty I32 SAMT tensor shaped [4, N]\n");
        return 2;
    }
    PytorchPhiloxNormalRng rng(seed, distribution_blocks);
    size_t start = 0;
    while (start < normal_draws.size()) {
        const size_t end = normal_draws.find(',', start);
        const std::string token = normal_draws.substr(start, end == std::string::npos ? end : end - start);
        char* parse_end = nullptr;
        const unsigned long long count = strtoull(token.c_str(), &parse_end, 10);
        if (token.empty() || parse_end == token.c_str() || *parse_end != '\0' || count == 0 ||
            count > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "coords-downsample: invalid normal draw size '%s'\n", token.c_str());
            return 2;
        }
        std::string error;
        if (!rng.advance_normal(static_cast<size_t>(count), error)) {
            fprintf(stderr, "coords-downsample: cannot advance Philox state: %s\n", error.c_str());
            return 1;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    const auto* source = reinterpret_cast<const int32_t*>(coordinates_tensor.data.data());
    std::vector<int32_t> coordinates(source, source + coordinates_tensor.data.size() / sizeof(int32_t));
    std::vector<int32_t> downsampled;
    std::string error;
    bool randomly_subsampled = false;
    downsampled = downsample_sparse_coords_pytorch(coordinates, rng, error,
                                                    randomly_subsampled, max_coordinates,
                                                    downsample_factor);
    if (downsampled.empty()) {
        fprintf(stderr, "coords-downsample: %s\n", error.empty() ? "empty output" : error.c_str());
        return 1;
    }
    if (!save_raw_tensor_i32(output, {4, static_cast<int64_t>(downsampled.size() / 4)},
                             downsampled.data())) {
        fprintf(stderr, "coords-downsample: cannot write %s\n", output.c_str());
        return 1;
    }
    return 0;
}

struct MeshExportOpts {
    std::string vertices;
    std::string faces;
    std::string attributes;
    std::string texcoords;
    std::string texture;
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
    if (!options.attributes.empty() && (!options.texcoords.empty() || !options.texture.empty())) {
        LOGE("mesh-export: --attrs is the non-baked vertex-color path and cannot be combined "
             "with --uv or --texture");
        return 1;
    }
    if (options.texcoords.empty() != options.texture.empty()) {
        LOGE("mesh-export: the baked PBR path requires both --uv and --texture");
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
    std::string error;
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

    if (!options.texcoords.empty()) {
        RawTensor texcoords_tensor;
        std::vector<float> texcoords;
        if (!tensor_values_f32(options.texcoords, texcoords_tensor, texcoords, "UV coordinates")) {
            return 1;
        }
        if (texcoords_tensor.ne.size() != 2 || texcoords_tensor.ne[0] != 2 ||
            texcoords.size() != vertex_count * 2) {
            LOGE("mesh-export: UV coordinates must have official SAMT shape [2, vertex_count]");
            return 1;
        }
        mesh.texcoords = std::move(texcoords);
        if (!load_rgba_image(options.texture, mesh.material.base_color_texture, error)) {
            LOGE("mesh-export: %s", error.c_str());
            return 1;
        }
    }

    if (!write_pbr_glb(options.output, mesh, error)) {
        LOGE("mesh-export: %s", error.c_str());
        return 1;
    }
    const char* material = mesh.material.base_color_texture.rgba.empty()
                               ? (mesh.colors.empty() ? "" : ", COLOR_0")
                               : ", TEXCOORD_0, baseColorTexture";
    LOGI("mesh-export: wrote %s (%zu vertices, %zu triangles%s)", options.output.c_str(),
         vertex_count, mesh.indices.size() / 3, material);
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
    bool native_visibility = false;
    int visibility_views = 1000;
    int visibility_resolution = 1024;
};

struct MeshVisibilityFilterOpts {
    std::string vertices;
    std::string faces;
    std::string visibility;
    std::string vertices_output;
    std::string faces_output;
    std::string candidates_output;
};

struct MeshVisibilityOpts {
    std::string vertices;
    std::string faces;
    std::string output;
    std::string extrinsics;
    std::string intrinsics;
    std::string view;
    std::string projection;
    int view_count = 1000;
    int resolution = 1024;
};

struct MeshCameraDumpOpts {
    std::string extrinsics_output;
    std::string intrinsics_output;
    std::string views_output;
    std::string projections_output;
    int view_count = 1000;
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
    if (options.native_visibility && !options.visibility.empty()) {
        LOGE("mesh-postprocess accepts either --visibility or --native-visibility, not both");
        return 1;
    }
    MeshVisibilityFilterStats visibility_stats;
    if (!options.visibility.empty() || options.native_visibility) {
        RawTensor visibility_tensor;
        std::vector<float> visibility;
        if (options.native_visibility) {
            MeshRasterConfig raster;
            raster.width = options.visibility_resolution;
            raster.height = options.visibility_resolution;
            if (!mesh_visibility_frequency(mesh, options.visibility_views, raster, visibility, error)) {
                LOGE("mesh-postprocess native visibility: %s", error.c_str());
                return 1;
            }
        } else {
            if (!tensor_values_f32(options.visibility, visibility_tensor, visibility,
                                   "mesh-postprocess visibility") ||
                visibility_tensor.ne.size() != 1 ||
                visibility.size() != mesh.indices.size() / 3) {
                LOGE("mesh-postprocess visibility must be F32 [face_count] after VTK reduction");
                return 1;
            }
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
         "visibility=%s mincut candidates=%zu removed=%zu (MeshFix boundary repair is separate)",
         options.target_reduction, static_cast<long long>(vertex_count), static_cast<long long>(face_count),
         options.native_visibility ? "native" : (options.visibility.empty() ? "disabled" : "external"),
         visibility_stats.mincut_face_count, visibility_stats.removed_face_count);
    return 0;
#endif
}

static int cmd_mesh_visibility(const MeshVisibilityOpts& options) {
    if (options.vertices.empty() || options.faces.empty() || options.output.empty()) {
        LOGE("mesh-visibility requires --vertices, --faces and --out");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "mesh-visibility vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "mesh-visibility faces") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("mesh-visibility expects vertices [3,V] and faces [3,F] SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    MeshRasterConfig config;
    config.width = options.resolution;
    config.height = options.resolution;
    std::vector<float> visibility;
    std::string error;
    const bool has_opencv_cameras = !options.extrinsics.empty() || !options.intrinsics.empty();
    const bool has_opengl_cameras = !options.view.empty() || !options.projection.empty();
    const bool has_external_cameras = has_opencv_cameras || has_opengl_cameras;
    int effective_view_count = options.view_count;
    bool success = false;
    if (has_external_cameras) {
        if (has_opencv_cameras && has_opengl_cameras) {
            LOGE("mesh-visibility accepts one external camera convention at a time");
            return 1;
        }
        RawTensor first_tensor;
        RawTensor second_tensor;
        std::vector<float> first;
        std::vector<float> second;
        const std::string& first_path = has_opengl_cameras ? options.view : options.extrinsics;
        const std::string& second_path = has_opengl_cameras ? options.projection : options.intrinsics;
        if (first_path.empty() || second_path.empty() ||
            !tensor_values_f32(first_path, first_tensor, first, "mesh-visibility camera matrix") ||
            !tensor_values_f32(second_path, second_tensor, second, "mesh-visibility camera matrix") ||
            first_tensor.ne.size() != 3 || first_tensor.ne[0] != 4 || first_tensor.ne[1] != 4 ||
            second_tensor.ne.size() != 3 || second_tensor.ne[2] != first_tensor.ne[2] ||
            first.size() != static_cast<size_t>(first_tensor.ne[2]) * 16) {
            LOGE("mesh-visibility camera tensors must share F32 [V,4,4] matrices");
            return 1;
        }
        if (has_opencv_cameras && (second_tensor.ne[0] != 3 || second_tensor.ne[1] != 3 ||
                                  second.size() != static_cast<size_t>(second_tensor.ne[2]) * 9)) {
            LOGE("mesh-visibility intrinsics must be F32 [V,3,3]");
            return 1;
        }
        if (has_opengl_cameras && (second_tensor.ne[0] != 4 || second_tensor.ne[1] != 4 ||
                                  second.size() != static_cast<size_t>(second_tensor.ne[2]) * 16)) {
            LOGE("mesh-visibility projections must be F32 [V,4,4]");
            return 1;
        }
        std::vector<MeshRasterCamera> cameras(static_cast<size_t>(first_tensor.ne[2]));
        for (size_t index = 0; index < cameras.size(); ++index) {
            if (has_opengl_cameras) {
                std::copy_n(first.data() + index * 16, 16, cameras[index].view.begin());
                std::copy_n(second.data() + index * 16, 16, cameras[index].projection.begin());
                cameras[index].has_view_projection = true;
            } else {
                std::copy_n(first.data() + index * 16, 16, cameras[index].extrinsics.begin());
                std::copy_n(second.data() + index * 9, 9, cameras[index].intrinsics.begin());
            }
        }
        effective_view_count = static_cast<int>(cameras.size());
        success = mesh_visibility_frequency(mesh, cameras, config, visibility, error);
    } else {
        success = mesh_visibility_frequency(mesh, options.view_count, config, visibility, error);
    }
    if (!success) {
        LOGE("mesh-visibility: %s", error.c_str());
        return 1;
    }
    if (!save_raw_tensor_f32(options.output,
                             {static_cast<int64_t>(visibility.size())}, visibility.data())) {
        LOGE("mesh-visibility: failed to write %s", options.output.c_str());
        return 1;
    }
    LOGI("mesh-visibility: wrote %s (%zu faces, %d views, %dx%d)",
         options.output.c_str(), visibility.size(), effective_view_count,
         config.width, config.height);
    return 0;
}

static int cmd_mesh_camera_dump(const MeshCameraDumpOpts& options) {
    if (options.extrinsics_output.empty() || options.intrinsics_output.empty()) {
        LOGE("mesh-camera-dump requires --extrinsics-out and --intrinsics-out");
        return 1;
    }
    if (options.views_output.empty() != options.projections_output.empty()) {
        LOGE("mesh-camera-dump requires --views-out and --projections-out together");
        return 1;
    }
    MeshRasterConfig config;
    std::string error;
    const std::vector<MeshRasterCamera> cameras =
        make_mesh_hammersley_cameras(options.view_count, config, error);
    if (cameras.empty()) {
        LOGE("mesh-camera-dump: %s", error.c_str());
        return 1;
    }
    std::vector<float> extrinsics;
    std::vector<float> intrinsics;
    extrinsics.reserve(cameras.size() * 16);
    intrinsics.reserve(cameras.size() * 9);
    for (const MeshRasterCamera& camera : cameras) {
        extrinsics.insert(extrinsics.end(), camera.extrinsics.begin(), camera.extrinsics.end());
        intrinsics.insert(intrinsics.end(), camera.intrinsics.begin(), camera.intrinsics.end());
    }
    if (!save_raw_tensor_f32(options.extrinsics_output,
                             {4, 4, static_cast<int64_t>(cameras.size())}, extrinsics.data()) ||
        !save_raw_tensor_f32(options.intrinsics_output,
                             {3, 3, static_cast<int64_t>(cameras.size())}, intrinsics.data())) {
        LOGE("mesh-camera-dump: failed to write camera tensors");
        return 1;
    }
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    if (!options.views_output.empty()) {
        std::vector<float> views;
        std::vector<float> projections;
        if (!mesh_hammersley_camera_matrices_nvdiffrast_cuda(
                options.view_count, views, projections, error) ||
            !save_raw_tensor_f32(options.views_output,
                                 {4, 4, static_cast<int64_t>(options.view_count)}, views.data()) ||
            !save_raw_tensor_f32(options.projections_output,
                                 {4, 4, static_cast<int64_t>(options.view_count)}, projections.data())) {
            LOGE("mesh-camera-dump: failed to write CUDA visibility matrices: %s", error.c_str());
            return 1;
        }
    }
#elif !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    if (!options.views_output.empty()) {
        LOGE("mesh-camera-dump --views-out requires a CUDA nvdiffrast-native-PBR build");
        return 1;
    }
#endif
    LOGI("mesh-camera-dump: wrote %zu Hammersley cameras", cameras.size());
    return 0;
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

struct TextureBakeOpts {
    std::string vertices;
    std::string faces;
    std::string uv;
    std::string observations_dir;
    std::string extrinsics;
    std::string intrinsics;
    std::string output;
    std::string holes_output;
    std::string raw_texture_output;
    int texture_size = 1024;
    int steps = 2500;
    unsigned seed = 0;
    bool apply_telea = true;
};

struct TextureRasterOpts {
    std::string vertices;
    std::string faces;
    std::string uv;
    std::string extrinsics;
    std::string intrinsics;
    std::string uv_output;
    std::string uv_derivatives_output;
    std::string coverage_output;
    int view_index = 0;
    int resolution = 1024;
};

struct PbrAssembleOpts {
    std::string vertices;
    std::string faces;
    std::string ply;
    std::string output;
    int render_views = 100;
    int resolution = 1024;
    int texture_size = 1024;
    int texture_steps = 2500;
    unsigned seed = 0;
    bool already_clean = false;
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

static int cmd_texture_bake(const TextureBakeOpts& options) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    (void)options;
    LOGE("texture-bake requires CUDA native PBR with "
         "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON; it never falls back to PyTorch");
    return 1;
#else
    if (options.vertices.empty() || options.faces.empty() || options.uv.empty() ||
        options.observations_dir.empty() || options.extrinsics.empty() || options.intrinsics.empty() ||
        options.output.empty() || options.texture_size <= 0 || options.steps <= 0) {
        LOGE("texture-bake requires --vertices, --faces, --uv, --observations-dir, --extrinsics, "
             "--intrinsics, --out, positive --texture-size and positive --steps");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    RawTensor uv_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    std::vector<float> uv;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "texture-bake vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "texture-bake faces") ||
        !tensor_values_f32(options.uv, uv_tensor, uv, "texture-bake UVs") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        uv_tensor.ne.size() != 2 || uv_tensor.ne[0] != 2 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3 ||
        uv.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 2) {
        LOGE("texture-bake expects vertices [3,V], faces [3,F], and xatlas UVs [2,V] SAMT tensors");
        return 1;
    }
    RawTensor extrinsics_tensor;
    RawTensor intrinsics_tensor;
    if (!load_raw_tensor(options.extrinsics, extrinsics_tensor) ||
        !load_raw_tensor(options.intrinsics, intrinsics_tensor) ||
        extrinsics_tensor.type != GGML_TYPE_F32 || intrinsics_tensor.type != GGML_TYPE_F32 ||
        extrinsics_tensor.ne.size() != 3 || intrinsics_tensor.ne.size() != 3 ||
        extrinsics_tensor.ne[0] != 4 || extrinsics_tensor.ne[1] != 4 ||
        intrinsics_tensor.ne[0] != 3 || intrinsics_tensor.ne[1] != 3 ||
        extrinsics_tensor.ne[2] <= 0 || intrinsics_tensor.ne[2] != extrinsics_tensor.ne[2] ||
        extrinsics_tensor.data.size() != static_cast<size_t>(extrinsics_tensor.ne[2]) * 16 * sizeof(float) ||
        intrinsics_tensor.data.size() != static_cast<size_t>(intrinsics_tensor.ne[2]) * 9 * sizeof(float)) {
        LOGE("texture-bake camera SAMT tensors must be F32 [camera_count,4,4] and [camera_count,3,3]");
        return 1;
    }
    const size_t view_count = static_cast<size_t>(extrinsics_tensor.ne[2]);
    std::vector<GaussianCamera> cameras(view_count);
    for (size_t index = 0; index < view_count; ++index) {
        std::memcpy(cameras[index].extrinsics.data(),
                    extrinsics_tensor.data.data() + index * 16 * sizeof(float), 16 * sizeof(float));
        std::memcpy(cameras[index].intrinsics.data(),
                    intrinsics_tensor.data.data() + index * 9 * sizeof(float), 9 * sizeof(float));
    }
    std::vector<RgbaImage> observations;
    observations.reserve(view_count);
    std::string error;
    for (size_t index = 0; index < view_count; ++index) {
        char file_name[64];
        std::snprintf(file_name, sizeof(file_name), "view_%03zu.png", index);
        const std::string path = (std::filesystem::path(options.observations_dir) / file_name).string();
        RgbaImage observation;
        if (!load_rgba_image(path, observation, error)) {
            LOGE("texture-bake: failed to load official Gaussian observation %s: %s", path.c_str(),
                 error.c_str());
            return 1;
        }
        observations.push_back(std::move(observation));
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    mesh.texcoords = std::move(uv);
    TextureBakeConfig config;
    config.texture_size = options.texture_size;
    config.steps = options.steps;
    config.random_seed = options.seed;
    config.apply_telea = options.apply_telea;
    RgbaImage texture;
    std::vector<uint8_t> holes;
    std::vector<float> optimized_texture;
    std::vector<float>* optimized_texture_output =
        options.raw_texture_output.empty() ? nullptr : &optimized_texture;
    if (!bake_texture_official_cuda(mesh, observations, cameras, config, texture, &holes,
                                    optimized_texture_output, error)) {
        LOGE("texture-bake: %s", error.c_str());
        return 1;
    }
    if (!write_rgba_png(options.output, texture, error)) {
        LOGE("texture-bake: %s", error.c_str());
        return 1;
    }
    if (!options.raw_texture_output.empty() &&
        !save_raw_tensor_f32(options.raw_texture_output,
                             {3, static_cast<int64_t>(texture.width), static_cast<int64_t>(texture.height)},
                             optimized_texture.data())) {
        LOGE("texture-bake: failed to write F32 optimized texture %s",
             options.raw_texture_output.c_str());
        return 1;
    }
    if (!options.holes_output.empty()) {
        RgbaImage hole_image;
        hole_image.width = texture.width;
        hole_image.height = texture.height;
        hole_image.rgba.resize(holes.size() * 4);
        for (size_t pixel = 0; pixel < holes.size(); ++pixel) {
            const uint8_t value = holes[pixel] == 0 ? 0 : 255;
            hole_image.rgba[pixel * 4] = value;
            hole_image.rgba[pixel * 4 + 1] = value;
            hole_image.rgba[pixel * 4 + 2] = value;
            hole_image.rgba[pixel * 4 + 3] = 255;
        }
        if (!write_rgba_png(options.holes_output, hole_image, error)) {
            LOGE("texture-bake: %s", error.c_str());
            return 1;
        }
    }
    LOGI("texture-bake: wrote %s from %zu native Gaussian views (%dx%d, %d Adam/TV steps)",
         options.output.c_str(), view_count, texture.width, texture.height, config.steps);
    return 0;
#endif
}

static int cmd_texture_raster(const TextureRasterOpts& options) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    (void)options;
    LOGE("texture-raster requires CUDA native PBR with "
         "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON");
    return 1;
#else
    if (options.vertices.empty() || options.faces.empty() || options.uv.empty() ||
        options.extrinsics.empty() || options.intrinsics.empty() || options.uv_output.empty() ||
        options.uv_derivatives_output.empty() || options.coverage_output.empty() ||
        options.view_index < 0 || options.resolution <= 0) {
        LOGE("texture-raster requires mesh, camera, output tensors, non-negative --view-index and "
             "positive --resolution");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    RawTensor uv_tensor;
    RawTensor extrinsics_tensor;
    RawTensor intrinsics_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    std::vector<float> uv;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "texture-raster vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "texture-raster faces") ||
        !tensor_values_f32(options.uv, uv_tensor, uv, "texture-raster UVs") ||
        !load_raw_tensor(options.extrinsics, extrinsics_tensor) ||
        !load_raw_tensor(options.intrinsics, intrinsics_tensor) ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        uv_tensor.ne.size() != 2 || uv_tensor.ne[0] != 2 ||
        uv.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 2 ||
        extrinsics_tensor.type != GGML_TYPE_F32 || intrinsics_tensor.type != GGML_TYPE_F32 ||
        extrinsics_tensor.ne.size() != 3 || intrinsics_tensor.ne.size() != 3 ||
        extrinsics_tensor.ne[0] != 4 || extrinsics_tensor.ne[1] != 4 ||
        intrinsics_tensor.ne[0] != 3 || intrinsics_tensor.ne[1] != 3 ||
        intrinsics_tensor.ne[2] != extrinsics_tensor.ne[2] ||
        options.view_index >= extrinsics_tensor.ne[2]) {
        LOGE("texture-raster expects vertices [3,V], faces [3,F], UVs [2,V], and camera tensors "
             "F32 [camera_count,4,4] / [camera_count,3,3]");
        return 1;
    }
    GaussianCamera camera;
    std::memcpy(camera.extrinsics.data(),
                extrinsics_tensor.data.data() + static_cast<size_t>(options.view_index) * 16 * sizeof(float),
                16 * sizeof(float));
    std::memcpy(camera.intrinsics.data(),
                intrinsics_tensor.data.data() + static_cast<size_t>(options.view_index) * 9 * sizeof(float),
                9 * sizeof(float));
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    mesh.texcoords = std::move(uv);
    TextureBakeConfig config;
    TextureBakeRaster raster;
    std::string error;
    if (!rasterize_texture_bake_view_official_cuda(mesh, camera, config, options.resolution,
                                                   options.resolution, raster, error)) {
        LOGE("texture-raster: %s", error.c_str());
        return 1;
    }
    std::vector<int32_t> coverage(raster.coverage.begin(), raster.coverage.end());
    if (!save_raw_tensor_f32(options.uv_output,
                             {2, static_cast<int64_t>(raster.width), static_cast<int64_t>(raster.height)},
                             raster.uv.data()) ||
        !save_raw_tensor_f32(options.uv_derivatives_output,
                             {4, static_cast<int64_t>(raster.width), static_cast<int64_t>(raster.height)},
                             raster.uv_derivatives.data()) ||
        !save_raw_tensor_i32(options.coverage_output,
                             {static_cast<int64_t>(raster.width), static_cast<int64_t>(raster.height)},
                             coverage.data())) {
        LOGE("texture-raster: failed to write raster SAMT outputs");
        return 1;
    }
    LOGI("texture-raster: wrote view %d native UV, derivative, and coverage tensors", options.view_index);
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

static int cmd_pbr_assemble(const PbrAssembleOpts& options) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    (void)options;
    LOGE("pbr-assemble requires CUDA native PBR with "
         "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON; it never falls back to PyTorch");
    return 1;
#else
    if (options.vertices.empty() || options.faces.empty() || options.ply.empty() ||
        options.output.empty() || options.render_views <= 0 || options.resolution <= 0 ||
        options.texture_size <= 0 || options.texture_steps <= 0) {
        LOGE("pbr-assemble requires --vertices, --faces, --ply, --out and positive stage sizes");
        return 1;
    }
    RawTensor vertices_tensor;
    RawTensor faces_tensor;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    if (!tensor_values_f32(options.vertices, vertices_tensor, vertices, "pbr-assemble vertices") ||
        !tensor_indices(options.faces, faces_tensor, faces, "pbr-assemble faces") ||
        vertices_tensor.ne.size() != 2 || vertices_tensor.ne[0] != 3 ||
        vertices.size() != static_cast<size_t>(vertices_tensor.ne[1]) * 3 ||
        faces_tensor.ne.size() != 2 || faces_tensor.ne[0] != 3 ||
        faces.size() != static_cast<size_t>(faces_tensor.ne[1]) * 3) {
        LOGE("pbr-assemble expects vertices [3,V] F32 and faces [3,F] F32/I32 SAMT tensors");
        return 1;
    }
    NativeMesh mesh;
    mesh.positions = std::move(vertices);
    mesh.indices = std::move(faces);
    GaussianSplatSet splats;
    std::string error;
    if (!load_gaussian_splat_ply(options.ply, splats, error)) {
        LOGE("pbr-assemble: %s", error.c_str());
        return 1;
    }
    NativePbrPipelineConfig config;
    config.input_mesh_already_clean = options.already_clean;
    config.render_views = options.render_views;
    config.render_resolution = options.resolution;
    config.texture_size = options.texture_size;
    config.texture_steps = options.texture_steps;
    config.random_seed = options.seed;
    NativePbrPipelineResult result;
    if (!assemble_official_pbr_cuda(mesh, splats, config, result, error)) {
        LOGE("pbr-assemble: %s", error.c_str());
        return 1;
    }
    if (!write_pbr_glb(options.output, result.mesh, error)) {
        LOGE("pbr-assemble: %s", error.c_str());
        return 1;
    }
    auto texture_path = std::filesystem::path(options.output);
    texture_path.replace_extension(".base_color.png");
    if (!write_rgba_png(texture_path.string(), result.mesh.material.base_color_texture, error)) {
        LOGE("pbr-assemble: failed to write base-color texture: %s", error.c_str());
        return 1;
    }
    LOGI("pbr-assemble: wrote %s (%zu rendered views, %zu pre-MeshFix faces, %zu final triangles)",
         options.output.c_str(), result.rendered_views, result.pre_meshfix_faces, result.final_faces);
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
    bool native_attention = true;
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

    MeshDecoderRunOptions run_options;
    run_options.model_path = options.model;
    run_options.backend = options.backend;
    run_options.threads = options.threads;
    run_options.debug_stage = options.stage;
    run_options.native_attention = options.native_attention;
    MeshDecoderRunResult result;
    std::string error;
    if (!run_mesh_decoder(reinterpret_cast<const float*>(features_tensor.data.data()),
                          reinterpret_cast<const int32_t*>(coords_tensor.data.data()),
                          token_count, run_options, result, error)) {
        LOGE("mesh-decode: %s", error.c_str());
        return 1;
    }
    if (!save_raw_tensor_f32(options.output, {result.channels, result.token_count},
                             result.features.data())) {
        LOGE("mesh-decode: failed to write %s", options.output.c_str());
        return 1;
    }
    if (!options.output_coordinates.empty()) {
        RawTensor output_coords;
        output_coords.ne = {4, result.token_count};
        output_coords.type = GGML_TYPE_I32;
        output_coords.data.resize(result.coordinates.size() * sizeof(int32_t));
        std::memcpy(output_coords.data.data(), result.coordinates.data(), output_coords.data.size());
        if (!save_raw_tensor(options.output_coordinates, output_coords)) {
            LOGE("mesh-decode: failed to write %s", options.output_coordinates.c_str());
            return 1;
        }
    }
    LOGI("mesh-decode: %lld input cells -> [%lld, %lld] %s (%s)",
         static_cast<long long>(token_count), static_cast<long long>(result.channels),
         static_cast<long long>(result.token_count),
         options.stage.empty() ? "raw features" : options.stage.c_str(), result.backend_name.c_str());
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

struct MogeInferOpts {
    std::string model;
    std::string image_path;
    std::string output_prefix;
    std::string backend = "cpu";
    int num_tokens = 2500;
    int threads = 8;
    bool force_projection = false;
    bool apply_mask = true;
    bool dump_intermediates = false;
    bool dump_blocks = false;
};

struct ImageTo3DOpts {
    std::string models_dir = "cpp_ggml/models/gguf";
    std::string moge_model = "cpp_ggml/models/gguf/moge_vitl-f16.gguf";
    std::string backend = "auto";
    std::string dtype = "f16";
    std::string image;
    std::string mask;
    std::string output;
    std::string pbr_output;
    std::string mesh_vertices_output;
    std::string mesh_faces_output;
    std::string pose_output;
    std::string dtype_contract_output;
    std::string noise_dir;
    std::string conditions_output;
    int threads = 8;
    int seed = 42;
    // Default is the F16-KV flash throughput path (--ss-attention normal is
    // the CLI default since 2026-09-18): accepted by the 27-scene validation
    // (RGB MAE 9.6 / IoU 0.816 vs strict 9.9/0.804, 27/27 converged) and the
    // single-object A/B (neural MAE 0.02081 vs strict 0.02115) at ~2.9x SS
    // speed.  Strict F32 score materialization stays available as the parity
    // diagnostic path (--ss-attention strict); diagnostic entry points pin it
    // explicitly.
    bool strict_ss_attention = false;
    bool gs_portable_attention = false;
    uint32_t philox_blocks = 0;   // 0 = require the legacy env contract
    // Flow trajectory lengths; see ImageTo3DOptions for the caliber notes.
    int ss_steps = 25;
    int slat_steps = 25;
    // Batch mode: run every mask of `mask_list` in one session process.
    // Mutually exclusive with --mask; outputs land under out_dir as
    // objects/obj_<ID>/{output.ply,pose.json} plus a batch manifest.
    std::string mask_list;
    std::string out_dir;
    // Hot-session timing: N serial complete requests with the MoGe cache
    // disabled; writes hot_timing.json under out_dir.
    int hot_timing = 0;
};

struct PoseDecodeOpts {
    std::string rotation_6d;
    std::string log_scale;
    std::string translation;
    std::string log_translation_scale;
    std::string scene_scale;
    std::string scene_shift;
    std::string output;
};

static bool load_pose_f32(const std::string& path, size_t expected_count,
                          std::vector<float>& values, const char* label) {
    RawTensor tensor;
    if (path.empty() || !load_raw_tensor(path, tensor) || tensor.type != GGML_TYPE_F32 ||
        tensor.data.size() != expected_count * sizeof(float)) {
        LOGE("pose-decode: %s must be an F32[%zu] SAMT tensor", label, expected_count);
        return false;
    }
    const float* source = reinterpret_cast<const float*>(tensor.data.data());
    values.assign(source, source + expected_count);
    return true;
}

static int cmd_pose_decode(int argc, char** argv) {
    PoseDecodeOpts options;
    for (int index = 0; index < argc; ++index) {
        if (!std::strcmp(argv[index], "--rotation-6d") && index + 1 < argc) {
            options.rotation_6d = argv[++index];
        } else if (!std::strcmp(argv[index], "--log-scale") && index + 1 < argc) {
            options.log_scale = argv[++index];
        } else if (!std::strcmp(argv[index], "--translation") && index + 1 < argc) {
            options.translation = argv[++index];
        } else if (!std::strcmp(argv[index], "--log-translation-scale") && index + 1 < argc) {
            options.log_translation_scale = argv[++index];
        } else if (!std::strcmp(argv[index], "--scene-scale") && index + 1 < argc) {
            options.scene_scale = argv[++index];
        } else if (!std::strcmp(argv[index], "--scene-shift") && index + 1 < argc) {
            options.scene_shift = argv[++index];
        } else if (!std::strcmp(argv[index], "--out") && index + 1 < argc) {
            options.output = argv[++index];
        } else {
            LOGE("pose-decode: unrecognized or incomplete argument %s", argv[index]);
            return 2;
        }
    }
    std::vector<float> rotation_6d, log_scale, translation, log_translation_scale;
    std::vector<float> scene_scale, scene_shift;
    if (options.output.empty() ||
        !load_pose_f32(options.rotation_6d, 6, rotation_6d, "--rotation-6d") ||
        !load_pose_f32(options.log_scale, 3, log_scale, "--log-scale") ||
        !load_pose_f32(options.translation, 3, translation, "--translation") ||
        !load_pose_f32(options.log_translation_scale, 1, log_translation_scale,
                       "--log-translation-scale") ||
        !load_pose_f32(options.scene_scale, 3, scene_scale, "--scene-scale") ||
        !load_pose_f32(options.scene_shift, 3, scene_shift, "--scene-shift")) {
        return 2;
    }
    NativeInstancePose pose;
    std::string error;
    if (!decode_scale_shift_invariant_pose(rotation_6d.data(), log_scale.data(), translation.data(),
                                           log_translation_scale[0], scene_scale.data(),
                                           scene_shift.data(), pose, error) ||
        !write_native_pose_json(options.output, pose, error)) {
        LOGE("pose-decode: %s", error.c_str());
        return 1;
    }
    LOGI("pose-decode: wrote %s", options.output.c_str());
    return 0;
}

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

static int cmd_moge_infer(const MogeInferOpts& options) {
    if (options.model.empty() || options.image_path.empty() || options.output_prefix.empty()) {
        LOGE("moge-infer requires --model, --input and --out");
        return 1;
    }
    RgbaImage image;
    std::string error;
    if (!load_rgba_image(options.image_path, image, error)) {
        LOGE("moge-infer: %s", error.c_str());
        return 1;
    }
    auto backend = Backend::create(options.backend, options.threads);
    if (!backend) return 1;
    GGUFModel model;
    if (!model.load(options.model, backend->weights_buffer_type())) return 1;
    MogeInferenceOptions inference_options;
    inference_options.num_tokens = options.num_tokens;
    inference_options.apply_mask = options.apply_mask;
    inference_options.force_projection = options.force_projection;
    inference_options.capture_intermediates = options.dump_intermediates;
    inference_options.capture_block_outputs = options.dump_blocks;
    MogeInferenceResult result;
    const double started = now_ms();
    if (!run_moge_inference(image, model, *backend, inference_options, result, error)) {
        LOGE("moge-infer: %s", error.c_str());
        return 1;
    }
    const std::filesystem::path output_path(options.output_prefix);
    if (!output_path.parent_path().empty()) {
        std::error_code create_error;
        std::filesystem::create_directories(output_path.parent_path(), create_error);
        if (create_error) {
            LOGE("moge-infer: cannot create output directory: %s", create_error.message().c_str());
            return 1;
        }
    }
    std::vector<float> mask(result.mask.begin(), result.mask.end());
    const float scalar_values[] = {result.focal, result.shift};
    if (!save_raw_tensor_f32(options.output_prefix + ".moge_points.samt",
                             {3, result.width, result.height}, result.points_moge.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".pointmap_raw.samt",
                             {result.width, result.height, 3}, result.pointmap_pytorch3d.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".mask_logits.samt",
                             {result.width, result.height}, result.mask_logits.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".mask_probability.samt",
                             {result.width, result.height}, result.mask_probability.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".mask.samt",
                             {result.width, result.height}, mask.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".depth.samt",
                             {result.width, result.height}, result.depth.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".intrinsics.samt", {3, 3},
                             result.intrinsics.data()) ||
        !save_raw_tensor_f32(options.output_prefix + ".focal_shift.samt", {2}, scalar_values)) {
        LOGE("moge-infer: failed to write output tensors with prefix %s", options.output_prefix.c_str());
        return 1;
    }
    if (options.dump_intermediates &&
        (!save_raw_tensor_f32(options.output_prefix + ".resized_rgb.samt",
                              {3, result.resized_width, result.resized_height},
                              result.resized_rgb.data()) ||
         !save_raw_tensor_f32(options.output_prefix + ".forward_points.samt",
                              {3, result.width, result.height}, result.forward_points.data()))) {
        LOGE("moge-infer: failed to write regression intermediate tensors with prefix %s",
             options.output_prefix.c_str());
        return 1;
    }
    if (options.dump_blocks) {
        const auto save_block_series = [&](const char* name,
                                           const std::vector<std::vector<float>>& values,
                                           int channels) {
            if (values.empty() || channels <= 0) return false;
            for (size_t index = 0; index < values.size(); ++index) {
                const std::string path = options.output_prefix + "." + name +
                                         std::to_string(index) + ".samt";
                if (!save_raw_tensor_f32(path, {channels, result.backbone_tokens},
                                         values[index].data())) {
                    return false;
                }
            }
            return true;
        };
        if (!save_raw_tensor_f32(options.output_prefix + ".backbone_image.samt",
                                 {3, result.backbone_image_width, result.backbone_image_height},
                                 result.backbone_image.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".backbone_patch_tokens.samt",
                                 {result.backbone_hidden, result.backbone_tokens - 1},
                                 result.backbone_patch_tokens.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".backbone_position_tokens.samt",
                                 {result.backbone_hidden, result.backbone_tokens},
                                 result.backbone_position_tokens.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".backbone_input.samt",
                                 {result.backbone_hidden, result.backbone_tokens},
                                 result.backbone_input.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".attention_context.samt",
                                 {static_cast<int64_t>(result.backbone_attention_context.size())},
                                 result.backbone_attention_context.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".q.samt",
                                 {static_cast<int64_t>(result.backbone_q.size())},
                                 result.backbone_q.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".k.samt",
                                 {static_cast<int64_t>(result.backbone_k.size())},
                                 result.backbone_k.data()) ||
            !save_raw_tensor_f32(options.output_prefix + ".v.samt",
                                 {static_cast<int64_t>(result.backbone_v.size())},
                                 result.backbone_v.data()) ||
            !save_block_series("attention_projection", result.backbone_attention_projection,
                               result.backbone_hidden) ||
            !save_block_series("attention", result.backbone_attention, result.backbone_hidden) ||
            !save_block_series("mlp_fc1", result.backbone_mlp_fc1, result.backbone_mlp_hidden) ||
            !save_block_series("mlp_gelu", result.backbone_mlp_gelu, result.backbone_mlp_hidden) ||
            !save_block_series("mlp", result.backbone_mlp, result.backbone_hidden) ||
            !save_block_series("block", result.backbone_blocks, result.backbone_hidden)) {
            LOGE("moge-infer: failed to write DINO block regression tensors with prefix %s",
                 options.output_prefix.c_str());
            return 1;
        }
    }
    LOGI("moge-infer: backend=%s input=%dx%d tokens=%d pointmap=%s.pointmap_raw.samt %.1f ms",
         backend->backend_name(), image.width, image.height, options.num_tokens,
         options.output_prefix.c_str(), now_ms() - started);
    return 0;
}

// ---- image-to-3d batch mode ---------------------------------------------

// 64-bit FNV-1a content digest. Not a cryptographic hash: it exists so a
// resume pass can prove that the referenced inputs and outputs are byte-ident
// to the ones this manifest describes (stale artifacts cannot pass as fresh).
static std::string file_digest_64(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    uint64_t hash = 1469598103934665603ull;
    std::vector<char> buffer(1 << 20);
    size_t got;
    while ((got = std::fread(buffer.data(), 1, buffer.size(), f)) > 0) {
        for (size_t i = 0; i < got; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ull;
        }
    }
    std::fclose(f);
    char out[24];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(hash));
    return out;
}

struct BatchObject { int id; std::string mask_path; };

static bool parse_mask_list(const std::string& list_path,
                            std::vector<BatchObject>& objects, std::string& error) {
    std::ifstream in(list_path);
    if (!in) {
        error = "cannot open mask list '" + list_path + "'";
        return false;
    }
    const std::string base = std::filesystem::path(list_path).parent_path().string();
    std::string line;
    size_t line_no = 0;
    std::set<int> seen;
    while (std::getline(in, line)) {
        ++line_no;
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;  // blank line
        if (line[first] == '#') continue;           // comment
        const auto tab = line.find('\t', first);
        if (tab == std::string::npos || tab == first) {
            error = "mask list line " + std::to_string(line_no) +
                    " must be 'non-negative-integer-ID<TAB>mask-path'";
            return false;
        }
        const std::string id_text = line.substr(first, tab - first);
        if (id_text.find_first_not_of("0123456789") != std::string::npos) {
            error = "mask list line " + std::to_string(line_no) +
                    " has a non-numeric or negative ID '" + id_text + "'";
            return false;
        }
        const int id = std::atoi(id_text.c_str());
        if (!seen.insert(id).second) {
            error = "mask list line " + std::to_string(line_no) +
                    " repeats object ID " + std::to_string(id);
            return false;
        }
        std::string mask = line.substr(tab + 1);
        const auto mask_first = mask.find_first_not_of(" \t");
        const auto mask_last = mask.find_last_not_of(" \t\r");
        if (mask_first == std::string::npos) {
            error = "mask list line " + std::to_string(line_no) + " has an empty mask path";
            return false;
        }
        mask = mask.substr(mask_first, mask_last - mask_first + 1);
        if (!std::filesystem::path(mask).is_absolute() && !base.empty()) mask = base + "/" + mask;
        std::error_code ec;
        if (!std::filesystem::is_regular_file(mask, ec) || ec) {
            error = "mask list line " + std::to_string(line_no) +
                    " references a missing mask file '" + mask + "'";
            return false;
        }
        objects.push_back({id, mask});
    }
    if (objects.empty()) {
        error = "mask list '" + list_path + "' contains no entries";
        return false;
    }
    return true;
}

static int run_image_to_3d_batch(const ImageTo3DOpts& options,
                                 ImageTo3DOptions& native_options) {
    std::vector<BatchObject> objects;
    std::string error;
    if (!parse_mask_list(options.mask_list, objects, error)) {
        LOGE("image-to-3d: %s", error.c_str());
        return 1;
    }
    if (options.output.empty() && options.out_dir.empty()) {
        LOGE("image-to-3d: --mask-list requires --out-dir");
        return 1;
    }
    const std::filesystem::path root =
        options.out_dir.empty()
            ? std::filesystem::path(options.output).parent_path()
            : std::filesystem::path(options.out_dir);
    std::error_code fs_error;
    std::filesystem::create_directories(root / "objects", fs_error);
    if (fs_error) {
        LOGE("image-to-3d: cannot create batch output directory %s: %s",
             root.string().c_str(), fs_error.message().c_str());
        return 1;
    }

    // Resume: an object is only skipped when the previous run's manifest
    // marks it ok AND the recorded image/mask/config digests all still match
    // AND the recorded output digests match the files on disk. A stale
    // artifact or a changed binary/config re-runs the object; a bare FAILED
    // marker never permanently skips anything. Verified rows are carried
    // into the fresh manifest so repeated resumes keep skipping them.
    const std::string image_digest_now = file_digest_64(options.image);
    const std::string moge_digest_now = file_digest_64(options.moge_model);
    const std::filesystem::path manifest_path = root / "batch_manifest.json";
    std::map<int, std::string> carried_rows;  // verified ok rows from the previous run
    if (std::filesystem::is_regular_file(manifest_path, fs_error)) {
        std::ifstream in(manifest_path);
        std::string line;
        while (std::getline(in, line)) {
            const auto grab = [&](const char* key) -> std::string {
                const std::string needle = std::string("\"") + key + "\": \"";
                const auto at = line.find(needle);
                if (at == std::string::npos) return {};
                const auto end = line.find("\"", at + needle.size());
                return end == std::string::npos
                    ? std::string() : line.substr(at + needle.size(), end - at - needle.size());
            };
            const auto grab_number = [&](const char* key) -> std::string {
                const std::string needle = std::string("\"") + key + "\": ";
                const auto at = line.find(needle);
                if (at == std::string::npos) return {};
                const auto comma = line.find(',', at);
                return comma == std::string::npos
                    ? std::string() : line.substr(at + needle.size(), comma - at - needle.size());
            };
            if (grab("status") != "ok") continue;
            const std::string id_text = grab_number("id");
            if (id_text.empty()) continue;
            carried_rows[std::atoi(id_text.c_str())] = line;
        }
    }
    std::vector<BatchObject> pending;
    for (const BatchObject& object : objects) {
        auto carried = carried_rows.find(object.id);
        if (carried == carried_rows.end() ||
            carried_rows[object.id].find(sam3d::json_escape(object.mask_path)) == std::string::npos) {
            pending.push_back(object);
            continue;
        }
        const std::string case_dir = (root / "objects" /
            ("obj_" + std::to_string(object.id))).string();
        const std::string& row = carried->second;
        const auto grab_digest = [&](const char* key) -> std::string {
            const std::string needle = std::string("\"") + key + "\": \"";
            const auto at = row.find(needle);
            if (at == std::string::npos) return {};
            const auto end = row.find("\"", at + needle.size());
            return end == std::string::npos
                ? std::string() : row.substr(at + needle.size(), end - at - needle.size());
        };
        const std::string stored_ply = grab_digest("ply_digest");
        const std::string stored_pose = grab_digest("pose_digest");
        if (!stored_ply.empty() && !stored_pose.empty() &&
            file_digest_64(case_dir + "/output.ply") == stored_ply &&
            file_digest_64(case_dir + "/pose.json") == stored_pose) {
            LOGI("image-to-3d: resume: obj_%d outputs verified, skipping", object.id);
            continue;
        }
        pending.push_back(object);
    }
    if (pending.empty()) {
        LOGI("image-to-3d: batch resume: every object verified, nothing to run");
        return 0;
    }
    // The manifest must describe every requested object, so the write loop
    // walks the full list; resume-verified entries carry their previous row
    // verbatim while the pending ones are re-run here.
    const std::vector<BatchObject> all_objects = objects;
    objects = pending;
    std::vector<int> pending_ids;
    for (const BatchObject& object : objects) pending_ids.push_back(object.id);

    ImageTo3DSession session;
    if (!session.init(native_options, error)) {
        LOGE("image-to-3d: %s", error.c_str());
        return 1;
    }
    LOGI("image-to-3d: batch session ready (%d masks, backend=%s dtype=%s)",
         static_cast<int>(objects.size()), options.backend.c_str(), options.dtype.c_str());

    const auto batch_started = std::chrono::steady_clock::now();
    std::string manifest = "{\n";
    manifest += "  \"format\": \"sam3d-batch-manifest/1\",\n";
    manifest += "  \"image\": \"" + sam3d::json_escape(options.image) + "\",\n";
    manifest += "  \"image_digest\": \"" + file_digest_64(options.image) + "\",\n";
    manifest += "  \"moge_model_digest\": \"" + file_digest_64(options.moge_model) + "\",\n";
    manifest += "  \"config\": {\"models_dir\": \"" + sam3d::json_escape(options.models_dir) +
                "\", \"backend\": \"" + sam3d::json_escape(options.backend) +
                "\", \"dtype\": \"" + sam3d::json_escape(options.dtype) +
                "\", \"seed\": " + std::to_string(options.seed) +
                ", \"threads\": " + std::to_string(options.threads) +
                ", \"philox_blocks\": " + std::to_string(options.philox_blocks) + "},\n";
    manifest += "  \"objects\": [\n";

    bool all_ok = true;
    for (size_t index = 0; index < all_objects.size(); ++index) {
        const BatchObject& object = all_objects[index];
        // A resume-verified object keeps its previous manifest row verbatim;
        // it was neither re-run nor re-timed this round.
        auto carried = carried_rows.find(object.id);
        const bool skipped = carried != carried_rows.end() &&
            std::none_of(pending_ids.begin(), pending_ids.end(),
                         [&](int id) { return id == object.id; });
        if (skipped) {
            // The carried row is the verbatim previous manifest line; strip
            // its trailing separator so the write loop owns the comma logic.
            std::string carried_row = carried->second;
            while (!carried_row.empty() &&
                   (carried_row.back() == ',' || carried_row.back() == '\n' ||
                    carried_row.back() == ' ' || carried_row.back() == '\r')) {
                carried_row.pop_back();
            }
            manifest += "    " + carried_row;
            manifest += index + 1 == all_objects.size() ? "\n" : ",\n";
            continue;
        }
        const std::string case_dir = (root / "objects" /
            ("obj_" + std::to_string(object.id))).string();
        std::filesystem::create_directories(case_dir, fs_error);
        ImageTo3DOptions request = native_options;
        request.mask_path = object.mask_path;
        request.out_ply = case_dir + "/output.ply";
        request.out_pose = case_dir + "/pose.json";
        const auto request_started = std::chrono::steady_clock::now();
        RunResult result = session.run(request);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - request_started).count();
        const bool ok = result.ok;
        all_ok &= ok;
        manifest += "    {\"id\": " + std::to_string(object.id) +
                    ", \"mask\": \"" + sam3d::json_escape(object.mask_path) + "\"" +
                    ", \"mask_digest\": \"" + file_digest_64(object.mask_path) + "\"";
        if (ok) {
            manifest += ", \"status\": \"ok\"";
            manifest += ", \"ply_digest\": \"" + file_digest_64(request.out_ply) + "\"";
            manifest += ", \"pose_digest\": \"" + file_digest_64(request.out_pose) + "\"";
            manifest += ", \"moge_pointmap_reused\": " +
                        std::string(result.output.stats.moge_pointmap_reused ? "true" : "false");
            manifest += ", \"elapsed_s\": " + std::to_string(elapsed);
            LOGI("image-to-3d: batch obj_%d ok (%.1fs) -> %s", object.id, elapsed,
                 request.out_ply.c_str());
        } else {
            // A failed object is recorded and the batch continues; the final
            // exit code stays non-zero so a partial scene is never a success.
            manifest += ", \"status\": \"failed\"";
            manifest += ", \"error\": \"" + sam3d::json_escape(result.error) + "\"";
            manifest += ", \"elapsed_s\": " + std::to_string(elapsed);
            LOGE("image-to-3d: batch obj_%d FAILED: %s", object.id, result.error.c_str());
        }
        manifest += "}";
        manifest += index + 1 == all_objects.size() ? "\n" : ",\n";
    }
    const double total_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_started).count();
    manifest += "  ],\n";
    manifest += "  \"total_elapsed_s\": " + std::to_string(total_elapsed) + "\n";
    manifest += "}\n";

    const std::string manifest_out_path = (root / "batch_manifest.json").string();
    std::ofstream out(manifest_out_path);
    out << manifest;
    LOGI("image-to-3d: batch manifest -> %s (%.1fs total, %s)", manifest_out_path.c_str(),
         total_elapsed, all_ok ? "all objects ok" : "WITH FAILURES");
    return all_ok ? 0 : 2;
}

// Hot-session timing: N serial complete requests in one session. The MoGe
// point-map cache is disabled so every sample carries the real per-request
// MoGe cost; session resources (backend, stage weight uploads) are shared,
// matching the production batch's hot state. Emits a self-contained JSON
// receipt with the per-sample wall time and the median.
static int run_image_to_3d_hot_timing(const ImageTo3DOpts& options,
                                      ImageTo3DOptions& native_options, int samples) {
    std::string error;
    ImageTo3DSession session;
    if (!session.init(native_options, error)) {
        LOGE("image-to-3d: %s", error.c_str());
        return 1;
    }
    const std::filesystem::path root = options.out_dir;
    std::error_code fs_error;
    std::filesystem::create_directories(root, fs_error);
    if (fs_error) {
        LOGE("image-to-3d: cannot create hot-timing output directory %s: %s",
             root.string().c_str(), fs_error.message().c_str());
        return 1;
    }
    std::vector<double> sample_ms;
    std::string manifest = "{\n  \"format\": \"sam3d-hot-timing/1\",\n";
    manifest += "  \"timer_contract\": \"session-hot: serial complete requests in one "
                "session; MoGe point-map cache disabled; covers MoGe through PLY/pose "
                "file close, including per-stage weight uploads\",\n";
    manifest += "  \"samples\": [\n";
    for (int i = 0; i < samples; ++i) {
        ImageTo3DOptions request = native_options;
        request.disable_moge_pointmap_cache = true;
        request.out_ply = (root / "hot_sample_output.ply").string();
        request.out_pose = (root / "hot_sample_pose.json").string();
        const auto started = std::chrono::steady_clock::now();
        RunResult result = session.run(request);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (!result.ok) {
            LOGE("image-to-3d: hot-timing sample %d failed: %s", i, result.error.c_str());
            return 1;
        }
        sample_ms.push_back(ms);
        manifest += "    {\"sample\": " + std::to_string(i) +
                    ", \"elapsed_ms\": " + std::to_string(ms) + "}"
                    + (i + 1 == samples ? "\n" : ",\n");
        LOGI("image-to-3d: hot-timing sample %d/%d: %.3f s", i + 1, samples, ms / 1000.0);
    }
    std::sort(sample_ms.begin(), sample_ms.end());
    const double median_ms = sample_ms.size() % 2
        ? sample_ms[sample_ms.size() / 2]
        : 0.5 * (sample_ms[sample_ms.size() / 2 - 1] + sample_ms[sample_ms.size() / 2]);
    manifest += "  ],\n  \"median_ms\": " + std::to_string(median_ms) + ",\n";
    manifest += "  \"sample_count\": " + std::to_string(samples) + "\n}\n";
    const std::string out_path = (root / "hot_timing.json").string();
    std::ofstream out(out_path);
    out << manifest;
    LOGI("image-to-3d: hot-timing receipt -> %s (median %.3f s over %d samples)",
         out_path.c_str(), median_ms / 1000.0, samples);
    return 0;
}

static int cmd_image_to_3d(const ImageTo3DOpts& options) {
    ImageTo3DOptions native_options;
    native_options.models_dir = options.models_dir;
    native_options.moge_model = options.moge_model;
    native_options.backend = options.backend;
    native_options.dtype = options.dtype;
    native_options.image_path = options.image;
    native_options.mask_path = options.mask;
    native_options.out_ply = options.output;
    native_options.out_pbr = options.pbr_output;
    native_options.out_mesh_vertices = options.mesh_vertices_output;
    native_options.out_mesh_faces = options.mesh_faces_output;
    native_options.out_pose = options.pose_output;
    native_options.out_dtype_contract = options.dtype_contract_output;
    native_options.noise_dir = options.noise_dir;
    native_options.conditions_out = options.conditions_output;
    native_options.n_threads = options.threads;
    native_options.seed = options.seed;
    native_options.strict_ss_attention = options.strict_ss_attention;
    native_options.gs_portable_attention = options.gs_portable_attention;
    native_options.philox_blocks = options.philox_blocks;
    native_options.ss_steps = options.ss_steps;
    native_options.slat_steps = options.slat_steps;

    if (!options.mask_list.empty()) {
        return run_image_to_3d_batch(options, native_options);
    }
    if (options.hot_timing > 0) {
        if (options.out_dir.empty()) {
            LOGE("image-to-3d: --hot-timing requires --out-dir");
            return 1;
        }
        return run_image_to_3d_hot_timing(options, native_options, options.hot_timing);
    }

    RunResult result = run_image_to_3d(native_options);
    if (!result.ok) {
        LOGE("image-to-3d: %s", result.error.c_str());
        return 1;
    }
    LOGI("image-to-3d: wrote %s%s%s%s", options.output.c_str(),
         options.pbr_output.empty() ? "" : " and native PBR GLB",
         options.mesh_vertices_output.empty() ? "" : " and raw FlexiCubes mesh",
         options.pose_output.empty() ? "" : " and official pose JSON");
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
    std::error_code directory_error;
    std::filesystem::create_directories(out_dir, directory_error);
    if (directory_error) {
        LOGE("gs-decode: cannot create %s: %s", out_dir.c_str(),
             directory_error.message().c_str());
        return 1;
    }
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
#ifdef SAM3D_USE_CUDA
    gb.use_pytorch_cuda_attention = std::strstr(backend->backend_name(), "CUDA") != nullptr &&
                                      getenv("SAM3D_GS_PORTABLE_ATTN") == nullptr;
#endif
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
    gb.use_cuda_spconv = be != nullptr && std::strcmp(be, "cuda") == 0;
    if (const char* st = getenv("SAM3D_DEBUG_STAGE")) gb.debug_stage = st;
    // Trace every input_blocks.1 boundary in one run: the debug-stage graph
    // prunes nodes, which changes allocation layout and hides layout-dependent
    // failures. Dumping all traced boundaries from the unpruned graph keeps
    // every dump in the same allocation context.
    gb.dump_block_outputs = getenv("SAM3D_DEBUG_DUMP_BLOCKS") != nullptr;
    std::vector<ggml_tensor*> outs = gb.build();

    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto* o : outs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }
    // The allocator may recycle non-output intermediate buffers before the
    // post-run diagnostic readback below. Pin requested debug tensors as graph
    // outputs so each SAMT file represents the value produced by this run.
    if (getenv("SAM3D_DEBUG_DUMP_BLOCKS") != nullptr) {
        for (auto* block : gb.debug_block_outputs) {
            ggml_set_output(block);
            ggml_build_forward_expand(graph, block);
        }
    }
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
            if (getenv("SAM3D_DEBUG_INPUT_TYPES") != nullptr) {
                LOGI("slat-step input[%zu]: name=%s type=%s elements=%lld",
                     ti, t->name, ggml_type_name(t->type),
                     (long long)ggml_nelements(t));
            }
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
    // The official SS backbone runs entirely in F32 (use_fp16=false), so its
    // attention contract is F32 SDPA. Keep K/V in F32 by default; the F16-K/V
    // throughput path is an explicit opt-out.
    gb.strict_attention = getenv("SAM3D_SS_FAST_ATTN") == nullptr;
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

// Materialize one GGUF tensor through the selected backend's cast path. This
// is a numerical diagnostic: it makes the values consumed by a graph
// independently comparable with the Python GGUF dequantization oracle.
static int cmd_tensor_dump(const std::string& model_path, const std::string& tensor_name,
                           const std::string& output_path, const std::string& output_dtype) {
    const char* backend_name = getenv("SAM3D_BACKEND");
    auto backend = Backend::create(backend_name ? backend_name : "cpu",
                                   getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8);
    if (!backend) return 1;

    GGUFModel model;
    if (!model.load(model_path, backend->weights_buffer_type())) return 1;
    ggml_tensor* source = model.get(tensor_name);
    if (!source) {
        LOGE("tensor-dump: tensor not found: %s", tensor_name.c_str());
        return 1;
    }
    const ggml_type target_type = output_dtype == "f32" ? GGML_TYPE_F32 :
                                  output_dtype == "f16" ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
    if (target_type == GGML_TYPE_COUNT) {
        LOGE("tensor-dump: --tensor-dtype must be f32 or f16");
        return 1;
    }

    GraphContext context;
    // CUDA has no direct Q8_0/Q4_0-to-F16 copy kernel. Keep the diagnostic
    // path identical to SlatFlowGraph::as_f16(): quantized -> F32 -> F16.
    ggml_tensor* materialized = source;
    if (materialized->type != target_type) {
        if (target_type == GGML_TYPE_F16 && ggml_is_quantized(materialized->type)) {
            materialized = ggml_cast(context.ctx(), materialized, GGML_TYPE_F32);
        }
        materialized = ggml_cast(context.ctx(), materialized, target_type);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 64, false);
    ggml_set_output(materialized);
    ggml_build_forward_expand(graph, materialized);
    if (!backend->alloc(graph) || !backend->run(graph)) return 1;

    std::vector<float> values;
    if (!backend->get_tensor_f32(materialized, values)) return 1;
    std::vector<int64_t> shape(materialized->ne, materialized->ne + ggml_n_dims(materialized));
    if (!save_raw_tensor_f32(output_path, shape, values.data())) return 1;
    printf("tensor-dump: %s (%s -> %s, %lld elements) -> %s\n", tensor_name.c_str(),
           ggml_type_name(source->type), ggml_type_name(target_type),
           (long long)values.size(), output_path.c_str());
    return 0;
}

namespace sam3d {
int cmd_e2e(const E2eOptions& opt);
}  // namespace sam3d

#if defined(SAM3D_NATIVE_PBR_CUDA)
// Normalize with the reference variant's recorded contract when supplied so
// every variant renders the identical scene frame; otherwise use the scene's
// own opacity>0.9 bounds (official normalized_gaussian).
static bool normalize_scene_with_reference(GaussianSplatSet& scene,
                                           const std::string& normalization_json,
                                           std::string& error) {
    if (normalization_json.empty()) {
        return normalize_scene(scene, error);
    }
    float inv_scale = 0.0f;
    std::array<float, 3> center{};
    if (!load_scene_normalization_json(normalization_json, inv_scale, center, error)) {
        return false;
    }
    return normalize_scene_with(scene, inv_scale, center, error);
}

// Native multi-object scene assembly: per-object Gaussian PLYs plus the
// official-schema pose receipts are composed, normalized, and rendered on the
// orbit of the official render_video without any Python stage.
static int cmd_scene_assemble(int argc, char** argv) {
    std::string objects_list, out_dir, ply_out = "scene_posed.ply";
    std::string frames_dir = "frames", manifest_out = "scene_manifest.json";
    std::string normalization_json;
    int num_frames = 300;
    int resolution = 512;
    float radius = 1.0f;
    float fov = 60.0f;
    for (int index = 0; index < argc; ++index) {
        const char* flag = argv[index];
        const auto needs_value = [&](std::string& target) -> bool {
            if (index + 1 >= argc) {
                LOGE("scene-assemble: missing value for %s", flag);
                return false;
            }
            target = argv[++index];
            return true;
        };
        if (!strcmp(flag, "--objects-list")) {
            if (!needs_value(objects_list)) return 2;
        } else if (!strcmp(flag, "--out-dir")) {
            if (!needs_value(out_dir)) return 2;
        } else if (!strcmp(flag, "--ply-out")) {
            if (!needs_value(ply_out)) return 2;
        } else if (!strcmp(flag, "--frames-dir")) {
            if (!needs_value(frames_dir)) return 2;
        } else if (!strcmp(flag, "--manifest-out")) {
            if (!needs_value(manifest_out)) return 2;
        } else if (!strcmp(flag, "--normalization-json")) {
            if (!needs_value(normalization_json)) return 2;
        } else if (!strcmp(flag, "--num-frames")) {
            std::string value;
            if (!needs_value(value)) return 2;
            num_frames = std::atoi(value.c_str());
        } else if (!strcmp(flag, "--resolution")) {
            std::string value;
            if (!needs_value(value)) return 2;
            resolution = std::atoi(value.c_str());
        } else if (!strcmp(flag, "--radius")) {
            std::string value;
            if (!needs_value(value)) return 2;
            radius = std::atof(value.c_str());
        } else if (!strcmp(flag, "--fov")) {
            std::string value;
            if (!needs_value(value)) return 2;
            fov = std::atof(value.c_str());
        } else {
            LOGE("scene-assemble: unrecognized or incomplete argument %s", flag);
            return 2;
        }
    }
    if (objects_list.empty() || out_dir.empty()) {
        LOGE("scene-assemble: --objects-list and --out-dir are required");
        return 2;
    }
    if (num_frames <= 0 || resolution <= 0 || !(radius > 0.0f) || !(fov > 0.0f)) {
        LOGE("scene-assemble: invalid render configuration");
        return 2;
    }
    struct SceneObjectEntry {
        std::string ply;
        std::string pose;
        size_t gaussians = 0;
    };
    std::vector<SceneObjectEntry> entries;
    {
        std::ifstream list_stream(objects_list);
        if (!list_stream) {
            LOGE("scene-assemble: cannot open object list: %s", objects_list.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(list_stream, line)) {
            if (line.empty()) continue;
            const size_t split = line.find('\t');
            if (split == std::string::npos) {
                LOGE("scene-assemble: list line must be '<ply>\\t<pose>': %s", line.c_str());
                return 1;
            }
            entries.push_back({line.substr(0, split), line.substr(split + 1), 0});
        }
    }
    if (entries.empty()) {
        LOGE("scene-assemble: object list is empty: %s", objects_list.c_str());
        return 1;
    }

    std::string error;
    GaussianSplatSet scene;
    for (auto& entry : entries) {
        GaussianSplatSet object;
        if (!load_gaussian_splat_ply(entry.ply, object, error)) {
            LOGE("scene-assemble: %s", error.c_str());
            return 1;
        }
        ScenePose pose;
        if (!load_scene_pose_json(entry.pose, pose, error)) {
            LOGE("scene-assemble: %s", error.c_str());
            return 1;
        }
        // The representation's 3d_filter_kernel_size (session.cpp GS_MIN_KERNEL).
        if (!apply_scene_pose(object, pose, 0.0009f, error)) {
            LOGE("scene-assemble: %s", error.c_str());
            return 1;
        }
        entry.gaussians = object.size();
        append_splat_set(scene, object);
    }
    if (!scene.valid()) {
        LOGE("scene-assemble: composed scene is invalid");
        return 1;
    }

    std::filesystem::create_directories(out_dir);
    const std::filesystem::path out_path(out_dir);
    const std::string ply_path = (out_path / ply_out).string();
    if (!write_scene_ply(ply_path, scene, error)) {
        LOGE("scene-assemble: %s", error.c_str());
        return 1;
    }

    if (!normalize_scene_with_reference(scene, normalization_json, error)) {
        LOGE("scene-assemble: %s", error.c_str());
        return 1;
    }
    const std::vector<GaussianCamera> cameras =
        make_orbit_cameras(num_frames, radius, fov, error);
    if (cameras.empty()) {
        LOGE("scene-assemble: %s", error.c_str());
        return 1;
    }
    GaussianRenderConfig config;
    config.width = resolution;
    config.height = resolution;
    config.radius = radius;
    config.fov_degrees = fov;
    // The orbit stays outside the 100-view bake's tight depth envelope:
    // the normalized scene spans roughly [-0.9, 0.9] and the camera sits at
    // radius, so a wide envelope covers every object without clipping.
    config.near_plane = 0.05f;
    config.far_plane = 20.0f;
    // gsplat parity probe: the official scene render path goes through the
    // gsplat backend (render_frames backend="gsplat"), where pipe.kernel_size
    // is never consumed and the rasterizer applies its default eps2d=0.3
    // screen-space low-pass. The inria-style fork reads kernel_size directly,
    // so 0.3 reproduces the gsplat dilation instead of the inria 0.1.
    config.kernel_size = 0.3f;
    std::vector<RgbaImage> frames;
    if (!render_gaussian_views_cuda(scene, cameras, config, frames, error)) {
        LOGE("scene-assemble: %s", error.c_str());
        return 1;
    }
    const std::string frames_path = (out_path / frames_dir).string();
    std::filesystem::create_directories(frames_path);
    for (size_t frame = 0; frame < frames.size(); ++frame) {
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%04zu.png", frame);
        if (!write_rgba_png((std::filesystem::path(frames_path) / name).string(),
                            frames[frame], error)) {
            LOGE("scene-assemble: %s", error.c_str());
            return 1;
        }
    }

    const std::string manifest_path = (out_path / manifest_out).string();
    {
        FILE* manifest = fopen(manifest_path.c_str(), "wb");
        if (!manifest) {
            LOGE("scene-assemble: cannot open manifest output: %s", manifest_path.c_str());
            return 1;
        }
        fprintf(manifest,
                "{\n  \"schema\": \"sam3d.scene-native.v1\",\n"
                "  \"gaussians_total\": %zu,\n"
                "  \"render\": {\"frames\": %zu, \"radius\": %.6f, \"fov\": %.6f,"
                " \"resolution\": %d},\n"
                "  \"objects\": [\n",
                scene.size(), frames.size(), static_cast<double>(radius),
                static_cast<double>(fov), resolution);
        for (size_t index = 0; index < entries.size(); ++index) {
            fprintf(manifest,
                    "    {\"ply\": \"%s\", \"pose\": \"%s\", \"gaussians\": %zu}%s\n",
                    entries[index].ply.c_str(), entries[index].pose.c_str(),
                    entries[index].gaussians,
                    index + 1 < entries.size() ? "," : "");
        }
        fprintf(manifest,
                "  ],\n  \"outputs\": {\"posed_ply\": \"%s\", \"frames_dir\": \"%s\"}\n}\n",
                ply_path.c_str(), frames_path.c_str());
        if (fclose(manifest) != 0) {
            LOGE("scene-assemble: failed while writing manifest: %s", manifest_path.c_str());
            return 1;
        }
    }
    LOGI("scene-assemble: wrote %s (%zu objects, %zu gaussians) and %zu frames under %s",
         ply_path.c_str(), entries.size(), scene.size(), frames.size(), out_dir.c_str());
    return 0;
}
#endif

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
    if (cmd == "rng-dump") return cmd_rng_dump(argc - 2, argv + 2);
    if (cmd == "coords-downsample") return cmd_coords_downsample(argc - 2, argv + 2);
    if (cmd == "pose-decode") return cmd_pose_decode(argc - 2, argv + 2);
#if defined(SAM3D_NATIVE_PBR_CUDA)
    if (cmd == "scene-assemble") return cmd_scene_assemble(argc - 2, argv + 2);
#endif
    std::string model, input, out, tensor_name, tensor_dtype = "f32";
    std::vector<std::string> pos;  // bare positional args
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--tensor") && i + 1 < argc) tensor_name = argv[++i];
        else if (!strcmp(argv[i], "--tensor-dtype") && i + 1 < argc) tensor_dtype = argv[++i];
        else pos.emplace_back(argv[i]);
    }
    if (cmd == "info") {
        if (model.empty()) {
            print_usage();
            return 1;
        }
        return cmd_info(model);
    }
    if (cmd == "tensor-dump") {
        if (model.empty() || tensor_name.empty() || out.empty()) {
            print_usage();
            return 1;
        }
        return cmd_tensor_dump(model, tensor_name, out, tensor_dtype);
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
            else if (!std::strcmp(argv[i], "--uv") && i + 1 < argc) options.texcoords = argv[++i];
            else if (!std::strcmp(argv[i], "--texture") && i + 1 < argc) options.texture = argv[++i];
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
    if (cmd == "pbr-assemble") {
        PbrAssembleOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--ply") && i + 1 < argc) options.ply = argv[++i];
            else if (!std::strcmp(argv[i], "--views") && i + 1 < argc) options.render_views = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--resolution") && i + 1 < argc) options.resolution = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--texture-size") && i + 1 < argc) options.texture_size = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) options.texture_steps = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) options.seed = static_cast<unsigned>(std::atoi(argv[++i]));
            else if (!std::strcmp(argv[i], "--already-clean")) options.already_clean = true;
        }
        return cmd_pbr_assemble(options);
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
    if (cmd == "texture-bake") {
        TextureBakeOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--uv") && i + 1 < argc) options.uv = argv[++i];
            else if (!std::strcmp(argv[i], "--observations-dir") && i + 1 < argc) {
                options.observations_dir = argv[++i];
            } else if (!std::strcmp(argv[i], "--extrinsics") && i + 1 < argc) {
                options.extrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--intrinsics") && i + 1 < argc) {
                options.intrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--texture-size") && i + 1 < argc) {
                options.texture_size = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) {
                options.steps = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
                options.seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
            } else if (!std::strcmp(argv[i], "--holes-out") && i + 1 < argc) {
                options.holes_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--no-inpaint")) {
                options.apply_telea = false;
            } else if (!std::strcmp(argv[i], "--raw-texture-out") && i + 1 < argc) {
                options.raw_texture_output = argv[++i];
            }
        }
        return cmd_texture_bake(options);
    }
    if (cmd == "texture-raster") {
        TextureRasterOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--uv") && i + 1 < argc) options.uv = argv[++i];
            else if (!std::strcmp(argv[i], "--extrinsics") && i + 1 < argc) {
                options.extrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--intrinsics") && i + 1 < argc) {
                options.intrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--view-index") && i + 1 < argc) {
                options.view_index = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--resolution") && i + 1 < argc) {
                options.resolution = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--uv-out") && i + 1 < argc) {
                options.uv_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--uv-dr-out") && i + 1 < argc) {
                options.uv_derivatives_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--coverage-out") && i + 1 < argc) {
                options.coverage_output = argv[++i];
            }
        }
        return cmd_texture_raster(options);
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
            } else if (!std::strcmp(argv[i], "--portable-attention")) {
                options.native_attention = false;
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
            } else if (!std::strcmp(argv[i], "--native-visibility")) {
                options.native_visibility = true;
            } else if (!std::strcmp(argv[i], "--visibility-views") && i + 1 < argc) {
                options.visibility_views = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--visibility-resolution") && i + 1 < argc) {
                options.visibility_resolution = std::atoi(argv[++i]);
            }
        }
        return cmd_mesh_postprocess(options);
    }
    if (cmd == "mesh-visibility") {
        MeshVisibilityOpts options;
        options.output = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vertices") && i + 1 < argc) options.vertices = argv[++i];
            else if (!std::strcmp(argv[i], "--faces") && i + 1 < argc) options.faces = argv[++i];
            else if (!std::strcmp(argv[i], "--views") && i + 1 < argc) {
                options.view_count = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--resolution") && i + 1 < argc) {
                options.resolution = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--extrinsics") && i + 1 < argc) {
                options.extrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--intrinsics") && i + 1 < argc) {
                options.intrinsics = argv[++i];
            } else if (!std::strcmp(argv[i], "--view") && i + 1 < argc) {
                options.view = argv[++i];
            } else if (!std::strcmp(argv[i], "--projection") && i + 1 < argc) {
                options.projection = argv[++i];
            }
        }
        return cmd_mesh_visibility(options);
    }
    if (cmd == "mesh-camera-dump") {
        MeshCameraDumpOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--extrinsics-out") && i + 1 < argc) {
                options.extrinsics_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--intrinsics-out") && i + 1 < argc) {
                options.intrinsics_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--views-out") && i + 1 < argc) {
                options.views_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--projections-out") && i + 1 < argc) {
                options.projections_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--views") && i + 1 < argc) {
                options.view_count = std::atoi(argv[++i]);
            }
        }
        return cmd_mesh_camera_dump(options);
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
    if (cmd == "moge-infer") {
        MogeInferOpts options;
        options.model = model;
        options.image_path = input;
        options.output_prefix = out;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) options.backend = argv[++i];
            else if (!std::strcmp(argv[i], "--num-tokens") && i + 1 < argc) {
                options.num_tokens = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
                options.threads = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--force-projection")) {
                options.force_projection = true;
            } else if (!std::strcmp(argv[i], "--no-apply-mask")) {
                options.apply_mask = false;
            } else if (!std::strcmp(argv[i], "--dump-intermediates")) {
                options.dump_intermediates = true;
            } else if (!std::strcmp(argv[i], "--dump-blocks")) {
                options.dump_blocks = true;
            }
        }
        return cmd_moge_infer(options);
    }
    if (cmd == "image-to-3d") {
        ImageTo3DOpts options;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--image") && i + 1 < argc) options.image = argv[++i];
            else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) options.mask = argv[++i];
            else if (!std::strcmp(argv[i], "--mask-list") && i + 1 < argc) {
                options.mask_list = argv[++i];
            } else if (!std::strcmp(argv[i], "--out-dir") && i + 1 < argc) {
                options.out_dir = argv[++i];
            } else if (!std::strcmp(argv[i], "--hot-timing") && i + 1 < argc) {
                options.hot_timing = std::atoi(argv[++i]);
                if (options.hot_timing < 1) {
                    LOGE("image-to-3d: --hot-timing requires a positive sample count");
                    return 1;
                }
            } else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) options.output = argv[++i];
            else if (!std::strcmp(argv[i], "--pbr-out") && i + 1 < argc) {
                options.pbr_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--mesh-vertices-out") && i + 1 < argc) {
                options.mesh_vertices_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--mesh-faces-out") && i + 1 < argc) {
                options.mesh_faces_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--pose-out") && i + 1 < argc) {
                options.pose_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--dtype-contract-out") && i + 1 < argc) {
                options.dtype_contract_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--noise-dir") && i + 1 < argc) {
                options.noise_dir = argv[++i];
            } else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) {
                options.models_dir = argv[++i];
            } else if (!std::strcmp(argv[i], "--moge-model") && i + 1 < argc) {
                options.moge_model = argv[++i];
            } else if (!std::strcmp(argv[i], "--conditions-out") && i + 1 < argc) {
                options.conditions_output = argv[++i];
            } else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
                options.backend = argv[++i];
            } else if (!std::strcmp(argv[i], "--dtype") && i + 1 < argc) {
                options.dtype = argv[++i];
            } else if (!std::strcmp(argv[i], "--ss-attention") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value == "strict") {
                    options.strict_ss_attention = true;
                } else if (value == "normal") {
                    options.strict_ss_attention = false;
                } else {
                    LOGE("image-to-3d: --ss-attention must be normal or strict");
                    return 1;
                }
            } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
                options.seed = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
                options.threads = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--philox-blocks") && i + 1 < argc) {
                options.philox_blocks = (uint32_t)strtoul(argv[++i], nullptr, 10);
            } else if (!std::strcmp(argv[i], "--ss-steps") && i + 1 < argc) {
                options.ss_steps = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--slat-steps") && i + 1 < argc) {
                options.slat_steps = std::atoi(argv[++i]);
            } else if (!std::strcmp(argv[i], "--gs-portable-attention")) {
                options.gs_portable_attention = true;
            } else {
                LOGE("image-to-3d: unrecognized or incomplete argument %s", argv[i]);
                return 1;
            }
        }
        if (!options.mask_list.empty()) {
            // One object source per run: the list form writes obj_<ID> trees
            // under --out-dir, while --mask is the single-object form.
            if (!options.mask.empty()) {
                LOGE("image-to-3d: --mask-list and --mask are mutually exclusive");
                return 1;
            }
            if (!options.output.empty()) {
                LOGE("image-to-3d: --mask-list requires --out-dir; --out is the "
                     "single-object output path");
                return 1;
            }
            if (options.image.empty()) {
                LOGE("image-to-3d: --mask-list requires --image");
                return 1;
            }
        }
        return cmd_image_to_3d(options);
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
        E2eOptions e2e;
        e2e.models_dir = model.empty() ? "cpp_ggml/models/gguf" : model;
        e2e.cond_dir = pos[0];
        e2e.out_ply = out.empty() ? "output.ply" : out;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--noise-dir") && i + 1 < argc) e2e.noise_dir = argv[++i];
            else if (!strcmp(argv[i], "--dbg-dir") && i + 1 < argc) e2e.dbg_dir = argv[++i];
            else if (!strcmp(argv[i], "--pbr-out") && i + 1 < argc) e2e.out_pbr = argv[++i];
            else if (!strcmp(argv[i], "--pose-out") && i + 1 < argc) e2e.out_pose = argv[++i];
            else if (!strcmp(argv[i], "--dtype-contract-out") && i + 1 < argc) e2e.out_dtype_contract = argv[++i];
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc) e2e.seed = (unsigned)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) e2e.threads = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--backend") && i + 1 < argc) e2e.backend = argv[++i];
            else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) e2e.dtype = argv[++i];
            else if (!strcmp(argv[i], "--ss-dtype") && i + 1 < argc) e2e.ss_dtype = argv[++i];
            else if (!strcmp(argv[i], "--ss-decoder-dtype") && i + 1 < argc) e2e.ss_decoder_dtype = argv[++i];
            else if (!strcmp(argv[i], "--slat-dtype") && i + 1 < argc) e2e.slat_dtype = argv[++i];
            else if (!strcmp(argv[i], "--gs-dtype") && i + 1 < argc) e2e.gs_dtype = argv[++i];
            else if (!strcmp(argv[i], "--mesh-dtype") && i + 1 < argc) e2e.mesh_dtype = argv[++i];
            else if (!strcmp(argv[i], "--stage") && i + 1 < argc) e2e.stage = argv[++i];
            else if (!strcmp(argv[i], "--ss-cond-path") && i + 1 < argc) e2e.ss_cond_path = argv[++i];
            else if (!strcmp(argv[i], "--slat-cond-path") && i + 1 < argc) e2e.slat_cond_path = argv[++i];
            else if (!strcmp(argv[i], "--coords-path") && i + 1 < argc) e2e.coords_path = argv[++i];
            else if (!strcmp(argv[i], "--reference-coords")) e2e.reference_coords = true;
            else if (!strcmp(argv[i], "--ss-steps") && i + 1 < argc) e2e.ss_steps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--slat-steps") && i + 1 < argc) e2e.slat_steps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--ss-flow-only")) e2e.ss_flow_only = true;
            else if (!strcmp(argv[i], "--slat-flow-only")) e2e.slat_flow_only = true;
            else if (!strcmp(argv[i], "--dump-slat-steps")) e2e.dump_slat_steps = true;
            else if (!strcmp(argv[i], "--dino-dbg") && i + 1 < argc) { e2e.dino_dbg = true; e2e.dino_dbg_out = argv[++i]; }
            else if (!strcmp(argv[i], "--debug-stage") && i + 1 < argc) e2e.debug_stage = argv[++i];
            else if (!strcmp(argv[i], "--keep-quant-gemm")) e2e.keep_quant_gemm = true;
            else if (!strcmp(argv[i], "--gs-portable-attention")) e2e.gs_portable_attention = true;
            else if (!strcmp(argv[i], "--philox-blocks") && i + 1 < argc) e2e.philox_blocks = (uint32_t)strtoul(argv[++i], nullptr, 10);
            else if (!strcmp(argv[i], "--ss-fast-attention")) e2e.ss_strict_attention = false;
            else if (!strcmp(argv[i], "--cond-manual-attention")) e2e.cond_manual_attention = true;
            else if (!strcmp(argv[i], "--verify")) e2e.verify = true;
        }
        return cmd_e2e(e2e);
    }
    if (cmd == "run" && pos.size() >= 1) {
        const char* be = getenv("SAM3D_BACKEND");
        std::string backend = be ? be : "auto";
        std::string noise_dir, dbg_dir, pbr_out, pose_out, dtype_contract_out;
        unsigned seed = 42;
        int nthreads = getenv("SAM3D_NTHREADS") ? atoi(getenv("SAM3D_NTHREADS")) : 8;
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
            else if (!strcmp(argv[i], "--noise-dir") && i + 1 < argc) noise_dir = argv[++i];
            else if (!strcmp(argv[i], "--dbg-dir") && i + 1 < argc) dbg_dir = argv[++i];
            else if (!strcmp(argv[i], "--pbr-out") && i + 1 < argc) pbr_out = argv[++i];
            else if (!strcmp(argv[i], "--pose-out") && i + 1 < argc) pose_out = argv[++i];
            else if (!strcmp(argv[i], "--dtype-contract-out") && i + 1 < argc) dtype_contract_out = argv[++i];
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        }
        if (model.empty()) model = "cpp_ggml/models/gguf";
        if (out.empty()) out = "output.ply";
        E2eOptions e2e;
        e2e.models_dir = model;
        e2e.cond_dir = pos[0];
        e2e.noise_dir = noise_dir;
        e2e.out_ply = out;
        e2e.dbg_dir = dbg_dir;
        e2e.out_pbr = pbr_out;
        e2e.out_pose = pose_out;
        e2e.out_dtype_contract = dtype_contract_out;
        e2e.backend = backend;
        e2e.seed = seed;
        e2e.threads = nthreads;
        return cmd_e2e(e2e);
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
