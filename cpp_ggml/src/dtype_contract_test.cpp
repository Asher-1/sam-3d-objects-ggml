#include "dtype_contract.hpp"
#include "graph_builder.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path output = std::filesystem::temp_directory_path() /
        ("sam3d-dtype-contract-" + std::to_string(nonce) + ".json");
    sam3d::DtypeContractMetadata metadata;
    metadata.backend = "test";
    metadata.base_dtype = "f16";
    metadata.ss_dtype = "f16";
    metadata.ss_decoder_dtype = "f16";
    metadata.slat_dtype = "f16";
    metadata.gs_dtype = "f16";
    metadata.mesh_dtype = "f16";
    std::string error;
    sam3d::ScopedDtypeContractRecording recording(output.string(), metadata, error);
    if (!recording.enabled()) {
        std::fprintf(stderr, "failed to start dtype recording: %s\n", error.c_str());
        return 1;
    }
    {
        sam3d::GraphContext context;
        ggml_tensor* input = context.input_f32("input", {2, 3});
        ggml_tensor* weight = ggml_new_tensor_2d(context.ctx(), GGML_TYPE_F16, 2, 2);
        ggml_tensor* output_tensor = ggml_mul_mat(context.ctx(), weight, input);
        ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 16, false);
        ggml_build_forward_expand(graph, output_tensor);
        sam3d::record_dtype_contract_graph("unit_test", graph);
    }
    if (!recording.complete(error)) {
        std::fprintf(stderr, "failed to write dtype contract: %s\n", error.c_str());
        return 1;
    }
    std::ifstream stream(output);
    const std::string content((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
    if (content.find("\"complete\": true") == std::string::npos ||
        content.find("\"stage\": \"unit_test\"") == std::string::npos ||
        content.find("\"op\": \"MUL_MAT\"") == std::string::npos ||
        content.find("\"f16\"") == std::string::npos) {
        std::fprintf(stderr, "dtype contract did not contain the observed graph evidence\n");
        return 1;
    }
    return 0;
}
