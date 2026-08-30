#include "asset_io.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    sam3d::NativeMesh mesh;
    mesh.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    mesh.normals = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    mesh.texcoords = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    mesh.colors = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    mesh.indices = {0, 1, 2};
    mesh.material.base_color_texture.width = 1;
    mesh.material.base_color_texture.height = 1;
    mesh.material.base_color_texture.rgba = {255, 64, 32, 255};

    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "sam3dggml_asset_io_test.glb";
    std::string error;
    if (!sam3d::write_pbr_glb(output.string(), mesh, error)) {
        std::fprintf(stderr, "write_pbr_glb failed: %s\n", error.c_str());
        return 1;
    }

    std::ifstream stream(output, std::ios::binary);
    std::array<unsigned char, 12> header{};
    stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    const bool valid = stream.gcount() == static_cast<std::streamsize>(header.size()) &&
                       header[0] == 'g' && header[1] == 'l' && header[2] == 'T' && header[3] == 'F';
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
    if (!valid) {
        std::fprintf(stderr, "GLB header is invalid\n");
        return 1;
    }
    return 0;
}
