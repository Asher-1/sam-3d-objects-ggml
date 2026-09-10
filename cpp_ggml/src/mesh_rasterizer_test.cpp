#include "mesh_rasterizer.hpp"

#include <cstdio>
#include <string>

namespace {

sam3d::MeshRasterCamera identity_camera() {
    sam3d::MeshRasterCamera camera;
    camera.extrinsics = {1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f};
    camera.intrinsics = {0.5f, 0.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f};
    return camera;
}

}  // namespace

int main() {
    sam3d::NativeMesh mesh;
    // The first triangle is one unit closer to the camera than the identical
    // second triangle. Its UVs identify whether the depth test won.
    mesh.positions = {-0.5f, -0.5f, 1.0f, 0.5f, -0.5f, 1.0f, 0.0f, 0.5f, 1.0f,
                      -0.5f, -0.5f, 2.0f, 0.5f, -0.5f, 2.0f, 0.0f, 0.5f, 2.0f};
    mesh.texcoords = {0.1f, 0.2f, 0.9f, 0.2f, 0.5f, 0.8f,
                      0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
    mesh.indices = {0, 1, 2, 3, 4, 5};
    sam3d::MeshRasterConfig config;
    config.width = 32;
    config.height = 32;
    std::string error;
    sam3d::MeshRasterResult raster;
    if (!sam3d::rasterize_mesh_uv(mesh, identity_camera(), config, raster, error)) {
        std::fprintf(stderr, "mesh rasterizer failed: %s\n", error.c_str());
        return 1;
    }
    const size_t center = static_cast<size_t>(16 * config.width + 16);
    if (raster.coverage[center] == 0 || raster.face_ids[center] != 0 ||
        !(raster.uv[center * 2] > 0.1f && raster.uv[center * 2] < 0.9f) ||
        !(raster.uv[center * 2 + 1] > 0.2f && raster.uv[center * 2 + 1] < 0.8f)) {
        std::fprintf(stderr, "mesh rasterizer lost the front triangle or UV interpolation\n");
        return 1;
    }

    std::vector<float> visibility;
    if (!sam3d::mesh_visibility_frequency(mesh, 8, config, visibility, error) ||
        visibility.size() != 2 || visibility[0] < 0.0f || visibility[0] > 1.0f ||
        visibility[1] != 0.0f) {
        std::fprintf(stderr, "mesh visibility frequency failed: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
