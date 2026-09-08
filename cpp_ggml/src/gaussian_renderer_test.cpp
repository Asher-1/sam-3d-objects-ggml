#include "gaussian_renderer.hpp"

#include <cstdio>
#include <fstream>
#include <limits>

int main() {
    const char* ply_path = "sam3d-gaussian-opacity-limit-test.ply";
    {
        std::ofstream ply(ply_path, std::ios::binary);
        ply << "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
               "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
               "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
               "property float rot_2\nproperty float rot_3\nend_header\n";
        const float payload[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                                 std::numeric_limits<float>::infinity(), -3.0f, -3.0f,
                                 -3.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        ply.write(reinterpret_cast<const char*>(payload), sizeof(payload));
    }
    sam3d::GaussianSplatSet loaded;
    std::string error;
    if (!sam3d::load_gaussian_splat_ply(ply_path, loaded, error) ||
        loaded.opacities.size() != 1 || loaded.opacities[0] != 1.0f) {
        std::fprintf(stderr, "official +inf opacity loading failed: %s\n", error.c_str());
        std::remove(ply_path);
        return 1;
    }
    std::remove(ply_path);

    sam3d::GaussianSplatSet splats;
    splats.positions = {0.0f, 0.0f, 0.0f};
    splats.sh0 = {1.0f, 1.0f, 1.0f};
    splats.opacities = {0.9f};
    splats.scales = {0.12f, 0.12f, 0.12f};
    splats.rotations = {1.0f, 0.0f, 0.0f, 0.0f};

    sam3d::GaussianRenderConfig config;
    config.width = 32;
    config.height = 32;
    const std::vector<sam3d::GaussianCamera> cameras =
        sam3d::make_gaussian_hammersley_cameras(1, config, error);
    if (cameras.size() != 1) {
        std::fprintf(stderr, "camera construction failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<sam3d::RgbaImage> images;
    if (!sam3d::render_gaussian_views_cuda(splats, cameras, config, images, error)) {
        std::fprintf(stderr, "Gaussian renderer failed: %s\n", error.c_str());
        return 1;
    }
    if (images.size() != 1 || images[0].rgba.size() != 32u * 32u * 4u) {
        std::fprintf(stderr, "Gaussian renderer emitted an invalid image\n");
        return 1;
    }
    bool has_color = false;
    for (size_t index = 0; index < images[0].rgba.size(); index += 4) {
        has_color = has_color || images[0].rgba[index] != 0 || images[0].rgba[index + 1] != 0 ||
                    images[0].rgba[index + 2] != 0;
    }
    if (!has_color) {
        std::fprintf(stderr, "Gaussian renderer produced only the black background\n");
        return 1;
    }
    return 0;
}
