#include "texture_baker.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    sam3d::TextureBakeConfig schedule_config;
    schedule_config.steps = 4;
    schedule_config.learning_rate = 0.01f;
    schedule_config.eta_min = 0.001f;
    const auto expected_schedule = [](int step) {
        constexpr double kPi = 3.14159265358979323846;
        return static_cast<float>(0.001 + 0.5 * (0.01 - 0.001) *
                                  (1.0 + std::cos(kPi * static_cast<double>(step) / 4.0)));
    };
    const float update_zero = sam3d::texture_bake_learning_rate_for_update(schedule_config, 0);
    const float update_one = sam3d::texture_bake_learning_rate_for_update(schedule_config, 1);
    const float update_two = sam3d::texture_bake_learning_rate_for_update(schedule_config, 2);
    if (std::fabs(update_zero - schedule_config.learning_rate) > 1.0e-7f ||
        std::fabs(update_one - expected_schedule(0)) > 1.0e-7f ||
        std::fabs(update_two - expected_schedule(1)) > 1.0e-7f) {
        std::fprintf(stderr, "texture bake learning-rate schedule does not match update-then-schedule semantics\n");
        return 1;
    }

    sam3d::NativeMesh mesh;
    mesh.positions = {-0.8f, -0.8f, 1.0f, 0.8f, -0.8f, 1.0f, 0.0f, 0.8f, 1.0f};
    mesh.texcoords = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
    mesh.indices = {0, 1, 2};

    sam3d::RgbaImage observation;
    observation.width = 32;
    observation.height = 32;
    observation.rgba.resize(32u * 32u * 4u, 255);
    for (size_t index = 0; index < observation.rgba.size(); index += 4) {
        observation.rgba[index] = 196;
        observation.rgba[index + 1] = 101;
        observation.rgba[index + 2] = 47;
    }

    sam3d::GaussianCamera camera;
    camera.extrinsics = {1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f};
    camera.intrinsics = {0.5f, 0.0f, 0.5f,
                         0.0f, 0.5f, 0.5f,
                         0.0f, 0.0f, 1.0f};
    sam3d::TextureBakeConfig config;
    config.texture_size = 32;
    config.steps = 2;
    config.random_seed = 0;
    sam3d::RgbaImage texture;
    std::vector<uint8_t> holes;
    std::string error;
    if (!sam3d::bake_texture_official_cuda(mesh, {observation}, {camera}, config, texture, &holes,
                                           nullptr, error)) {
        std::fprintf(stderr, "native texture bake failed: %s\n", error.c_str());
        return 1;
    }
    if (texture.width != 32 || texture.height != 32 || texture.rgba.size() != 32u * 32u * 4u ||
        holes.size() != 32u * 32u) {
        std::fprintf(stderr, "native texture bake emitted an invalid atlas\n");
        return 1;
    }
    bool has_color = false;
    bool has_coverage = false;
    for (size_t pixel = 0; pixel < holes.size(); ++pixel) {
        has_coverage = has_coverage || holes[pixel] == 0;
        has_color = has_color || texture.rgba[pixel * 4] != 0 || texture.rgba[pixel * 4 + 1] != 0 ||
                    texture.rgba[pixel * 4 + 2] != 0;
        if (texture.rgba[pixel * 4 + 3] != 255) {
            std::fprintf(stderr, "native texture bake did not normalize atlas alpha\n");
            return 1;
        }
    }
    if (!has_coverage || !has_color) {
        std::fprintf(stderr, "native texture bake produced no covered color texels\n");
        return 1;
    }
    return 0;
}
