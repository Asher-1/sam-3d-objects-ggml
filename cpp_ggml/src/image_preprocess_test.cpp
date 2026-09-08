#include "image_preprocess.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

int main() {
    sam3d::RgbaImage image;
    image.width = 6;
    image.height = 4;
    image.rgba.resize(static_cast<size_t>(image.width) * image.height * 4, 0);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const size_t pixel = (static_cast<size_t>(y) * image.width + x) * 4;
            image.rgba[pixel + 0] = static_cast<uint8_t>(20 * x);
            image.rgba[pixel + 1] = static_cast<uint8_t>(30 * y);
            image.rgba[pixel + 2] = 100;
            image.rgba[pixel + 3] = (x >= 1 && x <= 4 && y >= 1 && y <= 3) ? 255 : 0;
        }
    }
    std::vector<float> pointmap(static_cast<size_t>(3) * image.width * image.height);
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                pointmap[(static_cast<size_t>(channel) * image.height + y) * image.width + x] =
                    static_cast<float>(channel * 10 + x + y);
            }
        }
    }
    sam3d::NativePreprocessConfig config;
    config.target_size = 8;
    sam3d::NativeConditionInputs output;
    std::string error;
    if (!sam3d::preprocess_ss_conditions(image, pointmap, image.width, image.height,
                                         config, output, error)) {
        std::fprintf(stderr, "preprocess failed: %s\n", error.c_str());
        return 1;
    }
    const size_t image_size = static_cast<size_t>(3) * config.target_size * config.target_size;
    const size_t mask_size = static_cast<size_t>(config.target_size) * config.target_size;
    if (output.width != config.target_size || output.height != config.target_size ||
        output.image.size() != image_size || output.rgb_image.size() != image_size ||
        output.pointmap.size() != image_size || output.rgb_pointmap.size() != image_size ||
        output.mask.size() != mask_size || output.rgb_image_mask.size() != mask_size) {
        std::fprintf(stderr, "unexpected native preprocessing output dimensions\n");
        return 1;
    }
    if (!std::isfinite(output.pointmap_scale[0]) || output.pointmap_scale[0] <= 0.0f ||
        output.pointmap_scale[0] != output.pointmap_scale[1] ||
        output.pointmap_scale[1] != output.pointmap_scale[2]) {
        std::fprintf(stderr, "invalid ObjectCentricSSI scale\n");
        return 1;
    }
    const auto active = [](const std::vector<float>& mask) {
        size_t count = 0;
        for (float value : mask) count += value > 0.5f;
        return count;
    };
    if (active(output.mask) == 0 || active(output.rgb_image_mask) == 0) {
        std::fprintf(stderr, "mask was lost during preprocessing\n");
        return 1;
    }
    return 0;
}
