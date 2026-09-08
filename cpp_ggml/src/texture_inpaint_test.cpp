#include "texture_inpaint.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    sam3d::RgbaImage texture;
    texture.width = 7;
    texture.height = 5;
    texture.rgba.resize(static_cast<size_t>(texture.width) * texture.height * 4);
    for (int y = 0; y < texture.height; ++y) {
        for (int x = 0; x < texture.width; ++x) {
            const size_t index = static_cast<size_t>(y * texture.width + x) * 4;
            texture.rgba[index] = static_cast<uint8_t>(x * 31);
            texture.rgba[index + 1] = static_cast<uint8_t>(y * 47);
            texture.rgba[index + 2] = static_cast<uint8_t>((x + y) * 17);
            texture.rgba[index + 3] = 19;
        }
    }
    const std::vector<uint8_t> original = texture.rgba;
    std::vector<uint8_t> holes(static_cast<size_t>(texture.width) * texture.height, 0);
    holes[static_cast<size_t>(2 * texture.width + 3)] = 1;
    holes[static_cast<size_t>(2 * texture.width + 4)] = 1;
    std::string error;
    if (!sam3d::inpaint_texture_telea(texture, holes, 3.0f, error)) {
        std::fprintf(stderr, "Telea inpaint failed: %s\n", error.c_str());
        return 1;
    }
    bool changed_hole = false;
    for (size_t pixel = 0; pixel < holes.size(); ++pixel) {
        const size_t index = pixel * 4;
        if (texture.rgba[index + 3] != 255) {
            std::fprintf(stderr, "Telea inpaint did not normalize output alpha\n");
            return 1;
        }
        if (holes[pixel] != 0) {
            changed_hole = changed_hole || texture.rgba[index] != original[index] ||
                           texture.rgba[index + 1] != original[index + 1] ||
                           texture.rgba[index + 2] != original[index + 2];
        } else if (texture.rgba[index] != original[index] ||
                   texture.rgba[index + 1] != original[index + 1] ||
                   texture.rgba[index + 2] != original[index + 2]) {
            std::fprintf(stderr, "Telea inpaint changed a known texel\n");
            return 1;
        }
    }
    if (!changed_hole) {
        std::fprintf(stderr, "Telea inpaint did not reconstruct either requested texel\n");
        return 1;
    }
    return 0;
}
