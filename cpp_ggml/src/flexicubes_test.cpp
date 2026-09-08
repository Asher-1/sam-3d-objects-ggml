#include "flexicubes.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    constexpr int resolution = 2;
    constexpr int channels = 101;
    std::vector<float> features;
    std::vector<int32_t> coordinates;
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                coordinates.insert(coordinates.end(), {0, x, y, z});
                const size_t base = features.size();
                features.resize(base + channels, 0.0f);
                for (int corner = 0; corner < 8; ++corner) {
                    const int vertex_x = x + (corner & 1);
                    // The native extractor adds the official -1 / resolution
                    // SDF bias, so offset the desired plane accordingly.
                    features[base + corner] = (vertex_x == 0 ? -1.0f : 1.0f) + 0.5f;
                }
            }
        }
    }
    sam3d::FlexiCubesResult mesh;
    std::string error;
    if (!sam3d::decode_flexicubes(features.data(), coordinates.data(), 8, resolution, mesh, error)) {
        std::fprintf(stderr, "FlexiCubes extraction failed: %s\n", error.c_str());
        return 1;
    }
    if (!mesh.success() || mesh.positions.size() % 3 != 0 ||
        mesh.vertex_attributes.size() != mesh.positions.size() * 2 || mesh.indices.size() % 3 != 0) {
        std::fprintf(stderr, "FlexiCubes synthetic plane did not produce a valid triangle mesh\n");
        return 1;
    }
    for (float value : mesh.positions) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FlexiCubes emitted a non-finite position\n");
            return 1;
        }
    }
    for (uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size() / 3) {
            std::fprintf(stderr, "FlexiCubes emitted an out-of-range triangle index\n");
            return 1;
        }
    }
    return 0;
}
