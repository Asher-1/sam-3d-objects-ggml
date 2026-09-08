#include "sparse_ops.hpp"

#include <cstdio>

int main() {
    const int32_t input[] = {0, 1, 2, 3};
    sam3d::MeshTables tables;
    if (!tables.build(input, 1, 64)) {
        std::fprintf(stderr, "MeshTables::build unexpectedly failed\n");
        return 1;
    }
    if (tables.levels[0].n != 1 || tables.levels[1].n != 8 || tables.levels[2].n != 64 ||
        tables.levels[1].resolution != 128 || tables.levels[2].resolution != 256) {
        std::fprintf(stderr, "MeshTables subdivision sizes are incorrect\n");
        return 1;
    }

    const int32_t* first = tables.levels[2].coords.data();
    const int32_t* last = first + 63 * 4;
    // The first and final 256^3 cells demonstrate the exact torch.nonzero
    // (x, y, z) child ordering used by SparseSubdivide.
    if (first[0] != 0 || first[1] != 4 || first[2] != 8 || first[3] != 12 ||
        last[0] != 0 || last[1] != 7 || last[2] != 11 || last[3] != 15) {
        std::fprintf(stderr, "MeshTables child coordinates are not torch-order compatible\n");
        return 1;
    }

    const auto& conv = tables.levels[2].conv;
    for (int64_t row = 0; row < tables.levels[2].n; ++row) {
        const int32_t center = conv[static_cast<size_t>(row) * 27 + 13];
        if (center != row) {
            std::fprintf(stderr, "MeshTables convolution center mapping is incorrect\n");
            return 1;
        }
    }
    return 0;
}
