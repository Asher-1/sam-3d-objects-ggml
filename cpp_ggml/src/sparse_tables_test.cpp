#include "sparse_ops.hpp"

#include <cstdio>
#include <vector>

int main() {
    // Three sparse cells make the mask ordering observable: the disconnected
    // cell has only the center bit, while the adjacent cells have asymmetric
    // neighbour bitsets. The expected permutation is the stable sort used by
    // spconv's implicit-GEMM indice generator.
    const int32_t coords[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 3, 0, 0,
    };
    sam3d::SlatTables tables;
    if (!tables.build(coords, 3, 1024)) {
        std::fprintf(stderr, "SlatTables::build unexpectedly failed\n");
        return 1;
    }

    const std::vector<int32_t> expected_masks = {
        1 << 13,
        (1 << 4) | (1 << 13),
        (1 << 13) | (1 << 22),
    };
    const std::vector<int32_t> expected_argsort = {2, 1, 0};
    if (tables.conv_fine_mask != expected_masks ||
        tables.conv_fine_argsort != expected_argsort) {
        std::fprintf(stderr, "spconv mask sort/permutation is not reference compatible\n");
        return 1;
    }

    for (size_t i = 1; i < tables.conv_fine_mask.size(); ++i) {
        if (tables.conv_fine_mask[i - 1] > tables.conv_fine_mask[i]) {
            std::fprintf(stderr, "spconv masks are not sorted\n");
            return 1;
        }
    }
    return 0;
}
