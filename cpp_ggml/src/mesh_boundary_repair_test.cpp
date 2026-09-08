#include "mesh_postprocess.hpp"

#include <cstdio>
#include <string>

int main() {
    // Three sides of a tetrahedron leave one three-edge boundary. MeshFix's
    // `fillSmallBoundaries(55, true)` must close it without creating invalid
    // topology. The full real-mesh exact check lives in
    // scripts/verify_meshfix_reference.py because its fixture is intentionally
    // not committed to the source tree.
    sam3d::NativeMesh mesh;
    mesh.positions = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    mesh.indices = {
        0, 2, 1,
        0, 1, 3,
        1, 2, 3,
    };
    std::string error;
    if (!sam3d::repair_mesh_boundaries_official_meshfix(mesh, 55, true, error)) {
        std::fprintf(stderr, "MeshFix boundary repair failed: %s\n", error.c_str());
        return 1;
    }
    if (mesh.indices.size() / 3 != 4 || mesh.positions.size() / 3 != 4) {
        std::fprintf(stderr, "MeshFix repair emitted unexpected %zu vertices / %zu triangles\n",
                     mesh.positions.size() / 3, mesh.indices.size() / 3);
        return 1;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size() / 3) {
            std::fprintf(stderr, "MeshFix repair emitted an out-of-range index\n");
            return 1;
        }
    }
    return 0;
}
