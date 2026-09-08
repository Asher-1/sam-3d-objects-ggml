#include "mesh_postprocess.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

sam3d::NativeMesh make_tetrahedron(float scale) {
    sam3d::NativeMesh mesh;
    mesh.positions = {
        0.0f, 0.0f, 0.0f,
        scale, 0.0f, 0.0f,
        0.0f, scale, 0.0f,
        0.0f, 0.0f, scale,
    };
    mesh.indices = {
        0, 2, 1,
        0, 1, 3,
        1, 2, 3,
        2, 0, 3,
    };
    return mesh;
}

bool test_visibility_mincut(float scale, size_t expected_removed_faces) {
    sam3d::NativeMesh mesh = make_tetrahedron(scale);
    // The official component threshold is clamped to 0.5. Face 0 is the
    // source, faces 2/3 are targets, and face 1 checks the weighted dual cut.
    const std::vector<float> visibility = {0.0f, 0.1f, 0.8f, 0.9f};
    sam3d::MeshVisibilityFilterStats stats;
    std::string error;
    if (!sam3d::filter_invisible_faces_official(mesh, visibility, {}, &stats, error)) {
        std::fprintf(stderr, "visibility/mincut test failed: %s\n", error.c_str());
        return false;
    }
    if (stats.invisible_face_count != 1 || stats.removed_face_count != expected_removed_faces ||
        mesh.indices.size() / 3 != 4 - expected_removed_faces) {
        std::fprintf(stderr, "visibility/mincut removed %zu faces; expected %zu\n",
                     stats.removed_face_count, expected_removed_faces);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    constexpr int kWidth = 16;
    sam3d::NativeMesh mesh;
    for (int y = 0; y < kWidth; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            mesh.positions.insert(mesh.positions.end(), {static_cast<float>(x), static_cast<float>(y),
                                                          0.05f * static_cast<float>((x + y) % 3)});
        }
    }
    for (int y = 0; y + 1 < kWidth; ++y) {
        for (int x = 0; x + 1 < kWidth; ++x) {
            const uint32_t a = static_cast<uint32_t>(y * kWidth + x);
            const uint32_t b = a + 1;
            const uint32_t c = a + kWidth;
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, b, d, a, d, c});
        }
    }
    const size_t original_faces = mesh.indices.size() / 3;
    std::string error;
    if (!sam3d::simplify_mesh_official_vtk(mesh, {}, error)) {
        std::fprintf(stderr, "VTK simplify test failed: %s\n", error.c_str());
        return 1;
    }
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0 ||
        mesh.indices.size() / 3 >= original_faces) {
        std::fprintf(stderr, "VTK simplify test did not reduce the mesh\n");
        return 1;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size() / 3) {
            std::fprintf(stderr, "VTK simplify test emitted an invalid index\n");
            return 1;
        }
    }
    if (!test_visibility_mincut(0.1f, 1)) return 1;
    // The same cut is rejected when the new boundary proxy exceeds the
    // official 0.04 default. This guards against deleting a visible-scale cap.
    if (!test_visibility_mincut(1.0f, 0)) return 1;
    return 0;
}
