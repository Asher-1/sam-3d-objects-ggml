#include "mesh_uv.hpp"

#include <cstdio>
#include <string>

int main() {
    sam3d::NativeMesh mesh;
    mesh.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    mesh.colors = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    mesh.vertex_attributes = {
        1.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.3f,
        0.0f, 1.0f, 0.0f, 0.4f, 0.5f, 0.6f,
        0.0f, 0.0f, 1.0f, 0.7f, 0.8f, 0.9f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.1f, 1.2f,
    };
    mesh.indices = {0, 1, 2, 0, 2, 3};
    std::string error;
    if (!sam3d::parameterize_mesh_xatlas(mesh, error)) {
        std::fprintf(stderr, "xatlas UV test failed: %s\n", error.c_str());
        return 1;
    }
    if (mesh.texcoords.size() != mesh.positions.size() / 3 * 2 || mesh.indices.size() != 6 ||
        mesh.colors.size() != mesh.positions.size() ||
        mesh.vertex_attributes.size() != mesh.positions.size() * 2) {
        std::fprintf(stderr, "xatlas UV test emitted invalid remapped attributes\n");
        return 1;
    }
    return 0;
}
