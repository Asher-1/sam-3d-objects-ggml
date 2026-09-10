#include "native_pbr_pipeline.hpp"
#include "mesh_uv.hpp"

#include <cmath>
#include <cstdio>
#include <string>

int main() {
    // Both incident corners are right angles, but the second face has eight
    // times the area. Area-weighted normals would be close to +X; trimesh's
    // angle-weighted result is exactly halfway between +X and +Z.
    sam3d::NativeMesh normal_fixture;
    normal_fixture.positions = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 4.0f,
        9.0f, 9.0f, 9.0f,
    };
    normal_fixture.indices = {0, 1, 2, 0, 3, 4, 0, 1, 1};
    sam3d::compute_official_angle_weighted_vertex_normals(normal_fixture);
    constexpr float expected_component = 0.70710677f;
    constexpr float tolerance = 1.0e-5f;
    if (std::fabs(normal_fixture.normals[0] - expected_component) > tolerance ||
        std::fabs(normal_fixture.normals[1]) > tolerance ||
        std::fabs(normal_fixture.normals[2] - expected_component) > tolerance ||
        normal_fixture.normals[15] != 0.0f || normal_fixture.normals[16] != 0.0f ||
        normal_fixture.normals[17] != 0.0f) {
        std::fprintf(stderr, "angle-weighted normal regression failed\n");
        return 1;
    }

    sam3d::NativeMesh mesh;
    // Decoder-space Z-up mesh centered at the native Gaussian renderer's
    // origin. The test intentionally marks it clean: GPL MeshFix is tested
    // separately under its explicit build option.
    mesh.positions = {-0.6f, -0.6f, 0.0f, 0.6f, -0.6f, 0.0f, 0.0f, 0.6f, 0.0f};
    mesh.indices = {0, 1, 2};

    sam3d::GaussianSplatSet splats;
    splats.positions = {0.0f, 0.0f, 0.0f};
    splats.sh0 = {1.0f, 0.5f, 0.25f};
    splats.opacities = {0.9f};
    splats.scales = {0.12f, 0.12f, 0.12f};
    splats.rotations = {1.0f, 0.0f, 0.0f, 0.0f};

    sam3d::NativePbrPipelineConfig config;
    config.input_mesh_already_clean = true;
    config.render_views = 1;
    config.render_resolution = 32;
    config.texture_size = 32;
    config.texture_steps = 1;

    sam3d::NativePbrPipelineResult result;
    std::string error;
    sam3d::NativeMesh bake_coordinates = mesh;
    if (!sam3d::parameterize_mesh_xatlas(bake_coordinates, error)) {
        std::fprintf(stderr, "reference UV parameterization failed: %s\n", error.c_str());
        return 1;
    }
    if (!sam3d::assemble_official_pbr_cuda(mesh, splats, config, result, error)) {
        std::fprintf(stderr, "native PBR pipeline failed: %s\n", error.c_str());
        return 1;
    }
    if (result.mesh.texcoords.size() != bake_coordinates.texcoords.size()) {
        std::fprintf(stderr, "native PBR GLB export changed UV count\n");
        return 1;
    }
    for (size_t uv = 0; uv < result.mesh.texcoords.size(); uv += 2) {
        if (result.mesh.texcoords[uv] != bake_coordinates.texcoords[uv] ||
            result.mesh.texcoords[uv + 1] != 1.0f - bake_coordinates.texcoords[uv + 1]) {
            std::fprintf(stderr, "native PBR GLB export did not flip V after baking\n");
            return 1;
        }
    }
    if (result.mesh.material.metallic != 1.0f || result.mesh.material.roughness != 1.0f) {
        std::fprintf(stderr, "native PBR material differs from official glTF effective defaults\n");
        return 1;
    }
    if (result.rendered_views != 1 || result.pre_meshfix_faces != 1 || result.final_faces != 1 ||
        result.mesh.positions.size() != 9 || result.mesh.texcoords.size() != 6 ||
        result.mesh.normals.size() != 9 ||
        result.mesh.material.base_color_texture.width != 32 ||
        result.mesh.material.base_color_texture.height != 32 ||
        result.mesh.material.base_color_texture.rgba.size() != 32u * 32u * 4u) {
        std::fprintf(stderr, "native PBR pipeline emitted an invalid textured mesh\n");
        return 1;
    }
    for (size_t vertex = 0; vertex < result.mesh.normals.size(); vertex += 3) {
        if (result.mesh.normals[vertex] != 0.0f || result.mesh.normals[vertex + 1] != 1.0f ||
            result.mesh.normals[vertex + 2] != 0.0f) {
            std::fprintf(stderr, "native PBR pipeline did not rotate normals with Z-up to Y-up mesh\n");
            return 1;
        }
    }
    bool has_color = false;
    for (size_t offset = 0; offset < result.mesh.material.base_color_texture.rgba.size(); offset += 4) {
        has_color = has_color || result.mesh.material.base_color_texture.rgba[offset] != 0 ||
                    result.mesh.material.base_color_texture.rgba[offset + 1] != 0 ||
                    result.mesh.material.base_color_texture.rgba[offset + 2] != 0;
    }
    if (!has_color) {
        std::fprintf(stderr, "native PBR pipeline emitted an all-black texture\n");
        return 1;
    }
    return 0;
}
