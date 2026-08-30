// Native image and GLB asset I/O used by the raw-image C++ pipeline.
//
// These routines deliberately have no Python, Torch, cuDNN, or renderer
// dependency.  The model stages exchange normal C++ containers with this
// layer, so the final asset cannot silently fall back to Python tooling.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

struct RgbaImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
};

struct PbrMaterial {
    float base_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 1.0f;
    // Optional RGBA texture.  When non-empty it is encoded in the GLB BIN
    // chunk as PNG and referenced by the PBR base-color texture slot.
    RgbaImage base_color_texture;
};

struct NativeMesh {
    std::vector<float> positions;  // xyz, one triplet per vertex
    std::vector<float> normals;    // xyz, optional
    std::vector<float> texcoords;  // uv, optional
    // Optional linear RGB vertex colors. This is the official non-baked mesh
    // fallback: glTF combines COLOR_0 with the PBR base-color material.
    std::vector<float> colors;
    std::vector<uint32_t> indices; // triangles
    PbrMaterial material;
};

// Decode a PNG/JPEG/etc. source into four-channel RGBA pixels.  Keeping the
// decoded form explicit makes raw image and mask handling deterministic.
bool load_rgba_image(const std::string& path, RgbaImage& out, std::string& error);

// Encode the mesh as GLB 2.0.  The resulting file has a triangle primitive,
// PBR metallic-roughness material and (when supplied) an embedded PNG texture.
bool write_pbr_glb(const std::string& path, const NativeMesh& mesh, std::string& error);

}  // namespace sam3d
