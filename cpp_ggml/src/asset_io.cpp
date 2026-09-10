#include "asset_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <jpeglib.h>

#include <algorithm>
#include <array>
#include <csetjmp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>

namespace sam3d {
namespace {

constexpr uint32_t kGlbMagic = 0x46546C67;      // glTF
constexpr uint32_t kGlbVersion = 2;
constexpr uint32_t kJsonChunk = 0x4E4F534A;     // JSON
constexpr uint32_t kBinChunk = 0x004E4942;      // BIN\0

void append_bytes(std::vector<uint8_t>& dst, const void* data, size_t size) {
    const auto* first = static_cast<const uint8_t*>(data);
    dst.insert(dst.end(), first, first + size);
}

template <typename T>
void append_values(std::vector<uint8_t>& dst, const std::vector<T>& values) {
    if (!values.empty()) append_bytes(dst, values.data(), values.size() * sizeof(T));
}

void pad4(std::vector<uint8_t>& dst, uint8_t value = 0) {
    while (dst.size() % 4 != 0) dst.push_back(value);
}

void append_u32(std::vector<uint8_t>& dst, uint32_t value) {
    append_bytes(dst, &value, sizeof(value));
}

struct PngSink {
    std::vector<uint8_t> bytes;
};

struct JpegErrorManager {
    jpeg_error_mgr base;
    jmp_buf jump_buffer;
    char message[JMSG_LENGTH_MAX]{};
};

extern "C" void on_jpeg_error(j_common_ptr cinfo) {
    auto* error = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, error->message);
    longjmp(error->jump_buffer, 1);
}

bool has_jpeg_signature(const std::string& path) {
    std::array<unsigned char, 2> signature{};
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    const bool is_jpeg = std::fread(signature.data(), 1, signature.size(), file) ==
                             signature.size() &&
                         signature[0] == 0xFF && signature[1] == 0xD8;
    std::fclose(file);
    return is_jpeg;
}

bool load_jpeg_rgba(const std::string& path, RgbaImage& out, std::string& error) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open JPEG image '" + path + "'";
        return false;
    }

    jpeg_decompress_struct decompressor{};
    JpegErrorManager jpeg_error{};
    std::vector<JSAMPLE> scanline;
    bool created = false;
    decompressor.err = jpeg_std_error(&jpeg_error.base);
    jpeg_error.base.error_exit = on_jpeg_error;
    if (setjmp(jpeg_error.jump_buffer) != 0) {
        if (created) jpeg_destroy_decompress(&decompressor);
        std::fclose(file);
        error = "cannot decode JPEG image '" + path + "': " + jpeg_error.message;
        return false;
    }

    jpeg_create_decompress(&decompressor);
    created = true;
    jpeg_stdio_src(&decompressor, file);
    jpeg_read_header(&decompressor, TRUE);
    // The official input loader returns eight-bit RGB for JPEG. Request that
    // color space explicitly instead of accepting a library-selected layout.
    decompressor.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decompressor);
    if (decompressor.output_width == 0 || decompressor.output_height == 0 ||
        decompressor.output_components != 3) {
        error = "JPEG decoder did not produce RGB pixels for '" + path + "'";
        jpeg_destroy_decompress(&decompressor);
        std::fclose(file);
        return false;
    }

    const size_t width = decompressor.output_width;
    const size_t height = decompressor.output_height;
    if (width > std::numeric_limits<size_t>::max() / height / 4) {
        error = "JPEG image dimensions overflow the RGBA buffer for '" + path + "'";
        jpeg_destroy_decompress(&decompressor);
        std::fclose(file);
        return false;
    }
    out.width = static_cast<int>(width);
    out.height = static_cast<int>(height);
    out.rgba.resize(width * height * 4);
    scanline.resize(width * 3);
    while (decompressor.output_scanline < decompressor.output_height) {
        JSAMPROW row = scanline.data();
        jpeg_read_scanlines(&decompressor, &row, 1);
        const size_t y = static_cast<size_t>(decompressor.output_scanline - 1);
        uint8_t* destination = out.rgba.data() + y * width * 4;
        for (size_t x = 0; x < width; ++x) {
            destination[x * 4 + 0] = scanline[x * 3 + 0];
            destination[x * 4 + 1] = scanline[x * 3 + 1];
            destination[x * 4 + 2] = scanline[x * 3 + 2];
            destination[x * 4 + 3] = 255;
        }
    }
    jpeg_finish_decompress(&decompressor);
    jpeg_destroy_decompress(&decompressor);
    std::fclose(file);
    return true;
}

void write_png_callback(void* context, void* data, int size) {
    auto* sink = static_cast<PngSink*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    sink->bytes.insert(sink->bytes.end(), bytes, bytes + size);
}

bool png_from_rgba(const RgbaImage& image, std::vector<uint8_t>& out, std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() != static_cast<size_t>(image.width) * image.height * 4) {
        error = "PBR texture must be a non-empty RGBA image";
        return false;
    }
    PngSink sink;
    const int stride = image.width * 4;
    if (!stbi_write_png_to_func(write_png_callback, &sink, image.width, image.height, 4,
                                image.rgba.data(), stride)) {
        error = "stb_image_write failed to encode PBR texture";
        return false;
    }
    out = std::move(sink.bytes);
    return true;
}

std::string json_number(float value) {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(7);
    stream << value;
    return stream.str();
}

}  // namespace

bool load_rgba_image(const std::string& path, RgbaImage& out, std::string& error) {
    if (has_jpeg_signature(path)) return load_jpeg_rgba(path, out, error);

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (pixels == nullptr) {
        error = "cannot decode image '" + path + "': " + stbi_failure_reason();
        return false;
    }
    const size_t size = static_cast<size_t>(width) * height * 4;
    out.width = width;
    out.height = height;
    out.rgba.assign(pixels, pixels + size);
    stbi_image_free(pixels);
    return true;
}

bool write_rgba_png(const std::string& path, const RgbaImage& image, std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() != static_cast<size_t>(image.width) * image.height * 4) {
        error = "PNG output must be a non-empty RGBA image";
        return false;
    }
    if (!stbi_write_png(path.c_str(), image.width, image.height, 4, image.rgba.data(),
                        image.width * 4)) {
        error = "stb_image_write failed to write PNG '" + path + "'";
        return false;
    }
    return true;
}

bool write_pbr_glb(const std::string& path, const NativeMesh& mesh, std::string& error) {
    const size_t vertex_count = mesh.positions.size() / 3;
    if (vertex_count == 0 || mesh.positions.size() % 3 != 0) {
        error = "GLB mesh requires non-empty xyz positions";
        return false;
    }
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        error = "GLB mesh requires triangle indices";
        return false;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count) {
            error = "GLB index is outside the vertex array";
            return false;
        }
    }
    if (!mesh.normals.empty() && mesh.normals.size() != mesh.positions.size()) {
        error = "GLB normals must contain one xyz tuple per vertex";
        return false;
    }
    if (!mesh.texcoords.empty() && mesh.texcoords.size() != vertex_count * 2) {
        error = "GLB texture coordinates must contain one uv tuple per vertex";
        return false;
    }
    if (!mesh.colors.empty() && mesh.colors.size() != vertex_count * 3) {
        error = "GLB vertex colors must contain one rgb tuple per vertex";
        return false;
    }

    std::vector<uint8_t> png;
    const bool has_texture = !mesh.material.base_color_texture.rgba.empty();
    if (has_texture && mesh.texcoords.empty()) {
        error = "GLB base-color texture requires one uv tuple per vertex";
        return false;
    }
    if (has_texture && !png_from_rgba(mesh.material.base_color_texture, png, error)) return false;

    std::vector<uint8_t> bin;
    const uint32_t positions_offset = static_cast<uint32_t>(bin.size());
    append_values(bin, mesh.positions);
    pad4(bin);
    const uint32_t normals_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.normals.empty()) append_values(bin, mesh.normals);
    pad4(bin);
    const uint32_t texcoords_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.texcoords.empty()) append_values(bin, mesh.texcoords);
    pad4(bin);
    const uint32_t colors_offset = static_cast<uint32_t>(bin.size());
    if (!mesh.colors.empty()) append_values(bin, mesh.colors);
    pad4(bin);
    const uint32_t indices_offset = static_cast<uint32_t>(bin.size());
    append_values(bin, mesh.indices);
    pad4(bin);
    const uint32_t texture_offset = static_cast<uint32_t>(bin.size());
    if (has_texture) append_values(bin, png);
    pad4(bin);

    std::array<float, 3> min_pos{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    std::array<float, 3> max_pos{
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (size_t i = 0; i < mesh.positions.size(); i += 3) {
        for (size_t c = 0; c < 3; ++c) {
            min_pos[c] = std::min(min_pos[c], mesh.positions[i + c]);
            max_pos[c] = std::max(max_pos[c], mesh.positions[i + c]);
        }
    }

    std::ostringstream json;
    json << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"sam3d-ggml-native\"},"
         << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
         << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
    int accessor_count = 1;
    if (!mesh.normals.empty()) json << ",\"NORMAL\":" << accessor_count++;
    if (!mesh.texcoords.empty()) json << ",\"TEXCOORD_0\":" << accessor_count++;
    if (!mesh.colors.empty()) json << ",\"COLOR_0\":" << accessor_count++;
    const int indices_accessor = accessor_count++;
    json << "},\"indices\":" << indices_accessor << ",\"material\":0,\"mode\":4}]}],";

    json << "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":["
         << json_number(mesh.material.base_color[0]) << "," << json_number(mesh.material.base_color[1])
         << "," << json_number(mesh.material.base_color[2]) << "," << json_number(mesh.material.base_color[3])
         << "],\"metallicFactor\":" << json_number(mesh.material.metallic)
         << ",\"roughnessFactor\":" << json_number(mesh.material.roughness);
    if (has_texture) json << ",\"baseColorTexture\":{\"index\":0}";
    json << "}}],";
    const int texture_view = 2 + (!mesh.normals.empty() ? 1 : 0) +
                             (!mesh.texcoords.empty() ? 1 : 0) +
                             (!mesh.colors.empty() ? 1 : 0);
    if (has_texture) {
        json << "\"textures\":[{\"source\":0}],\"images\":[{\"bufferView\":"
             << texture_view << ",\"mimeType\":\"image/png\"}],";
    }
    json << "\"buffers\":[{\"byteLength\":" << bin.size() << "}],\"bufferViews\":["
         << "{\"buffer\":0,\"byteOffset\":" << positions_offset << ",\"byteLength\":"
         << mesh.positions.size() * sizeof(float) << ",\"target\":34962}";
    int view_index = 1;
    if (!mesh.normals.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << normals_offset << ",\"byteLength\":"
             << mesh.normals.size() * sizeof(float) << ",\"target\":34962}";
        ++view_index;
    }
    if (!mesh.texcoords.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << texcoords_offset << ",\"byteLength\":"
             << mesh.texcoords.size() * sizeof(float) << ",\"target\":34962}";
        ++view_index;
    }
    if (!mesh.colors.empty()) {
        json << ",{\"buffer\":0,\"byteOffset\":" << colors_offset << ",\"byteLength\":"
             << mesh.colors.size() * sizeof(float) << ",\"target\":34962}";
        ++view_index;
    }
    const int indices_view = view_index++;
    json << ",{\"buffer\":0,\"byteOffset\":" << indices_offset << ",\"byteLength\":"
         << mesh.indices.size() * sizeof(uint32_t) << ",\"target\":34963}";
    if (has_texture) {
        json << ",{\"buffer\":0,\"byteOffset\":" << texture_offset << ",\"byteLength\":"
             << png.size() << "}";
    }
    json << "],\"accessors\":["
         << "{\"bufferView\":0,\"componentType\":5126,\"count\":" << vertex_count
         << ",\"type\":\"VEC3\",\"min\":[" << json_number(min_pos[0]) << "," << json_number(min_pos[1])
         << "," << json_number(min_pos[2]) << "],\"max\":[" << json_number(max_pos[0]) << ","
         << json_number(max_pos[1]) << "," << json_number(max_pos[2]) << "]}";
    int next_view = 1;
    if (!mesh.normals.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC3\"}";
    }
    if (!mesh.texcoords.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC2\"}";
    }
    if (!mesh.colors.empty()) {
        json << ",{\"bufferView\":" << next_view++
             << ",\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC3\"}";
    }
    json << ",{\"bufferView\":" << indices_view
         << ",\"componentType\":5125,\"count\":" << mesh.indices.size() << ",\"type\":\"SCALAR\"}]}";

    std::string json_bytes = json.str();
    std::vector<uint8_t> glb;
    std::vector<uint8_t> json_chunk(json_bytes.begin(), json_bytes.end());
    pad4(json_chunk, ' ');
    append_u32(glb, kGlbMagic);
    append_u32(glb, kGlbVersion);
    append_u32(glb, static_cast<uint32_t>(12 + 8 + json_chunk.size() + 8 + bin.size()));
    append_u32(glb, static_cast<uint32_t>(json_chunk.size()));
    append_u32(glb, kJsonChunk);
    append_values(glb, json_chunk);
    append_u32(glb, static_cast<uint32_t>(bin.size()));
    append_u32(glb, kBinChunk);
    append_values(glb, bin);

    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        error = "cannot open GLB output '" + path + "'";
        return false;
    }
    const size_t written = std::fwrite(glb.data(), 1, glb.size(), file);
    const int close_status = std::fclose(file);
    if (written != glb.size() || close_status != 0) {
        error = "failed to write complete GLB output '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace sam3d
