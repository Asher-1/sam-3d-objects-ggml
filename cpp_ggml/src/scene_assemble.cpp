#include "scene_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace sam3d {
namespace {

constexpr float kPi = 3.14159265358979323846f;
// GS_MIN_KERNEL in session.cpp: the decoder representation's
// 3d_filter_kernel_size (checkpoints/hf/slat_decoder_gs.yaml).
constexpr float kMinKernel = 0.0009f;

struct Vec3 {
    float x, y, z;
};

Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Same pytorch3d quaternion conventions as pose_decoder.cpp (wxyz, Hamilton
// product, unit-quaternion inverse = conjugate).
std::array<float, 4> quat_conjugate(const std::array<float, 4>& q) {
    return {q[0], -q[1], -q[2], -q[3]};
}

std::array<float, 4> quat_multiply(const std::array<float, 4>& a,
                                   const std::array<float, 4>& b) {
    const float aw = a[0], ax = a[1], ay = a[2], az = a[3];
    const float bw = b[0], bx = b[1], by = b[2], bz = b[3];
    return {
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    };
}

// pytorch3d.transforms.quaternion_to_matrix (row-major 3x3).
std::array<std::array<float, 3>, 3> quat_to_matrix(const std::array<float, 4>& q) {
    const float r = q[0], i = q[1], j = q[2], k = q[3];
    const float norm2 = r * r + i * i + j * j + k * k;
    const float two_s = norm2 == 0.0f ? std::numeric_limits<float>::infinity()
                                      : 2.0f / norm2;
    return {{
        {{1.0f - two_s * (j * j + k * k), two_s * (i * j - k * r),
          two_s * (i * k + j * r)}},
        {{two_s * (i * j + k * r), 1.0f - two_s * (i * i + k * k),
          two_s * (j * k - i * r)}},
        {{two_s * (i * k - j * r), two_s * (j * k + i * r),
          1.0f - two_s * (i * i + j * j)}},
    }};
}

// Scan a JSON text for "key"<whitespace>:<whitespace>[ and parse `count`
// nested-array floats. Keys are matched with their quotes so "rotation" never
// matches inside "rotation_wxyz", and only the first occurrence is consumed -
// the sam3d.native-pose.v2 "native" block repeats keys with different values.
bool extract_float_array(const std::string& text, const char* key, size_t count,
                         float* out) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) return false;
    size_t cursor = text.find('[', key_pos + needle.size());
    if (cursor == std::string::npos) return false;
    for (size_t index = 0; index < count; ++index) {
        const size_t start = text.find_first_not_of(" \t\r\n,[", cursor);
        if (start == std::string::npos) return false;
        const char* begin = text.c_str() + start;
        char* end = nullptr;
        const float value = std::strtof(begin, &end);
        if (end == begin) return false;
        out[index] = value;
        cursor = start + static_cast<size_t>(end - begin);
    }
    return true;
}

}  // namespace

bool load_scene_pose_json(const std::string& path, ScenePose& pose,
                          std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot open pose receipt: " + path;
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    if (!extract_float_array(text, "rotation", 4, pose.rotation_wxyz.data()) ||
        !extract_float_array(text, "translation", 3, pose.translation.data()) ||
        !extract_float_array(text, "scale", 3, pose.scale.data())) {
        error = "pose receipt is missing the official rotation/translation/scale keys: " + path;
        return false;
    }
    for (float value : pose.rotation_wxyz) {
        if (!std::isfinite(value)) {
            error = "pose receipt rotation is not finite: " + path;
            return false;
        }
    }
    for (float value : pose.translation) {
        if (!std::isfinite(value)) {
            error = "pose receipt translation is not finite: " + path;
            return false;
        }
    }
    return true;
}

bool load_scene_normalization_json(const std::string& path, float& inv_scale,
                                   std::array<float, 3>& center,
                                   std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot open normalization contract: " + path;
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    float buffer[3]{};
    if (!extract_float_array(text, "center", 3, buffer)) {
        error = "normalization contract is missing center: " + path;
        return false;
    }
    center = {buffer[0], buffer[1], buffer[2]};
    // "inv_scale" is a scalar, not an array: scan for the key, then the
    // colon, then the number (extract_float_array would wrongly latch onto
    // the '[' of the following "center" array).
    const size_t key_pos = text.find("\"inv_scale\"");
    if (key_pos == std::string::npos) {
        error = "normalization contract is missing inv_scale: " + path;
        return false;
    }
    const size_t colon = text.find(':', key_pos + 11);
    if (colon == std::string::npos) {
        error = "normalization contract inv_scale has no value: " + path;
        return false;
    }
    const size_t start = text.find_first_not_of(" \t\r\n", colon + 1);
    if (start == std::string::npos) {
        error = "normalization contract inv_scale has no value: " + path;
        return false;
    }
    char* end = nullptr;
    inv_scale = std::strtof(text.c_str() + start, &end);
    if (end == text.c_str() + start) {
        error = "normalization contract inv_scale is not a number: " + path;
        return false;
    }
    return true;
}

bool apply_scene_pose(GaussianSplatSet& splats, const ScenePose& pose,
                      float min_kernel, std::string& error) {
    if (splats.size() == 0) {
        error = "cannot apply a scene pose to an empty splat set";
        return false;
    }
    const auto rotation = quat_to_matrix(pose.rotation_wxyz);
    const std::array<float, 4> inverse = quat_conjugate(pose.rotation_wxyz);
    for (size_t index = 0; index < splats.size(); ++index) {
        float* position = &splats.positions[index * 3];
        const Vec3 point{position[0], position[1], position[2]};
        // pytorch3d Transform3d.transform_points multiplies the row vector by
        // the composed matrix WITHOUT transposing it, so the actual point
        // action of Transform3d().scale(s).rotate(R).translate(t) is
        //   p' = R^T @ diag(s) @ p + t  (scale first, then the inverse
        // rotation, then translate). Verified against the official
        // object_pointcloud numerically; the naive R @ diag(s) @ p reading is
        // wrong by exactly R vs R^T.
        const Vec3 scaled{pose.scale[0] * point.x,
                          pose.scale[1] * point.y,
                          pose.scale[2] * point.z};
        for (size_t column = 0; column < 3; ++column) {
            position[column] =
                rotation[0][column] * scaled.x +
                rotation[1][column] * scaled.y +
                rotation[2][column] * scaled.z +
                pose.translation[column];
        }
        float* quaternion = &splats.rotations[index * 4];
        std::array<float, 4> rotated =
            quat_multiply(inverse, {quaternion[0], quaternion[1], quaternion[2],
                                    quaternion[3]});
        const float norm = std::sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1] +
                                     rotated[2] * rotated[2] + rotated[3] * rotated[3]);
        if (!(norm > 0.0f) || !std::isfinite(norm)) {
            error = "scene pose produced a degenerate gaussian rotation";
            return false;
        }
        for (float& value : rotated) value /= norm;
        // pytorch3d.quaternion_multiply ends with standardize_quaternion: the
        // versor is forced to a nonnegative real part. Without this the stored
        // quaternion differs from the official one by a global sign (the same
        // rotation, but not the same receipt bytes).
        if (rotated[0] < 0.0f) {
            for (float& value : rotated) value = -value;
        }
        for (size_t channel = 0; channel < 4; ++channel) {
            quaternion[channel] = rotated[channel];
        }
        // Official make_scene: adjusted = scale * pose_scale, floored at
        // 1.1 * (min_kernel * pose_scale) after the per-object floor grows.
        for (size_t channel = 0; channel < 3; ++channel) {
            float* scale = &splats.scales[index * 3 + channel];
            *scale = std::max(*scale * pose.scale[channel],
                              1.1f * min_kernel * pose.scale[channel]);
        }
    }
    return true;
}

void append_splat_set(GaussianSplatSet& scene, const GaussianSplatSet& object) {
    scene.positions.insert(scene.positions.end(),
                           object.positions.begin(), object.positions.end());
    scene.sh0.insert(scene.sh0.end(), object.sh0.begin(), object.sh0.end());
    scene.opacities.insert(scene.opacities.end(), object.opacities.begin(),
                           object.opacities.end());
    scene.scales.insert(scene.scales.end(), object.scales.begin(),
                        object.scales.end());
    scene.rotations.insert(scene.rotations.end(), object.rotations.begin(),
                           object.rotations.end());
}

bool normalize_scene_with(GaussianSplatSet& scene, float inv_scale,
                          const std::array<float, 3>& center, std::string& error) {
    const size_t count = scene.size();
    if (count == 0) {
        error = "cannot normalize an empty scene";
        return false;
    }
    if (!(inv_scale > 0.0f) || !std::isfinite(inv_scale)) {
        error = "normalization inv_scale must be positive and finite";
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        for (size_t axis = 0; axis < 3; ++axis) {
            scene.positions[index * 3 + axis] =
                scene.positions[index * 3 + axis] / inv_scale - center[axis];
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            scene.scales[index * 3 + channel] /= inv_scale;
        }
    }
    return true;
}

bool normalize_scene(GaussianSplatSet& scene, std::string& error) {
    const size_t count = scene.size();
    if (count == 0) {
        error = "cannot normalize an empty scene";
        return false;
    }
    // Official normalized_gaussian: the opacity > 0.9 active bounds drive a
    // uniform rescale and centering. fix_alignment=False keeps the axes.
    Vec3 lower{std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()};
    Vec3 upper{-lower.x, -lower.y, -lower.z};
    const auto axis_value = [](const Vec3& value, size_t axis) {
        return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
    };
    size_t active = 0;
    for (size_t index = 0; index < count; ++index) {
        if (scene.opacities[index] <= 0.9f) continue;
        ++active;
        const Vec3 position{scene.positions[index * 3],
                            scene.positions[index * 3 + 1],
                            scene.positions[index * 3 + 2]};
        for (size_t axis = 0; axis < 3; ++axis) {
            const float value = axis_value(position, axis);
            if (axis == 0) {
                lower.x = std::min(lower.x, value);
                upper.x = std::max(upper.x, value);
            } else if (axis == 1) {
                lower.y = std::min(lower.y, value);
                upper.y = std::max(upper.y, value);
            } else {
                lower.z = std::min(lower.z, value);
                upper.z = std::max(upper.z, value);
            }
        }
    }
    if (active == 0) {
        error = "scene has no gaussian with opacity > 0.9; cannot normalize";
        return false;
    }
    float inverse_scale = std::max(upper.x - lower.x, upper.y - lower.y);
    inverse_scale = std::max(inverse_scale, upper.z - lower.z);
    if (!(inverse_scale > 0.0f) || !std::isfinite(inverse_scale)) {
        error = "scene active bounds are degenerate; cannot normalize";
        return false;
    }
    const Vec3 center{(lower.x + upper.x) * 0.5f / inverse_scale,
                      (lower.y + upper.y) * 0.5f / inverse_scale,
                      (lower.z + upper.z) * 0.5f / inverse_scale};
    for (size_t index = 0; index < count; ++index) {
        for (size_t axis = 0; axis < 3; ++axis) {
            float* value = &scene.positions[index * 3 + axis];
            *value = *value / inverse_scale - axis_value(center, axis);
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            scene.scales[index * 3 + channel] /= inverse_scale;
        }
    }
    return true;
}

std::vector<GaussianCamera> make_orbit_cameras(int frame_count, float radius,
                                               float fov_degrees,
                                               std::string& error) {
    if (frame_count <= 0 || !(radius > 0.0f) || !(fov_degrees > 0.0f) ||
        fov_degrees >= 180.0f) {
        error = "invalid orbit camera configuration";
        return {};
    }
    const float focal = 0.5f / std::tan(fov_degrees * kPi / 360.0f);
    std::vector<GaussianCamera> cameras;
    cameras.reserve(static_cast<size_t>(frame_count));
    for (int frame = 0; frame < frame_count; ++frame) {
        // torch.linspace(0, 2*pi, frames) + radians(-90), pitch = 0.
        // torch.linspace includes BOTH endpoints, so the divisor is
        // frames - 1. The previous divisor (frames) produced a per-frame yaw
        // drift of 2*pi/(frames*(frames-1)) that grew to ~1 deg by frame 299
        // and dominated the scene comparison MAE.
        const float yaw =
            (frame_count == 1)
                ? -kPi * 0.5f
                : 2.0f * kPi * static_cast<float>(frame) /
                        static_cast<float>(frame_count - 1) - kPi * 0.5f;
        const Vec3 eye{std::sin(yaw) * radius, 0.0f, std::cos(yaw) * radius};
        // utils3d extrinsics_look_at(eye, origin, up=[0,1,0]):
        // z = normalize(look_at - eye); x = normalize(cross(-up, z)); y = z x x.
        const Vec3 z_axis{-eye.x / radius, 0.0f, -eye.z / radius};
        const Vec3 x_axis = cross3({0.0f, -1.0f, 0.0f}, z_axis);
        const float x_norm = std::sqrt(x_axis.x * x_axis.x + x_axis.y * x_axis.y +
                                       x_axis.z * x_axis.z);
        if (!(x_norm > 0.0f)) {
            error = "orbit camera coincides with the up axis";
            return {};
        }
        const Vec3 x_unit{x_axis.x / x_norm, x_axis.y / x_norm, x_axis.z / x_norm};
        const Vec3 y_axis = cross3(z_axis, x_unit);
        GaussianCamera camera;
        camera.extrinsics = {x_unit.x, x_unit.y, x_unit.z,
                             -(x_unit.x * eye.x + x_unit.y * eye.y + x_unit.z * eye.z),
                             y_axis.x, y_axis.y, y_axis.z,
                             -(y_axis.x * eye.x + y_axis.y * eye.y + y_axis.z * eye.z),
                             z_axis.x, z_axis.y, z_axis.z,
                             -(z_axis.x * eye.x + z_axis.y * eye.y + z_axis.z * eye.z),
                             0.0f, 0.0f, 0.0f, 1.0f};
        camera.intrinsics = {focal, 0.0f, 0.5f, 0.0f, focal, 0.5f, 0.0f, 0.0f, 1.0f};
        cameras.push_back(camera);
    }
    return cameras;
}

bool write_scene_ply(const std::string& path, const GaussianSplatSet& scene,
                     std::string& error) {
    if (scene.size() == 0) {
        error = "cannot write an empty scene PLY";
        return false;
    }
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) {
        error = "cannot open scene PLY output: " + path;
        return false;
    }
    char header[1024];
    std::snprintf(header, sizeof(header),
                  "ply\nformat binary_little_endian 1.0\ncomment NeoGaussian\n"
                  "element vertex %zu\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float nx\nproperty float ny\nproperty float nz\n"
                  "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                  "property float opacity\n"
                  "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                  "property float rot_0\nproperty float rot_1\nproperty float rot_2\n"
                  "property float rot_3\nend_header\n",
                  scene.size());
    bool ok = fwrite(header, 1, std::strlen(header), file) == std::strlen(header);
    std::vector<float> row(17);
    for (size_t index = 0; index < scene.size() && ok; ++index) {
        // Storage semantics of Gaussian.save_ply: xyz is the activated
        // position; opacity/scale store the inverse activations; the loader
        // renormalizes quaternions, so the normalized value round-trips.
        for (size_t axis = 0; axis < 3; ++axis) {
            row[axis] = scene.positions[index * 3 + axis];
            row[3 + axis] = 0.0f;
            row[6 + axis] = scene.sh0[index * 3 + axis];
            row[10 + axis] = std::log(scene.scales[index * 3 + axis]);
        }
        row[9] = std::log(scene.opacities[index] / (1.0f - scene.opacities[index]));
        for (size_t channel = 0; channel < 4; ++channel) {
            row[13 + channel] = scene.rotations[index * 4 + channel];
        }
        ok = fwrite(row.data(), sizeof(float), row.size(), file) == row.size();
    }
    if (fclose(file) != 0) ok = false;
    if (!ok) {
        error = "failed while writing scene PLY: " + path;
        return false;
    }
    return true;
}

}  // namespace sam3d
