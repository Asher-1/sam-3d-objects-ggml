#include "pose_decoder.hpp"

#include <cmath>
#include <cstdio>
#include <string>

int main() {
    constexpr float rotation_mean[] = {
        -0.06366084883674913f, 0.008438224692279752f, 0.00017084786438302483f,
        0.0007126610473540038f, -0.0030916726538816417f, 0.5166093753457688f,
    };
    constexpr float rotation_std[] = {
        0.6656971967514863f, 0.6787012271867754f, 0.30345010594844524f,
        0.4394504420678794f, 0.39817973931717104f, 0.6176286868761914f,
    };
    // Choose the normalized latent whose decoded 6D representation is the
    // identity basis: a1=(1,0,0), a2=(0,1,0).
    const float target_rotation[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float normalized_rotation[6]{};
    for (size_t index = 0; index < 6; ++index) {
        normalized_rotation[index] = (target_rotation[index] - rotation_mean[index]) /
                                     rotation_std[index];
    }
    const float log_scale[] = {std::log(2.0f), std::log(3.0f), std::log(4.0f)};
    const float translation[] = {1.0f, 2.0f, 3.0f};
    const float scene_scale[] = {2.0f, 3.0f, 4.0f};
    const float scene_shift[] = {10.0f, 20.0f, 30.0f};
    sam3d::NativeInstancePose pose;
    std::string error;
    if (!sam3d::decode_scale_shift_invariant_pose(normalized_rotation, log_scale, translation,
                                                  std::log(5.0f), scene_scale, scene_shift,
                                                  pose, error)) {
        std::fprintf(stderr, "pose decode failed: %s\n", error.c_str());
        return 1;
    }
    constexpr float tolerance = 1.0e-5f;
    const float expected_rotation[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float expected_scale[] = {4.0f, 9.0f, 16.0f};
    const float expected_translation[] = {12.0f, 26.0f, 42.0f};
    for (size_t index = 0; index < 4; ++index) {
        if (std::fabs(pose.rotation_wxyz[index] - expected_rotation[index]) > tolerance) {
            std::fprintf(stderr, "unexpected quaternion component %zu\n", index);
            return 1;
        }
    }
    for (size_t index = 0; index < 3; ++index) {
        if (std::fabs(pose.scale[index] - expected_scale[index]) > tolerance ||
            std::fabs(pose.translation[index] - expected_translation[index]) > tolerance) {
            std::fprintf(stderr, "unexpected transformed pose component %zu\n", index);
            return 1;
        }
    }
    if (std::fabs(pose.translation_scale - 5.0f) > tolerance) {
        std::fprintf(stderr, "translation scale was not decoded with exp\n");
        return 1;
    }
    // Downsample-factor rescale (official scale *= downsample_factor after
    // decode): the scale doubles while the rotation stays untouched.
    sam3d::NativeInstancePose rescaled;
    if (!sam3d::decode_scale_shift_invariant_pose(normalized_rotation, log_scale, translation,
                                                  std::log(5.0f), scene_scale, scene_shift,
                                                  rescaled, error, 2)) {
        std::fprintf(stderr, "pose decode with downsample factor failed: %s\n", error.c_str());
        return 1;
    }
    for (size_t index = 0; index < 4; ++index) {
        if (std::fabs(rescaled.rotation_wxyz[index] - pose.rotation_wxyz[index]) > tolerance) {
            std::fprintf(stderr, "downsample factor changed the rotation component %zu\n", index);
            return 1;
        }
    }
    for (size_t index = 0; index < 3; ++index) {
        if (std::fabs(rescaled.scale[index] - 2.0f * pose.scale[index]) > tolerance ||
            std::fabs(rescaled.translation[index] - pose.translation[index]) > tolerance) {
            std::fprintf(stderr, "downsample factor did not only rescale the scale component %zu\n",
                         index);
            return 1;
        }
    }
    return 0;
}
