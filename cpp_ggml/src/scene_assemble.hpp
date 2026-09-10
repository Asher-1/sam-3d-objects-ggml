// Native multi-object scene assembly: the official demo_multi_object flow
// (N single-object reconstructions, pose application, concatenation,
// normalization, orbit rendering) with no Python at any stage.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "gaussian_renderer.hpp"

namespace sam3d {

// The official-schema pose receipt written by pose_decoder.cpp: top-level
// rotation (wxyz quaternion), translation, and the uniform mean scale.
struct ScenePose {
    std::array<float, 4> rotation_wxyz{};
    std::array<float, 3> translation{};
    std::array<float, 3> scale{};
};

// Parse the pose receipt. Only the first occurrence of each official key is
// read; the sam3d.native-pose.v2 "native" diagnostics block is ignored.
bool load_scene_pose_json(const std::string& path, ScenePose& pose,
                          std::string& error);

// Parse the reference variant's normalization contract
// ({"inv_scale": float, "center": [x, y, z]}).
bool load_scene_normalization_json(const std::string& path, float& inv_scale,
                                   std::array<float, 3>& center,
                                   std::string& error);

// Official make_scene pose semantics on the activated splat representation
// (Gaussian.get_xyz/get_rotation/get_scaling):
//   positions: p' = diag(pose_scale) @ (R(pose_q) @ p) + t  (rotate, scale,
//              translate - verified against pytorch3d Transform3d)
//   rotations: q' = inv(pose_q) (x) q                       (Hamilton, wxyz)
//   scales:    s' = max(s * pose_scale, 1.1 * min_kernel * pose_scale)
bool apply_scene_pose(GaussianSplatSet& splats, const ScenePose& pose,
                      float min_kernel, std::string& error);

// Concatenate an object's splats onto the accumulating scene.
void append_splat_set(GaussianSplatSet& scene, const GaussianSplatSet& object);

// Official normalized_gaussian: uniform rescale and centering driven by the
// opacity > 0.9 active bounds (fix_alignment=False, no axis permutation).
bool normalize_scene(GaussianSplatSet& scene, std::string& error);

// Same normalization with externally supplied parameters (the reference
// variant's recorded inv_scale and post-scale center) so all variants render
// the identical scene frame.
bool normalize_scene_with(GaussianSplatSet& scene, float inv_scale,
                          const std::array<float, 3>& center, std::string& error);

// Official render_video orbit: yaw = linspace(0, 2*pi, frames) - pi/2,
// pitch = 0, eye = radius * (sin yaw cos pitch, sin pitch, cos yaw cos pitch),
// utils3d extrinsics_look_at(eye, origin, up=[0, 1, 0]), and the normalized
// intrinsics_from_fov_xy (focal = 0.5 / tan(fov/2), principal point 0.5).
std::vector<GaussianCamera> make_orbit_cameras(int frame_count, float radius,
                                               float fov_degrees,
                                               std::string& error);

// Write the composed scene in the native/official save_ply PLY contract from
// the activated representation (opacity logits, log scales, raw quaternions).
bool write_scene_ply(const std::string& path, const GaussianSplatSet& scene,
                     std::string& error);

}  // namespace sam3d
