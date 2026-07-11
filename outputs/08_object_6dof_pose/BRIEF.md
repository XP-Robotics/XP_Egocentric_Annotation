# Stage 08 — Object 6DoF Pose

**XP Robotics · Egocentric Perception Stack**

## What it does
Estimates each tracked object's **6DoF pose** (3D rotation + translation) and an
**oriented 3D bounding box**, in the shared metric camera frame — the geometry a
robot needs to grasp or place the object.

## Input
Object masks (Stage 04) + metric depth (Stage 05) + camera intrinsics.

## Output
| Output | Format |
|---|---|
| Per-object R (3×3) + t (m) + 3D box extent | `object_6dof_poses.json` |
| 3D box + pose-axes overlay video | `object_6dof_pose.mp4` |

## Method
- **Mask+depth PCA (delivered):** the object mask is back-projected with metric
  depth to a 3D point cloud; principal-axis analysis yields an oriented box and a
  6DoF pose. **No CAD model required** — runs on any text-prompted object.
- **FoundationPose / BundleSDF (upgrade):** for higher-fidelity, temporally-tracked
  pose. FoundationPose needs a CAD/mesh per object; BundleSDF is model-free but
  heavy. Recommended when survey-grade pose is required.

## Why it matters
- **Grasp/place geometry** — 6DoF pose + box is directly actionable for a robot.
- **CAD-free** — immediate pose for arbitrary objects from the mask+depth path.

© XP Robotics