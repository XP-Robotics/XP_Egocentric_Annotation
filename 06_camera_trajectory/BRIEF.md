# Stage 06 — Camera Trajectory / Egomotion

**XP Robotics · Egocentric Perception Stack**

## What it does
Recovers the **6DoF motion of the camera** (the wearer's head/body) through the
clip — the trajectory that lets per-frame reconstructions be fused into one stable
world frame instead of drifting with the moving camera.

## Input
| Input | Notes |
|---|---|
| Rectified RGB frames | front-camera pinhole frames |
| Pinhole intrinsics | `camera.json` (fx, fy, cx, cy) |

## Output
| Output | Format |
|---|---|
| Per-frame camera pose (6DoF) | rotation + translation |
| Camera trajectory | `.npy` / plot |
| Sparse scene structure | point cloud (optional) |

## Method
**DROID-SLAM** — a deep visual SLAM system that jointly optimises dense
correspondence and camera poses over a keyframe graph. Runs on the rectified
monocular stream with the known intrinsics from `camera.json`.

> **Note (ego + hardware).** On a **static or slowly-moving** capture, egomotion is
> small and per-frame reconstruction already lives in a consistent frame. DROID-SLAM
> adds the most value on **freely-moving** captures. If the platform is Project Aria
> glasses, the on-device **Aria MPS** SLAM + factory calibration is the easier,
> higher-accuracy path and can replace this stage.

## Why it matters
- **Stable world frame** — fuses moving-camera observations into one coordinate
  system; prerequisite for long-horizon trajectories.
- **Egomotion signal** — the wearer's motion is itself useful context for learning.

© XP Robotics