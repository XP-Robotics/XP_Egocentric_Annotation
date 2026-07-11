# Stage 05 — Metric Depth

**XP Robotics · Egocentric Perception Stack**

## What it does
Predicts **per-pixel metric depth** (in metres) for every frame from the single
RGB stream — the geometry that places hands and objects in a common 3D world
frame. No depth sensor required.

## Input
| Input | Notes |
|---|---|
| Rectified RGB frames | front-camera pinhole frames |
| Known focal length | from `camera.json` — passed as a prior (critical, see below) |

## Output
| Output | Format |
|---|---|
| Per-frame metric depth map | float metres |
| Depth colormap video | MP4 |
| Back-projected scene point cloud | fed to 3D reconstruction |

## Method
Two interchangeable metric-depth backends:
- **MoGe-2** — metric depth + focal, used as the pipeline default.
- **Depth Anything V2 (metric)** — strong generalist, provided as an alternative /
  cross-check backend.

**Known-focal prior (ego-specific).** A body-worn fisheye rectified to a wide
virtual pinhole has a genuinely wide field of view. Left to guess, a depth model
mis-estimates the focal and *bends flat surfaces*. We feed the true rectified focal
(`camera.json`) as a prior so the geometry is correct — flat desks stay flat and
hands sit on them.

## Why it matters
- **Sensor-free metric 3D** — depth from ordinary RGB footage.
- **Correct geometry on ego views** — the known-focal prior is what makes a
  fisheye-derived view metrically usable.

© XP Robotics

---

## Stereo variant — HITNet (triangulated depth)
Because the front camera is a **stereo fisheye pair**, depth can also be recovered
by **geometric triangulation** rather than monocular inference:

- Both eyes are stereo-rectified (from the calibration `R,T`, 62.35 mm baseline).
- **HITNet** (learned stereo matcher) predicts disparity per pixel.
- Metric depth = `focal · baseline / disparity` — a *measured* depth, not inferred.

Runs on-GPU (~47 ms/frame). Sharper on textured edges; monocular MoGe/DA-v2 stay
smoother on textureless regions. Output: `depth_hitnet_stereo.mp4` + compressed
per-frame depth (`depth_hitnet_stereo_f16.npz`). Model: HITNet middlebury_d400.