# Stage 06 — Implementation

`run_droid.sh <frames_dir> <camera.json> <out_dir>` — runs DROID-SLAM on the
rectified monocular stream with the pinhole intrinsics from `camera.json`, writing
the per-frame 6DoF camera trajectory. Uses a dedicated `droid-venv` (Blackwell/cu128).
Weights: `../../models/droid_slam/droid.pth`. DROID-SLAM repo under `DROID-SLAM/`.
