# External video — monocular depth heatmap

Depth map on `inputs/WhatsApp Video 2026-07-08 at 7.07.46 PM.mp4` (a top-down
monocular clip of hands working with fabric). No stereo/calibration available, so
depth is from **Depth Anything V2 (metric)** — the same monocular backend as
Stage 05. Rendered in the heatmap style (full JET, temporally smoothed, tight
normalised range, HUD + nearest-object distance labels).

- `tailoring_depth_heatmap.mp4` — 906 frames @ 12 fps, near = red, far = blue.
- Raw per-frame metric depth available on request.

Note: monocular depth is scale-approximate (no stereo baseline here), so the
distance labels are indicative, not survey-grade.
