# Stage 05 — Implementation

- `run_depth_anything.py <frames_dir> <out_dir>` — Depth Anything V2 (metric, ViT-L
  Hypersim) per-frame metric depth. Weights: `../../models/depth_anything_v2/`.
- `render_depth_overlay.py {moge|da2} <src> <frames_dir> <out.mp4> [alpha plo phi]`
  — egocentric-tuned viz: robust per-clip range (p5–p92) + alpha overlay on the input
  so the scene shows through. `moge` src = pipeline pkl; `da2` src = depth npy.

MoGe-2 runs inside the shared pipeline with a **known-focal prior** (from
`camera.json`) so metric geometry is correct on the wide ego view.
