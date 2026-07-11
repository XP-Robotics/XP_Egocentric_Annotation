# Stage 02 — Implementation

- `render_mano_overlay.py <pkl> <out_dir>` — overlays the HaMeR/MANO mesh on each frame.
- `export_meshes.py <pkl> <out_dir>` — writes per-frame `frame_NNNNNN.obj` (778 verts/hand,
  labelled hand_left/hand_right, metric camera frame).

HaMeR weights: `../../models/hamer/`.
