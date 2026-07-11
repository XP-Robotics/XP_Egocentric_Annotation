# Stage 03 — Implementation

`render_tracking_2d.py <pkl> <frames_dir> <out_dir>` — draws 21-joint hand
skeletons + tracked object boxes. `hand_tracks.json` holds per-frame 2D+3D
keypoints with stable left/right track_ids. SAM2 (`../../models/sam2.1_hiera_small.pt`)
propagates hand identity; MANO is temporally smoothed.
