# Stage 04 — Implementation

`export_object_masks.py <pkl> <frames_dir> <out_dir>` — renders open-vocab object
masks + boxes + labels, and `object_trajectories.png`. Objects are set by text in
the clip manifest `objects[]`. Open-vocab model: SAM 3.1 (`../../models/sam3.1/`),
Grounded-SAM fallback (`../../models/grounded_sam/`).
