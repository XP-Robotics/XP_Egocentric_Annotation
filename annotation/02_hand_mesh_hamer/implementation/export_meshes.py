#!/usr/bin/env python
"""
Stage 02 — export per-frame HaMeR/MANO hand meshes as Wavefront .obj.
Each file holds the hand mesh(es) for that frame (778 verts/hand) in the metric
camera frame, labelled hand_left / hand_right.

Usage: export_meshes.py <pipeline_result.pkl.gz> <out_dir>
"""
import sys, os, gzip, pickle
import numpy as np

PKL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
faces = np.asarray(d["mano_faces"], int)   # (F,3)
fd = d["frame_data"]

n_written = 0
for fi, f in enumerate(fd):
    V = f.get("vertices_3d")
    if not V:
        continue
    sides = f.get("hand_is_right") or []
    lines = []
    voff = 0
    for hi, verts in enumerate(V):
        verts = np.asarray(verts, float)
        if verts.ndim != 2 or verts.shape[1] != 3:
            continue
        name = "hand_right" if (hi < len(sides) and sides[hi]) else "hand_left"
        lines.append(f"o {name}")
        for x, y, z in verts:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for a, b, c in faces:
            lines.append(f"f {a+1+voff} {b+1+voff} {c+1+voff}")
        voff += len(verts)
    if lines:
        with open(os.path.join(OUT, f"frame_{fi:06d}.obj"), "w") as fo:
            fo.write("\n".join(lines) + "\n")
        n_written += 1
print(f"stage02: wrote {n_written} .obj mesh files -> {OUT}")
