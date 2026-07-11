#!/usr/bin/env python
"""
Stage 09 — Gaze (PROXY).

True gaze needs an eye-tracking stream (e.g. Aria) — not present on this rig. As a
stand-in, we render a **manipulation-attention proxy**: on a manual task the eyes
overwhelmingly fixate where the hands are working, so the fingertip / active-object
centroid is a strong prior on the gaze point. This is a documented approximation,
NOT measured gaze.

Usage: attention_proxy.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>
"""
import sys, os, gzip, pickle, json
import numpy as np, cv2

PKL, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
FR = os.path.join(OUT, "_frames"); os.makedirs(FR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
TIPS = [4, 8, 12, 16, 20]
pt = None; trail = []; rec = []
for fi, f in enumerate(fd):
    p = os.path.join(FRAMES, f"{fi+1:06d}.jpg"); im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    pts = []
    for J in (f.get("joints_2d_pred") or []):
        J = np.asarray(J, float)
        if J.ndim == 2 and len(J) > 20:
            pts += [J[t] for t in TIPS]
    if pts:
        target = np.median(np.asarray(pts), 0)
        pt = target if pt is None else 0.7 * pt + 0.3 * target      # smooth
    if pt is not None:
        c = tuple(np.round(pt).astype(int))
        trail.append(c); trail[:] = trail[-25:]
        overlay = im.copy()
        cv2.circle(overlay, c, 46, (0, 230, 255), -1)
        im = cv2.addWeighted(overlay, 0.25, im, 0.75, 0)
        for k in range(1, len(trail)):
            cv2.line(im, trail[k-1], trail[k], (0, 230, 255), 1, cv2.LINE_AA)
        cv2.circle(im, c, 46, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(im, c, (0, 230, 255), cv2.MARKER_CROSS, 26, 2)
        rec.append({"frame": fi, "attention_px": [int(c[0]), int(c[1])]})
    cv2.putText(im, "gaze PROXY (manipulation attention - not eye-tracked)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FR, f"{fi:06d}.png"), im)

json.dump({"note": "manipulation-attention proxy; not measured gaze", "frames": rec},
          open(os.path.join(OUT, "attention_proxy.json"), "w"))
os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 15 -i {FR}/%06d.png "
          f"-c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 {OUT}/gaze_attention_proxy.mp4")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {OUT}/gaze_attention_proxy.mp4 -vf "
          f"\"fps=8,scale=440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" {OUT}/gaze_attention_proxy.gif")
import shutil; shutil.rmtree(FR, ignore_errors=True)
print(f"stage09: attention proxy over {len(rec)} frames -> {OUT}")
