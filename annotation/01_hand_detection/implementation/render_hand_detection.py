#!/usr/bin/env python
"""
Stage 01 — Hand Detection + Side (L/R).
Draws per-frame hand bounding boxes with left/right labels + confidence, from the
pipeline result. Box is the tight bound of the 2D hand joints, padded.

Usage: render_hand_detection.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>
"""
import sys, os, gzip, pickle, json
import numpy as np, cv2

PKL, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
FR = os.path.join(OUT, "_frames"); os.makedirs(FR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
COL = {True: (60, 90, 240), False: (240, 160, 40)}   # right=red-ish, left=blue-ish (BGR)

summary = []
for fi, f in enumerate(fd):
    p = os.path.join(FRAMES, f"{fi+1:06d}.jpg")
    im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    j2d = f.get("joints_2d_pred") or []
    sides = f.get("hand_is_right") or []
    meta = f.get("hand_meta") or []
    rec = {"frame": fi, "hands": []}
    for hi, J in enumerate(j2d):
        J = np.asarray(J, float)
        if J.ndim != 2 or len(J) < 5:
            continue
        x0, y0 = J[:, 0].min(), J[:, 1].min()
        x1, y1 = J[:, 0].max(), J[:, 1].max()
        pad = 0.15 * max(x1 - x0, y1 - y0)
        x0, y0, x1, y1 = [int(v) for v in (x0 - pad, y0 - pad, x1 + pad, y1 + pad)]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W - 1, x1), min(H - 1, y1)
        is_right = bool(sides[hi]) if hi < len(sides) else (hi == 1)
        conf = float(meta[hi]["confidence"]) if hi < len(meta) and isinstance(meta[hi], dict) else 0.0
        c = COL[is_right]; lbl = f"{'RIGHT' if is_right else 'LEFT'} {conf:.2f}"
        cv2.rectangle(im, (x0, y0), (x1, y1), c, 2)
        cv2.rectangle(im, (x0, y0 - 20), (x0 + 130, y0), c, -1)
        cv2.putText(im, lbl, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rec["hands"].append({"side": "right" if is_right else "left",
                             "confidence": round(conf, 3), "box_xyxy": [x0, y0, x1, y1]})
    cv2.putText(im, f"Hand detection + L/R | frame {fi}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FR, f"{fi:06d}.png"), im)
    summary.append(rec)

with open(os.path.join(OUT, "hand_detections.json"), "w") as fo:
    json.dump({"n_frames": len(fd), "frames": summary}, fo)
n_both = sum(1 for r in summary if len(r["hands"]) >= 2)
print(f"stage01: {len(summary)} frames, both-hand frames={n_both}")
