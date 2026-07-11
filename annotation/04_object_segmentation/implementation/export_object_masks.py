#!/usr/bin/env python
"""Export extra client deliverables from pipeline_result.pkl.gz:
  depth_frames/   -> metric depth colormap PNGs (+ encode to depth.mp4)
  mask_frames/    -> object segmentation overlay PNGs (+ object_masks.mp4)
  trajectories.png-> per-object 3D translation over time
  metrics.json    -> summary metrics
Usage: export_extras.py <pkl> <frames_dir> <out_dir>
"""
import sys
import os
import gzip
import pickle
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DFR = os.path.join(OUT, "_depth"); MFR = os.path.join(OUT, "_mask")
os.makedirs(DFR, exist_ok=True); os.makedirs(MFR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
mapping = d.get("sam3_prompt_mapping", [])
COLS = [(0, 90, 255), (0, 220, 0), (255, 100, 0), (0, 210, 255),
        (255, 0, 200), (170, 0, 255)]


def clean(fi, fb):
    p = os.path.join(FRAMES, f"{fi+1:06d}.jpg")
    if os.path.isfile(p):
        return cv2.imread(p)
    return (cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
            if isinstance(fb, (bytes, bytearray)) else np.ascontiguousarray(fb[..., ::-1]))


# object trajectory accumulator: oid -> list of (frame, t[3])
traj = {}
for fi, f in enumerate(fd):
    # depth colormap
    dp = f["depth_png"]
    depth = (cv2.imdecode(np.frombuffer(dp, np.uint8), cv2.IMREAD_UNCHANGED)
             if isinstance(dp, (bytes, bytearray)) else np.asarray(dp))
    dm = depth.astype(np.float32) / 1000.0
    vis = np.clip((dm - 0.3) / (3.0 - 0.3), 0, 1)          # 0.3-3 m -> 0..1
    cm = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[dm <= 0.3] = 0
    cv2.putText(cm, f"metric depth (0.3-3.0 m) | frame {fi}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(DFR, f"{fi:06d}.png"), cm)

    # object masks overlay
    ov = clean(fi, f["img_rgb"]); H, W = ov.shape[:2]
    od = f.get("sam3_obj_data") or {}
    for i, oid in enumerate(sorted(od)):
        v = od[oid]; mp, ms = v.get("mask_packed"), v.get("mask_shape")
        if mp is None:
            continue
        m = np.unpackbits(np.asarray(mp, np.uint8))[:int(ms[0])*int(ms[1])]
        m = m.reshape(int(ms[0]), int(ms[1])).astype(bool)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), 0, 0, cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue
        c = COLS[i % len(COLS)]
        ov[m] = (0.45*np.array(c) + 0.55*ov[m]).astype(np.uint8)
        ys, xs = np.where(m)
        lbl = mapping[oid]["prompt"] if oid < len(mapping) else str(oid)
        cv2.rectangle(ov, (xs.min(), ys.min()), (xs.max(), ys.max()), c, 1)
        cv2.putText(ov, lbl, (xs.min(), max(11, ys.min()-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        if v.get("pose_t") is not None:
            traj.setdefault(oid, []).append((fi, np.asarray(v["pose_t"], float)))
    cv2.imwrite(os.path.join(MFR, f"{fi:06d}.png"), ov)

# trajectory plot (object translation over time)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axlabel = ["X (m)", "Y (m)", "Z / depth (m)"]
for oid, pts in sorted(traj.items()):
    fr = [p[0] for p in pts]; T = np.array([p[1] for p in pts])
    lbl = mapping[oid]["prompt"] if oid < len(mapping) else str(oid)
    for k in range(3):
        axs[k].plot(fr, T[:, k], label=lbl, linewidth=1.5)
for k in range(3):
    axs[k].set_xlabel("frame"); axs[k].set_ylabel(axlabel[k]); axs[k].grid(alpha=0.3)
axs[0].legend(fontsize=8, loc="best")
fig.suptitle("Tracked object 3D position over time", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=120)

# metrics summary
n_hand_frames = sum(1 for f in fd if (f.get("joints_3d_pred") or []))
both = sum(1 for f in fd if len(f.get("joints_3d_pred") or []) >= 2)
objs = [{"label": m.get("prompt"), "detection_confidence": round(float(m.get("score", 0)), 3),
         "frames_with_6dof_pose": len(traj.get(i, []))}
        for i, m in enumerate(mapping)]
metrics = {
    "clip": {"n_frames": len(fd), "fps": 15,
             "duration_s": round(len(fd)/15.0, 2),
             "image_size": [int(fd[0]["depth_png"] and 0) or clean(0, fd[0]["img_rgb"]).shape[1],
                            clean(0, fd[0]["img_rgb"]).shape[0]]},
    "hands": {"frames_with_hands": n_hand_frames,
              "frames_with_both_hands": both,
              "hand_coverage_pct": round(100.0*n_hand_frames/len(fd), 1)},
    "objects": {"n_tracked": len(mapping), "detail": objs},
    "camera": {"focal_px": round(float(d["dp_focal"]), 2),
               "cx": float(d["cx"]), "cy": float(d["cy"])},
}
with open(os.path.join(OUT, "metrics.json"), "w") as fo:
    json.dump(metrics, fo, indent=2)
print("done: depth frames + mask frames + trajectories.png + metrics.json")
