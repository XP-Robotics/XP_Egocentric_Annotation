#!/usr/bin/env python
"""
Stage 08 — Object 6DoF pose + oriented 3D bounding box.

Model-free 6DoF: each tracked object's mask (Stage 04) is back-projected with the
metric depth (Stage 05) to a 3D point cloud; PCA gives an oriented 3D box (pose_R,
pose_t, obb) — already computed inside the pipeline. This renders the boxes + pose
axes projected into the image and exports the poses.

(FoundationPose / BundleSDF would give higher-fidelity, CAD-anchored pose — see the
client brief. This mask+depth PCA path needs no CAD model and runs on any object.)

Usage: render_6dof_pose.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>
"""
import sys, os, gzip, pickle, json
import numpy as np, cv2

PKL, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
FR = os.path.join(OUT, "_frames"); os.makedirs(FR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
f_px, cx, cy = float(d["dp_focal"]), float(d["cx"]), float(d["cy"])
mapping = d.get("sam3_prompt_mapping", [])
COLS = [(0, 90, 255), (0, 220, 0), (255, 120, 0), (0, 210, 255), (255, 0, 200), (170, 0, 255)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def proj(P):
    P = np.asarray(P, float)
    z = np.clip(P[:, 2], 1e-3, None)
    u = f_px * P[:, 0] / z + cx; v = f_px * P[:, 1] / z + cy
    return np.stack([u, v], 1)


out_json = {"fps": 15, "n_frames": len(fd), "camera": {"focal_px": f_px, "cx": cx, "cy": cy},
            "objects": [{"id": i, "label": m.get("prompt")} for i, m in enumerate(mapping)], "frames": []}

for fi, f in enumerate(fd):
    p = os.path.join(FRAMES, f"{fi+1:06d}.jpg"); im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    rec = {"frame": fi, "objects": []}
    od = f.get("sam3_obj_data") or {}
    for oid in sorted(od):
        v = od[oid]
        R, t, corners = v.get("pose_R"), v.get("pose_t"), v.get("obb_corners")
        if R is None or t is None or corners is None:
            continue
        R = np.asarray(R, float); t = np.asarray(t, float); corners = np.asarray(corners, float)
        # sanity filter: drop implausible boxes (noisy far mis-detections)
        ext = v.get("obb_extent")
        if ext is not None and float(np.max(ext)) > 0.35:      # >35 cm -> not a handled box
            continue
        if t[2] <= 0.05 or t[2] > 1.8:                          # too near/far
            continue
        c2 = proj(corners).astype(int)
        bw = c2[:, 0].max() - c2[:, 0].min(); bh = c2[:, 1].max() - c2[:, 1].min()
        if bw > 0.7 * W or bh > 0.7 * H:                        # projects larger than the object could be
            continue
        c = COLS[oid % len(COLS)]
        for a, b in EDGES:
            cv2.line(im, tuple(c2[a]), tuple(c2[b]), c, 2, cv2.LINE_AA)
        # pose axes (5 cm) at the object centre
        ax = proj(np.vstack([t, t + R[:, 0] * 0.05, t + R[:, 1] * 0.05, t + R[:, 2] * 0.05])).astype(int)
        for k, col in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(im, tuple(ax[0]), tuple(ax[k]), col, 2, cv2.LINE_AA)
        lbl = mapping[oid]["prompt"] if oid < len(mapping) else str(oid)
        cv2.putText(im, lbl, tuple(c2[:, :].min(0) + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        rec["objects"].append({"id": int(oid), "label": lbl,
                               "rotation_3x3": R.round(4).tolist(),
                               "translation_m": t.round(4).tolist(),
                               "obb_extent_m": np.asarray(v.get("obb_extent"), float).round(4).tolist()
                               if v.get("obb_extent") is not None else None})
    cv2.putText(im, f"6DoF object pose + 3D box | frame {fi}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FR, f"{fi:06d}.png"), im)
    out_json["frames"].append(rec)

json.dump(out_json, open(os.path.join(OUT, "object_6dof_poses.json"), "w"))
os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 15 -i {FR}/%06d.png "
          f"-c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 "
          f"{OUT}/object_6dof_pose.mp4")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {OUT}/object_6dof_pose.mp4 -vf "
          f"\"fps=8,scale=440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
          f"{OUT}/object_6dof_pose.gif")
import shutil; shutil.rmtree(FR, ignore_errors=True)
n = sum(len(r["objects"]) for r in out_json["frames"])
print(f"stage08: {len(out_json['frames'])} frames, {n} object-pose instances -> {OUT}")
