#!/usr/bin/env python
"""
Export all per-scene DATA files (JSON + meshes) from a pipeline result, matching
the client deliverable set. Writes into <out_dir>:

  hand_detections.json      per-frame hand boxes + side + confidence
  hand_tracks.json          per-frame 2D+3D hand keypoints + track ids
  objects.json              tracked objects + detection confidence
  object_6dof_poses.json    per-object 6DoF R,t + 3D box extent
  tracking.json             CONSOLIDATED per-frame hands + objects (client file)
  metrics.json              summary metrics (client file)
  hand_meshes/frame_*.obj   per-frame 3D hand meshes

Usage: export_scene_data.py <pipeline_result.pkl.gz> <out_dir>
"""
import sys, os, gzip, pickle, json
import numpy as np

PKL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
mp = d.get("sam3_prompt_mapping", [])
f_px, cx, cy = float(d["dp_focal"]), float(d["cx"]), float(d["cy"])
faces = np.asarray(d.get("mano_faces", []), int)
n = len(fd)


def A(x):
    return np.asarray(x, float)


# ---- per-stage JSONs ----
det = []; tracks = []; poses_out = []
for fi, f in enumerate(fd):
    j2 = f.get("joints_2d_pred") or []; j3 = f.get("joints_3d_pred") or []
    sd = f.get("hand_is_right") or []; meta = f.get("hand_meta") or []
    # detection
    dh = []
    for hi in range(len(j2)):
        J = A(j2[hi])
        if J.ndim != 2:
            continue
        x0, y0, x1, y1 = J[:, 0].min(), J[:, 1].min(), J[:, 0].max(), J[:, 1].max()
        pad = 0.15 * max(x1 - x0, y1 - y0)
        dh.append({"side": "right" if (hi < len(sd) and sd[hi]) else "left",
                   "confidence": round(float(meta[hi]["confidence"]) if hi < len(meta) and isinstance(meta[hi], dict) else 0.0, 3),
                   "box_xyxy": [int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad)]})
    det.append({"frame": fi, "hands": dh})
    # tracks
    th = []
    for hi in range(len(j3)):
        th.append({"side": "right" if (hi < len(sd) and sd[hi]) else "left",
                   "track_id": int(meta[hi]["track_id"]) if hi < len(meta) and isinstance(meta[hi], dict) else hi,
                   "joints_3d_m": A(j3[hi]).round(4).tolist(),
                   "joints_2d_px": A(j2[hi]).round(1).tolist() if hi < len(j2) else None})
    tracks.append({"frame": fi, "hands": th})
    # object poses
    po = []
    for oid, v in (f.get("sam3_obj_data") or {}).items():
        if v.get("pose_R") is None or v.get("pose_t") is None:
            continue
        po.append({"id": int(oid), "label": mp[oid]["prompt"] if oid < len(mp) else str(oid),
                   "rotation_3x3": A(v["pose_R"]).round(4).tolist(),
                   "translation_m": A(v["pose_t"]).round(4).tolist(),
                   "obb_extent_m": A(v["obb_extent"]).round(4).tolist() if v.get("obb_extent") is not None else None})
    poses_out.append({"frame": fi, "objects": po})

objects = [{"id": i, "label": m.get("prompt"), "detection_confidence": round(float(m.get("score", 0)), 3)}
           for i, m in enumerate(mp)]
json.dump({"n_frames": n, "frames": det}, open(f"{OUT}/hand_detections.json", "w"))
json.dump({"fps": 15, "n_frames": n, "frames": tracks}, open(f"{OUT}/hand_tracks.json", "w"))
json.dump({"objects": objects}, open(f"{OUT}/objects.json", "w"), indent=2)
json.dump({"fps": 15, "n_frames": n, "camera": {"focal_px": round(f_px, 2), "cx": cx, "cy": cy},
           "objects": objects, "frames": poses_out}, open(f"{OUT}/object_6dof_poses.json", "w"))

# ---- consolidated tracking.json (client) ----
frames = []
for fi, f in enumerate(fd):
    j2 = f.get("joints_2d_pred") or []; j3 = f.get("joints_3d_pred") or []; sd = f.get("hand_is_right") or []
    hands = [{"side": "right" if (hi < len(sd) and sd[hi]) else "left",
              "joints_3d_m": A(j3[hi]).round(4).tolist(),
              "joints_2d_px": A(j2[hi]).round(1).tolist() if hi < len(j2) else None} for hi in range(len(j3))]
    objs = []
    for oid, v in (f.get("sam3_obj_data") or {}).items():
        pose = ({"rotation_3x3": A(v["pose_R"]).round(4).tolist(), "translation_m": A(v["pose_t"]).round(4).tolist()}
                if v.get("pose_R") is not None and v.get("pose_t") is not None else None)
        obb = ({"corners_m": A(v["obb_corners"]).round(4).tolist(),
                "extent_m": A(v["obb_extent"]).round(4).tolist() if v.get("obb_extent") is not None else None}
               if v.get("obb_corners") is not None else None)
        objs.append({"id": int(oid), "label": mp[oid]["prompt"] if oid < len(mp) else str(oid), "pose": pose, "obb": obb})
    frames.append({"frame": fi, "hands": hands, "objects": objs})
meta = {"fps": 15, "n_frames": n, "camera": {"focal_px": round(f_px, 2), "cx": cx, "cy": cy, "model": "pinhole"},
        "coordinate_frame": "camera-centred, metres; +X right, +Y down, +Z forward",
        "hand_joint_order": "MANO 21: 0=wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20"}
json.dump({"metadata": meta, "objects": objects, "frames": frames}, open(f"{OUT}/tracking.json", "w"))

# ---- metrics.json (client) ----
hf = sum(1 for f in fd if (f.get("joints_3d_pred") or []))
both = sum(1 for f in fd if len(f.get("joints_3d_pred") or []) >= 2)
json.dump({"clip": {"n_frames": n, "fps": 15, "duration_s": round(n / 15, 2),
                    "camera": {"focal_px": round(f_px, 2), "cx": cx, "cy": cy}},
           "hands": {"frames_with_hands": hf, "frames_with_both_hands": both,
                     "coverage_pct": round(100 * hf / n, 1)},
           "objects": {"n_tracked": len(objects), "detail": objects}},
          open(f"{OUT}/metrics.json", "w"), indent=2)

# ---- hand meshes ----
if faces.size:
    md = os.path.join(OUT, "hand_meshes"); os.makedirs(md, exist_ok=True)
    for fi, f in enumerate(fd):
        V = f.get("vertices_3d")
        if not V:
            continue
        sd = f.get("hand_is_right") or []; lines = []; voff = 0
        for hi, verts in enumerate(V):
            verts = A(verts)
            if verts.ndim != 2 or verts.shape[1] != 3:
                continue
            lines.append(f"o {'hand_right' if (hi < len(sd) and sd[hi]) else 'hand_left'}")
            for x, y, z in verts:
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            for a, b, c in faces:
                lines.append(f"f {a+1+voff} {b+1+voff} {c+1+voff}")
            voff += len(verts)
        if lines:
            open(os.path.join(md, f"frame_{fi:06d}.obj"), "w").write("\n".join(lines) + "\n")

print(f"exported data + tracking.json + metrics.json + hand_meshes -> {OUT}")
