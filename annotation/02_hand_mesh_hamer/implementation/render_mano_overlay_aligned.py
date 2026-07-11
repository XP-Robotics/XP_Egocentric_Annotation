#!/usr/bin/env python
"""Overlay MANO meshes on frames, ANCHORED to the accurate 2D hand keypoints.

The pipeline's 2D keypoints (joints_2d_pred) sit tightly on the hand, but the
metric 3D mesh projects with a slightly different camera (~10 px offset). For each
hand we solve a 2D similarity (isotropic scale s + translation t) mapping the
projected 3D joints onto joints_2d_pred, then realise it in 3D:
  - scale: divide vertex depth by s   (projection scales by s about the centre)
  - translation: back-project the residual at the hand depth and shift X,Y
so the rendered mesh lands on the hand. Headless via EGL.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, gzip, pickle
import numpy as np, cv2, trimesh, pyrender

PKL, OUTDIR = sys.argv[1], sys.argv[2]
ONLY = int(sys.argv[3]) if len(sys.argv) > 3 else -1
ALPHA = 0.9
os.makedirs(OUTDIR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]; faces = np.asarray(d["mano_faces"])
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(PKL)), "extract_frames", "frames")
fx = fy = float(d["dp_focal"]); cx = float(d["cx"]); cy = float(d["cy"])
N = len(fd)
CAM_POSE = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], float)


def proj(P):
    z = np.clip(P[:, 2], 1e-4, None)
    return np.stack([fx * P[:, 0] / z + cx, fy * P[:, 1] / z + cy], 1)


def anchor(verts, j3, j2):
    """Return verts transformed so projected j3 matches j2 (scale+translation)."""
    pj = proj(j3)
    p = pj - pj.mean(0); q = j2 - j2.mean(0)
    denom = (p * p).sum()
    s = float((q * p).sum() / denom) if denom > 1e-6 else 1.0
    s = float(np.clip(s, 0.6, 1.6))                      # guard against outliers
    Zc = float(np.median(j3[:, 2]))
    v = verts.copy()
    v[:, 2] = v[:, 2] / s                                # depth scale -> 2D scale s
    # residual translation at the (new) hand depth
    pj2 = proj(np.column_stack([verts[:, 0], verts[:, 1], verts[:, 2] / s]))
    du, dv = (j2.mean(0) - pj2.mean(0))
    Zn = Zc / s
    v[:, 0] += du * Zn / fx
    v[:, 1] += dv * Zn / fy
    return v


renderer = None
frames = [ONLY] if ONLY >= 0 else range(N)
for fi in frames:
    fr = fd[fi]
    clean = os.path.join(FRAMES_DIR, f"{fi + 1:06d}.jpg")
    rgb = cv2.imread(clean)[..., ::-1] if os.path.isfile(clean) else np.ascontiguousarray(fr["img_rgb"])
    H, W = rgb.shape[:2]
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(W, H)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
    verts_list = fr.get("vertices_3d") or []
    j3l = fr.get("joints_3d_pred") or []; j2l = fr.get("joints_2d_pred") or []
    n_hand = 0
    for hi, vv in enumerate(verts_list):
        vv = np.asarray(vv, float)
        if vv.shape[0] != 778:
            continue
        if hi < len(j3l) and hi < len(j2l):
            j3 = np.asarray(j3l[hi], float); j2 = np.asarray(j2l[hi], float)
            if j3.shape[0] >= 21 and j2.shape[0] >= 21:
                vv = anchor(vv, j3, j2)
        tm = trimesh.Trimesh(vertices=vv, faces=faces, process=False)
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1, 1, 1, 1.0),
                                                 metallicFactor=0.1, roughnessFactor=0.6)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))
        n_hand += 1
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if n_hand:
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=20.0)
        scene.add(cam, pose=CAM_POSE)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=CAM_POSE)
        color, depth = renderer.render(scene)
        mask = depth > 0
        col_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        out[mask] = (ALPHA * col_bgr[mask] + (1 - ALPHA) * out[mask]).astype(np.uint8)
    cv2.putText(out, f"XP Robotics  |  MANO hand mesh  |  frame {fi}/{N-1}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUTDIR}/{fi:06d}.png", out)
if renderer:
    renderer.delete()
print("rendered", 1 if ONLY >= 0 else N, "aligned frame(s) ->", OUTDIR)
