#!/usr/bin/env python
"""
Stage 05 — full-frame depth HEATMAP with HUD (like the reference).

- Full-opacity metric-depth heatmap (near = warm/red, far = cool/blue).
- HUD text: frame index, camera/head-motion state, depth frame.
- Per-hand distance labels in metres (median depth at the hand), drawn at the hand.

Usage: render_depth_heatmap.py <pipeline_result.pkl.gz> <frames_dir> <out_mp4>
       [--cmap jet|turbo] [--plo 2] [--phi 98]
"""
import sys, os, gzip, pickle, argparse
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("pkl"); ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--cmap", default="jet")
ap.add_argument("--plo", type=float, default=2.0)
ap.add_argument("--phi", type=float, default=98.0)
a = ap.parse_args()
CMAP = {"jet": cv2.COLORMAP_JET, "turbo": cv2.COLORMAP_TURBO}[a.cmap]

d = pickle.load(gzip.open(a.pkl, "rb"))
fd = d["frame_data"]
N = len(fd)


def dec_depth(x):
    return (cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_UNCHANGED)
            if isinstance(x, (bytes, bytearray)) else np.asarray(x)).astype(np.float32) / 1000.0


# clip-wide robust range for a stable colour scale
samp = []
for f in fd[::5]:
    dm = dec_depth(f["depth_png"]); v = dm[dm > 0.05]
    if v.size:
        samp.append(v[::37])
samp = np.concatenate(samp)
lo, hi = np.percentile(samp, a.plo), np.percentile(samp, a.phi)

# camera/head motion state from inter-frame image change
prev = None; motion = np.zeros(N)
for i in range(N):
    p = os.path.join(a.frames, f"{i+1:06d}.jpg")
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (160, 120))
    if prev is not None:
        motion[i] = np.mean(np.abs(g.astype(np.float32) - prev))
    prev = g.astype(np.float32)
# smooth + classify
mk = np.convolve(motion, np.ones(5) / 5, mode="same")
t1, t2 = np.percentile(mk[mk > 0], 50), np.percentile(mk[mk > 0], 90)


def state(m):
    return "static" if m < t1 else ("normal" if m < t2 else "moving")


tmp = a.out + "_frames"; os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(fd):
    dm = dec_depth(f["depth_png"])
    H0, W0 = dm.shape
    valid = dm > 0.05
    # near = warm/red -> invert so small depth maps to JET high (red)
    vis = 1.0 - np.clip((dm - lo) / (hi - lo), 0, 1)
    heat = cv2.applyColorMap((vis * 255).astype(np.uint8), CMAP)
    heat[~valid] = (0, 0, 0)
    # size to the input frame
    ip = os.path.join(a.frames, f"{i+1:06d}.jpg")
    base = cv2.imread(ip); H, W = base.shape[:2]
    if heat.shape[:2] != (H, W):
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
        vis = cv2.resize(vis, (W, H)); valid = cv2.resize(valid.astype(np.uint8), (W, H)) > 0
        dm = cv2.resize(dm, (W, H), interpolation=cv2.INTER_NEAREST)
    # faint structure from the input so hands/edges read through the heatmap
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edge = cv2.cvtColor(cv2.Canny(g, 40, 120), cv2.COLOR_GRAY2BGR)
    frame = cv2.addWeighted(heat, 0.9, (0.12 * base).astype(np.uint8), 1.0, 0)
    frame = cv2.addWeighted(frame, 1.0, edge, 0.15, 0)

    # HUD
    cv2.putText(frame, f"frame {i:04d}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"camera/head state: {state(mk[i])}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"depth frame: {i:04d}", (10, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)

    # per-hand distance labels
    j2 = f.get("joints_2d_pred") or []; j3 = f.get("joints_3d_pred") or []; sd = f.get("hand_is_right") or []
    for hi in range(len(j2)):
        J2 = np.asarray(j2[hi], float)
        if J2.ndim != 2:
            continue
        cx, cy = int(np.median(J2[:, 0])), int(np.median(J2[:, 1]))
        # distance = median metric depth at the hand joints (fallback to 3D Z)
        zs = []
        for (u, v) in J2.astype(int):
            if 0 <= v < H and 0 <= u < W and dm[v, u] > 0.05:
                zs.append(dm[v, u])
        if zs:
            dist = float(np.median(zs))
        elif hi < len(j3):
            dist = float(np.median(np.asarray(j3[hi], float)[:, 2]))
        else:
            continue
        side = "R" if (hi < len(sd) and sd[hi]) else "L"
        txt = f"{side}: {dist:.2f}m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (cx - 4, cy - th - 6), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
        cv2.putText(frame, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), frame)

os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 15 -i {tmp}/%06d.png "
          f"-c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 {a.out}")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {a.out} -vf "
          f"\"fps=8,scale=440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
          f"{a.out.replace('.mp4','.gif')}")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(f"depth heatmap -> {a.out}  (range {lo:.2f}-{hi:.2f} m, cmap={a.cmap})")
