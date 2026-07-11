#!/usr/bin/env python
"""
HITNet stereo depth -> full-frame HEATMAP (reference style).

- Full JET heatmap (near = red/warm, far = blue/cool).
- Temporal EMA-median normalisation -> stable colour scale (no flicker).
- Reduced (tight) percentile range, tuned for close egocentric scenes.
- HUD: frame, camera/head-motion state, depth frame.
- Distance labels (absolute metres) on the nearest object blobs.

Usage: render_hitnet_heatmap.py <depth_npz> <left_frames_dir> <out_mp4>
       [--plo 20 --phi 80 --ema 0.15]
"""
import sys, os, glob, argparse
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("left"); ap.add_argument("out")
ap.add_argument("--plo", type=float, default=20.0)   # tighter = reduced range
ap.add_argument("--phi", type=float, default=80.0)
ap.add_argument("--ema", type=float, default=0.15)
ap.add_argument("--tsmooth", type=int, default=5)    # per-pixel temporal window (odd); kills flicker
ap.add_argument("--no-labels", action="store_true")  # hide the per-object distance labels
a = ap.parse_args()

dep = np.load(a.npz)["depth_m"].astype(np.float32)     # (N,H,W) metric metres
left = sorted(glob.glob(os.path.join(a.left, "*.png")))
N = dep.shape[0]
VMAX = 5.0


def vmask(d):
    return (d > 0.05) & (d < VMAX)


# --- per-pixel TEMPORAL smoothing: centred moving average over valid pixels ---
# HITNet runs each frame independently -> frame-to-frame flicker. Averaging each
# pixel over a small temporal window removes the jitter without global lag.
if a.tsmooth and a.tsmooth > 1:
    w = a.tsmooth // 2
    valid = ((dep > 0.05) & (dep < VMAX)).astype(np.float32)
    dv = np.where(valid > 0, dep, 0.0)
    sm = np.zeros_like(dep); cnt = np.zeros_like(dep)
    for k in range(-w, w + 1):
        sm += np.roll(dv, k, axis=0); cnt += np.roll(valid, k, axis=0)
    dep = np.where(cnt > 0, sm / np.maximum(cnt, 1e-6), dep)
    print(f"hitnet heatmap: temporal smoothing window={a.tsmooth}")


# camera/head motion state from inter-frame change
prev = None; motion = np.zeros(N)
for i, fp in enumerate(left):
    g = cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY), (160, 100)).astype(np.float32)
    if prev is not None:
        motion[i] = np.mean(np.abs(g - prev))
    prev = g
mk = np.convolve(motion, np.ones(5) / 5, mode="same")
t1, t2 = np.percentile(mk[mk > 0], 50), np.percentile(mk[mk > 0], 90)
st = lambda m: "static" if m < t1 else ("normal" if m < t2 else "moving")

# temporal normalisation: divide each frame by its EMA-smoothed median depth
meds = np.array([np.median(d[vmask(d)]) if vmask(d).any() else 1.0 for d in dep])
ema = meds.copy()
for i in range(1, N):
    ema[i] = a.ema * meds[i] + (1 - a.ema) * ema[i - 1]
depn = [d / s for d, s in zip(dep, ema)]

samp = np.concatenate([d[vmask(d)][::37] for d in depn if vmask(d).any()])
lo, hi = np.percentile(samp, a.plo), np.percentile(samp, a.phi)
print(f"hitnet heatmap: norm range {lo:.2f}-{hi:.2f} x-med (p{a.plo:.0f}-p{a.phi:.0f})")

tmp = a.out + "_frames"; os.makedirs(tmp, exist_ok=True)
for i, (dn, dm, fp) in enumerate(zip(depn, dep, left)):
    base = cv2.imread(fp); H, W = base.shape[:2]
    valid = vmask(dn)
    vis = 1.0 - np.clip((dn - lo) / (hi - lo), 0, 1)     # near -> 1 -> JET red
    heat = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[~valid] = (0, 0, 0)
    # faint edges so hands/objects read through the heatmap
    edge = cv2.cvtColor(cv2.Canny(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), 40, 120), cv2.COLOR_GRAY2BGR)
    frame = cv2.addWeighted(heat, 0.9, (0.12 * base).astype(np.uint8), 1.0, 0)
    frame = cv2.addWeighted(frame, 1.0, edge, 0.18, 0)

    # HUD
    cv2.putText(frame, f"frame {i:04d}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"camera/head state: {st(mk[i])}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"depth frame: {i:04d}", (10, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)

    # distance labels on the nearest blobs (absolute metres from raw depth)
    if a.no_labels:
        cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), frame)
        continue
    near = valid & (dm < np.percentile(dm[valid], 25))
    n = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnt, _ = cv2.findContours(n, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:2]
    for c in cnt:
        if cv2.contourArea(c) < 800:
            continue
        M = cv2.moments(c); cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        dist = float(np.median(dm[valid & (cv2.drawContours(np.zeros((H, W), np.uint8), [c], -1, 1, -1) > 0)]))
        txt = f"{dist:.2f}m"
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
print(f"done -> {a.out}")
