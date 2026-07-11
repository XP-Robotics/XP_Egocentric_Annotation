#!/usr/bin/env python
"""
Stage 05 — depth visualisation tuned for egocentric close-range scenes.

- Uses a robust per-clip range (p_lo..p_hi percentiles of valid depth) so the
  desk/hand contrast is visible instead of being crushed into the far end.
- Alpha-blends the depth colormap over the input frame so the scene shows through.

Usage:
  render_depth_overlay.py moge <pkl> <frames_dir> <out_mp4> [alpha] [plo] [phi]
  render_depth_overlay.py da2  <npy> <frames_dir> <out_mp4> [alpha] [plo] [phi]
"""
import sys, os, glob, gzip, pickle
import numpy as np, cv2

MODE, SRC, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.55
PLO = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
PHI = float(sys.argv[7]) if len(sys.argv) > 7 else 95.0
# 8th arg "norm" -> temporal normalisation: divide each frame by its EMA-smoothed
# median depth so the global colour scale is stable frame-to-frame (kills flicker).
NORM = (len(sys.argv) > 8 and sys.argv[8] == "norm")
LABEL = {"moge": "MoGe-2", "da2": "Depth Anything V2"}[MODE]


def load_depths():
    if MODE == "da2":
        arr = np.load(SRC)                       # (N,H,W) metres
        return [arr[i] for i in range(arr.shape[0])]
    d = pickle.load(gzip.open(SRC, "rb"))
    out = []
    for f in d["frame_data"]:
        x = f["depth_png"]
        dm = (cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_UNCHANGED)
              if isinstance(x, (bytes, bytearray)) else np.asarray(x)).astype(np.float32) / 1000.0
        out.append(dm)
    return out


depths = load_depths()
frames = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))

if NORM:
    # per-frame median, EMA-smoothed -> divide each frame so global scale is constant
    meds = np.array([np.median(d[d > 0.05]) if (d > 0.05).any() else 1.0 for d in depths])
    ema = meds.copy()
    a = 0.15
    for i in range(1, len(ema)):
        ema[i] = a * meds[i] + (1 - a) * ema[i - 1]
    depths = [d / s for d, s in zip(depths, ema)]      # now in "x median" units
    LABEL_SUF = " (temporally normalised)"
else:
    ema = None
    LABEL_SUF = ""

# robust clip-wide range from a subsample of valid pixels
samp = np.concatenate([d[d > 0.05][::37] for d in depths[::5] if (d > 0.05).any()])
lo, hi = np.percentile(samp, PLO), np.percentile(samp, PHI)
if hi - lo < 1e-3:
    hi = lo + 1.0
unit = "x med" if NORM else "m"
print(f"stage05/{MODE}: range {lo:.2f}-{hi:.2f} {unit} (p{PLO:.0f}-p{PHI:.0f}), "
      f"alpha={ALPHA}, norm={NORM}")

tmp = OUT + "_frames"; os.makedirs(tmp, exist_ok=True)
for i, (dm, fp) in enumerate(zip(depths, frames)):
    rgb = cv2.imread(fp)
    H, W = rgb.shape[:2]
    if dm.shape != (H, W):
        dm = cv2.resize(dm, (W, H), interpolation=cv2.INTER_NEAREST)
    valid = dm > 0.05
    vis = np.clip((dm - lo) / (hi - lo), 0, 1)
    cm = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    # blend only where depth is valid; keep raw input elsewhere
    out = rgb.copy()
    m3 = valid[..., None]
    blended = (ALPHA * cm + (1 - ALPHA) * rgb).astype(np.uint8)
    out = np.where(m3, blended, rgb)
    tag = (f"{LABEL} depth{LABEL_SUF}" if NORM
           else f"{LABEL} metric depth  {lo:.1f}-{hi:.1f} m")
    cv2.putText(out, tag, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), out)

os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 15 -i {tmp}/%06d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 {OUT}")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {OUT} -vf "
          f"\"fps=8,scale=440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
          f"{OUT.replace('.mp4','.gif')}")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(f"wrote {OUT}")
