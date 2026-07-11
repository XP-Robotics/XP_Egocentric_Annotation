#!/usr/bin/env python
"""
NeuroDepth — neuromorphic (event-camera) processing simulated from RGB video.

From an ordinary camera stream it derives the signals a neuromorphic vision
front-end produces, in four panels:
  RAW CAMERA          - the input frame
  ON/OFF EVENTS       - per-pixel log-intensity change; ON (+) = red, OFF (-) = green
  TIME SURFACE        - exponentially-decaying map of the most recent event time
  RECONSTRUCTED EDGES - motion/contrast edges accumulated from the event stream

HUD mimics an event-core readout (mode, event density, key events, fps, latency).

Usage: render_neurodepth.py <frames_dir> <out_mp4> [--fps 30 --thresh 0.12 --tau 6]
"""
import os, sys, glob, argparse
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--thresh", type=float, default=0.12)   # log-intensity change to fire an event
ap.add_argument("--tau", type=float, default=6.0)       # time-surface decay (frames)
ap.add_argument("--pw", type=int, default=440)          # panel width
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
PW = args.pw
# panel height from input aspect
h0, w0 = cv2.imread(files[0]).shape[:2]
PH = int(PW * h0 / w0)

last_t = np.full((PH, PW), -1e9, np.float32)     # last event time per pixel
edge_acc = np.zeros((PH, PW), np.float32)        # decaying event accumulation
prev_log = None
tmp = args.out + "_frames"; os.makedirs(tmp, exist_ok=True)


def panel(img, title, tl=(0, 0)):
    cv2.rectangle(img, (0, 0), (PW - 1, PH - 1), (60, 60, 60), 1)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


for i, fp in enumerate(files):
    raw = cv2.resize(cv2.imread(fp), (PW, PH))
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    log = np.log(gray + 1.0)
    # ---- ON/OFF events (log-intensity change polarity) ----
    ev = np.zeros((PH, PW), np.int8)
    if prev_log is not None:
        d = log - prev_log
        ev[d > args.thresh] = 1
        ev[d < -args.thresh] = -1
    prev_log = log
    n_on = int((ev == 1).sum()); n_off = int((ev == -1).sum()); n_ev = n_on + n_off
    onoff = np.zeros((PH, PW, 3), np.uint8)
    onoff[ev == 1] = (40, 40, 255)      # ON  -> red   (BGR)
    onoff[ev == -1] = (40, 255, 40)     # OFF -> green
    # ---- time surface ----
    fired = ev != 0
    last_t[fired] = i
    ts = np.exp(-(i - last_t) / args.tau)               # 0..1, recent = high
    ts_c = cv2.applyColorMap((np.clip(ts, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ts_c[ts < 0.02] = (40, 0, 0)
    # ---- reconstructed edges (accumulated events -> contours) ----
    edge_acc = edge_acc * 0.6 + fired.astype(np.float32)
    em = np.clip(edge_acc * 200, 0, 255).astype(np.uint8)
    edges = np.zeros((PH, PW, 3), np.uint8)
    edges[..., 0] = em; edges[..., 1] = (em * 0.7).astype(np.uint8)   # cyan-ish
    # ---- compose: raw on top, 3 panels below ----
    raw = panel(raw.copy(), "RAW CAMERA")
    row = np.hstack([panel(onoff, "ON/OFF EVENTS"),
                     panel(ts_c, "TIME SURFACE"),
                     panel(edges, "RECONSTRUCTED EDGES")])
    top = np.full((PH, row.shape[1], 3), 0, np.uint8)
    x0 = (row.shape[1] - PW) // 2
    top[:, x0:x0 + PW] = raw
    canvas = np.vstack([top, row])
    # ---- HUD ----
    density = 100.0 * n_ev / (PH * PW)
    lat = 20 + 20 * np.tanh(n_ev / 4e4)                 # simulated pipeline latency (ms)
    cv2.putText(canvas, f"NeuroDepth v0.1  |  mode=high_speed  events={n_ev:6d}  "
                        f"on={n_on}  off={n_off}  density={density:4.1f}%",
                (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"recv_fps={args.fps:.0f}   pipeline={lat:4.1f}ms   frame {i:04d}",
                (canvas.shape[1] - 360, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), canvas)

os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate {args.fps} -i {tmp}/%06d.png "
          f"-c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 {args.out}")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {args.out} -vf "
          f"\"fps=10,scale=520:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
          f"{args.out.replace('.mp4','.gif')}")
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(f"neurodepth -> {args.out}  ({len(files)} frames)")
