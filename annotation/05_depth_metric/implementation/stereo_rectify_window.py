#!/usr/bin/env python
"""
Stereo rectification for the front fisheye pair (HITNet input prep).

Decodes the front stereo frames (3840x1200 = two 1920x1200 eyes) for a window,
rectifies BOTH eyes to a common pinhole with aligned epipolar rows, and writes
matched left/right pairs. The stored P1/P2/Q in the calib are degenerate, so we
rebuild the projection from the reliable R1/R2 rotations + a chosen focal.

Outputs:
  <out>/left/000001.png, <out>/right/000001.png   rectified rows-aligned pairs
  <out>/stereo_rect.json                            {focal, cx, cy, baseline_mm, size}

Usage: stereo_rectify_window.py <front.h265> <out_dir>
       [--start 250 --num 360 --stride 2 --focal 430 --ow 960 --oh 600]
"""
import argparse, json, os, subprocess, glob
import numpy as np, cv2

CALIB = "/home/raush/Documents/Ego_Infinity/front_camera_calibration.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h265"); ap.add_argument("out")
    ap.add_argument("--start", type=int, default=250)
    ap.add_argument("--num", type=int, default=360)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--focal", type=float, default=430.0)
    ap.add_argument("--ow", type=int, default=960)
    ap.add_argument("--oh", type=int, default=600)
    a = ap.parse_args()

    c = np.load(CALIB, allow_pickle=True)
    Kl, Dl = c["K_left"].astype(np.float64), c["D_left"].astype(np.float64).reshape(4, 1)
    Kr, Dr = c["K_right"].astype(np.float64), c["D_right"].astype(np.float64).reshape(4, 1)
    R1, R2 = c["R1"].astype(np.float64), c["R2"].astype(np.float64)
    base_mm = float(np.linalg.norm(c["T"]))
    OW, OH, f = a.ow, a.oh, a.focal
    P1 = np.array([[f, 0, OW / 2, 0], [0, f, OH / 2, 0], [0, 0, 1, 0]], np.float64)
    P2 = P1.copy(); P2[0, 3] = -f * base_mm
    m1x, m1y = cv2.fisheye.initUndistortRectifyMap(Kl, Dl, R1, P1, (OW, OH), cv2.CV_16SC2)
    m2x, m2y = cv2.fisheye.initUndistortRectifyMap(Kr, Dr, R2, P2, (OW, OH), cv2.CV_16SC2)

    ldir = os.path.join(a.out, "left"); rdir = os.path.join(a.out, "right")
    os.makedirs(ldir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
    raw = os.path.join(a.out, "_raw"); os.makedirs(raw, exist_ok=True)
    b0, b1 = a.start, a.start + a.num - 1
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", a.h265,
                    "-vf", f"select='between(n,{b0},{b1})'", "-vsync", "0", "-q:v", "2",
                    os.path.join(raw, "%06d.png")], check=True)
    files = sorted(glob.glob(os.path.join(raw, "*.png"))); o = 1
    for i, fn in enumerate(files):
        if i % a.stride:
            continue
        full = cv2.imread(fn)
        L = full[:, :1920]; Rr = full[:, 1920:]
        cv2.imwrite(os.path.join(ldir, f"{o:06d}.png"), cv2.remap(L, m1x, m1y, cv2.INTER_LINEAR))
        cv2.imwrite(os.path.join(rdir, f"{o:06d}.png"), cv2.remap(Rr, m2x, m2y, cv2.INTER_LINEAR))
        o += 1
    meta = {"focal_px": f, "cx": OW / 2, "cy": OH / 2, "baseline_mm": base_mm,
            "width": OW, "height": OH, "n_pairs": o - 1,
            "depth_from_disparity": "depth_mm = focal_px * baseline_mm / disparity_px"}
    json.dump(meta, open(os.path.join(a.out, "stereo_rect.json"), "w"), indent=2)
    import shutil; shutil.rmtree(raw, ignore_errors=True)
    print(f"wrote {o-1} rectified pairs -> {a.out}  (f={f}, baseline={base_mm:.1f}mm)")


if __name__ == "__main__":
    main()
