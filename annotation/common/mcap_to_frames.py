#!/usr/bin/env python
"""
mcap_to_frames.py — Ego-video ingest for the annotation showcase.

Reads a multi-fisheye egocentric .mcap (foxglove.CompressedVideo, H.265),
extracts the FRONT camera's left eye, and rectifies it with a TILTED virtual
pinhole camera so both hands sit large and centred on the desk (the projection
that makes HaMeR / depth / SLAM behave well on a body-worn view).

Outputs:
  <out_dir>/frames/000001.jpg ...      undistorted RGB frames (1280x960, 15 fps)
  <out_dir>/input.mp4                  encoded preview of those frames
  <out_dir>/camera.json                pinhole intrinsics of the rectified frames

Usage:
  python mcap_to_frames.py <input.mcap> <out_dir> \
      [--calib front_camera_calibration.npz] [--topic /camera/front/image_raw/compressed] \
      [--start-frame N] [--num 360] [--stride 2] [--pitch -30] [--focal 530]

The rectified frames carry a *known* focal length (--focal) — pass the same value
to the depth stage as a known-focal prior so metric 3D is geometrically correct.
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np

# ---- config: real front-camera fisheye calibration (Kannala-Brandt, per-eye) ----
DEFAULT_CALIB = "/home/raush/Documents/Ego_Infinity/front_camera_calibration.npz"
DEFAULT_TOPIC = "/camera/front/image_raw/compressed"
EYE_W, EYE_H = 1920, 1200          # left-eye crop from the side-by-side stereo frame
OUT_W, OUT_H = 1280, 960           # rectified pinhole output size


def _field3(b):
    """Extract field 3 (bytes `data`) from a foxglove.CompressedVideo protobuf msg."""
    i = 0
    while i < len(b):
        tag = b[i]; i += 1
        fn, wt = tag >> 3, tag & 7
        if wt == 2:
            l = 0; s = 0
            while True:
                by = b[i]; i += 1; l |= (by & 0x7f) << s; s += 7
                if not by & 0x80:
                    break
            val = b[i:i + l]; i += l
            if fn == 3:
                return val
        elif wt == 0:
            while b[i] & 0x80:
                i += 1
            i += 1
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return None


def extract_front_h265(mcap_path, topic, out_h265):
    """Concatenate the H.265 bitstream for `topic` into a raw .265 file."""
    from mcap.reader import make_reader
    n = 0
    with open(out_h265, "wb") as fo, open(mcap_path, "rb") as fi:
        reader = make_reader(fi)
        # resolve channel id for the topic
        summary = reader.get_summary()
        chan_ids = {ch.id for ch in summary.channels.values() if ch.topic == topic}
        if not chan_ids:
            topics = sorted({ch.topic for ch in summary.channels.values()})
            raise SystemExit(f"topic {topic!r} not found. Available: {topics}")
        for _schema, ch, msg in reader.iter_messages():
            if ch.id not in chan_ids:
                continue
            data = _field3(msg.data)
            if data:
                fo.write(data); n += 1
    return n


def build_maps(calib_path, pitch_deg, focal):
    import cv2
    c = np.load(calib_path, allow_pickle=True)
    K = c["K_left"].astype(np.float64)
    D = c["D_left"].astype(np.float64).reshape(4, 1)
    t = np.deg2rad(pitch_deg); ct, st = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], np.float64)   # tilt up toward desk
    Knew = np.array([[focal, 0, OUT_W / 2], [0, focal, OUT_H / 2], [0, 0, 1]], np.float64)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, Knew, (OUT_W, OUT_H), cv2.CV_16SC2)
    return m1, m2, Knew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("out_dir")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num", type=int, default=360, help="source frames to take")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth (2 -> 15fps from 30fps)")
    ap.add_argument("--pitch", type=float, default=-30.0)
    ap.add_argument("--focal", type=float, default=530.0)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    import cv2
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    h265 = os.path.join(args.out_dir, "_front.h265")
    print(f"[1/4] extracting {args.topic} -> {h265}")
    total = extract_front_h265(args.mcap, args.topic, h265)
    print(f"      {total} frames in bitstream")

    # decode the requested window (left eye) to temp jpgs
    raw_dir = os.path.join(args.out_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    a, b = args.start_frame, args.start_frame + args.num - 1
    print(f"[2/4] decoding source frames {a}..{b} (left eye {EYE_W}x{EYE_H})")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", h265,
         "-vf", f"select='between(n,{a},{b})',crop={EYE_W}:{EYE_H}:0:0",
         "-vsync", "0", "-q:v", "2", os.path.join(raw_dir, "%06d.jpg")], check=True)

    m1, m2, Knew = build_maps(args.calib, args.pitch, args.focal)
    print(f"[3/4] rectifying (tilt pitch={args.pitch}, focal={args.focal}) -> {frames_dir}")
    raw = sorted(os.listdir(raw_dir)); o = 1
    for i, fn in enumerate(raw):
        if i % args.stride:
            continue
        im = cv2.imread(os.path.join(raw_dir, fn))
        und = cv2.remap(im, m1, m2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(frames_dir, f"{o:06d}.jpg"), und); o += 1
    n_out = o - 1
    print(f"      wrote {n_out} rectified frames")

    # encode preview + write intrinsics
    mp4 = os.path.join(args.out_dir, "input.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", os.path.join(frames_dir, "%06d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
    cam = {"model": "pinhole", "width": OUT_W, "height": OUT_H,
           "fx": float(Knew[0, 0]), "fy": float(Knew[1, 1]),
           "cx": float(Knew[0, 2]), "cy": float(Knew[1, 2]),
           "fps": args.fps, "n_frames": n_out,
           "source": {"mcap": os.path.abspath(args.mcap), "topic": args.topic,
                      "eye": "left", "pitch_deg": args.pitch,
                      "rectified_focal_px": args.focal,
                      "calib": os.path.abspath(args.calib)}}
    with open(os.path.join(args.out_dir, "camera.json"), "w") as f:
        json.dump(cam, f, indent=2)
    print(f"[4/4] done: {n_out} frames, {mp4}, camera.json (focal={args.focal}px)")

    # cleanup bulky temporaries
    import shutil
    shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
