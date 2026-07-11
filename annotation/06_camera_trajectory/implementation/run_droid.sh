#!/bin/bash
# Stage 06 — DROID-SLAM camera trajectory from rectified mono frames + intrinsics.
# Usage: run_droid.sh <frames_dir> <camera.json> <out_dir>
set -uo pipefail
ROOT=/home/raush/Documents/Ego_Infinity
DROID="$ROOT/annotation/06_camera_trajectory/implementation/DROID-SLAM"
VENV="$ROOT/droid-venv"
FRAMES="${1:?frames_dir}"; CAM="${2:?camera.json}"; OUT="${3:?out_dir}"
mkdir -p "$OUT"

# pinhole intrinsics -> DROID calib.txt ("fx fy cx cy")
FX=$("$VENV/bin/python" -c "import json;c=json.load(open('$CAM'));print(c['fx'],c['fy'],c['cx'],c['cy'])")
echo "$FX" > "$OUT/calib.txt"
echo "[droid] calib: $FX"

source "$VENV/bin/activate"
cd "$DROID"
python demo.py \
  --imagedir "$FRAMES" \
  --calib "$OUT/calib.txt" \
  --weights "$ROOT/annotation/models/droid_slam/droid.pth" \
  --stride 1 --disable_vis \
  --reconstruction_path "$OUT/recon"
echo "[droid] done -> $OUT/recon"
