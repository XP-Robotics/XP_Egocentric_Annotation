#!/bin/bash
# Egocentric perception stack — run every stage on an ingested clip.
# Usage: bash run_all.sh <ingest_dir>            (dir from common/mcap_to_frames.py)
#        bash run_all.sh <input.mcap> --ingest   (ingest first, then run)
set -uo pipefail
ROOT=/home/raush/Documents/Ego_Infinity
cd "$ROOT"
source egoinfinity_env.sh

ING="${1:?usage: run_all.sh <ingest_dir | input.mcap --ingest>}"
if [ "${2:-}" == "--ingest" ]; then
    OUT=annotation/outputs/ingest
    python annotation/common/mcap_to_frames.py "$ING" "$OUT" \
        --start-frame "${START:-250}" --num "${NUM:-360}" --stride 2 --pitch -30 --focal 530
    ING="$OUT"
fi
FR="$ING/frames"
FOCAL=$(python -c "import json;print(int(json.load(open('$ING/camera.json'))['fx']))")

# Shared engine: HaMeR hands + SAM3.1 objects + MoGe depth (known-focal prior)
CLIP=artifacts/annot_run
rm -rf "$CLIP"; mkdir -p "$CLIP/extract_frames/frames" "$CLIP/frames"
cp "$FR"/*.jpg "$CLIP/extract_frames/frames/"; cp "$FR"/*.jpg "$CLIP/frames/"
# NOTE: edit manifest.json objects[] for your scene's text prompts before running.
cp annotation/outputs/ingest/../_manifest_template.json "$CLIP/manifest.json" 2>/dev/null || true
EGOINFINITY_KNOWN_FOCAL=$FOCAL bash run_clip_sam3.sh "$ROOT/$CLIP" --no-resume
PKL="$CLIP/pipeline_result.pkl.gz"

# Per-stage outputs
A=annotation
python $A/01_hand_detection/implementation/render_hand_detection.py $PKL "$CLIP/extract_frames/frames" $A/01_hand_detection/outputs
python $A/02_hand_mesh_hamer/implementation/export_meshes.py $PKL $A/02_hand_mesh_hamer/outputs/hand_meshes
python $A/02_hand_mesh_hamer/implementation/render_mano_overlay.py $PKL /tmp/_mano
python $A/03_hand_tracking/implementation/render_tracking_2d.py $PKL "$CLIP/extract_frames/frames" /tmp/_track
python $A/04_object_segmentation/implementation/export_object_masks.py $PKL "$CLIP/extract_frames/frames" /tmp/_obj
python $A/05_depth_metric/implementation/render_depth_overlay.py moge $PKL "$FR" $A/05_depth_metric/outputs/depth_moge2.mp4 0.55 5 92
python $A/05_depth_metric/implementation/run_depth_anything.py "$FR" /tmp/_da2
python $A/05_depth_metric/implementation/render_depth_overlay.py da2 /tmp/_da2/depth_da2_metres.npy "$FR" $A/05_depth_metric/outputs/depth_anything_v2.mp4 0.55 5 92

# Stage 06 — camera trajectory (separate venv; see 06_camera_trajectory/implementation)
bash $A/06_camera_trajectory/implementation/run_droid.sh "$FR" "$ING/camera.json" \
     $A/06_camera_trajectory/outputs 2>/dev/null || echo "[stage06] DROID-SLAM step skipped/failed — see its README"

echo "done — see annotation/*/outputs and annotation/outputs/"
