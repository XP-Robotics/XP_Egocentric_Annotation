#!/bin/bash
# Render all applicable annotation stages for ONE scene into by_input/<scene>/.
# Usage: render_scene.sh <scene> <pkl> <frames_dir> <input_mp4> [nd_frames_dir]
set -uo pipefail
ROOT=/home/raush/Documents/Ego_Infinity
cd "$ROOT"; source egoinfinity_env.sh 2>/dev/null
SCENE="$1"; PKL="$(realpath "$2")"; FR="$(realpath "$3")"; INP="$(realpath "$4")"
NDF="$(realpath "${5:-$3}")"
A=annotation
OUT="$ROOT/annoatation_all/$SCENE"; mkdir -p "$OUT"
SS="$(echo "$SCENE" | tr '/ ' '__')"      # slash/space-free id for temp files
enc(){ ffmpeg -y -hide_banner -loglevel error -framerate 15 -i "$1/%06d.png" -c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 "$2"; }
cp "$INP" "$OUT/00_input.mp4"

echo "[$SCENE] 01 hand detection"
rm -rf /tmp/rs; python $A/01_hand_detection/implementation/render_hand_detection.py "$PKL" "$FR" /tmp/rs >/dev/null 2>&1
enc /tmp/rs/_frames "$OUT/01_hand_detection.mp4"

echo "[$SCENE] 02 hand mesh"
rm -rf /tmp/rs2; python $A/02_hand_mesh_hamer/implementation/render_mano_overlay_aligned.py "$PKL" /tmp/rs2 >/dev/null 2>&1
enc /tmp/rs2 "$OUT/02_hand_mesh.mp4"

echo "[$SCENE] 03 hand tracking"
rm -rf /tmp/rs3; python $A/03_hand_tracking/implementation/render_tracking_2d.py "$PKL" "$FR" /tmp/rs3 >/dev/null 2>&1
enc /tmp/rs3 "$OUT/03_hand_tracking.mp4"

echo "[$SCENE] 04 object segmentation"
rm -rf /tmp/rs4; python $A/04_object_segmentation/implementation/export_object_masks.py "$PKL" "$FR" /tmp/rs4 >/dev/null 2>&1
enc /tmp/rs4/_mask "$OUT/04_object_segmentation.mp4"

echo "[$SCENE] 05 depth (MoGe + Depth Anything V2)"
python $A/05_depth_metric/implementation/render_depth_overlay.py moge "$PKL" "$FR" "/tmp/${SS}_moge.mp4" 0.55 25 75 norm >/dev/null 2>&1
cp "/tmp/${SS}_moge.mp4" "$OUT/05_depth_moge2.mp4" 2>/dev/null
rm -rf /tmp/rs5; python $A/05_depth_metric/implementation/run_depth_anything.py "$FR" /tmp/rs5 >/dev/null 2>&1
python $A/05_depth_metric/implementation/render_depth_overlay.py da2 /tmp/rs5/depth_da2_metres.npy "$FR" "/tmp/${SS}_da2.mp4" 0.55 25 75 norm >/dev/null 2>&1
cp "/tmp/${SS}_da2.mp4" "$OUT/05_depth_anything_v2.mp4" 2>/dev/null

echo "[$SCENE] 06 camera trajectory (DROID-SLAM)"
"$ROOT/egoinfinity-venv/bin/python" -c "import gzip,pickle,json;d=pickle.load(gzip.open('$PKL','rb'));json.dump({'fx':float(d['dp_focal']),'fy':float(d['dp_focal']),'cx':float(d['cx']),'cy':float(d['cy'])},open('/tmp/cam_${SS}.json','w'))"
rm -rf "/tmp/droid_${SS}"
bash $A/06_camera_trajectory/implementation/run_droid.sh "$FR" "/tmp/cam_${SS}.json" "/tmp/droid_${SS}" >/dev/null 2>&1
"$ROOT/droid-venv/bin/python" $A/06_camera_trajectory/implementation/plot_traj.py "/tmp/droid_${SS}/recon" "$OUT/06_camera_trajectory.png" >/dev/null 2>&1 || echo "  [06 skipped]"

echo "[$SCENE] 07 active object"
rm -rf /tmp/rs7; python $A/07_active_object/implementation/active_object.py "$PKL" "$FR" /tmp/rs7 >/dev/null 2>&1
cp /tmp/rs7/active_object.mp4 "$OUT/07_active_object.mp4" 2>/dev/null

echo "[$SCENE] 08 6DoF pose"
rm -rf /tmp/rs8; python $A/08_object_6dof_pose/implementation/render_6dof_pose.py "$PKL" "$FR" /tmp/rs8 >/dev/null 2>&1
cp /tmp/rs8/object_6dof_pose.mp4 "$OUT/08_object_6dof_pose.mp4" 2>/dev/null

echo "[$SCENE] 09 gaze proxy"
rm -rf /tmp/rs9; python $A/09_gaze/implementation/attention_proxy.py "$PKL" "$FR" /tmp/rs9 >/dev/null 2>&1
cp /tmp/rs9/gaze_attention_proxy.mp4 "$OUT/09_gaze.mp4" 2>/dev/null

echo "[$SCENE] 10 neurodepth"
python $A/10_neurodepth/implementation/render_neurodepth.py "$NDF" "/tmp/${SS}_nd.mp4" --fps 15 >/dev/null 2>&1
cp "/tmp/${SS}_nd.mp4" "$OUT/10_neurodepth.mp4" 2>/dev/null

echo "[$SCENE] data (JSON + meshes + tracking.json + metrics.json)"
python $A/export_scene_data.py "$PKL" "$OUT" >/dev/null 2>&1
cp /tmp/rs7/active_object.json "$OUT/active_object.json" 2>/dev/null
cp /tmp/rs9/attention_proxy.json "$OUT/attention_proxy.json" 2>/dev/null

rm -f "$OUT"/*.gif                       # keep MP4s only (no gifs)
echo "[$SCENE] done -> $OUT"; ls "$OUT"
