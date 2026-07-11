# Stage 01 — Implementation

`render_hand_detection.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>`

Draws per-frame hand boxes (from the 2D hand joints, padded) with LEFT/RIGHT
labels + confidence. Writes `hand_detection.mp4/.gif` and `hand_detections.json`.
Detector weights: `../../models/yolo_hand_detector.pt` (WiLoR-family).
