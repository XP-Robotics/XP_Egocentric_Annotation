# Egocentric Video Perception Stack — XP Robotics

A modular perception stack for **egocentric (head/body-worn) RGB video**. Each
stage recovers one layer of hand–object interaction understanding; together they
produce a temporally consistent 4D reconstruction that can drive robot learning.

Input is a multi-fisheye `.mcap` capture. The front camera is rectified to a
tilted virtual pinhole (both hands large and centred on the workspace) and fed
through the stages below.

---

## Recommended stack for ego video

| # | Stage | Model | Folder | Status |
|---|---|---|---|---|
| 01 | Hand detection + side (L/R) | YOLO hand detector (WiLoR-family) | [`01_hand_detection/`](01_hand_detection/) | ✅ |
| 02 | 3D hand mesh (MANO) | **HaMeR** | [`02_hand_mesh_hamer/`](02_hand_mesh_hamer/) | ✅ |
| 03 | Hand tracking over time | SAM 2 propagation + HaMeR per frame | [`03_hand_tracking/`](03_hand_tracking/) | ✅ |
| 04 | Object segmentation (open-vocab) | Grounded-SAM → SAM 2 / SAM 3.1 | [`04_object_segmentation/`](04_object_segmentation/) | ✅ |
| 05 | Depth (metric) | MoGe-2 + **Depth Anything V2** + **HITNet** (stereo) | [`05_depth_metric/`](05_depth_metric/) | ✅ |
| 06 | Camera trajectory / egomotion | **DROID-SLAM** | [`06_camera_trajectory/`](06_camera_trajectory/) | ✅ (near-static clip) |
| 07 | Active / next-active object | **100DOH** (learned) + contact-geometry | [`07_active_object/`](07_active_object/) | ✅ |
| 08 | Object 6DoF pose | mask+depth PCA (+FoundationPose upgrade) | [`08_object_6dof_pose/`](08_object_6dof_pose/) | ✅ |
| 09 | Gaze | attention proxy (eye-tracking = hardware-gated) | [`09_gaze/`](09_gaze/) | ◑ proxy |
| 10 | NeuroDepth (event/neuromorphic) | simulated event camera (ON/OFF, time surface, edges) | [`10_neurodepth/`](10_neurodepth/) | ✅ |

Depth stage ships **three** backends: two monocular (MoGe-2, Depth Anything V2) and
one **stereo/triangulated** (HITNet, using both fisheye eyes + the 62 mm baseline).

**All 9 table stages implemented.** Active-object & 6DoF pose use CAD-free geometry
methods (100DOH / FoundationPose are documented upgrade paths); gaze ships as a
manipulation-attention proxy pending an eye-tracking hardware stream.

---

## Folder layout

```
annotation/
├── models/                     ← ALL model weights live here (one place)
│   ├── hamer/  sam3.1/  sam2.1_hiera_small.pt  wilor_final.ckpt
│   ├── depth_anything_v2/  droid_slam/  grounded_sam/  moge2/
├── common/
│   └── mcap_to_frames.py       ← ego .mcap → rectified pinhole frames + camera.json
├── 01_hand_detection/
│   ├── implementation/         ← runnable code
│   ├── docs/CLIENT_BRIEF.md    ← client-facing capability sheet
│   └── outputs/                ← results for the delivered clip
├── 02_hand_mesh_hamer/ …       ← same shape for every stage
│   …
├── outputs/                    ← combined montage + master metrics
└── run_all.sh                  ← ingest an .mcap and run every stage
```

Every stage folder is self-contained: **`implementation/`** (how it runs),
**`docs/`** (what it delivers, for the client), **`outputs/`** (results).

---

## Running the stack

```bash
# 1. Ingest an egocentric .mcap → rectified frames + intrinsics
python common/mcap_to_frames.py <input.mcap> outputs/ingest \
    --start-frame 1400 --num 360 --stride 2 --pitch -30 --focal 530

# 2. Run every stage on the ingested frames
bash run_all.sh outputs/ingest
```

Each stage reads `outputs/ingest/frames/` + `outputs/ingest/camera.json` and
writes to its own `outputs/`. The rectified frames carry a **known focal length**
(`--focal`), which is passed to the depth stage as a prior so metric 3D is
geometrically correct (flat surfaces stay flat).

---

## Why this stack

- **Sensor-free 3D** — metric depth + articulated hands from a single RGB stream.
- **Open-vocabulary** — objects specified in plain text, no per-object training.
- **Ego-tuned** — the front-camera rectification and known-focal depth prior are
  what make a body-worn fisheye view usable for hand/object reconstruction.
- **Modular** — every stage is swappable and independently documented.

© XP Robotics — egocentric perception stack.
