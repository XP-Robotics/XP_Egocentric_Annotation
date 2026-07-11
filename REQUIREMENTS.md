# Requirement Coverage — Recommended Stack for Ego Video

**XP Robotics** · mapping of the requested ego-video stack to delivered results.

| # | Stage | Requested approach | Delivered | Result |
|---|---|---|---|---|
| 1 | Hand detection + side (L/R) | WiLoR / 100DOH | ✅ | both hands, L/R, 100% of frames — [`01_hand_detection/`](01_hand_detection/) |
| 2 | 3D hand mesh (MANO) | HaMeR | ✅ | per-frame MANO mesh, both hands — [`02_hand_mesh_hamer/`](02_hand_mesh_hamer/) |
| 3 | Hand tracking over time | SAM 2 + HaMeR | ✅ | temporally-stable 2D+3D tracks — [`03_hand_tracking/`](03_hand_tracking/) |
| 4 | Object segmentation (open-vocab) | Grounded-SAM → SAM 2 | ✅ | text-prompted, tracked masks — [`04_object_segmentation/`](04_object_segmentation/) |
| 5 | Active / next-active object | 100DOH detector | ✅ | contact state + active object — [`07_active_object/`](07_active_object/) |
| 6 | Object 6DoF pose | FoundationPose / BundleSDF | ✅ | oriented 3D box + 6DoF pose — [`08_object_6dof_pose/`](08_object_6dof_pose/) |
| 7 | Depth (metric) | Depth Anything v2 / Metric3D / UniDepth | ✅✅✅ | **three** backends inc. **stereo** — [`05_depth_metric/`](05_depth_metric/) |
| 8 | Camera trajectory / egomotion | DROID-SLAM / Aria MPS | ✅ | 6DoF camera path — [`06_camera_trajectory/`](06_camera_trajectory/) |
| 9 | Gaze | eye-tracking / gaze net | ◑ | attention proxy (see note) — [`09_gaze/`](09_gaze/) |

## Coverage: **8 of 9 fully delivered + 1 proxy**

### Notes
- **Depth (row 7):** delivered **three** interchangeable backends — two monocular
  and one **stereo/triangulated** depth that uses both camera eyes and the physical
  baseline for a *measured* (not inferred) depth. This exceeds the single-model ask.
- **Object 6DoF pose (row 6):** delivered CAD-free (works on any object, no per-object
  mesh required). A CAD-anchored high-fidelity variant is available on request.
- **Gaze (row 9):** true gaze requires an **eye-tracking hardware stream**, which
  this capture rig does not provide. Delivered as a clearly-labelled
  **manipulation-attention proxy**; a real gaze stream integrates directly when the
  hardware supplies it.

<div align="center"><sub>© XP Robotics.</sub></div>
