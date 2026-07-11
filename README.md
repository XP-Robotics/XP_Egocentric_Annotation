<div align="center">

# Egocentric Video Perception — Hand · Object · Scene Understanding

**XP Robotics**

A complete perception stack for **egocentric (head/body-worn) RGB video** — turning
a raw wearable-camera clip into hands, objects, depth, interaction labels, 6DoF
poses, and camera motion, in one metric world frame.

📄 [Capability summary](CAPABILITY.md) &nbsp;·&nbsp; ✅ [Requirement coverage](REQUIREMENTS.md) &nbsp;·&nbsp; 🗂️ [Data format](DATA_FORMAT.md) &nbsp;·&nbsp; 📦 [`tracking.json`](tracking.json)

</div>

---

## Results — the full ego stack

<table>
  <tr>
    <td width="33%" align="center"><b>Hand Detection + L/R</b><br><sub>both hands, side-labelled</sub><br><br><img src="01_hand_detection/hand_detection.gif" width="100%"></td>
    <td width="33%" align="center"><b>3D Hand Mesh</b><br><sub>MANO, per frame</sub><br><br><img src="02_hand_mesh_hamer/hand_mesh_overlay.gif" width="100%"></td>
    <td width="33%" align="center"><b>Hand Tracking</b><br><sub>21-joint, temporally stable</sub><br><br><img src="03_hand_tracking/hand_tracking.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Object Segmentation</b><br><sub>open-vocabulary, text-prompted</sub><br><br><img src="04_object_segmentation/object_segmentation.gif" width="100%"></td>
    <td align="center"><b>Metric Depth</b><br><sub>heatmap</sub><br><br><img src="05_depth_metric/tailoring_depth_heatmap.gif" width="100%"></td>
    <td align="center"><b>Camera Trajectory</b><br><sub>egomotion / SLAM</sub><br><br><img src="06_camera_trajectory/camera_trajectory.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Active / Next-Active Object</b><br><sub>hand-object contact</sub><br><br><img src="07_active_object/hand_object_100doh.gif" width="100%"></td>
    <td align="center"><b>Object 6DoF Pose</b><br><sub>oriented 3D box + axes</sub><br><br><img src="08_object_6dof_pose/object_6dof_pose.gif" width="100%"></td>
    <td align="center"><b>Gaze (attention proxy)</b><br><sub>manipulation focus</sub><br><br><img src="09_gaze/gaze_attention_proxy.gif" width="100%"></td>
  </tr>
</table>

<div align="center">

**NeuroDepth — neuromorphic event vision** (simulated event camera: ON/OFF events, time surface, reconstructed edges)

<img src="10_neurodepth/neurodepth_bandage.gif" width="70%">

</div>

<div align="center"><sub>
Full-resolution MP4s live in each stage folder. Combined overview:
<a href="showcase_montage.jpg">showcase_montage.jpg</a>.
</sub></div>

---

## What's in each folder

Every stage folder contains its **result videos** (MP4 + GIF), its **structured
data** (JSON / mesh / trajectory), and a one-page **`BRIEF.md`** describing the
capability, its inputs, and its outputs.

| # | Stage | Folder | Key deliverables |
|---|---|---|---|
| 01 | Hand detection + side (L/R) | [`01_hand_detection/`](01_hand_detection/) | video · `hand_detections.json` |
| 02 | 3D hand mesh (MANO) | [`02_hand_mesh_hamer/`](02_hand_mesh_hamer/) | video · `hand_meshes/` (per-frame `.obj`) |
| 03 | Hand tracking over time | [`03_hand_tracking/`](03_hand_tracking/) | video · `hand_tracks.json` (2D+3D) |
| 04 | Object segmentation (open-vocab) | [`04_object_segmentation/`](04_object_segmentation/) | video · `objects.json` · trajectory plot |
| 05 | Metric depth | [`05_depth_metric/`](05_depth_metric/) | monocular + **stereo** depth videos |
| 06 | Camera trajectory / egomotion | [`06_camera_trajectory/`](06_camera_trajectory/) | trajectory plot · `camera_poses.npy` |
| 07 | Active / next-active object | [`07_active_object/`](07_active_object/) | video · contact + active-object JSON |
| 08 | Object 6DoF pose | [`08_object_6dof_pose/`](08_object_6dof_pose/) | video · `object_6dof_poses.json` |
| 09 | Gaze (attention proxy) | [`09_gaze/`](09_gaze/) | video · `attention_proxy.json` |

A depth result on an **additional client clip** is included in
[`external_video_depth/`](external_video_depth/).

---

## Coordinate frame & units

Everything is in **metric metres**, in a **camera-centred** frame
(`+X` right, `+Y` down, `+Z` forward), consistent across the clip. Hands, objects,
depth, and poses share this single world frame. See per-stage `BRIEF.md` for the
exact schema of each JSON.

<div align="center"><sub>© XP Robotics — egocentric perception delivery.</sub></div>
