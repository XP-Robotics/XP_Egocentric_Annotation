<div align="center">

# Egocentric Video Perception — Hand · Object · Scene Understanding

**XP Robotics**

A complete perception stack for **egocentric (head/body-worn) RGB video** — turning
a raw wearable-camera clip into hands, objects, depth, interaction labels, 6DoF
poses, and camera motion, in one metric world frame.

📄 [Capability summary](CAPABILITY.md) &nbsp;·&nbsp; ✅ [Requirement coverage](REQUIREMENTS.md) &nbsp;·&nbsp; 🗂️ [Data format](DATA_FORMAT.md) &nbsp;·&nbsp; 📦 [`tracking.json`](tracking.json)

</div>

---

## Results — one stage per scene

The stack runs across many everyday egocentric scenes. Each panel below is a
**different stage** shown on a **different real scene** — demonstrating the same
pipeline generalising across healthcare, industrial, office and household tasks.

<table>
  <tr>
    <td width="33%" align="center"><b>01 · Hand Detection + L/R</b><br><sub>healthcare · bandage rolls</sub><br><br><img src="showcase/01_hand_detection.gif" width="100%"></td>
    <td width="33%" align="center"><b>02 · 3D Hand Mesh (MANO)</b><br><sub>maintenance · plumbing</sub><br><br><img src="showcase/02_hand_mesh.gif" width="100%"></td>
    <td width="33%" align="center"><b>03 · Hand Tracking</b><br><sub>industrial · electrician</sub><br><br><img src="showcase/03_hand_tracking.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>04 · Object Segmentation</b><br><sub>office · open-vocab</sub><br><br><img src="showcase/04_object_segmentation.gif" width="100%"></td>
    <td align="center"><b>05 · Metric Depth</b><br><sub>tailoring · fabric heatmap</sub><br><br><img src="showcase/05_depth.gif" width="100%"></td>
    <td align="center"><b>06 · Camera Trajectory</b><br><sub>household · room organising</sub><br><br><img src="showcase/06_trajectory.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>07 · Active / Next-Active Object</b><br><sub>fold-cloth packing</sub><br><br><img src="showcase/07_active_object.gif" width="100%"></td>
    <td align="center"><b>08 · Object 6DoF Pose</b><br><sub>cleaning kitchen slab</sub><br><br><img src="showcase/08_6dof.gif" width="100%"></td>
    <td align="center"><b>09 · Gaze (attention proxy)</b><br><sub>sticker-in-cloth</sub><br><br><img src="showcase/09_gaze.gif" width="100%"></td>
  </tr>
</table>

<div align="center">

**10 · NeuroDepth — neuromorphic event vision** &nbsp;<sub>(office-data scene)</sub>
<br>ON/OFF events · time surface · reconstructed edges

<img src="showcase/10_neurodepth.gif" width="70%">

</div>

<div align="center"><sub>
Full-resolution MP4s + per-frame JSON for <b>every</b> scene live in each stage folder.
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
