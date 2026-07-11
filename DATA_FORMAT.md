# Data Format

All geometry is in **metric metres**, in a **camera-centred** frame
(`+X` right, `+Y` down, `+Z` forward), consistent across the clip.

## `tracking.json` (consolidated, per-frame)
```jsonc
{
  "metadata": {
    "fps": 15, "n_frames": 180,
    "image_width": 1280, "image_height": 960,
    "camera": { "focal_px": ..., "cx": ..., "cy": ..., "model": "pinhole" },
    "coordinate_frame": "camera-centred, metres; +X right, +Y down, +Z forward",
    "hand_joint_order": "MANO 21: 0=wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20"
  },
  "objects": [ { "id": 0, "label": "red folder", "detection_confidence": 0.88 } ],
  "frames": [
    {
      "frame": 0,
      "hands": [
        { "side": "right",
          "joints_3d_m": [[x,y,z], … 21],
          "joints_2d_px": [[u,v], … 21] }
      ],
      "objects": [
        { "id": 0, "label": "red folder",
          "pose": { "rotation_3x3": [[…],[…],[…]], "translation_m": [x,y,z] },
          "obb":  { "corners_m": [[x,y,z], … 8], "extent_m": [dx,dy,dz] } }
      ]
    }
  ]
}
```
`pose` / `obb` may be `null` on frames where an object is occluded. Projecting
3D→2D: `u = focal_px·X/Z + cx`, `v = focal_px·Y/Z + cy`.

## Per-stage JSON / data
| File | Contents |
|---|---|
| `01_hand_detection/hand_detections.json` | per-frame hand boxes, side, confidence |
| `02_hand_mesh_hamer/hand_meshes/*.obj` | per-frame 3D hand meshes (778 verts/hand) |
| `03_hand_tracking/hand_tracks.json` | per-frame 2D+3D hand keypoints, stable L/R track IDs |
| `04_object_segmentation/objects.json` | tracked objects + detection confidence |
| `06_camera_trajectory/camera_poses.npy` | per-keyframe camera pose (tx ty tz qx qy qz qw) |
| `07_active_object/active_object.json` | per-frame active / next-active object |
| `07_active_object/detections_all.json` | per-frame hand boxes + contact state + active-object box |
| `08_object_6dof_pose/object_6dof_poses.json` | per-object 6DoF R,t + 3D box extent |
| `09_gaze/attention_proxy.json` | per-frame attention point (proxy) |
| `metrics.json` | summary metrics across all stages |

Per-frame metric depth maps and per-object segmentation masks are available on
request (kept out of this package to keep it light).

© XP Robotics.
