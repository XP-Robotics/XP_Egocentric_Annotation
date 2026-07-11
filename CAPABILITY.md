# Capability Sheet — Egocentric Video Perception

**XP Robotics** · Hand · Object · Scene understanding from wearable-camera video

---

## What it does
From an egocentric (head/body-worn) RGB clip, the stack reconstructs a temporally
consistent understanding of the hands, the objects, and the scene — in one metric
world frame:

- **Hands** — detected, left/right labelled, reconstructed as 3D meshes, and tracked
  smoothly over time.
- **Objects** — segmented by **plain-text prompt** (no per-object training), tracked,
  and given **6DoF pose** + oriented 3D boxes.
- **Interaction** — which object each hand is manipulating now (active / next-active).
- **Scene** — per-pixel **metric depth** (monocular and stereo) and the **camera's own
  motion** (egomotion / SLAM).
- **Attention** — a manipulation-focus proxy (real gaze integrates with eye-tracking
  hardware).

## Inputs
| Input | Notes |
|---|---|
| Egocentric RGB video | wearable camera; stereo pair used when available |
| Object names (text) | e.g. `"red folder"`, `"papers"` — no per-object training |

## Outputs (delivered)
| Output | Format |
|---|---|
| Hand detection + L/R, contact state, active object | MP4 · JSON |
| 3D hand meshes (both hands, per frame) | Wavefront `.obj` |
| 2D + 3D hand keypoint tracks | JSON |
| Open-vocabulary object masks + trajectories | MP4 · JSON · PNG |
| Metric depth (monocular + stereo) | MP4 |
| Object 6DoF pose + oriented 3D boxes | MP4 · JSON |
| Camera trajectory (egomotion) | PNG · `.npy` |
| Consolidated per-frame tracking | `tracking.json` |

## Why it matters
- **Sensor-free 3D** — hands, objects and depth from ordinary wearable RGB.
- **Open-vocabulary** — any object, specified in plain language.
- **Robot-ready** — 3D hands + 6DoF object poses connect perception to action.
- **Structured, documented data** — not just visualisations; drop-in for downstream
  learning and analysis.

---
© XP Robotics — capability demonstration.
