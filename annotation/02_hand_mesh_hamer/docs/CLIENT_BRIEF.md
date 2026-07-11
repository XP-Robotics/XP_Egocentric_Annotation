# Stage 02 — 3D Hand Mesh (MANO) via HaMeR

**XP Robotics · Egocentric Perception Stack**

## What it does
Reconstructs a full **articulated 3D hand mesh** for each detected hand — 778
vertices and 21 joints in the MANO parametric model — from a single RGB crop.
This turns a 2D hand detection into metric 3D geometry that a robot can be
retargeted to.

## Input
| Input | Notes |
|---|---|
| Per-hand RGB crop + side | from Stage 01 |
| Camera focal length | from `camera.json` (metric placement) |

## Output
| Output | Format |
|---|---|
| MANO pose + shape (θ, β) per hand per frame | parameters |
| 3D joints (21) + mesh (778 verts) | metric metres, camera frame |
| Mesh overlay video | MP4 |
| Per-frame hand meshes | Wavefront `.obj` |

## Method
**HaMeR** (Hand Mesh Recovery, transformer-based) predicts MANO parameters and a
weak-perspective camera per hand crop; the crop translation is rescaled with the
real camera focal so the hand sits at metric depth. Best-in-class for the
occluded, oblique hand poses typical of egocentric capture; run per-hand.

## Why it matters
- **Robot-ready geometry** — MANO maps directly onto a dexterous/bimanual robot.
- **Handles ego occlusion** — recovers plausible full-hand pose even when fingers
  are hidden, where keypoint-only methods fail.

## Model & weights
- HaMeR checkpoint: `models/hamer/hamer_ckpts/`
- ViTPose (crop keypoints): `models/hamer/vitpose_ckpts/`

© XP Robotics
