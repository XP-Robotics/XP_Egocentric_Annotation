# Stage 01 — Hand Detection + Side (Left/Right)

**XP Robotics · Egocentric Perception Stack**

## What it does
Finds every hand in each frame and labels it **left** or **right** — the entry
point for all downstream hand reconstruction. On an egocentric view the hands
enter from the bottom of the frame and are frequently self-occluded, so robust
detection + correct side assignment is what keeps the 3D hand meshes from
swapping identity between frames.

## Input
| Input | Notes |
|---|---|
| Rectified RGB frames | front-camera pinhole frames from `common/mcap_to_frames.py` |

## Output
| Output | Format |
|---|---|
| Per-frame hand bounding boxes | pixel coords |
| Left / right label + confidence | per detection |
| Overlay video | MP4 (boxes + L/R tags) |

## Method
A YOLO-based hand detector (WiLoR-family, trained partly on egocentric data)
proposes hand boxes; side is assigned from wrist/keypoint geometry and kept
temporally consistent. Feeds the per-hand crop to Stage 02 (HaMeR).

## Why it matters
- **Foundation for the whole hand pipeline** — mesh, tracking and retargeting all
  depend on correct hand boxes + side.
- **Ego-robust** — tuned for hands entering the frame edge and mutual occlusion.

© XP Robotics