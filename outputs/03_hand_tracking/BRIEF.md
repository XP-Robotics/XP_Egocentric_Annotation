# Stage 03 — Hand Tracking Over Time

**XP Robotics · Egocentric Perception Stack**

## What it does
Links per-frame hand reconstructions into **smooth, identity-stable trajectories**
across the whole clip. Raw per-frame MANO jitters and can flip left/right during
occlusion; this stage propagates hand identity and temporally smooths the pose so
the motion is clean enough to retarget.

## Input
| Input | Notes |
|---|---|
| Per-frame hand masks + MANO | Stages 01–02 |
| RGB frames | for mask propagation |

## Output
| Output | Format |
|---|---|
| Temporally smoothed MANO per hand | parameters |
| Consistent left/right track IDs | across all frames |
| 2D + 3D hand keypoint tracks | pixels + metric metres |
| Tracking overlay video | MP4 (keypoint skeletons) |

## Method
**SAM 2** propagates each hand's mask through the video to hold identity across
occlusions; HaMeR runs per frame; MANO parameters are smoothed over time to remove
jitter. The result is a continuous hand-motion signal, not a stack of independent
per-frame guesses.

## Why it matters
- **Retarget-quality motion** — smooth, identity-stable trajectories are what a
  robot policy actually consumes.
- **Survives occlusion** — mask propagation keeps the correct hand labelled through
  hand–hand and hand–object overlap.

© XP Robotics