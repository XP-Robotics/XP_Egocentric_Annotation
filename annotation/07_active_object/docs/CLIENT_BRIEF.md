# Stage 07 — Active / Next-Active Object

**XP Robotics · Egocentric Perception Stack**

## What it does
Identifies, per frame, **which object the hand is actually interacting with** (the
*active* object) and the most likely *next* object. This is the core interaction
label for imitation learning — it turns "hands + objects in view" into "hand is
manipulating THIS object now".

## Input
Hand keypoints (Stages 01–03) + tracked object masks (Stage 04).

## Output
| Output | Format |
|---|---|
| Per-frame active + next-active object | `active_object.json` |
| Overlay video (ACTIVE highlighted) | `active_object.mp4` |

## Method
Two paths:
- **Contact-geometry (delivered):** fingertip-to-object-mask distance → object in
  contact = ACTIVE, nearest untouched = next-active. No extra model; reuses the stack.
- **100DOH (reference learned detector):** Faster-RCNN trained partly on egocentric
  data, predicts hand contact-state + active-object box directly. Installed under
  `implementation/hand_object_detector/` when the build supports the GPU.

## Why it matters
- **Interaction labels** — the key supervision signal for learning manipulation.
- **Model-free option** — works from the stack's own outputs, any object.

© XP Robotics
