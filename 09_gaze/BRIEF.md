# Stage 09 — Gaze

**XP Robotics · Egocentric Perception Stack**

## What it does
Recovers where the wearer is **looking**. True gaze is measured from an
**eye-tracking stream** (e.g. Project Aria's on-device eye cameras).

## Status — hardware-gated
This capture rig provides **no eye-tracking stream**, so measured gaze is not
available. We deliver two things:
- **Documented integration point** — if the platform is Aria (or any rig with eye
  cameras / a gaze-estimation net), that stream drops straight into the stack here.
- **Manipulation-attention proxy (delivered):** on a manual task the gaze fixates
  where the hands work; we render the fingertip / active-object centroid as an
  attention marker + trail. Clearly labelled as a **proxy, not measured gaze**.

## Output
| Output | Format |
|---|---|
| Attention proxy overlay | `gaze_attention_proxy.mp4` |
| Per-frame attention point | `attention_proxy.json` |

## Why it matters
- **Drop-in ready** — real gaze integrates immediately once hardware provides it.
- **Useful prior now** — the attention proxy already gives a usable focus signal.

© XP Robotics