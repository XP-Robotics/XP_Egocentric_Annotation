# Stage 10 — NeuroDepth (Neuromorphic Event Vision)

**XP Robotics · Egocentric Perception Stack**

## What it does
Derives the signals of a **neuromorphic / event-based camera** from an ordinary RGB
stream — the sparse, high-temporal-resolution representation used for low-latency,
high-speed perception. Four synchronized views:

- **RAW CAMERA** — the input frame.
- **ON/OFF EVENTS** — per-pixel log-intensity change polarity (ON/+ = red, OFF/- = green).
- **TIME SURFACE** — exponentially-decaying map of the most recent event time; encodes
  motion direction and recency.
- **RECONSTRUCTED EDGES** — contours accumulated from the event stream.

## Input / Output
| | |
|---|---|
| Input | RGB video (any camera) |
| Output | `neurodepth_bandage.mp4` — 4-panel event visualization + live event HUD |

## Method
Log-intensity temporal contrast produces ON/OFF events per pixel; a per-pixel
last-event-time buffer yields the time surface; a decaying event accumulator gives
the reconstructed edges. A readout HUD reports event count, ON/OFF split, event
density, input fps and pipeline latency — mirroring a neuromorphic event-core.

## Why it matters
- **Low-latency, high-speed** — event representations react to change in microseconds,
  ideal for fast manipulation and motion.
- **Sparse & efficient** — only changing pixels carry information.
- **Sensor-ready** — the same pipeline ingests a real event camera (DVS) directly.

© XP Robotics
