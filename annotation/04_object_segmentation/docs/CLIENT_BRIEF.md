# Stage 04 — Open-Vocabulary Object Segmentation

**XP Robotics · Egocentric Perception Stack**

## What it does
Detects, segments, and tracks the objects a person interacts with — specified by
**plain-text prompts**, with no per-object training. Ask for `"parts organizer"`,
`"screwdriver"`, `"cardboard box"` and get a tracked mask per object across the clip.

## Input
| Input | Notes |
|---|---|
| Rectified RGB frames | front-camera pinhole frames |
| Object names (text) | e.g. `"clear plastic parts tray"`, `"connector"` |

## Output
| Output | Format |
|---|---|
| Per-object segmentation masks | per frame, tracked |
| Boxes + labels + detection confidence | per object |
| Segmentation overlay video | MP4 |
| (with depth) object 3D point clouds | fed to 3D reconstruction |

## Method
**Grounded-SAM → SAM 2** (or **SAM 3.1** open-vocab): a text-grounded detector
proposes objects from the prompts, SAM produces pixel-accurate masks, and SAM 2
propagates them through the video. Combined with Stage 05 depth, each mask lifts to
a 3D object point cloud in the shared metric frame.

## Why it matters
- **No training per object** — new tasks/objects need only a text prompt.
- **Interaction context** — object masks + hand meshes give the full
  hand–object relationship, the core signal for imitation learning.

## Model & weights
- SAM 3.1 (open-vocab): `models/sam3.1/`
- Grounded-SAM detector + SAM 2: `models/grounded_sam/`, `models/sam2.1_hiera_small.pt`

© XP Robotics
