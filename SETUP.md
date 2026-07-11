# Setup — external models & dependencies

Model weights and large third-party repos are **not** stored in this repo (size
limits + upstream licenses). Fetch them locally:

## Python env
Python 3.10, PyTorch **cu128** (RTX 40/50-series). A separate env is used for
DROID-SLAM (`droid-venv`), HITNet (`hitnet-venv`), and 100DOH (`hod-venv`).

## Model weights  (place under `annotation/models/`)
| Model | Source |
|---|---|
| HaMeR | https://github.com/geopavlakos/hamer (checkpoints via their download) |
| SAM 3.1 | Gated — request access from the authors |
| SAM 2 | https://github.com/facebookresearch/sam2 |
| MoGe-2 | HuggingFace `Ruicheng/moge-2-vitl` (auto-downloaded) |
| Depth Anything V2 (metric) | HF `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` |
| HITNet | PINTO0309 model zoo (`142_HITNET`, middlebury_d400 ONNX) |
| DROID-SLAM | `droid.pth` (repo README Google Drive) |
| 100DOH | `faster_rcnn_1_8_132028.pth` (repo model link) |

## Third-party repos (clone under the matching `*/implementation/`)
- DROID-SLAM → `06_camera_trajectory/implementation/DROID-SLAM`
- Depth-Anything-V2 → `05_depth_metric/implementation/Depth-Anything-V2`
- HITNet (ONNX-HITNET-Stereo-Depth-estimation) → `05_depth_metric/implementation/HITNet`
- hand_object_detector (100DOH) → `07_active_object/implementation/hand_object_detector`

CUDA extensions (lietorch / droid_backends / 100DOH ops) build with
`TORCH_CUDA_ARCH_LIST="9.0+PTX"` (PTX JIT to sm_120 on Blackwell).

© XP Robotics
