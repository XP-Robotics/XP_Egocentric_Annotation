#!/usr/bin/env python
"""
Stage 05 (Depth Anything V2, metric) — per-frame metric depth for a frames dir.
Writes colormap PNGs + a raw .npy stack. Uses the Hypersim (indoor) ViT-L metric model.

Usage: run_depth_anything.py <frames_dir> <out_dir>
"""
import sys, os, glob
import numpy as np, cv2, torch

REPO = "/home/raush/Documents/Ego_Infinity/annotation/05_depth_metric/implementation/Depth-Anything-V2/metric_depth"
CKPT = "/home/raush/Documents/Ego_Infinity/annotation/models/depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth"
sys.path.insert(0, REPO)
from depth_anything_v2.dpt import DepthAnythingV2

FRAMES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
CFR = os.path.join(OUT, "_color"); os.makedirs(CFR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = {"encoder": "vitl", "features": 256,
       "out_channels": [256, 512, 1024, 1024], "max_depth": 20.0}
model = DepthAnythingV2(**cfg)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(DEVICE).eval()

files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
stack = []
VMIN, VMAX = 0.3, 3.0
for i, fn in enumerate(files):
    raw = cv2.imread(fn)                      # BGR, as DA-v2 expects
    with torch.no_grad():
        depth = model.infer_image(raw, 518)   # HxW float32 metric metres
    stack.append(depth.astype(np.float32))
    vis = np.clip((depth - VMIN) / (VMAX - VMIN), 0, 1)
    cm = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(cm, f"Depth Anything V2 (metric) 0.3-3.0 m | frame {i}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(CFR, f"{i:06d}.png"), cm)
np.save(os.path.join(OUT, "depth_da2_metres.npy"), np.stack(stack))
print(f"stage05/DA2: {len(files)} frames, depth range "
      f"{np.min(stack):.2f}-{np.max(stack):.2f} m -> {OUT}")
