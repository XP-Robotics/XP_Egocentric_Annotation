#!/usr/bin/env python
"""Plot a DROID-SLAM reconstruction's camera trajectory to a PNG.
Usage: plot_traj.py <recon_file> <out_png>   (run with droid-venv python)"""
import sys, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

recon, out = sys.argv[1], sys.argv[2]
d = torch.load(recon, map_location="cpu", weights_only=False)
p = d["poses"].numpy() if hasattr(d["poses"], "numpy") else np.asarray(d["poses"])
t = p[:, :3] * 100.0                                  # cm
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(t[:, 0], t[:, 1], t[:, 2], "-o", c="tab:blue")
if len(t):
    ax.scatter(*t[0], c="g", s=60, label="start"); ax.scatter(*t[-1], c="r", s=60, label="end")
ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)"); ax.legend()
ax.set_title("DROID-SLAM camera path")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(t[:, 0], t[:, 2], "-o", c="tab:blue"); ax2.set_xlabel("X (cm)"); ax2.set_ylabel("Z (cm)")
ax2.set_title("Top-down (X-Z)"); ax2.grid(alpha=0.3); ax2.axis("equal")
span = (t.max(0) - t.min(0)) if len(t) else np.zeros(3)
fig.suptitle(f"Camera trajectory — span {span[0]:.1f}x{span[1]:.1f}x{span[2]:.1f} cm ({len(t)} keyframes)")
fig.tight_layout(); fig.savefig(out, dpi=120)
print("saved", out)
