#!/usr/bin/env python
"""
Stage 07 — Active / next-active object (hand-object contact).

The active object is the one a hand is currently interacting with. We detect it
from geometry we already have: for each hand (Stage 01-03 keypoints) we find the
tracked object mask (Stage 04) the fingertips are touching / closest to. The object
in contact is flagged ACTIVE; the nearest not-yet-touched object is NEXT-ACTIVE.

(100DOH is the reference learned detector for this signal; see the client brief.
This contact-geometry method needs no extra model and reuses the stack's outputs.)

Usage: active_object.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>
"""
import sys, os, gzip, pickle, json
import numpy as np, cv2

PKL, FRAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
FR = os.path.join(OUT, "_frames"); os.makedirs(FR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
mapping = d.get("sam3_prompt_mapping", [])
TIP_IDS = [4, 8, 12, 16, 20]          # MANO fingertips
CONTACT_PX = 22                        # fingertip-to-mask distance for "contact"


def unpack(v, H, W):
    mp, ms = v.get("mask_packed"), v.get("mask_shape")
    if mp is None:
        return None
    m = np.unpackbits(np.asarray(mp, np.uint8))[:int(ms[0])*int(ms[1])].reshape(int(ms[0]), int(ms[1])).astype(np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m


APPROACH_W = 120.0     # weight for "hand moving toward object" in next-active score
prev_cent = None
next_hist = []         # short history for hysteresis
summary = []
for fi, f in enumerate(fd):
    p = os.path.join(FRAMES, f"{fi+1:06d}.jpg"); im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    j2 = f.get("joints_2d_pred") or []
    od = f.get("sam3_obj_data") or {}
    # skip 'hand' prompts as candidate objects
    obj_ids = [o for o in sorted(od) if (mapping[o]["prompt"] if o < len(mapping) else "").lower() != "hand"]
    dist_maps = {}; centroids = {}
    for o in obj_ids:
        m = unpack(od[o], H, W)
        if m is None or not m.any():
            continue
        dist_maps[o] = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
        ys, xs = np.where(m > 0); centroids[o] = np.array([xs.mean(), ys.mean()])
    # all hand joints + per-hand bounding boxes (the "grasp region")
    allj = []; hand_boxes = []
    for J in j2:
        J = np.asarray(J, float)
        if J.ndim == 2 and len(J) > 20:
            allj += list(J)
            hand_boxes.append([J[:, 0].min(), J[:, 1].min(), J[:, 0].max(), J[:, 1].max()])
    tips = [np.asarray(J, float)[t] for J in j2 for t in TIP_IDS
            if np.asarray(J, float).ndim == 2 and len(np.asarray(J, float)) > 20]
    cent = np.mean(np.asarray(tips), 0) if tips else prev_cent
    vel = (cent - prev_cent) if (cent is not None and prev_cent is not None) else np.zeros(2)
    prev_cent = cent
    CONTACT = max(CONTACT_PX, 0.03 * W)              # scale contact radius to image

    def bbox_overlap_ratio(hb, ob):
        ix0, iy0 = max(hb[0], ob[0]), max(hb[1], ob[1])
        ix1, iy1 = min(hb[2], ob[2]), min(hb[3], ob[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        oa = max(1.0, (ob[2] - ob[0]) * (ob[3] - ob[1]))
        return inter / oa                            # fraction of the OBJECT inside a hand box

    # hand GRIP region: filled convex hull of each hand's joints, slightly dilated.
    # A HELD object's visible pixels fall inside this grip; a background object the
    # hand merely hovers over has its visible pixels OUTSIDE the hand silhouette.
    hull = np.zeros((H, W), np.uint8)
    for J in j2:
        J = np.asarray(J, float)
        if J.ndim == 2 and len(J) > 20:
            cv2.fillConvexPoly(hull, cv2.convexHull(J.astype(np.int32)), 1)
    kd = max(9, int(0.02 * W)) | 1
    hull = cv2.dilate(hull, np.ones((kd, kd), np.uint8))
    hull_area = float(hull.sum()) + 1.0

    dmins = {}; dtip = {}; grip_frac = {}; area_frac = {}
    for o, dm in dist_maps.items():
        dmins[o] = min((dm[int(np.clip(v, 0, H-1)), int(np.clip(u, 0, W-1))] for (u, v) in allj), default=1e9)
        dtip[o] = min((dm[int(np.clip(v, 0, H-1)), int(np.clip(u, 0, W-1))] for (u, v) in tips), default=1e9)
        m = unpack(od[o], H, W)
        grip_frac[o] = float(((m > 0) & (hull > 0)).sum()) / hull_area   # object share of the grip
        area_frac[o] = float((m > 0).sum()) / (H * W)
    ovr = grip_frac
    # ACTIVE = object filling the grip most (held), gated so it's genuinely at the hand.
    active = set()
    if dist_maps:
        best_o = max(dist_maps, key=lambda o: grip_frac[o] - 0.5 * area_frac[o])
        if grip_frac[best_o] >= 0.03 and dtip[best_o] <= 1.5 * CONTACT:
            active = {best_o}
    # next-active = the non-active object the hand is nearest to / moving toward (ALWAYS predicted)
    nextact = None; best = 1e9
    for o in dist_maps:
        if o in active:
            continue
        score = dmins[o]
        if cent is not None and np.linalg.norm(vel) > 1.0:
            d = centroids[o] - cent; dn = np.linalg.norm(d)
            if dn > 1e-3:
                toward = float(np.dot(vel / (np.linalg.norm(vel) + 1e-6), d / dn))   # -1..1
                score -= APPROACH_W * max(0.0, toward)
        if score < best:
            best = score; nextact = o
    # temporal hysteresis: keep next-active stable over a few frames
    next_hist.append(nextact); next_hist[:] = next_hist[-5:]
    if next_hist:
        vals = [x for x in next_hist if x is not None]
        if vals:
            nextact = max(set(vals), key=vals.count)
    # draw (label size scales with image so it's readable at any resolution)
    fs = max(0.6, W / 1600.0); th = max(1, int(round(fs * 2)))

    def label(img, x, y, text, col, filled=True):
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        yb = max(tht + 6, y)
        if filled:
            cv2.rectangle(img, (x, yb - tht - 8), (x + tw + 8, yb + 2), col, -1)
            cv2.putText(img, text, (x + 4, yb - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        else:
            cv2.putText(img, text, (x + 4, yb - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)

    # draw non-active first, active last (on top)
    for o in [x for x in dist_maps if x not in active and x != nextact]:
        m = unpack(od[o], H, W); ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        cv2.rectangle(im, (xs.min(), ys.min()), (xs.max(), ys.max()), (150, 150, 150), 1)
        label(im, xs.min(), ys.min(), mapping[o]["prompt"] if o < len(mapping) else str(o), (110, 110, 110), filled=False)
    if nextact is not None and nextact in dist_maps:
        m = unpack(od[nextact], H, W); ys, xs = np.where(m > 0)
        if len(xs):
            im[m > 0] = (0.25 * np.array((0, 200, 255)) + 0.75 * im[m > 0]).astype(np.uint8)
            cv2.rectangle(im, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 200, 255), 2)
            label(im, xs.min(), ys.min(), f"NEXT: {mapping[nextact]['prompt'] if nextact < len(mapping) else nextact}", (0, 170, 220))
    for o in active:
        m = unpack(od[o], H, W); ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        im[m > 0] = (0.45 * np.array((0, 0, 255)) + 0.55 * im[m > 0]).astype(np.uint8)
        cv2.rectangle(im, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 0, 255), 3)
        label(im, xs.min(), ys.min(), f"ACTIVE: {mapping[o]['prompt'] if o < len(mapping) else o}", (0, 0, 220))
    label(im, 8, 20, "active / next-active object", (40, 40, 40))
    cv2.imwrite(os.path.join(FR, f"{fi:06d}.png"), im)
    summary.append({"frame": fi,
                    "active": [mapping[o]["prompt"] if o < len(mapping) else str(o) for o in sorted(active)],
                    "next_active": (mapping[nextact]["prompt"] if nextact is not None and nextact < len(mapping) else None)})

json.dump({"n_frames": len(fd), "frames": summary}, open(os.path.join(OUT, "active_object.json"), "w"))
os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 15 -i {FR}/%06d.png "
          f"-c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -crf 20 {OUT}/active_object.mp4")
os.system(f"ffmpeg -y -hide_banner -loglevel error -i {OUT}/active_object.mp4 -vf "
          f"\"fps=8,scale=440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" {OUT}/active_object.gif")
import shutil; shutil.rmtree(FR, ignore_errors=True)
na = sum(1 for s in summary if s["active"])
print(f"stage07: {len(summary)} frames, {na} with an active object -> {OUT}")
