"""Tune pedestrian detection precision without retraining.

Two levers, both cheap:
  1. score threshold  -- raise the Faster R-CNN confidence cut (0.5 -> higher)
  2. temporal filter   -- only accept a person that persists >= K consecutive frames
                          (a real pedestrian is stable; a false one flickers)

The detector runs ONCE per frame at a low score (0.3); every candidate threshold /
temporal setting is then evaluated by filtering, so the sweep is fast. GT = CARLA
bev_gt pedestrian instances (visible region), matched within a 2.5 m gate.

  python tune_ped.py --data ../data/carla_dualcam_hazard_2k_v2 --stride 3
"""
import os, sys, cv2, json, math, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
import pedestrian_detect as PD
from render_prof_style import fov_visible_mask

BEV, RES, X0, Y0 = HZ.BEV, HZ.RES, HZ.X0, HZ.Y0


def gt_peds(label, vis, min_area=6):
    out = []
    n, _, st, ct = cv2.connectedComponentsWithStats((label == 2).astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        col, row = ct[i]; r, c = int(round(row)), int(round(col))
        if 0 <= r < BEV and 0 <= c < BEV and vis[r, c]:
            out.append(HZ.px_to_ego(col, row))
    return out


def match(gts, dets, gate=2.5):
    used = [False] * len(dets); hit = 0
    for gx, gy in gts:
        best, bd = -1, gate
        for j, (dx, dy) in enumerate(dets):
            if used[j]:
                continue
            d = math.hypot(gx - dx, gy - dy)
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used[best] = True; hit += 1
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_hazard_2k_v2")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=2000)
    ap.add_argument("--gate", type=float, default=2.5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = args.data
    calib = json.load(open(f"{D}/calibration.json"))
    VIS = fov_visible_mask(BEV, BEV)
    poses = json.load(open(f"{D}/poses.json"))
    frames = [f"{i:06d}.png" for i in range(0, min(args.nmax, 2000), args.stride)]
    frames = [f for f in frames if os.path.exists(f"{D}/bev_gt/{f}")]

    # per frame: list of (x, y, score); plus GT peds; plus ego pose for world tracking
    per_frame = []
    for k, n in enumerate(frames):
        lab = cv2.imread(f"{D}/bev_gt/{n}", 0)
        gts = gt_peds(lab, VIS)
        dets = []
        for (x, y, cf, cam, box) in PD.pedestrians_in_frame(D, n, calib, dev, score=0.30):
            col = int((y - Y0) / RES); row = int((BEV - 1) - (x - X0) / RES)
            if 0 <= row < BEV and 0 <= col < BEV and VIS[row, col]:
                dets.append((x, y, cf))
        per_frame.append({"n": n, "pose": poses[n], "gts": gts, "dets": dets})
        if k % 40 == 0:
            print(f"{k}/{len(frames)}", flush=True)

    def evaluate(thr, persist):
        """persist: require a det's world-track to have >= persist hits (1 = off)."""
        # build world tracks for temporal filter
        keep_flags = []
        if persist > 1:
            tracks = []  # each: {wx,wy,frames:set,idxs:[(fi,di)]}
            for fi, f in enumerate(per_frame):
                ex, ey, yaw = f["pose"]; a = math.radians(yaw); c, s = math.cos(a), math.sin(a)
                for di, (x, y, sc) in enumerate(f["dets"]):
                    if sc < thr:
                        continue
                    wx = ex + c * x - s * y; wy = ey + s * x + c * y
                    best = None
                    for t in tracks:
                        if math.hypot(t["wx"] - wx, t["wy"] - wy) < 2.0:
                            best = t; break
                    if best is None:
                        tracks.append({"wx": wx, "wy": wy, "cnt": 1, "members": [(fi, di)]})
                    else:
                        best["wx"], best["wy"] = wx, wy; best["cnt"] += 1; best["members"].append((fi, di))
            ok = set()
            for t in tracks:
                if t["cnt"] >= persist:
                    ok.update(t["members"])
            keepset = ok
        gt_tot = det_tot = hit_tot = 0
        for fi, f in enumerate(per_frame):
            dets = [(x, y) for di, (x, y, sc) in enumerate(f["dets"])
                    if sc >= thr and (persist <= 1 or (fi, di) in keepset)]
            gt_tot += len(f["gts"]); det_tot += len(dets)
            hit_tot += match(f["gts"], dets, args.gate)
        rec = hit_tot / max(gt_tot, 1); prec = hit_tot / max(det_tot, 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        return rec, prec, f1, det_tot

    print(f"\n{'='*66}\nPedestrian precision tuning ({len(frames)} frames)\n{'='*66}")
    print(f"{'setting':<28}{'Recall':>9}{'Prec':>9}{'F1':>9}{'#det':>8}")
    print(f"{'baseline (score 0.5)':<28}", end="")
    r, p, f1, nd = evaluate(0.5, 1); print(f"{r:>9.3f}{p:>9.3f}{f1:>9.3f}{nd:>8}")
    for thr in (0.6, 0.7, 0.8):
        r, p, f1, nd = evaluate(thr, 1)
        print(f"{'score '+str(thr):<28}{r:>9.3f}{p:>9.3f}{f1:>9.3f}{nd:>8}")
    for thr in (0.5, 0.6, 0.7):
        for persist in (2, 3):
            r, p, f1, nd = evaluate(thr, persist)
            print(f"{'score '+str(thr)+' + persist>='+str(persist):<28}{r:>9.3f}{p:>9.3f}{f1:>9.3f}{nd:>8}")


if __name__ == "__main__":
    main()
