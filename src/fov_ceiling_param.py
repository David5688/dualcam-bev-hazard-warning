"""Parameterized FOV recall-ceiling: how many GT vehicle px fall inside the
2-camera field of view, for an arbitrary hfov. Pure-geometry visible-cell %
is data-independent; per-class ceiling is computed over the dataset's frames.

Reproduces fov_recall_ceiling.py when --fov 90 --data .../gt_lane --split test.txt
"""
import os, cv2, glob, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--fov", type=float, required=True, help="full horizontal FOV (deg)")
ap.add_argument("--split", default=None, help="split .txt; if omitted, use ALL bev_gt frames")
ap.add_argument("--limit", type=int, default=0, help="cap frames (0=all)")
args = ap.parse_args()

BEV_H = BEV_W = 240
RES, X_MIN, Y_MIN = 0.25, -30.0, -30.0
D0, D1 = 2.0, 42.0            # LSS dbound (unchanged by FOV)
CAM_X = 1.5                   # front cam at +1.5m, rear at -1.5m

rows = np.arange(BEV_H)[:, None]
cols = np.arange(BEV_W)[None, :]
X = np.broadcast_to(X_MIN + (BEV_H - 1 - rows) * RES, (BEV_H, BEV_W))
Y = np.broadcast_to(Y_MIN + cols * RES, (BEV_H, BEV_W))

half = np.radians(args.fov / 2.0)
tan_half = np.tan(half)       # inside cone iff |lateral| <= |along| * tan(half)

fx, fy = X - CAM_X, Y
front = (fx > 0) & (np.abs(fy) <= np.abs(fx) * tan_half) & (np.hypot(fx, fy) >= D0) & (np.hypot(fx, fy) <= D1)
rx, ry = X + CAM_X, Y
rear = (rx < 0) & (np.abs(ry) <= np.abs(rx) * tan_half) & (np.hypot(rx, ry) >= D0) & (np.hypot(rx, ry) <= D1)
visible = front | rear

print(f"=== fov={args.fov:g}  data={os.path.basename(args.data)} ===")
print(f"visible cells: {visible.mean():.1%} of the {BEV_H}x{BEV_W} BEV grid  (pure geometry)")

if args.split:
    files = [l.strip() for l in open(os.path.join(args.data, args.split)) if l.strip()]
else:
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data, "bev_gt", "*.png")))
if args.limit:
    files = files[::max(1, len(files) // args.limit)][:args.limit]

tot = vis = ped_tot = ped_vis = 0
for n in files:
    g = cv2.imread(os.path.join(args.data, "bev_gt", n), cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    v = (g == 1); tot += int(v.sum()); vis += int((v & visible).sum())
    p = (g == 2); ped_tot += int(p.sum()); ped_vis += int((p & visible).sum())

print(f"{len(files)} frames"
      + (f" (sampled from split {args.split})" if args.split else " (all)"))
print(f"  vehicle    GT px {tot:>10,}   inside FOV {vis:>10,}  = {vis/max(tot,1):.1%}  <- hard recall ceiling")
print(f"  pedestrian GT px {ped_tot:>10,}   inside FOV {ped_vis:>10,}  = {ped_vis/max(ped_tot,1):.1%}")
