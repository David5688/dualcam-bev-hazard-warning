"""How many GT vehicle pixels are even inside the 2-camera field of view?

That is a hard upper bound on recall, independent of the model.
BEV convention from carla_dualcam_collect_gt.draw_map_bev:
    x = X_MIN + (BEV_H - 1 - row) * RES     (ego forward)
    y = Y_MIN + col * RES                   (ego right)
"""
import os, cv2, numpy as np

D = "E:/dualcam_project/data/carla_dualcam_gt_lane"
BEV_H = BEV_W = 240
RES, X_MIN, Y_MIN = 0.25, -30.0, -30.0
D0, D1 = 2.0, 42.0          # dbound used by the LSS model

rows = np.arange(BEV_H)[:, None]
cols = np.arange(BEV_W)[None, :]
X = X_MIN + (BEV_H - 1 - rows) * RES
Y = Y_MIN + cols * RES
X = np.broadcast_to(X, (BEV_H, BEV_W))
Y = np.broadcast_to(Y, (BEV_H, BEV_W))

# 90 deg hfov: a point is inside the cone iff |along-axis| >= |lateral|
fx, fy = X - 1.5, Y
front = (fx > 0) & (np.abs(fx) >= np.abs(fy)) & (np.hypot(fx, fy) >= D0) & (np.hypot(fx, fy) <= D1)
rx, ry = X + 1.5, Y
rear = (rx < 0) & (np.abs(rx) >= np.abs(ry)) & (np.hypot(rx, ry) >= D0) & (np.hypot(rx, ry) <= D1)
visible = front | rear

print(f"visible cells: {visible.mean():.1%} of the {BEV_H}x{BEV_W} BEV grid")

for split in ["train.txt", "test.txt"]:
    files = [l.strip() for l in open(os.path.join(D, split)) if l.strip()]
    tot = vis = 0
    ped_tot = ped_vis = 0
    for n in files:
        g = cv2.imread(os.path.join(D, "bev_gt", n), cv2.IMREAD_GRAYSCALE)
        v = (g == 1)
        tot += int(v.sum()); vis += int((v & visible).sum())
        p = (g == 2)
        ped_tot += int(p.sum()); ped_vis += int((p & visible).sum())
    print(f"\n[{split}]  {len(files)} frames")
    print(f"  vehicle    GT px {tot:>8,}   inside FOV {vis:>8,}  = {vis/max(tot,1):.1%}  <- hard recall ceiling")
    print(f"  pedestrian GT px {ped_tot:>8,}   inside FOV {ped_vis:>8,}  = {ped_vis/max(ped_tot,1):.1%}")
