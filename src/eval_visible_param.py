"""Parameterized visible-region vehicle metrics (all vs FOV-visible).
Adapts eval_visible_region.py to arbitrary dataset / checkpoint / FOV.
Builds the LSS frustum from the dataset's own calibration.json, and the
visible mask from --fov (so 120-deg models get the correct 120-deg mask).
"""
import os, sys, cv2, argparse, numpy as np, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS
from dualcam_bev_fpn_resnet import DualCamBEVFPNResNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--fov", type=float, required=True)
ap.add_argument("--arch", default="lss", choices=["lss", "resnet"])
ap.add_argument("--num_classes", type=int, default=6)
ap.add_argument("--split", default="test.txt")
ap.add_argument("--img_h", type=int, default=224)
ap.add_argument("--img_w", type=int, default=400)
args = ap.parse_args()

IMG_H, IMG_W, VEH, dev = args.img_h, args.img_w, 1, "cuda" if torch.cuda.is_available() else "cpu"
BEV_H = BEV_W = 240
RES, X_MIN, Y_MIN, D0, D1 = 0.25, -30.0, -30.0, 2.0, 42.0

rows = np.arange(BEV_H)[:, None]; cols = np.arange(BEV_W)[None, :]
X = np.broadcast_to(X_MIN + (BEV_H - 1 - rows) * RES, (BEV_H, BEV_W))
Y = np.broadcast_to(Y_MIN + cols * RES, (BEV_H, BEV_W))
tan_half = np.tan(np.radians(args.fov / 2.0))
fx, fy = X - 1.5, Y
front = (fx > 0) & (np.abs(fy) <= np.abs(fx) * tan_half) & (np.hypot(fx, fy) >= D0) & (np.hypot(fx, fy) <= D1)
rx, ry = X + 1.5, Y
rear = (rx < 0) & (np.abs(ry) <= np.abs(rx) * tan_half) & (np.hypot(rx, ry) >= D0) & (np.hypot(rx, ry) <= D1)
VIS = front | rear

def read_img(p):
    img = cv2.imread(p)
    img = cv2.resize(img, (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

meta = torch.load(args.model, map_location="cpu", weights_only=False)
ds = meta.get("lss_downsample", 8) if isinstance(meta, dict) else 8
pool = meta.get("lss_pool", 0.5) if isinstance(meta, dict) else 0.5
if args.arch == "lss":
    m = DualCamBEVLSS.from_calib_file(os.path.join(args.data, "calibration.json"),
                                      num_classes=args.num_classes, bev_h=240, bev_w=240,
                                      img_h=IMG_H, img_w=IMG_W, downsample=ds, pool_res=pool, pretrained=False)
else:
    m = DualCamBEVFPNResNet(num_classes=args.num_classes, bev_h=240, bev_w=240, pretrained=False)
state = meta["model_state_dict"] if isinstance(meta, dict) and "model_state_dict" in meta else meta
m.load_state_dict(state, strict=True); m = m.to(dev).eval()

files = [l.strip() for l in open(os.path.join(args.data, args.split)) if l.strip()]
print(f"=== {os.path.basename(args.data)}  fov={args.fov:g}  {os.path.basename(args.model)} ===")
print(f"{len(files)} frames.  visible region = {VIS.mean():.1%} of BEV grid (fov={args.fov:g})\n")
print(f"{'region':10s} {'IoU':>8} {'recall':>8} {'prec':>8}")
print("-" * 40)
acc = {"all": [0, 0, 0], "visible": [0, 0, 0]}
with torch.no_grad():
    for n in files:
        imgs = torch.stack([read_img(os.path.join(args.data, "front_rgb", n)),
                            read_img(os.path.join(args.data, "rear_rgb", n))], 0).unsqueeze(0).to(dev)
        p = torch.argmax(m(imgs)[0], 0).cpu().numpy() == VEH
        g = cv2.imread(os.path.join(args.data, "bev_gt", n), cv2.IMREAD_GRAYSCALE) == VEH
        for key, mask in (("all", np.ones_like(VIS)), ("visible", VIS)):
            pp, gg = p & mask, g & mask
            acc[key][0] += int((pp & gg).sum()); acc[key][1] += int(pp.sum()); acc[key][2] += int(gg.sum())
for key in ("all", "visible"):
    i, pv, gv = acc[key]
    print(f"{key:10s} {i/max(pv+gv-i,1):>8.4f} {i/max(gv,1):>8.4f} {i/max(pv,1):>8.4f}")
