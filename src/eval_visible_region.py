"""Vehicle IoU restricted to the BEV region the two cameras can actually see."""
import os, cv2, numpy as np, torch
from dualcam_bev_fpn_resnet import DualCamBEVFPNResNet
from dualcam_bev_lss import DualCamBEVLSS

D = "E:/dualcam_project/data/carla_dualcam_gt_lane"
CK = "E:/dualcam_project/checkpoints"
IMG_H, IMG_W, VEH, dev = 224, 400, 1, "cuda"
BEV_H = BEV_W = 240
RES, X_MIN, Y_MIN, D0, D1 = 0.25, -30.0, -30.0, 2.0, 42.0

rows = np.arange(BEV_H)[:, None]; cols = np.arange(BEV_W)[None, :]
X = np.broadcast_to(X_MIN + (BEV_H - 1 - rows) * RES, (BEV_H, BEV_W))
Y = np.broadcast_to(Y_MIN + cols * RES, (BEV_H, BEV_W))
fx, fy = X - 1.5, Y
front = (fx > 0) & (np.abs(fx) >= np.abs(fy)) & (np.hypot(fx, fy) >= D0) & (np.hypot(fx, fy) <= D1)
rx, ry = X + 1.5, Y
rear = (rx < 0) & (np.abs(rx) >= np.abs(ry)) & (np.hypot(rx, ry) >= D0) & (np.hypot(rx, ry) <= D1)
VIS = front | rear

def read_img(p):
    img = cv2.imread(p)
    img = cv2.resize(img, (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def load(arch):
    if arch == "lss":
        m = DualCamBEVLSS.from_calib_file(os.path.join(D, "calibration.json"),
                                          num_classes=6, bev_h=240, bev_w=240, pretrained=False)
        f = f"{CK}/dualcam_bev_lss_contig_best.pt"
    else:
        m = DualCamBEVFPNResNet(num_classes=6, bev_h=240, bev_w=240, pretrained=False)
        f = f"{CK}/dualcam_bev_fpn_resnet_contig_best.pt"
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.to(dev).eval()

files = [l.strip() for l in open(os.path.join(D, "test.txt")) if l.strip()]
print(f"test split, {len(files)} frames.  visible region = {VIS.mean():.1%} of the BEV grid\n")
print(f"{'model':8s} {'region':10s} {'IoU':>8} {'recall':>8} {'prec':>8}")
print("-" * 48)
for arch in ["resnet", "lss"]:
    m = load(arch)
    acc = {"all": [0,0,0], "visible": [0,0,0]}
    with torch.no_grad():
        for n in files:
            imgs = torch.stack([read_img(os.path.join(D, "front_rgb", n)),
                                read_img(os.path.join(D, "rear_rgb", n))], 0).unsqueeze(0).to(dev)
            p = torch.argmax(m(imgs)[0], 0).cpu().numpy() == VEH
            g = cv2.imread(os.path.join(D, "bev_gt", n), cv2.IMREAD_GRAYSCALE) == VEH
            for key, mask in (("all", np.ones_like(VIS)), ("visible", VIS)):
                pp, gg = p & mask, g & mask
                acc[key][0] += int((pp & gg).sum()); acc[key][1] += int(pp.sum()); acc[key][2] += int(gg.sum())
    for key in ("all", "visible"):
        i, pv, gv = acc[key]
        print(f"{arch:8s} {key:10s} {i/max(pv+gv-i,1):>8.4f} {i/max(gv,1):>8.4f} {i/max(pv,1):>8.4f}")
    del m; torch.cuda.empty_cache()
