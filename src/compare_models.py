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

MODELS = [
    ("resnet",    "dualcam_bev_fpn_resnet_contig_best.pt"),
    ("lss",       "dualcam_bev_lss_contig_best.pt"),
    ("lss+aug",   "dualcam_bev_lss_aug_best.pt"),
]

def read_img(p):
    img = cv2.imread(p)
    img = cv2.resize(img, (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def load(name, f):
    if name == "resnet":
        m = DualCamBEVFPNResNet(num_classes=6, bev_h=240, bev_w=240, pretrained=False)
    else:
        m = DualCamBEVLSS.from_calib_file(os.path.join(D, "calibration.json"),
                                          num_classes=6, bev_h=240, bev_w=240, pretrained=False)
    ck = torch.load(f"{CK}/{f}", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.to(dev).eval(), ck.get("epoch")

def run(m, split, mask=None):
    files = [l.strip() for l in open(os.path.join(D, split)) if l.strip()]
    i = pv = gv = 0
    with torch.no_grad():
        for n in files:
            imgs = torch.stack([read_img(os.path.join(D, "front_rgb", n)),
                                read_img(os.path.join(D, "rear_rgb", n))], 0).unsqueeze(0).to(dev)
            p = torch.argmax(m(imgs)[0], 0).cpu().numpy() == VEH
            g = cv2.imread(os.path.join(D, "bev_gt", n), cv2.IMREAD_GRAYSCALE) == VEH
            if mask is not None: p, g = p & mask, g & mask
            i += int((p & g).sum()); pv += int(p.sum()); gv += int(g.sum())
    return i/max(pv+gv-i,1), i/max(gv,1), i/max(pv,1)

print(f"{'model':10s} {'ep':>3} | {'train IoU':>9} {'test IoU':>8} {'gap':>6} | "
      f"{'test recall':>11} {'test prec':>9} | {'visible IoU':>11} {'vis recall':>10}")
print("-" * 96)
for name, f in MODELS:
    m, ep = load(name, f)
    tr, _, _ = run(m, "train.txt")
    te, rec, pre = run(m, "test.txt")
    vi, vrec, _ = run(m, "test.txt", VIS)
    print(f"{name:10s} {ep:>3} | {tr:>9.4f} {te:>8.4f} {tr-te:>6.3f} | "
          f"{rec:>11.4f} {pre:>9.4f} | {vi:>11.4f} {vrec:>10.4f}")
    del m; torch.cuda.empty_cache()

print(f"\nhard recall ceiling on test (GT vehicle px inside FOV): 34.4%")
print(f"visible region = {VIS.mean():.1%} of the BEV grid")
