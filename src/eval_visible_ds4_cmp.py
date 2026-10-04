import os, cv2, numpy as np, torch
from dualcam_bev_lss import DualCamBEVLSS

D = "E:/dualcam_project/data/carla_dualcam_gt_lane"
CK = "E:/dualcam_project/checkpoints"
IMG_H, IMG_W, dev = 224, 400, "cuda"
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

def load(f):
    ck = torch.load(f"{CK}/{f}", map_location="cpu", weights_only=False)
    ds = ck.get("lss_downsample", 8); pool = ck.get("lss_pool", 0.5)
    m = DualCamBEVLSS.from_calib_file(os.path.join(D, "calibration.json"),
            num_classes=6, bev_h=240, bev_w=240, img_h=IMG_H, img_w=IMG_W,
            downsample=ds, pool_res=pool, pretrained=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.to(dev).eval(), ck.get("epoch")

def metric(pred, gt, cls, mask=None):
    p, g = pred == cls, gt == cls
    if mask is not None: p, g = p & mask, g & mask
    i = int((p & g).sum()); u = int((p | g).sum())
    return i, u, int(p.sum()), int(g.sum())

MODELS = [("lss+aug (ds8)", "dualcam_bev_lss_aug_best.pt"),
          ("lss+aug (ds4)", "dualcam_bev_lss_aug_ds4_best.pt")]
files = [l.strip() for l in open(os.path.join(D, "test.txt")) if l.strip()]

print(f"{'model':16s} {'ep':>3} | {'veh all':>8} {'veh vis':>8} {'vis rec':>8} {'vis prec':>9} | {'road all':>8} {'road vis':>8}")
print("-"*82)
for name, f in MODELS:
    m, ep = load(f)
    acc = {c: {'all':[0,0,0,0], 'vis':[0,0,0,0]} for c in (1,3)}
    with torch.no_grad():
        for n in files:
            imgs = torch.stack([read_img(os.path.join(D,"front_rgb",n)),
                                read_img(os.path.join(D,"rear_rgb",n))],0).unsqueeze(0).to(dev)
            pr = torch.argmax(m(imgs)[0],0).cpu().numpy()
            gt = cv2.imread(os.path.join(D,"bev_gt",n), cv2.IMREAD_GRAYSCALE)
            for c in (1,3):
                for k,msk in (('all',None),('vis',VIS)):
                    i,u,pv,gv = metric(pr,gt,c,msk)
                    a=acc[c][k]; a[0]+=i; a[1]+=u; a[2]+=pv; a[3]+=gv
    v_all=acc[1]['all']; v_vis=acc[1]['vis']; r_all=acc[3]['all']; r_vis=acc[3]['vis']
    print(f"{name:16s} {ep:>3} | {v_all[0]/max(v_all[1],1):>8.4f} {v_vis[0]/max(v_vis[1],1):>8.4f} "
          f"{v_vis[0]/max(v_vis[3],1):>8.4f} {v_vis[0]/max(v_vis[2],1):>9.4f} | "
          f"{r_all[0]/max(r_all[1],1):>8.4f} {r_vis[0]/max(r_vis[1],1):>8.4f}")
    del m; torch.cuda.empty_cache()
