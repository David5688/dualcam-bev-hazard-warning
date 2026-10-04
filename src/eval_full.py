"""Complete, no-shortcuts evaluation against CARLA ground truth.

Four metric families, each with the exact formula it uses (so the report can cite
them). GT = CARLA (bev_gt labels + depth camera). Everything is measured in the
camera-visible region where relevant (the 54.7% blind zone is hallucination and is
reported separately, not hidden).

  A. Semantic segmentation (pixel-level)   IoU / Precision / Recall / mIoU / PixelAcc
  B. Instance detection (object-level)     Recall / Precision / localisation error
                                           (pedestrian: BEV vs image+depth, the 0.16 vs 0.71)
  C. Depth (model vs GT depth)             MAE(EPE) / RMSE / AbsRel / delta<1.25^k, per range band
  D. Hazard warning (event-level)          Precision / Recall vs GT hazards
                                           (variant-D vs trust-blind -> the 31->3 claim, rigorous)

Formulae
  TP = pred & gt ;  FP = pred & !gt ;  FN = !pred & gt
  IoU        = TP / (TP + FP + FN)
  Precision  = TP / (TP + FP)
  Recall     = TP / (TP + FN)
  mIoU       = mean_c IoU_c ;  PixelAcc = correct / total
  Instance recall/precision use nearest-neighbour matching within a distance gate.
  Depth: MAE=mean|dp-dg| ; RMSE=sqrt(mean(dp-dg)^2) ; AbsRel=mean(|dp-dg|/dg) ;
         delta_k = %[ max(dp/dg, dg/dp) < 1.25^k ].
  TTC = range / closing_rate ;  warning = level in {CAUTION,DANGER}.

  python eval_full.py --data ../data/carla_dualcam_hazard_2k_v2 --stride 3
"""
import os, sys, cv2, json, math, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
import pedestrian_detect as PD
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask
from depth_to_pointcloud import depth_png_to_metres

NAMES = ["background", "vehicle", "pedestrian", "road", "sidewalk", "lane_marking"]
C = 6
BEV, RES, X0, Y0 = HZ.BEV, HZ.RES, HZ.X0, HZ.Y0
IMG_H, IMG_W = 224, 400
DYN = (1, 2)


# ---------- metric helpers ----------
def cm_iou_pr(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    return iou, prec, rec


def clusters(label, cls, vis, min_area=6):
    """(x_ego, y_ego) centroids of `cls` blobs inside the visible region."""
    out = []
    mask = (label == cls).astype(np.uint8)
    n, _, st, ct = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        col, row = ct[i]
        r, c = int(round(row)), int(round(col))
        if 0 <= r < BEV and 0 <= c < BEV and vis[r, c]:
            out.append(HZ.px_to_ego(col, row))
    return out


def match(gts, dets, gate):
    """Greedy nearest-neighbour match. Returns (n_matched, sum_loc_err over matches)."""
    used = [False] * len(dets)
    hit, err = 0, 0.0
    for gx, gy in gts:
        best, bd = -1, gate
        for j, (dx, dy) in enumerate(dets):
            if used[j]:
                continue
            d = math.hypot(gx - dx, gy - dy)
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used[best] = True; hit += 1; err += bd
    return hit, err


def depth_metrics(dp, dg, lo, hi):
    """dp, dg flat arrays of predicted/GT depth in [lo,hi). Standard depth errors."""
    m = (dg >= lo) & (dg < hi) & (dp > 0)
    if m.sum() < 10:
        return None
    dp, dg = dp[m], dg[m]
    ae = np.abs(dp - dg)
    ratio = np.maximum(dp / dg, dg / dp)
    return dict(n=int(m.sum()), MAE=float(ae.mean()), RMSE=float(np.sqrt(((dp - dg) ** 2).mean())),
                AbsRel=float((ae / dg).mean()),
                d1=float((ratio < 1.25).mean()), d2=float((ratio < 1.25 ** 2).mean()),
                d3=float((ratio < 1.25 ** 3).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_hazard_2k_v2")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=2000)
    ap.add_argument("--gate_veh", type=float, default=3.0)
    ap.add_argument("--gate_ped", type=float, default=2.5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out_csv", default="../metrics/eval_full.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = args.data
    calib = json.load(open(f"{D}/calibration.json"))
    poses = json.load(open(f"{D}/poses.json"))
    VIS = fov_visible_mask(BEV, BEV); BLIND = ~VIS
    ALL = np.ones((BEV, BEV), bool)

    meta = torch.load(args.model, map_location="cpu", weights_only=False)
    m = DualCamBEVLSS.from_calib_file(f"{D}/calibration.json", num_classes=C, bev_h=BEV, bev_w=BEV,
                                      img_h=IMG_H, img_w=IMG_W, downsample=meta.get("lss_downsample", 8),
                                      pool_res=meta.get("lss_pool", 0.5), pretrained=False)
    m.load_state_dict(meta.get("model_state_dict", meta), strict=True); m = m.to(dev).eval()
    centers = (m.d0 + (torch.arange(m.D) + 0.5) * m.dd).float().to(dev)

    def rd(p):
        im = cv2.resize(cv2.imread(p), (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
        return torch.from_numpy(im).permute(2, 0, 1)

    def infer(n):
        x = torch.stack([rd(f"{D}/front_rgb/{n}"), rd(f"{D}/rear_rgb/{n}")], 0).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits, dl = m(x, return_depth=True)
            lab = torch.argmax(logits[0], 0).cpu().numpy().astype(np.uint8)
            prob = torch.softmax(dl, 1)
            expd = (prob * centers[None, :, None, None]).sum(1)      # (2, fH, fW) expected depth
        return lab, expd.cpu().numpy()

    # accumulators
    cm_glob = np.zeros((C, C), np.int64); cm_vis = np.zeros((C, C), np.int64)
    inst = {k: [0, 0, 0, 0.0] for k in ("veh", "ped_bev", "ped_img")}   # [gt, pred, hit, err_sum]
    BANDS = [(2, 10), (10, 20), (20, 42)]
    dsum = {b: [] for b in BANDS}                                       # per-band metric dicts
    warn = {"D": [0, 0, 0], "raw": [0, 0, 0]}                           # [TP, FP, FN]
    trk = {"D": HZ.Tracker(), "raw": HZ.Tracker(), "gt": HZ.Tracker()}
    dt = args.stride / args.fps

    frames = [f"{i:06d}.png" for i in range(0, min(args.nmax, 2000), args.stride)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{D}/bev_gt/{f}")]
    Kf = {c: m._scaled_K(calib["cameras"][c]["K"], calib["image"]["width"], calib["image"]["height"]).numpy()
          for c in ("front", "rear")}
    Tf = {c: calib["cameras"][c]["T_ego_cam"] for c in ("front", "rear")}

    def cm_add(cm, gt, pr, mask):
        g = gt[mask].astype(np.int64); p = pr[mask].astype(np.int64)
        ok = (g >= 0) & (g < C) & (p >= 0) & (p < C)
        cm += np.bincount(g[ok] * C + p[ok], minlength=C * C).reshape(C, C)

    def gt_depth_feat(cam, n):
        """GT depth min-pooled to feature grid (foreground wins), matching model output."""
        d = depth_png_to_metres(cv2.imread(f"{D}/{cam}_depth/{n}"))
        d = cv2.resize(d, (IMG_W, IMG_H))
        t = torch.from_numpy(d)[None, None]
        t = -torch.nn.functional.adaptive_max_pool2d(-t, (m.fH, m.fW))[0, 0]
        return t.numpy()

    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]
        gt = cv2.imread(f"{D}/bev_gt/{n}", 0)
        pred, expd = infer(n)
        predD = pred.copy(); predD[BLIND & np.isin(predD, DYN)] = 0          # variant D

        # ---- A. segmentation ----
        cm_add(cm_glob, gt, pred, ALL)
        cm_add(cm_vis, gt, pred, VIS)

        # ---- C. depth (model expected vs GT), per band, front+rear ----
        for cam in ("front", "rear"):
            dg = gt_depth_feat(cam, n).ravel()
            dp = expd[0 if cam == "front" else 1].ravel()
            for b in BANDS:
                r = depth_metrics(dp, dg, *b)
                if r: dsum[b].append(r)

        # ---- B. instances ----
        gv = clusters(gt, 1, VIS); pv = clusters(predD, 1, VIS)
        h, e = match(gv, pv, args.gate_veh)
        inst["veh"][0] += len(gv); inst["veh"][1] += len(pv); inst["veh"][2] += h; inst["veh"][3] += e
        gp = clusters(gt, 2, VIS); pbev = clusters(predD, 2, VIS)
        h, e = match(gp, pbev, args.gate_ped)
        inst["ped_bev"][0] += len(gp); inst["ped_bev"][1] += len(pbev); inst["ped_bev"][2] += h; inst["ped_bev"][3] += e
        # image+depth pedestrians
        peds = PD.pedestrians_in_frame(D, n, calib, dev, 0.7)   # tuned threshold
        pimg = []
        for (x, y, cf, cam, box) in peds:
            col = int((y - Y0) / RES); row = int((BEV - 1) - (x - X0) / RES)
            if 0 <= row < BEV and 0 <= col < BEV and VIS[row, col]:
                pimg.append((x, y))
        h, e = match(gp, pimg, args.gate_ped)
        inst["ped_img"][0] += len(gp); inst["ped_img"][1] += len(pimg); inst["ped_img"][2] += h; inst["ped_img"][3] += e

        # ---- D. warning: build obstacle sets, track, level, match to GT hazards ----
        def obst(label, use_img_ped):
            o = []
            for (x, y, cls, a) in HZ.detect_obstacles(label):
                if cls == 2 and use_img_ped:                 # peds come from image for the system
                    continue
                wx, wy = HZ.world_of(x, y, ex, ey, yaw)
                o.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": cls, "range": math.hypot(x, y)})
            if use_img_ped:
                for (x, y) in pimg:
                    wx, wy = HZ.world_of(x, y, ex, ey, yaw)
                    o.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": 2, "range": math.hypot(x, y)})
            return o

        def leveled(tracker, obs, t):
            tracker.update(obs, t); res = []
            for tid, tr in tracker.active():
                d = tr["cur"]; ttc, _ = HZ.ttc_of(tr["hist"])
                lv, _ = HZ.risk_of(d["range"], ttc)
                if lv != "SAFE":
                    res.append((d["x"], d["y"], lv))
            return res

        gt_haz = leveled(trk["gt"], obst(gt, use_img_ped=False), k * dt)
        sysD = leveled(trk["D"], obst(predD, use_img_ped=True), k * dt)
        sysR = leveled(trk["raw"], obst(pred, use_img_ped=True), k * dt)   # raw: blind dyn kept

        for key, sysw in (("D", sysD), ("raw", sysR)):
            gpos = [(x, y) for (x, y, lv) in gt_haz]; spos = [(x, y) for (x, y, lv) in sysw]
            tp, _ = match(gpos, spos, args.gate_veh)
            warn[key][0] += tp
            warn[key][1] += len(spos) - tp                # FP = system warnings with no GT hazard
            warn[key][2] += len(gpos) - tp                # FN = GT hazards not warned
        if k % 40 == 0:
            print(f"{k}/{len(frames)}", flush=True)

    # ================= report =================
    rows = []
    def P(s): print(s);
    P(f"\n{'#'*70}\n# EVALUATION on {os.path.basename(D)}  ({len(frames)} frames, stride {args.stride})\n{'#'*70}")

    P("\n== A. Semantic segmentation (pixel-level) ==")
    for tag, cm in (("visible", cm_vis), ("global", cm_glob)):
        iou, pr, rc = cm_iou_pr(cm)
        acc = np.diag(cm).sum() / max(cm.sum(), 1)
        P(f"\n  [{tag}]  mIoU {np.nanmean(iou):.4f}  PixelAcc {acc:.4f}")
        P(f"  {'class':<13}{'IoU':>8}{'Prec':>8}{'Recall':>8}")
        for ci in range(C):
            P(f"  {NAMES[ci]:<13}{iou[ci]:>8.4f}{pr[ci]:>8.4f}{rc[ci]:>8.4f}")
            rows.append(("seg_" + tag, NAMES[ci], "IoU", iou[ci]))
            rows.append(("seg_" + tag, NAMES[ci], "precision", pr[ci]))
            rows.append(("seg_" + tag, NAMES[ci], "recall", rc[ci]))
        rows.append(("seg_" + tag, "all", "mIoU", float(np.nanmean(iou))))
        rows.append(("seg_" + tag, "all", "PixelAcc", float(acc)))

    P("\n== B. Instance detection (visible region, nearest-match) ==")
    P(f"  {'target':<22}{'GT':>6}{'Pred':>6}{'Recall':>8}{'Prec':>8}{'LocErr(m)':>10}")
    labelmap = {"veh": f"vehicle (gate {args.gate_veh}m)", "ped_bev": f"ped: BEV (gate {args.gate_ped}m)",
                "ped_img": f"ped: image+depth (gate {args.gate_ped}m)"}
    for key in ("veh", "ped_bev", "ped_img"):
        g, p, hit, err = inst[key]
        rec = hit / max(g, 1); prec = hit / max(p, 1); loc = err / max(hit, 1)
        P(f"  {labelmap[key]:<22}{g:>6}{p:>6}{rec:>8.3f}{prec:>8.3f}{loc:>10.2f}")
        rows.append(("inst", key, "recall", rec)); rows.append(("inst", key, "precision", prec))
        rows.append(("inst", key, "loc_err_m", loc))

    P("\n== C. Depth (model expected vs CARLA GT, front+rear pooled) ==")
    P(f"  {'range':<12}{'N':>10}{'MAE(m)':>9}{'RMSE(m)':>9}{'AbsRel':>9}{'d<1.25':>9}{'d<1.25^2':>9}{'d<1.25^3':>9}")
    for b in BANDS:
        ds = dsum[b]
        if not ds: continue
        agg = lambda key: float(np.average([d[key] for d in ds], weights=[d["n"] for d in ds]))
        N = sum(d["n"] for d in ds)
        P(f"  {str(b[0])+'-'+str(b[1])+'m':<12}{N:>10}{agg('MAE'):>9.3f}{agg('RMSE'):>9.3f}"
          f"{agg('AbsRel'):>9.3f}{agg('d1'):>9.3f}{agg('d2'):>9.3f}{agg('d3'):>9.3f}")
        for key in ("MAE", "RMSE", "AbsRel", "d1", "d2", "d3"):
            rows.append(("depth", f"{b[0]}-{b[1]}m", key, agg(key)))

    P("\n== D. Hazard warning (event-level, vs GT hazards) ==")
    P(f"  {'system':<26}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for key, tag in (("raw", "raw pred (trust blind)"), ("D", "variant D (deployed)")):
        tp, fp, fn = warn[key]
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        P(f"  {tag:<26}{tp:>6}{fp:>6}{fn:>6}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}")
        rows.append(("warn", key, "precision", prec)); rows.append(("warn", key, "recall", rec))
        rows.append(("warn", key, "TP", tp)); rows.append(("warn", key, "FP", fp)); rows.append(("warn", key, "FN", fn))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("section,item,metric,value\n")
        for sec, item, met, val in rows:
            f.write(f"{sec},{item},{met},{val:.6f}\n")
    P(f"\nsaved: {args.out_csv}")


if __name__ == "__main__":
    main()
