"""Quantify the temporal BEV fusion (companion to viz_temporal.py).

The fusion loop is line-for-line the same as viz_temporal.py; only the rendering
is swapped for confusion matrices. Four blind-zone policies are scored against
the same GT, so the comparison is one-variable -- the policy:

  A  single frame, raw            -- model output everywhere, blind zone included
  B  single frame, blind -> bg    -- the --clean_pred convention (blank the blind zone)
  C  temporal fusion              -- ego-motion-warped static memory fills the blind zone
  D  class-dependent blind policy -- keep predicted STATIC in blind, blank DYNAMIC only

All four are bit-identical inside the visible region, so every global difference
comes from the blind zone -- hence the blind-only table, which is what this
actually measures. vehicle is the negative control for C: fusion never remembers
dynamic classes, so C must equal B on vehicle exactly.

Frames are also split by yaw rate (turn vs straight) and road connected
components are counted, because the claim under test is specifically that
single-frame BEV *fragments the road during turns* and that fusion reconnects it.
"""
import os, sys, cv2, json, math, glob, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
ap.add_argument("--num_classes", type=int, default=6)
ap.add_argument("--img_h", type=int, default=224)
ap.add_argument("--img_w", type=int, default=400)
ap.add_argument("--maxage", type=int, default=20)
ap.add_argument("--turn_deg", type=float, default=1.0, help="|yaw rate| (deg/frame) above which a frame counts as a turn")
ap.add_argument("--out_csv", default="../metrics/eval_temporal_town01.csv")
args = ap.parse_args()

D_, C = args.data, args.num_classes
NAMES = ["background", "vehicle", "pedestrian", "road", "sidewalk", "lane_marking"][:C]
STATIC, DYNAMIC = (3, 4, 5), (1, 2)
BEV = 240
RES, X0, Y0 = 0.25, -30.0, -30.0
VIS = fov_visible_mask(BEV, BEV)
BLIND = ~VIS
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ego-motion warp (identical to viz_temporal.py) ----
def px_to_ego(col, row): return (X0 + (BEV - 1 - row) * RES, Y0 + col * RES)
def ego_to_px(x, y):     return ((y - Y0) / RES, (BEV - 1) - (x - X0) / RES)
def world_from_ego(x, y, ex, ey, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    return ex + c * x - s * y, ey + s * x + c * y
def ego_from_world(wx, wy, ex, ey, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a); dx, dy = wx - ex, wy - ey
    return c * dx + s * dy, -s * dx + c * dy
def align_M(pose_cur, pose_prev):
    src = [(10, 10), (BEV - 10, 10), (10, BEV - 10)]; dst = []
    for col, row in src:
        x, y = px_to_ego(col, row); wx, wy = world_from_ego(x, y, *pose_cur)
        x2, y2 = ego_from_world(wx, wy, *pose_prev); c2, r2 = ego_to_px(x2, y2); dst.append((c2, r2))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

def clean_static(lab):
    out = np.zeros_like(lab)
    for cls in STATIC:
        mm = cv2.morphologyEx((lab == cls).astype(np.uint8) * 255, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out[mm > 0] = cls
    return out

def road_fragments(lab, min_area=60):
    """(#connected road components, largest-component share) -- the fragmentation claim."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((lab == 3).astype(np.uint8), 8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not areas: return 0, np.nan
    return len(areas), max(areas) / sum(areas)

# ---- model ----
meta = torch.load(args.model, map_location="cpu", weights_only=False)
ds = meta.get("lss_downsample", 8); pool = meta.get("lss_pool", 0.5)
m = DualCamBEVLSS.from_calib_file(f"{D_}/calibration.json", num_classes=C, bev_h=BEV, bev_w=BEV,
                                  img_h=args.img_h, img_w=args.img_w, downsample=ds,
                                  pool_res=pool, pretrained=False)
st = meta["model_state_dict"] if isinstance(meta, dict) and "model_state_dict" in meta else meta
m.load_state_dict(st, strict=True); m = m.to(dev).eval()

def readimg(p):
    im = cv2.imread(p); im = cv2.resize(im, (args.img_w, args.img_h))[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1)
def predict(n):
    x = torch.stack([readimg(f"{D_}/front_rgb/{n}"), readimg(f"{D_}/rear_rgb/{n}")], 0).unsqueeze(0).to(dev)
    with torch.no_grad(): return torch.argmax(m(x)[0], 0).cpu().numpy().astype(np.uint8)

poses = json.load(open(f"{D_}/poses.json"))
files = sorted(os.path.basename(p) for p in glob.glob(f"{D_}/bev_gt/*.png"))
files = [f for f in files if f in poses]

# ---- turn / straight split by yaw rate ----
def dyaw(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)
is_turn = {}
for i, n in enumerate(files):
    if i == 0: is_turn[n] = False; continue
    is_turn[n] = dyaw(poses[n][2], poses[files[i - 1]][2]) >= args.turn_deg
n_turn = sum(is_turn.values())
print(f"{len(files)} frames | blind = {100*BLIND.mean():.1f}% of BEV | "
      f"turn frames (|yaw rate| >= {args.turn_deg} deg/f) = {n_turn} ({100*n_turn/len(files):.0f}%)")

VARIANTS = ["A_single_raw", "B_blind_bg", "C_temporal", "D_blind_static_only"]
REGIONS = ["global", "blind"]
SUBSETS = ["all", "turn", "straight"]
cm = {v: {(r, s): np.zeros((C, C), np.int64) for r in REGIONS for s in SUBSETS} for v in VARIANTS}
frag = {v: {s: [] for s in SUBSETS} for v in VARIANTS}

def accumulate(v, key, gt, pred, mask):
    g = gt[mask].astype(np.int64); p = pred[mask].astype(np.int64)
    ok = (g >= 0) & (g < C) & (p >= 0) & (p < C)
    cm[v][key] += np.bincount(g[ok] * C + p[ok], minlength=C * C).reshape(C, C)

memory = np.zeros((BEV, BEV), np.uint8)
age = np.full((BEV, BEV), 9999, np.int32)
prev_pose = None
ALL = np.ones((BEV, BEV), bool)
for i, n in enumerate(files):
    pred = predict(n); pose = poses[n]
    if prev_pose is not None:
        M = align_M(pose, prev_pose)
        memory = cv2.warpAffine(memory, M, (BEV, BEV), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
        age = cv2.warpAffine(age.astype(np.float32), M, (BEV, BEV), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=9999).astype(np.int32)
    age = age + 1
    memory[VIS] = np.where(np.isin(pred[VIS], STATIC), pred[VIS], 0).astype(np.uint8)
    age[VIS] = 0
    memory[age > args.maxage] = 0
    memory = clean_static(memory)
    fused = memory.copy(); fused[VIS] = pred[VIS]
    prev_pose = pose

    single_bg = pred.copy(); single_bg[BLIND] = 0
    static_only = pred.copy()
    static_only[BLIND & np.isin(pred, DYNAMIC)] = 0       # blank only what the model can't know
    gt = cv2.imread(f"{D_}/bev_gt/{n}", cv2.IMREAD_GRAYSCALE)

    subs = ["all", "turn" if is_turn[n] else "straight"]
    for v, pr in [("A_single_raw", pred), ("B_blind_bg", single_bg),
                  ("C_temporal", fused), ("D_blind_static_only", static_only)]:
        for s in subs:
            accumulate(v, ("global", s), gt, pr, ALL)
            accumulate(v, ("blind", s), gt, pr, BLIND)
            frag[v][s].append(road_fragments(pr))
    for s in subs: frag.setdefault("GT", {sx: [] for sx in SUBSETS})[s].append(road_fragments(gt))
    if i % 100 == 0: print(f"{i}/{len(files)}", flush=True)

def iou_of(k):
    tp = np.diag(k).astype(np.float64); fp = k.sum(0) - tp; fn = k.sum(1) - tp
    den = tp + fp + fn
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)
def rec_prec(k, cls):
    tp = k[cls, cls]; fn = k[cls].sum() - tp; fp = k[:, cls].sum() - tp
    return (tp / max(tp + fn, 1), tp / max(tp + fp, 1))

rows = []
for r in REGIONS:
    for s in SUBSETS:
        print(f"\n{'='*84}\n{r.upper()} region | {s} frames\n{'='*84}")
        print(f"{'class':<14}" + "".join(f"{v:>21}" for v in VARIANTS))
        ious = {v: iou_of(cm[v][(r, s)]) for v in VARIANTS}
        for ci, name in enumerate(NAMES):
            print(f"{name:<14}" + "".join(f"{ious[v][ci]:>21.4f}" for v in VARIANTS))
            for v in VARIANTS: rows.append((r, s, v, name, ious[v][ci]))
        print(f"{'mIoU':<14}" + "".join(f"{np.nanmean(ious[v]):>21.4f}" for v in VARIANTS))
        for v in VARIANTS: rows.append((r, s, v, "mIoU", np.nanmean(ious[v])))
        for ci in (3, 1):
            print(f"  {NAMES[ci]:<12} rec/prec " + " | ".join(
                f"{rec_prec(cm[v][(r,s)], ci)[0]:.3f}/{rec_prec(cm[v][(r,s)], ci)[1]:.3f}" for v in VARIANTS))
            for v in VARIANTS:
                rc, pc = rec_prec(cm[v][(r, s)], ci)
                rows.append((r, s, v, f"{NAMES[ci]}_recall", rc)); rows.append((r, s, v, f"{NAMES[ci]}_precision", pc))

print(f"\n{'='*84}\nROAD FRAGMENTATION (whole BEV) -- the actual claim under test\n{'='*84}")
print(f"{'variant':<22}" + "".join(f"{'#cc '+s:>16}{'top-share '+s:>18}" for s in ["turn", "straight"]))
for v in VARIANTS + ["GT"]:
    line = f"{v:<22}"
    for s in ["turn", "straight"]:
        cc = [a for a, _ in frag[v][s]]; sh = [b for _, b in frag[v][s] if not np.isnan(b)]
        line += f"{np.mean(cc):>16.2f}{np.mean(sh):>18.3f}"
        rows.append(("frag", s, v, "road_cc_mean", np.mean(cc)))
        rows.append(("frag", s, v, "road_top_share", np.mean(sh)))
    print(line)

os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
with open(args.out_csv, "w") as f:
    f.write("region,frames,variant,metric,value\n")
    for r, s, v, k, val in rows: f.write(f"{r},{s},{v},{k},{val:.6f}\n")
print(f"\nframes={len(files)} (turn={n_turn})  maxage={args.maxage}  ->  {args.out_csv}")
