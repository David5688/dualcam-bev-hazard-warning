import os
import cv2
import argparse
import numpy as np
import torch

from dualcam_bev_fpn_model import DualCamBEVFPN
from dualcam_bev_fpn_resnet import DualCamBEVFPNResNet
from dualcam_bev_lss import DualCamBEVLSS

IMG_H, IMG_W = 224, 400
ROAD_GRAY = (80, 80, 80)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)


def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def fov_visible_mask(bev_h, bev_w, x_min=-30.0, y_min=-30.0, res=0.25,
                     cam_x=1.5, d0=2.0, d1=42.0):
    """BEV cells inside either 90-deg camera's field of view (front + rear)."""
    rows = np.arange(bev_h)[:, None]
    cols = np.arange(bev_w)[None, :]
    x = np.broadcast_to(x_min + (bev_h - 1 - rows) * res, (bev_h, bev_w))
    y = np.broadcast_to(y_min + cols * res, (bev_h, bev_w))
    fx, fy = x - cam_x, y
    front = (fx > 0) & (np.abs(fx) >= np.abs(fy)) & (np.hypot(fx, fy) >= d0) & (np.hypot(fx, fy) <= d1)
    rx, ry = x + cam_x, y
    rear = (rx < 0) & (np.abs(rx) >= np.abs(ry)) & (np.hypot(rx, ry) >= d0) & (np.hypot(rx, ry) <= d1)
    return front | rear


def drop_small_blobs(lab, cls, min_area):
    """Remove speckle: connected components of `cls` smaller than min_area."""
    mask = (lab == cls).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            lab[comp == i] = 0
    return lab


def _ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def smooth_region(mask, close_k=5, open_k=3, min_area=60):
    """Tidy a predicted area mask for display: fill pinholes, shave the spiky
    edges the decoder produces, and drop isolated false-positive islands.

    The raw prediction has ragged borders; the green contour traces them and
    amplifies the raggedness into visual noise. This only affects how the
    PREDICTION panel is drawn -- never the ground truth, never the metrics.
    """
    m = mask.astype(np.uint8)
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ell(close_k))
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ell(open_k))
    if min_area:
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                m[comp == i] = 0
    return m.astype(bool)


DYNAMIC_CLASSES = (1, 2)                          # vehicle, pedestrian


def render_bev_prof(label, out_size=540, dim_blind=None, mask_blind=None,
                    blind_policy="dynamic", min_vehicle=0, min_lane=0, smooth_road=False):
    """Professor LSS style: black bg, gray drivable fill, green contour, red vehicles.

    dim_blind    cells outside the camera FOV, drawn at 45% brightness (both panels)
    mask_blind   blind-zone cells -- PREDICTION panel only. What to clear there is set
                 by blind_policy; the cameras see nothing in this region.
    blind_policy how to treat mask_blind cells (eval_temporal.py measured all three on
                 Town01, 500 frames):
                   "dynamic"  clear only vehicle/pedestrian -- static road/sidewalk/lane
                              predictions are kept. This is variant D and the BEST:
                              global mIoU 0.3076 vs 0.2515 for "all". The model's
                              blind-zone road IS a valid context prior (blind road
                              IoU 0.56 / recall 0.78); only its dynamic output is
                              noise (vehicle blind IoU 0.0012). DEFAULT.
                   "all"      blank the whole blind zone (the old --clean_pred). Throws
                              away the valid static prior: road 0.687 -> 0.548.
    smooth_road  tidy the ragged road border (the green contour traces it and turns
                 the raggedness into visual noise) and drop false-positive islands
    min_vehicle / min_lane  drop speckle blobs smaller than this

    Ground truth is never masked, smoothed or de-speckled.
    """
    lab = label.astype(np.uint8).copy()

    if mask_blind is not None:
        if blind_policy == "all":
            lab[mask_blind] = 0                   # blank everything the cameras can't see
        else:                                     # "dynamic" (variant D): keep static prior
            lab[mask_blind & np.isin(lab, DYNAMIC_CLASSES)] = 0
    if min_vehicle > 0:
        lab = drop_small_blobs(lab, 1, min_vehicle)
    if min_lane > 0:
        lab = drop_small_blobs(lab, 5, min_lane)
    if smooth_road:
        keep = np.isin(lab, (1, 2, 5))           # never overwrite vehicle/ped/lane
        # close_k=9 also fills the pinholes that confidence filtering punches into
        # the road interior -- those holes are a filtering artefact, not a model error
        road_m = smooth_region(lab == 3, close_k=9, open_k=3, min_area=60)
        lab[lab == 3] = 0
        lab[road_m & ~keep] = 3

    lab = cv2.resize(lab, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    blind = dim_blind
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    road = (lab == 3)
    mask = (road.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # straighten the wavy border into GT-like polygon segments; fill + green edge
    # come from the SAME polygon so the gray fill and the boundary line stay flush
    eps = out_size * 0.006
    polys = [cv2.approxPolyDP(c, eps, True) for c in contours if cv2.contourArea(c) > 40]
    cv2.drawContours(canvas, polys, -1, ROAD_GRAY, -1)   # filled road, straight edges
    cv2.drawContours(canvas, polys, -1, GREEN, 2)        # green boundary line

    # yellow lane markings on top of the road fill
    canvas[lab == 5] = YELLOW
    # vehicles as red, pedestrians as small magenta
    canvas[lab == 1] = RED
    canvas[lab == 2] = MAGENTA

    if blind is not None:
        b = cv2.resize(blind.astype(np.uint8), (out_size, out_size),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        # semi-transparent blind overlay: content (road/vehicle) shows through
        # darkened, and the blind wedge stays faintly visible over black bg
        a, fill = 0.55, np.float32((52, 52, 52))
        canvas[b] = (canvas[b].astype(np.float32) * (1 - a) + fill * a).astype(np.uint8)
    return canvas


def panel(img_bev, title, panel_w=960, panel_h=540):
    """Center a square BEV on a black panel and title it."""
    p = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    s = min(panel_h, panel_w)
    bev = cv2.resize(img_bev, (s, s), interpolation=cv2.INTER_NEAREST)
    x0 = (panel_w - s) // 2
    p[0:s, x0:x0 + s] = bev
    cv2.putText(p, title, (24, 48), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    return p


def rgb_panel(img, title, panel_w=960, panel_h=540):
    p = cv2.resize(img, (panel_w, panel_h))
    cv2.putText(p, title, (24, 48), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="carla_dualcam_gt_out")
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--model", default="dualcam_bev_fpn_best.pt")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--out", default="pred_vs_gt_profstyle.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--arch", default="fpn", choices=["fpn", "resnet", "lss"])
    ap.add_argument("--img_h", type=int, default=224)
    ap.add_argument("--img_w", type=int, default=400)
    ap.add_argument("--show_fov", action="store_true",
                    help="dim the BEV cells outside the 2-camera FOV (blind sectors)")
    ap.add_argument("--clean_pred", action="store_true",
                    help="prediction panel only: clean the blind zone (variant D by default: "
                         "clear only vehicle/pedestrian, KEEP road/sidewalk/lane -- the "
                         "static blind-zone prediction is a valid context prior, not "
                         "hallucination; measured best, mIoU 0.3076) and drop speckle blobs. "
                         "Ground truth is never altered.")
    ap.add_argument("--blind_all", action="store_true",
                    help="with --clean_pred: blank the ENTIRE blind zone instead of variant D. "
                         "This is the old behaviour; it throws away the valid static prior "
                         "(road 0.687->0.548). Kept only for reproducing older videos.")
    ap.add_argument("--keep_blind", action="store_true",
                    help="with --clean_pred: still DRAW the full prediction in the blind zone "
                         "(smoothing/de-speckling/confidence filtering still apply). "
                         "Note the cameras see nothing there -- dynamic output is hallucination.")
    ap.add_argument("--conf_road", type=float, default=0.0,
                    help="drop road predictions the model is less than this confident about. "
                         "0.7 raises road IoU 0.496->0.530 and precision 0.538->0.606 on Town01: "
                         "it removes low-confidence false positives, it does not hide errors.")
    ap.add_argument("--conf_lane", type=float, default=0.0,
                    help="same for lane markings (0.7: IoU 0.078->0.116, precision 0.095->0.191)")
    args = ap.parse_args()

    global IMG_H, IMG_W
    IMG_H, IMG_W = args.img_h, args.img_w

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    front_dir = os.path.join(args.data, "front_rgb")
    rear_dir = os.path.join(args.data, "rear_rgb")
    gt_dir = os.path.join(args.data, "bev_gt")
    with open(os.path.join(args.data, args.split)) as f:
        files = [l.strip() for l in f if l.strip()]

    sample = cv2.imread(os.path.join(gt_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    bev_h, bev_w = sample.shape
    if args.arch == "lss":
        meta = torch.load(args.model, map_location="cpu", weights_only=False)
        ds = meta.get("lss_downsample", 8) if isinstance(meta, dict) else 8
        pool = meta.get("lss_pool", 0.5) if isinstance(meta, dict) else 0.5
        del meta
        model = DualCamBEVLSS.from_calib_file(
            os.path.join(args.data, "calibration.json"),
            num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w,
            img_h=IMG_H, img_w=IMG_W, downsample=ds, pool_res=pool, pretrained=False).to(device)
    elif args.arch == "resnet":
        model = DualCamBEVFPNResNet(num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w, pretrained=False).to(device)
    else:
        model = DualCamBEVFPN(num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    blind = None
    if args.show_fov or args.clean_pred:
        blind = ~fov_visible_mask(bev_h, bev_w)
        print(f"blind zone = {100*blind.mean():.1f}% of the BEV")
    dim_blind = blind if args.show_fov else None
    mask_blind = blind if (args.clean_pred and not args.keep_blind) else None
    blind_policy = "all" if args.blind_all else "dynamic"
    if args.clean_pred:
        if args.keep_blind:
            where = "keeping full blind-zone output (dynamic part hallucinated)"
        elif args.blind_all:
            where = "blanking ENTIRE blind zone (old behaviour, discards static prior)"
        else:
            where = "variant D: clearing only dynamic in blind zone, keeping static prior"
        print(f"clean_pred: {where} + smoothing + de-speckling, PREDICTION panel only")

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))

    with torch.no_grad():
        for i, name in enumerate(files):
            front_raw = cv2.imread(os.path.join(front_dir, name))
            rear_raw = cv2.imread(os.path.join(rear_dir, name))
            gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE)

            imgs = torch.stack([read_img(os.path.join(front_dir, name)),
                                read_img(os.path.join(rear_dir, name))], 0).unsqueeze(0).to(device)
            logits = model(imgs)[0]
            if args.conf_road > 0 or args.conf_lane > 0:
                prob = torch.softmax(logits, dim=0)
                conf, cls = prob.max(0)
                conf = conf.cpu().numpy()
                pred = cls.cpu().numpy().astype(np.uint8)
                # drop predictions the model itself is unsure about (false positives)
                if args.conf_road > 0:
                    pred[(pred == 3) & (conf < args.conf_road)] = 0
                if args.conf_lane > 0:
                    pred[(pred == 5) & (conf < args.conf_lane)] = 0
            else:
                pred = torch.argmax(logits, 0).cpu().numpy().astype(np.uint8)

            top = np.hstack([rgb_panel(front_raw, "Front RGB"),
                             rgb_panel(rear_raw, "Rear RGB")])
            bottom = np.hstack([
                panel(render_bev_prof(pred, dim_blind=dim_blind, mask_blind=mask_blind,
                                      blind_policy=blind_policy, min_vehicle=15, min_lane=6,
                                      smooth_road=True), "Prediction"),
                panel(render_bev_prof(gt, dim_blind=dim_blind), "Ground Truth")])
            writer.write(np.vstack([top, bottom]))
            if i % 50 == 0:
                print("frame", i)

    writer.release()
    print("saved:", args.out)


if __name__ == "__main__":
    main()
