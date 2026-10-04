"""Image-space pedestrian detection + depth localisation (Path B).

Pedestrians are ~2-3 px in the 0.25 m/px BEV, so the BEV segmenter barely finds
them (recall 0.013). But a person is large and clear in the CAMERA image, where a
pretrained COCO detector finds them reliably. This module detects persons in the
front/rear RGB, reads their range from the depth map, and back-projects each to the
ego BEV -- the same geometry as depth_to_pointcloud.py -- so pedestrians can be
placed and warned on without retraining the BEV model.

  python pedestrian_detect.py --frame 000170     # proof: BEV-missed ped, found here
"""
import os, sys, cv2, json, math, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_to_pointcloud import R_CAM_OPT, depth_png_to_metres

_MODEL = None


def detector(device):
    global _MODEL
    if _MODEL is None:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        _MODEL = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
    return _MODEL


def _persons_of(out, score):
    res = []
    for box, lab, sc in zip(out["boxes"].float().cpu(), out["labels"].cpu(),
                            out["scores"].float().cpu()):
        if int(lab) == 1 and float(sc) >= score:            # COCO 1 == person
            x1, y1, x2, y2 = box.tolist()
            res.append((x1, y1, x2, y2, float(sc)))
    return res


def _to_tensor(bgr, device):
    rgb = bgr[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device)


def detect_persons_batch(bgrs, device, score=0.5):
    """Person boxes for several images in ONE forward pass: a list per image.

    One call for both cameras instead of two, and under fp16 autocast.  Measured
    on 40 test_town01 frame pairs (2026-10-02, GPU shared with a running demo):
    151.5 ms -> 122.6 ms per pair, 58 of the 59 boxes the fp32 run finds at
    score >= 0.7 found again, no extras, boxes within 0.12 px.  Keeping the
    ResNet50 backbone and the default 800 px rescale was deliberate: the
    MobileNet variant found 6 of 59 and running at the native 448 px found 19,
    because the pedestrians that matter are 10-40 px tall.
    """
    ts = [_to_tensor(b, device) for b in bgrs]
    use_amp = str(device).startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        outs = detector(device)(ts)
    return [_persons_of(o, score) for o in outs]


def detect_persons(bgr, device, score=0.5):
    """Return person boxes [(x1,y1,x2,y2,score)] (COCO label 1) in the RGB image."""
    return detect_persons_batch([bgr], device, score)[0]


def box_to_ego(box, depth_m, K, T_ego_cam):
    """Place a person box in the ego frame using depth at their feet (box bottom-centre)."""
    x1, y1, x2, y2, _ = box
    u = (x1 + x2) / 2.0
    vf = min(depth_m.shape[0] - 1, int(y2) - 2)              # feet row (ground contact)
    # robust range: nearest quartile of depths in a small patch around the feet
    xs = slice(max(0, int(x1)), min(depth_m.shape[1], int(x2) + 1))
    patch = depth_m[max(0, vf - 4):vf + 1, xs]
    patch = patch[(patch > 0.5) & (patch < 60)]
    if patch.size == 0:
        return None
    d = float(np.percentile(patch, 25))                      # person is nearer than background
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    optical = np.array([u * d, vf * d, d]) @ Kinv.T
    R = np.asarray(T_ego_cam)[:3, :3] @ R_CAM_OPT
    ego = R @ optical + np.asarray(T_ego_cam)[:3, 3]
    return float(ego[0]), float(ego[1]), d                   # (x fwd, y lateral, range-ish)


def pedestrians_in_frame(data, n, calib, device, score=0.7):   # tuned: 0.7 raises precision 0.62->0.79
    """All camera-detected pedestrians of a frame as (x_ego, y_ego, conf, cam)."""
    out = []
    for cam in ("front", "rear"):
        bgr = cv2.imread(f"{data}/{cam}_rgb/{n}")
        dep = depth_png_to_metres(cv2.imread(f"{data}/{cam}_depth/{n}"))
        K = calib["cameras"][cam]["K"]; T = calib["cameras"][cam]["T_ego_cam"]
        for box in detect_persons(bgr, device, score):
            placed = box_to_ego(box, dep, K, T)
            if placed:
                out.append((placed[0], placed[1], box[4], cam, box))
    return out


def pedestrians_from_arrays(rgb, depth, calib, device, score=0.7):
    """Same as pedestrians_in_frame, but for images already in memory.

    live_demo.py reads its frames straight off the CARLA sensor queues, so there
    is no file to point at.  `rgb` and `depth` are {"front": bgr, "rear": bgr};
    the depth image carries CARLA's own packed encoding, exactly as the PNG on
    disk does, so depth_png_to_metres applies unchanged.
    """
    out = []
    cams = [c for c in ("front", "rear")
            if rgb.get(c) is not None and depth.get(c) is not None]
    if not cams:
        return out
    boxes_per_cam = detect_persons_batch([rgb[c] for c in cams], device, score)
    for cam, boxes in zip(cams, boxes_per_cam):
        dep = depth_png_to_metres(depth[cam])
        K = calib["cameras"][cam]["K"]
        T = calib["cameras"][cam]["T_ego_cam"]
        for box in boxes:
            placed = box_to_ego(box, dep, K, T)
            if placed:
                out.append((placed[0], placed[1], box[4], cam, box))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_hazard_town10_long")
    ap.add_argument("--frame", default="000170")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--out", default="../results/PED_detect_proof.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = json.load(open(f"{args.data}/calibration.json"))
    n = args.frame + ".png"

    peds = pedestrians_in_frame(args.data, n, calib, device, args.score)
    # BEV-model pedestrian pixels for comparison
    gt = cv2.imread(f"{args.data}/bev_gt/{n}", 0)
    gt_ped = int((gt == 2).sum()) if gt is not None else 0
    print(f"frame {args.frame}: camera-detected pedestrians = {len(peds)} | GT ped px in BEV = {gt_ped}")
    for (x, y, c, cam, box) in peds:
        print(f"  {cam}: ego ({x:.1f}, {y:.1f})  range {math.hypot(x, y):.1f} m  conf {c:.2f}")

    # visual proof: front RGB with boxes + BEV dot placement
    from render_prof_style import render_bev_prof, fov_visible_mask
    front = cv2.imread(f"{args.data}/front_rgb/{n}").copy()
    for (x, y, c, cam, box) in peds:
        if cam == "front":
            x1, y1, x2, y2, _ = box
            cv2.rectangle(front, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)
            cv2.putText(front, f"person {c:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    front = cv2.resize(front, (620, 349))

    blind = ~fov_visible_mask(240, 240)
    lab = gt.copy(); lab[lab == 2] = 0                        # draw road/lane, place peds ourselves
    bev = render_bev_prof(lab, out_size=540, dim_blind=blind, smooth_road=True)
    BEVN, RES, X0, Y0 = 240, 0.25, -30.0, -30.0
    s = 540 / BEVN
    for (x, y, c, cam, box) in peds:
        col = (y - Y0) / RES; row = (BEVN - 1) - (x - X0) / RES
        px, py = int(col * s), int(row * s)
        cv2.circle(bev, (px, py), 8, (255, 0, 200), -1, cv2.LINE_AA)
        cv2.circle(bev, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, f"PED {math.hypot(x,y):.0f}m", (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 200), 1)
    ecx, ecy = int(((0 - Y0) / RES) * s), int(((BEVN - 1) - (0 - X0) / RES) * s)
    cv2.arrowedLine(bev, (ecx, ecy + 13), (ecx, ecy - 15), (255, 255, 255), 2, tipLength=0.45)

    fh = np.full((540, 620, 3), 20, np.uint8); fh[:349] = front
    cv2.putText(fh, f"Front RGB: {len(peds)} person(s) detected", (10, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)
    cv2.putText(fh, f"BEV segmenter found {gt_ped} ped px", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(fh, "-> localised via depth (magenta)", (10, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)
    canvas = np.hstack([fh, bev])
    cv2.imwrite(args.out, canvas)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
