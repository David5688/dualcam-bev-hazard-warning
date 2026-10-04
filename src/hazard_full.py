"""Full hazard warning: vehicles from the BEV model, pedestrians from the image
detector + depth (Path B). Hybrid by design -- large objects (vehicles) are found
in BEV, small objects (pedestrians, ~2-3 px in BEV) are found in the camera image
where they are large and clear, then localised into BEV via depth.

Both feed the same world tracker -> TTC -> risk (hazard_warning.risk_of) and the
same clean BEV panel (hazard_bev.bev_panel). Person boxes are drawn on the RGB.

  python hazard_full.py --start 150 --count 60   # -> results\HAZARD_FULL_town10.mp4
"""
import os, sys, cv2, json, math, argparse, subprocess, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
import pedestrian_detect as PD
from hazard_bev import bev_panel, ego_to_canvas
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask, render_bev_prof

FFMPEG = HZ.FFMPEG


def gt_panel(bev_gt, out_size=560):
    """Ground-truth BEV for side-by-side comparison: true road/lane/vehicle/ped,
    plus the same distance rings + ego arrow as the prediction panel. GT is never
    smoothed or de-speckled (prof-style convention)."""
    blind = ~fov_visible_mask(HZ.BEV, HZ.BEV)
    img = render_bev_prof(bev_gt, out_size=out_size, dim_blind=blind)
    s = out_size / HZ.BEV
    ecx, ecy = ego_to_canvas(0.0, 0.0, out_size)
    for r in (10, 20, 30):
        cv2.circle(img, (ecx, ecy), int(r / HZ.RES * s), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (ecx, ecy + 13), (ecx, ecy - 15), (255, 255, 255), 2, tipLength=0.45)
    return img


def titled(img, label):
    """Add a title strip above a panel (English: cv2 can't render CJK)."""
    bar = np.full((30, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def build_bev(data, model, device, img_h=224, img_w=400):
    # The input size is part of the geometry, not a preference: DualCamBEVLSS bakes
    # the frustum->BEV mapping into buffers from K scaled to (img_h, img_w).  Feed a
    # model images at a size it was not trained on and every feature lands in the
    # wrong BEV cell.  The v4 model is 800x448; everything before it was 400x224.
    meta = torch.load(model, map_location="cpu", weights_only=False)
    ckpt_h, ckpt_w = meta.get("img_h"), meta.get("img_w")
    if ckpt_h and ckpt_w and (ckpt_h, ckpt_w) != (img_h, img_w):
        print(f"WARNING: checkpoint was trained at {ckpt_w}x{ckpt_h} but rendering at "
              f"{img_w}x{img_h} -- the LSS projection will be wrong. Pass "
              f"--img_h {ckpt_h} --img_w {ckpt_w}.")
    m = DualCamBEVLSS.from_calib_file(f"{data}/calibration.json", num_classes=6, bev_h=240, bev_w=240,
                                      img_h=img_h, img_w=img_w, downsample=meta.get("lss_downsample", 8),
                                      pool_res=meta.get("lss_pool", 0.5), pretrained=False)
    m.load_state_dict(meta.get("model_state_dict", meta), strict=True)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_hazard_town10_long")
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--score", type=float, default=0.7)   # tuned for precision (was 0.5)
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out", default="../results/HAZARD_FULL_town10.mp4")
    ap.add_argument("--img_h", type=int, default=224,
                    help="model input height -- must match what the checkpoint was "
                         "trained at (v4 models: 448)")
    ap.add_argument("--img_w", type=int, default=400,
                    help="model input width (v4 models: 800)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = json.load(open(f"{args.data}/calibration.json"))
    poses = json.load(open(f"{args.data}/poses.json"))
    dt = args.step / args.fps
    m = build_bev(args.data, args.model, device, args.img_h, args.img_w)
    blind = ~fov_visible_mask(240, 240)

    def vehicles(n):
        def rd(p):
            im = cv2.resize(cv2.imread(p), (args.img_w, args.img_h))[:, :, ::-1].copy().astype(np.float32) / 255.0
            return torch.from_numpy(im).permute(2, 0, 1)
        x = torch.stack([rd(f"{args.data}/front_rgb/{n}"), rd(f"{args.data}/rear_rgb/{n}")], 0).unsqueeze(0).to(device)
        with torch.no_grad():
            lab = torch.argmax(m(x)[0], 0).cpu().numpy().astype(np.uint8)
        lab[blind & (lab == 1)] = 0                          # variant D on vehicles
        lab[lab == 2] = 0                                    # peds come from the image detector
        return lab

    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/front_rgb/{f}")]

    tracker = HZ.Tracker(); writer = None
    tmp = args.out.replace(".mp4", "_raw.mp4")
    n_ped_warn = 0
    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]
        veh_lab = vehicles(n)
        peds = PD.pedestrians_in_frame(args.data, n, calib, device, args.score)

        dw = []
        for (x, y, cls, area) in HZ.detect_obstacles(veh_lab):    # vehicles only (label has no class 2)
            wx, wy = HZ.world_of(x, y, ex, ey, yaw)
            dw.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": 1, "range": math.hypot(x, y)})
        for (x, y, conf, cam, box) in peds:                       # image-detected pedestrians
            wx, wy = HZ.world_of(x, y, ex, ey, yaw)
            dw.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": 2, "range": math.hypot(x, y)})
        tracker.update(dw, k * dt, ego=(ex, ey, yaw))

        obstacles = []
        for tid, tr in tracker.active():
            d = tr["cur"]; ttc, rate = HZ.ttc_of(tr["hist"])
            level, col = HZ.risk_of(d["range"], ttc, 2.0, 4.0, 5.0, 10.0)
            # temporal filter: a pedestrian must persist >= 2 frames before it can warn
            # (a real person is stable; a false detection flickers) -> raises ped precision
            if d["cls"] == 2 and len(tr["hist"]) < 2 and level != "SAFE":
                level, col = "SAFE", (0, 200, 0)
            obstacles.append((d["x"], d["y"], d["cls"], level, col, ttc))
            if d["cls"] == 2 and level != "SAFE":
                n_ped_warn += 1

        panel = bev_panel(veh_lab, obstacles)                     # road/lane base + all obstacles
        front = cv2.imread(f"{args.data}/front_rgb/{n}").copy()
        rear = cv2.imread(f"{args.data}/rear_rgb/{n}").copy()
        for (x, y, conf, cam, box) in peds:
            img = front if cam == "front" else rear
            x1, y1, x2, y2, _ = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)
            cv2.putText(img, "person", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        front = cv2.resize(front, (498, 280)); rear = cv2.resize(rear, (498, 280))
        cv2.putText(front, "FRONT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rear, "REAR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panel = cv2.resize(panel, (560, 560))
        gt = cv2.imread(f"{args.data}/bev_gt/{n}", 0)
        gtp = cv2.resize(gt_panel(gt), (560, 560))
        cols = [titled(np.vstack([front, rear]), "CAMERA (front / rear)")]
        # top-down bird's-eye RGB reference (if collected with --bev_rgb); NOT a model input
        bevrgb_path = f"{args.data}/bev_rgb/{n}"
        if os.path.exists(bevrgb_path):
            bevrgb = cv2.resize(cv2.imread(bevrgb_path), (560, 560))
            cols.append(titled(bevrgb, "BEV RGB (top-down reference)"))
        cols.append(titled(panel, "PREDICTION (system detection)"))
        cols.append(titled(gtp, "GROUND TRUTH"))
        canvas = np.hstack(cols)
        cap = np.full((30, canvas.shape[1], 3), 18, np.uint8)
        cv2.putText(cap, f"Full hazard: vehicles (BEV) + pedestrians (image+depth) vs GROUND TRUTH  frame {n}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        canvas = np.vstack([canvas, cap])
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(canvas)
        if k % 10 == 0:
            print(f"{k}/{len(frames)}", flush=True)
    writer.release()
    subprocess.run([FFMPEG, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
                   check=True, capture_output=True)
    os.remove(tmp)
    print(f"pedestrian warning events: {n_ped_warn}")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
