"""Uncertainty-aware hazard warning (the innovation): only commit to a warning when
the model is confident about the detection; show low-confidence detections but do
NOT raise the banner on them.

Per detection, confidence = mean over the obstacle's cluster pixels of the model's
peak softmax probability. A detection with confidence < --gate is "uncertain": it
is drawn (hollow grey "?") but never escalates FRONT/REAR beyond SAFE. This is the
safety-system version of the project's variant-D idea -- a false DANGER erodes
trust, so the system stays silent unless it is sure.

  python uncertainty_warning.py --probe               # print confidence distribution
  python uncertainty_warning.py --gate 0.6 --start 430 --count 56   # -> results/UNCERT_BEV_town01.mp4
"""
import os, sys, cv2, json, math, argparse, subprocess, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
from hazard_bev import bev_panel, ego_to_canvas
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask

FFMPEG = HZ.FFMPEG


def build_predictor(data, model, device):
    """Returns predict(n) -> (argmax label, per-cell peak-softmax confidence map)."""
    meta = torch.load(model, map_location="cpu", weights_only=False)
    m = DualCamBEVLSS.from_calib_file(f"{data}/calibration.json", num_classes=6, bev_h=240, bev_w=240,
                                      img_h=224, img_w=400, downsample=meta.get("lss_downsample", 8),
                                      pool_res=meta.get("lss_pool", 0.5), pretrained=False)
    m.load_state_dict(meta.get("model_state_dict", meta), strict=True)
    m = m.to(device).eval()
    blind = ~fov_visible_mask(240, 240)

    def predict(n):
        def rd(p):
            im = cv2.resize(cv2.imread(p), (400, 224))[:, :, ::-1].copy().astype(np.float32) / 255.0
            return torch.from_numpy(im).permute(2, 0, 1)
        x = torch.stack([rd(f"{data}/front_rgb/{n}"), rd(f"{data}/rear_rgb/{n}")], 0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = m(x)[0]
            prob = torch.softmax(logits, 0)
            conf, lab = prob.max(0)
        lab = lab.cpu().numpy().astype(np.uint8); conf = conf.cpu().numpy()
        lab[blind & np.isin(lab, (1, 2))] = 0                  # variant D
        return lab, conf
    return predict


def detect_conf(label, conf_map, min_area=8):
    """Cluster vehicle/ped pixels -> (x_ego, y_ego, cls, area, confidence)."""
    out = []
    for cls in (1, 2):
        mask = (label == cls).astype(np.uint8)
        n, lab_im, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            col, row = cent[i]
            x, y = HZ.px_to_ego(col, row)
            conf = float(conf_map[lab_im == i].mean())
            out.append((x, y, cls, int(stats[i, cv2.CC_STAT_AREA]), conf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
    ap.add_argument("--gate", type=float, default=0.6, help="min detection confidence to allow a warning")
    ap.add_argument("--start", type=int, default=430)
    ap.add_argument("--count", type=int, default=56)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--probe", action="store_true", help="print confidence distribution and exit")
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out", default="../results/UNCERT_BEV_town01.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    poses = json.load(open(f"{args.data}/poses.json"))
    predict = build_predictor(args.data, args.model, device)
    dt = args.step / args.fps
    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/front_rgb/{f}")]

    if args.probe:
        confs = []
        for n in frames:
            lab, conf = predict(n)
            for (x, y, cls, area, c) in detect_conf(lab, conf):
                confs.append((c, "veh" if cls == 1 else "ped", round(math.hypot(x, y), 1), area))
        confs.sort()
        cs = np.array([c[0] for c in confs])
        print(f"detections: {len(confs)} | confidence min {cs.min():.2f} p25 {np.percentile(cs,25):.2f} "
              f"median {np.median(cs):.2f} p75 {np.percentile(cs,75):.2f} max {cs.max():.2f}")
        print("lowest-confidence detections (candidates the gate would silence):")
        for c in confs[:10]:
            print(f"  conf {c[0]:.2f}  {c[1]}  range {c[2]}m  area {c[3]}")
        return

    tracker = HZ.Tracker(); writer = None
    tmp = args.out.replace(".mp4", "_raw.mp4")
    n_gated = n_warn = n_uncertain = 0
    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]
        label, conf_map = predict(n)
        dets = detect_conf(label, conf_map)
        dw = []
        for (x, y, cls, area, c) in dets:
            wx, wy = HZ.world_of(x, y, ex, ey, yaw)
            dw.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": cls, "range": math.hypot(x, y), "conf": c})
        tracker.update(dw, k * dt, ego=(ex, ey, yaw))
        obstacles = []
        for tid, tr in tracker.active():
            d = tr["cur"]; ttc, rate = HZ.ttc_of(tr["hist"])
            level, col = HZ.risk_of(d["range"], ttc, 2.0, 4.0, 5.0, 10.0)
            uncertain = d["conf"] < args.gate
            if level != "SAFE" and uncertain:
                n_gated += 1
            elif level != "SAFE":
                n_warn += 1
            obstacles.append((d["x"], d["y"], d["cls"], level, col, ttc, uncertain))

        panel = bev_panel(label, obstacles)
        front = cv2.resize(cv2.imread(f"{args.data}/front_rgb/{n}"), (498, 280))
        rear = cv2.resize(cv2.imread(f"{args.data}/rear_rgb/{n}"), (498, 280))
        cv2.putText(front, "FRONT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rear, "REAR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        left = np.vstack([front, rear]); panel = cv2.resize(panel, (560, 560))
        canvas = np.hstack([left, panel])
        cap = np.full((30, canvas.shape[1], 3), 18, np.uint8)
        cv2.putText(cap, f"Uncertainty-aware warning  gate={args.gate}  Town01  frame {n}  (\"?\"=silenced, low confidence)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        canvas = np.vstack([canvas, cap])
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(canvas)
        if k % 12 == 0:
            print(f"{k}/{len(frames)}", flush=True)
    writer.release()
    subprocess.run([FFMPEG, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
                   check=True, capture_output=True)
    os.remove(tmp)
    print(f"warnings kept (confident): {n_warn} | silenced by uncertainty gate ({args.gate}): {n_gated}")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
