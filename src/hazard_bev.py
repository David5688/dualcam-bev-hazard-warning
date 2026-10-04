"""Hazard warning on a clean prof-style BEV (the FINAL-video look + warnings).

Merges two things this project already has:
  * render_prof_style.render_bev_prof  -> clean top-down road (grey fill + green
    contour) and lane markings, blind zone dimmed;
  * the hazard layer (detect -> world-track -> TTC -> risk) from hazard_warning.py.

The BEV panel draws the clean road/lane as the base, then overlays: distance rings,
the ego marker, each obstacle as a hazard-coloured disc (green safe / yellow caution
/ red danger) with a range+TTC label, and a front/rear warning banner. Vehicles/peds
are removed from the road label first so they are only drawn once, hazard-coloured.

  python hazard_bev.py --source pred --start 430 --count 56    # -> results\HAZARD_BEV_town01.mp4
"""
import os, sys, cv2, json, math, argparse, subprocess, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
from render_prof_style import render_bev_prof, fov_visible_mask

BEV, RES, X0, Y0 = HZ.BEV, HZ.RES, HZ.X0, HZ.Y0
FFMPEG = HZ.FFMPEG


def ego_to_canvas(x, y, out_size):
    col = (y - Y0) / RES
    row = (BEV - 1) - (x - X0) / RES
    s = out_size / BEV
    return int(round(col * s)), int(round(row * s))


def bev_panel(label, obstacles, out_size=540):
    """Clean prof-style road/lane + hazard overlays. obstacles: (x,y,cls,level,col)."""
    blind = ~fov_visible_mask(BEV, BEV)
    road_label = label.copy()
    road_label[np.isin(road_label, (1, 2))] = 0                # obstacles drawn separately
    img = render_bev_prof(road_label, out_size=out_size, dim_blind=blind, smooth_road=True)

    s = out_size / BEV
    ecx, ecy = ego_to_canvas(0.0, 0.0, out_size)
    for r in (10, 20, 30):
        cv2.circle(img, (ecx, ecy), int(r / RES * s), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r}m", (ecx + 4, ecy - int(r / RES * s) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (ecx, ecy + 13), (ecx, ecy - 15), (255, 255, 255), 2, tipLength=0.45)

    order = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    front = rear = "SAFE"
    for o in sorted(obstacles, key=lambda o: -math.hypot(o[0], o[1])):
        x, y, cls, level, col, ttc = o[:6]
        uncertain = o[6] if len(o) > 6 else False
        path = o[7] if len(o) > 7 else None
        px, py = ego_to_canvas(x, y, out_size)
        if path and level != "SAFE" and not uncertain:
            # Where the predictor thinks it is going.  Drawn thinner than anything
            # else on the panel: it is a claim about the future and should not
            # read with the same weight as the measured position it grows out of.
            pts = [ego_to_canvas(qx, qy, out_size) for (_qt, qx, qy) in path[::3]]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
            if pts:
                cv2.circle(img, pts[-1], 3, col, 1, cv2.LINE_AA)
        if uncertain:
            # low-confidence detection: shown but NOT warned on (hollow grey + "?")
            draw_marker(img, px, py, cls, (150, 150, 150), fill=False)
            cv2.putText(img, "?", (px - 4, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        else:
            # shape encodes class (square=vehicle, triangle=pedestrian), colour encodes risk
            draw_marker(img, px, py, cls, col, fill=True)
            lbl = f"{math.hypot(x, y):.0f}m" + (f" {ttc:.1f}s" if ttc < 9 else "")
            if len(o) > 9 and o[9]:
                # On the watch list: caught driving erratically, and still marked
                # while it happens to be driving straight.  A ring rather than a
                # different colour -- colour already means risk level, and this is
                # a statement about the driver, not about the current geometry.
                cv2.circle(img, (px, py), 15, col, 2, cv2.LINE_AA)
                cv2.circle(img, (px, py), 15, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "WEAVING", (px - 26, py - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                lbl = "! " + lbl
                tx = px + 19          # clear the ring, which is wider than a marker
            else:
                tx = px + 13
            cv2.putText(img, lbl, (tx, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            d = "front" if x >= 0 else "rear"
            if d == "front" and order[level] > order[front]:
                front = level
            if d == "rear" and order[level] > order[rear]:
                rear = level

    cmap = {"SAFE": (0, 200, 0), "CAUTION": (0, 200, 230), "DANGER": (0, 0, 230)}
    for i, (d, lv) in enumerate((("FRONT", front), ("REAR", rear))):
        y0 = 8 + i * 34
        cv2.rectangle(img, (8, y0), (232, y0 + 28), cmap[lv], -1)
        cv2.putText(img, f"{d}: {lv}", (16, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
    legend(img, out_size)
    return img


def draw_marker(img, px, py, cls, col, fill=True):
    """Class shape: ■ square = vehicle, ▲ triangle = pedestrian. Colour = risk level."""
    edge = (245, 245, 245)
    th = -1 if fill else 1
    if cls == 2:                                              # pedestrian -> triangle
        r = 9
        pts = np.array([[px, py - r], [px - r, py + r - 2], [px + r, py + r - 2]], np.int32)
        if fill:
            cv2.fillConvexPoly(img, pts, col, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, edge, 1, cv2.LINE_AA)
    else:                                                    # vehicle -> square
        r = 8
        cv2.rectangle(img, (px - r, py - r), (px + r, py + r), col, th, cv2.LINE_AA)
        cv2.rectangle(img, (px - r, py - r), (px + r, py + r), edge, 1, cv2.LINE_AA)


def legend(img, out_size):
    """Shape legend so the class encoding is self-explanatory (English: cv2 can't draw CJK)."""
    x0, y0 = 8, out_size - 34
    cv2.rectangle(img, (x0, y0), (x0 + 200, y0 + 26), (35, 35, 35), -1)
    cv2.rectangle(img, (x0 + 8, y0 + 6), (x0 + 22, y0 + 20), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "Vehicle", (x0 + 28, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    tx = x0 + 110
    pts = np.array([[tx + 7, y0 + 5], [tx, y0 + 20], [tx + 14, y0 + 20]], np.int32)
    cv2.polylines(img, [pts], True, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "Ped", (x0 + 130, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def build_predictor(data, model, device):
    import torch
    from dualcam_bev_lss import DualCamBEVLSS
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
            lab = torch.argmax(m(x)[0], 0).cpu().numpy().astype(np.uint8)
        lab[blind & np.isin(lab, (1, 2))] = 0                  # variant D
        return lab
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
    ap.add_argument("--source", choices=["gt", "pred"], default="pred")
    ap.add_argument("--start", type=int, default=430)
    ap.add_argument("--count", type=int, default=56)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--frame", default=None, help="render just this frame -> PNG")
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out", default="../results/HAZARD_BEV_town01.mp4")
    args = ap.parse_args()

    device = "cpu"
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    poses = json.load(open(f"{args.data}/poses.json"))
    predict = build_predictor(args.data, args.model, device) if args.source == "pred" else None
    dt = args.step / args.fps

    def get_label(n):
        return cv2.imread(f"{args.data}/bev_gt/{n}", 0) if args.source == "gt" else predict(n)

    def frame_obstacles(label, ex, ey, yaw, tracker, t):
        dw = []
        for (x, y, cls, area) in HZ.detect_obstacles(label):
            wx, wy = HZ.world_of(x, y, ex, ey, yaw)
            dw.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": cls, "range": math.hypot(x, y)})
        tracker.update(dw, t, ego=(ex, ey, yaw))
        out = []
        for tid, tr in tracker.active():
            d = tr["cur"]; ttc, rate = HZ.ttc_of(tr["hist"])
            level, col = HZ.risk_of(d["range"], ttc, 2.0, 4.0, 5.0, 10.0)
            out.append((d["x"], d["y"], d["cls"], level, col, ttc))
        return out

    def compose(n, panel):
        front = cv2.resize(cv2.imread(f"{args.data}/front_rgb/{n}"), (498, 280))
        rear = cv2.resize(cv2.imread(f"{args.data}/rear_rgb/{n}"), (498, 280))
        cv2.putText(front, "FRONT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rear, "REAR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        left = np.vstack([front, rear])
        panel = cv2.resize(panel, (560, 560))
        canvas = np.hstack([left, panel])
        cap = np.full((30, canvas.shape[1], 3), 18, np.uint8)
        cv2.putText(cap, f"Clean BEV + hazard warning  {os.path.basename(args.data)}  frame {n}  source={args.source}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return np.vstack([canvas, cap])

    if args.frame:
        n = args.frame + ".png"; ex, ey, yaw = poses[n]
        obs = frame_obstacles(get_label(n), ex, ey, yaw, HZ.Tracker(), 0)
        out = compose(n, bev_panel(get_label(n), obs))
        p = args.out.replace(".mp4", ".png"); cv2.imwrite(p, out); print("saved:", p); return

    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/front_rgb/{f}")]
    tracker = HZ.Tracker(); writer = None
    tmp = args.out.replace(".mp4", "_raw.mp4")
    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]; label = get_label(n)
        obs = frame_obstacles(label, ex, ey, yaw, tracker, k * dt)
        canvas = compose(n, bev_panel(label, obs))
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
    print("saved:", args.out)


if __name__ == "__main__":
    main()
