"""Integrated 2x2 demo: reconstruction + hazard warning in one synchronized view.

  [ Front RGB ]      [ per-frame 3D point cloud ]
  [ Rear  RGB ]      [ BEV hazard radar         ]

Reuses the hazard layer (detection / world tracking / TTC / risk radar) from
hazard_warning.py and the depth back-projection from accumulate_pointcloud.py,
so this file only composes them -- no geometry or logic is re-implemented.

Detection uses the model's predicted BEV with variant D (blind-zone dynamic
predictions dropped); the 3D panel uses the model's learned depth. Everything is
the deployed system's own output, not ground truth.
"""
import os, sys, cv2, json, math, argparse, numpy as np, subprocess, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hazard_warning as HZ
from accumulate_pointcloud import build_model, model_depth, backproject
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask

IMG_H, IMG_W = 224, 400
FFMPEG = HZ.FFMPEG


def cloud_panel(xyz, rgb, size=360):
    """Render a per-frame 3D point cloud to an (size,size,3) BGR image (fixed view)."""
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    if len(xyz):
        sub = np.random.RandomState(0).permutation(len(xyz))[:12000]
        ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2], s=0.6,
                   c=np.clip(rgb[sub] / 255.0, 0, 1), depthshade=False)
    ax.set_xlim(-40, 40); ax.set_ylim(-25, 25); ax.set_zlim(-2, 10)
    ax.view_init(elev=20, azim=-88)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(img, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
    ap.add_argument("--start", type=int, default=430)
    ap.add_argument("--count", type=int, default=56)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--dmax", type=float, default=35.0)
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out", default="../results/INTEGRATED_demo_town01.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = json.load(open(f"{args.data}/calibration.json"))
    poses = json.load(open(f"{args.data}/poses.json"))
    dt = args.step / args.fps
    blind = ~fov_visible_mask(HZ.BEV, HZ.BEV)

    m, centers = build_model(args.data, device)
    ow, oh = calib["image"]["width"], calib["image"]["height"]
    K = {c: m._scaled_K(calib["cameras"][c]["K"], ow, oh).numpy() for c in ("front", "rear")}
    T = {c: calib["cameras"][c]["T_ego_cam"] for c in ("front", "rear")}

    def predict_bev(n):
        def rd(p):
            im = cv2.resize(cv2.imread(p), (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
            return torch.from_numpy(im).permute(2, 0, 1)
        x = torch.stack([rd(f"{args.data}/front_rgb/{n}"), rd(f"{args.data}/rear_rgb/{n}")], 0).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(m(x)[0], 0).cpu().numpy().astype(np.uint8)

    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/front_rgb/{f}")]

    tracker = HZ.Tracker()
    writer = None
    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]
        # --- hazard: predicted BEV + variant D ---
        label = predict_bev(n)
        label[blind & np.isin(label, (1, 2))] = 0
        dets_world = []
        for (x, y, cls, area) in HZ.detect_obstacles(label):
            wx, wy = HZ.world_of(x, y, ex, ey, yaw)
            dets_world.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": cls, "range": math.hypot(x, y)})
        tracker.update(dets_world, k * dt, ego=(ex, ey, yaw))
        frame_dets = []
        for tid, tr in tracker.active():
            d = tr["cur"]; ttc, rate = HZ.ttc_of(tr["hist"])
            level, col = HZ.risk_of(d["range"], ttc, 2.0, 4.0, 5.0, 10.0)
            frame_dets.append((d["x"], d["y"], d["cls"], d["range"], ttc, level, col))
        radar = HZ.render(frame_dets, None, None, size=360)

        # --- reconstruction: model depth -> point cloud ---
        df, dr = model_depth(m, centers, f"{args.data}/front_rgb/{n}", f"{args.data}/rear_rgb/{n}", device, args.conf)
        cl, co = [], []
        for cam, dep in (("front", df), ("rear", dr)):
            rgb = cv2.resize(cv2.imread(f"{args.data}/{cam}_rgb/{n}"), (IMG_W, IMG_H))
            ego, c = backproject(dep, rgb, K[cam], T[cam], 2.0, args.dmax)
            cl.append(ego); co.append(c)
        pc = cloud_panel(np.vstack(cl), np.vstack(co), size=360)

        # --- compose 2x2 ---
        front = cv2.resize(cv2.imread(f"{args.data}/front_rgb/{n}"), (500, 360))
        rear = cv2.resize(cv2.imread(f"{args.data}/rear_rgb/{n}"), (500, 360))
        cv2.putText(front, "FRONT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(rear, "REAR", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(pc, "3D reconstruction", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        left = np.vstack([front, rear])                       # 720 x 500
        right = np.vstack([pc, radar])                        # 720 x 360
        canvas = np.hstack([left, right])                     # 720 x 860
        cap = np.full((30, canvas.shape[1], 3), 18, np.uint8)
        cv2.putText(cap, f"Dual-cam 3D reconstruction + hazard warning  |  Town01  frame {n}  |  model pred + variant D",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        canvas = np.vstack([canvas, cap])
        if writer is None:
            h, w = canvas.shape[:2]
            tmp = args.out.replace(".mp4", "_raw.mp4")
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(canvas)
        if k % 10 == 0:
            print(f"{k}/{len(frames)}", flush=True)
    writer.release()

    tmp = args.out.replace(".mp4", "_raw.mp4")
    subprocess.run([FFMPEG, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", args.out], check=True, capture_output=True)
    os.remove(tmp)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
