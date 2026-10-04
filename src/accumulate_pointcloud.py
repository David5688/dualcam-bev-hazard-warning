"""Accumulate per-frame front+rear point clouds into a dense street reconstruction,
using ego pose (poses.json) to register every frame into a common world frame.

This is the "三維重建" centrepiece: a single frame sees two narrow FOV wedges, but
stacking many frames along the drive fills in a dense 3D map of the whole street.

Depth source is pluggable so the SAME accumulation works as we upgrade quality:
  --source model : learned monocular depth from the depth-supervised LSS model
                   (works on any RGB set that has poses.json, e.g. t01_temporal)
  --source gt    : CARLA depth-camera GT (for a re-collected depth+poses sequence)

Geometry matches dualcam_bev_lss.py / depth_to_pointcloud.py exactly. Pose is 2-D
(x, y, yaw) -- enough for a ground vehicle; z (height) is kept in ego/world directly.
"""
import os, sys, cv2, json, math, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dualcam_bev_lss import DualCamBEVLSS
from depth_to_pointcloud import R_CAM_OPT, depth_png_to_metres, write_ply

IMG_H, IMG_W = 224, 400


def world_from_ego(pts, ex, ey, yaw_deg):
    """Rotate+translate ego-frame xyz into world by the frame's pose (2-D, z kept)."""
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([ex + c * x - s * y, ey + s * x + c * y, z], axis=1)


def backproject(depth_m, rgb, K, T_ego_cam, dmin, dmax):
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us.ravel(); vs = vs.ravel(); d = depth_m.ravel()
    keep = (d >= dmin) & (d <= dmax)
    us, vs, d = us[keep], vs[keep], d[keep]
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    optical = np.stack([us * d, vs * d, d], axis=1) @ Kinv.T
    R = np.asarray(T_ego_cam)[:3, :3] @ R_CAM_OPT
    t = np.asarray(T_ego_cam)[:3, 3]
    ego = optical @ R.T + t
    col = rgb[vs, us][:, ::-1]
    return ego, col


def voxel_downsample(xyz, rgb, voxel):
    """One point per voxel, mean colour -- keeps the accumulated cloud manageable."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = inv.max() + 1
    out_xyz = np.zeros((n, 3)); out_rgb = np.zeros((n, 3)); cnt = np.zeros(n)
    np.add.at(out_xyz, inv, xyz); np.add.at(out_rgb, inv, rgb); np.add.at(cnt, inv, 1)
    return out_xyz / cnt[:, None], out_rgb / cnt[:, None]


def build_model(data, device):
    ck = "../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt"
    meta = torch.load(ck, map_location="cpu", weights_only=False)
    ds = meta.get("lss_downsample", 8); pool = meta.get("lss_pool", 0.5)
    m = DualCamBEVLSS.from_calib_file(f"{data}/calibration.json", num_classes=6, bev_h=240,
                                      bev_w=240, img_h=IMG_H, img_w=IMG_W, downsample=ds,
                                      pool_res=pool, pretrained=False)
    st = meta["model_state_dict"] if "model_state_dict" in meta else meta
    m.load_state_dict(st, strict=True)
    centers = (m.d0 + (torch.arange(m.D) + 0.5) * m.dd).float()
    return m.to(device).eval(), centers.to(device)


def model_depth(m, centers, front_rgb, rear_rgb, device, conf=0.0):
    """Expected depth (metres) at input resolution for front & rear, from the LSS model.

    conf>0 drops pixels where the model's peak depth-bin probability is below conf --
    the coarse 28x50 depth is noisy, and low-confidence pixels are what smear into
    radial streaks when accumulated. Masked pixels return depth 0 (filtered downstream).
    """
    def rd(p):
        im = cv2.resize(cv2.imread(p), (IMG_W, IMG_H))[:, :, ::-1].copy().astype(np.float32) / 255.0
        return torch.from_numpy(im).permute(2, 0, 1)
    x = torch.stack([rd(front_rgb), rd(rear_rgb)], 0).unsqueeze(0).to(device)
    with torch.no_grad():
        _, dl = m(x, return_depth=True)                 # (2, D, fH, fW)
        prob = torch.softmax(dl, dim=1)
        exp_d = (prob * centers[None, :, None, None]).sum(1)   # (2, fH, fW)
        if conf > 0:
            exp_d = exp_d.masked_fill(prob.max(1).values < conf, 0.0)
        exp_d = torch.nn.functional.interpolate(exp_d[:, None], size=(IMG_H, IMG_W),
                                                mode="nearest")[:, 0]
    return exp_d[0].cpu().numpy(), exp_d[1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
    ap.add_argument("--source", choices=["model", "gt"], default="model")
    ap.add_argument("--start", type=int, default=140)
    ap.add_argument("--count", type=int, default=80)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--dmin", type=float, default=2.0)
    ap.add_argument("--dmax", type=float, default=40.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--out", default="../results/pointcloud")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = json.load(open(f"{args.data}/calibration.json"))
    poses = json.load(open(f"{args.data}/poses.json"))
    os.makedirs(args.out, exist_ok=True)

    m = centers = None
    if args.source == "model":
        m, centers = build_model(args.data, device)
    # K scaled from original (640x360) to model input (400x224); reuse the model's scaler
    scaler = m if m is not None else DualCamBEVLSS.from_calib_file(
        f"{args.data}/calibration.json", num_classes=6, img_h=IMG_H, img_w=IMG_W, pretrained=False)
    ow, oh = calib["image"]["width"], calib["image"]["height"]
    K = {c: scaler._scaled_K(calib["cameras"][c]["K"], ow, oh).numpy() for c in ("front", "rear")}
    T = {c: calib["cameras"][c]["T_ego_cam"] for c in ("front", "rear")}

    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/front_rgb/{f}")]

    all_xyz, all_rgb = [], []
    p0 = poses[frames[0]]; dist = 0.0; prev = p0
    for n in frames:
        ex, ey, yaw = poses[n]
        dist += math.hypot(ex - prev[0], ey - prev[1]); prev = poses[n]
        if args.source == "model":
            df, dr = model_depth(m, centers, f"{args.data}/front_rgb/{n}", f"{args.data}/rear_rgb/{n}", device, args.conf)
        else:
            df = cv2.resize(depth_png_to_metres(cv2.imread(f"{args.data}/front_depth/{n}")), (IMG_W, IMG_H))
            dr = cv2.resize(depth_png_to_metres(cv2.imread(f"{args.data}/rear_depth/{n}")), (IMG_W, IMG_H))
        for cam, dep in (("front", df), ("rear", dr)):
            rgb = cv2.resize(cv2.imread(f"{args.data}/{cam}_rgb/{n}"), (IMG_W, IMG_H))
            ego, col = backproject(dep, rgb, K[cam], T[cam], args.dmin, args.dmax)
            all_xyz.append(world_from_ego(ego, ex, ey, yaw)); all_rgb.append(col)

    xyz = np.vstack(all_xyz); rgb = np.vstack(all_rgb)
    xyz[:, :2] -= np.array([p0[0], p0[1]])                  # recentre on start pose
    n_raw = len(xyz)
    xyz, rgb = voxel_downsample(xyz, rgb, args.voxel)
    print(f"{len(frames)} frames | drive {dist:.1f} m | {n_raw} raw -> {len(xyz)} voxels "
          f"({args.voxel} m) | source={args.source}")

    tag = f"accum_{os.path.basename(args.data)}_{args.source}_{args.start}"
    write_ply(os.path.join(args.out, tag + ".ply"), xyz, rgb)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = np.clip(rgb / 255.0, 0, 1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(xyz[:, 1], xyz[:, 0], s=0.3, c=c)
    ax.set_xlabel("y (m)"); ax.set_ylabel("x (m)")
    ax.set_title(f"Accumulated top-down  {os.path.basename(args.data)}  ({args.source}, {dist:.0f} m, {len(frames)} frames)")
    ax.set_aspect("equal"); ax.invert_xaxis(); ax.grid(alpha=0.2)
    td = os.path.join(args.out, tag + "_topdown.png")
    fig.savefig(td, dpi=110, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(11, 6)); ax = fig.add_subplot(111, projection="3d")
    sub = np.random.RandomState(0).permutation(len(xyz))[:120000]
    ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2], s=0.4, c=c[sub])
    ax.set_xlabel("x fwd"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
    ax.set_zlim(-2, 10); ax.view_init(elev=22, azim=-75)
    ax.set_title(f"Accumulated 3D reconstruction  ({args.source})")
    pv = os.path.join(args.out, tag + "_persp.png")
    fig.savefig(pv, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("saved:", td, "|", pv, "|", tag + ".ply")


if __name__ == "__main__":
    main()
