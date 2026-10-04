"""Back-project front+rear depth maps into an ego-frame 3D point cloud.

This is the reconstruction core of the hazard-warning system. The geometry is
bit-for-bit the same convention as dualcam_bev_lss.py::_build_geometry, so the
point cloud lives in the SAME ego frame the BEV model reasons in:

  optical pt = K^-1 @ [u*d, v*d, d]         (standard pinhole, d = planar depth)
  ego pt     = (R_ego_cam @ R_CAM_OPT) @ optical + t_ego_cam

R_CAM_OPT maps the optical basis (x-right, y-down, z-forward) to CARLA's camera
basis (x-forward, y-right, z-up); R_ego_cam/t_ego_cam come from calibration.json.
Ego frame is CARLA: x-forward, y-right(-left?), z-up. Camera height 1.6 m, so
ground points land near z=0 -- used below as a geometry sanity check.

Usage:
  python depth_to_pointcloud.py --data ..\\data\\carla_dualcam_town10_depth --frame 000100
Outputs (into results/pointcloud/): a .ply (open in MeshLab/CloudCompare) plus a
top-down and a perspective PNG.
"""
import os, sys, cv2, json, argparse, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# optical basis -> CARLA camera basis (identical to dualcam_bev_lss.R_CAM_OPT)
R_CAM_OPT = np.array([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]], dtype=np.float64)


def depth_png_to_metres(bgr):
    """CARLA depth: normalized = (R + G*256 + B*256^2)/(256^3-1), metres = 1000*norm.
    NOTE: this decodes CARLA's 3-channel encoding ONLY. Real RealSense D455 depth is
    16-bit Z16 (millimetres) -- use d455_z16_to_metres() for that instead."""
    b = bgr[:, :, 0].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    r = bgr[:, :, 2].astype(np.float64)
    return 1000.0 * (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)


def d455_z16_to_metres(z16, depth_scale=0.001):
    """Intel RealSense D455 depth -> metres. The camera returns a 16-bit Z16 image
    (uint16), each unit = depth_scale metres. depth_scale is device-specific; query
    it at runtime with `pipeline_profile.get_device().first_depth_sensor()
    .get_depth_scale()` (D455 default ~0.001 = millimetres). 0 = no reading (invalid).
    Load the raw PNG with cv2.imread(path, cv2.IMREAD_UNCHANGED) to keep 16-bit."""
    return z16.astype(np.float64) * depth_scale


def backproject(depth_m, rgb, K, T_ego_cam, dmin, dmax, stride):
    """(H,W) depth + (H,W,3) RGB -> (N,3) ego xyz, (N,3) rgb, keeping dmin<=d<=dmax."""
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us = us.ravel(); vs = vs.ravel()
    d = depth_m[vs, us]
    keep = (d >= dmin) & (d <= dmax)
    us, vs, d = us[keep], vs[keep], d[keep]

    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    pix = np.stack([us * d, vs * d, d], axis=1)            # (N,3)
    optical = pix @ Kinv.T                                  # (N,3) optical frame
    R = np.asarray(T_ego_cam)[:3, :3] @ R_CAM_OPT           # ego <- optical
    t = np.asarray(T_ego_cam)[:3, 3]
    ego = optical @ R.T + t                                 # (N,3) ego xyz
    col = rgb[vs, us][:, ::-1]                              # BGR->RGB
    return ego, col


def write_ply(path, xyz, rgb):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_town10_depth")
    ap.add_argument("--frame", default="000100")
    ap.add_argument("--dmin", type=float, default=0.5)
    ap.add_argument("--dmax", type=float, default=60.0)
    ap.add_argument("--stride", type=int, default=2, help="pixel subsample for point density")
    ap.add_argument("--out", default="../results/pointcloud")
    args = ap.parse_args()

    calib = json.load(open(os.path.join(args.data, "calibration.json")))
    K = calib["cameras"]["front"]["K"]
    os.makedirs(args.out, exist_ok=True)

    clouds, colors = [], []
    for cam in ("front", "rear"):
        dp = os.path.join(args.data, f"{cam}_depth", f"{args.frame}.png")
        rp = os.path.join(args.data, f"{cam}_rgb", f"{args.frame}.png")
        if not (os.path.exists(dp) and os.path.exists(rp)):
            print(f"skip {cam}: missing {dp} or {rp}"); continue
        depth = depth_png_to_metres(cv2.imread(dp))
        rgb = cv2.imread(rp)
        T = calib["cameras"][cam]["T_ego_cam"]
        ego, col = backproject(depth, rgb, K, T, args.dmin, args.dmax, args.stride)
        clouds.append(ego); colors.append(col)
        print(f"{cam}: {len(ego):6d} pts | depth {depth.min():.1f}-{depth.max():.1f}m "
              f"(kept {args.dmin}-{args.dmax}) | x[{ego[:,0].min():.1f},{ego[:,0].max():.1f}] "
              f"z[{ego[:,2].min():.1f},{ego[:,2].max():.1f}]")

    xyz = np.vstack(clouds); rgb = np.vstack(colors)
    print(f"total {len(xyz)} pts")

    # --- geometry sanity: ground should sit near z=0 (cameras at z=1.6) ---
    ground = xyz[(xyz[:, 2] > -0.3) & (xyz[:, 2] < 0.3)]
    print(f"SANITY ground band |z|<0.3m: {len(ground)} pts ({100*len(ground)/len(xyz):.0f}%) "
          f"-- road should dominate this band")

    ply = os.path.join(args.out, f"cloud_{os.path.basename(args.data)}_{args.frame}.ply")
    write_ply(ply, xyz, rgb)
    print("PLY:", ply)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.clip(rgb / 255.0, 0, 1)
    # top-down (x forward vs y lateral) -- must look like a road scene from above
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xyz[:, 1], xyz[:, 0], s=0.4, c=c)
    ax.set_xlabel("y (lateral, m)"); ax.set_ylabel("x (forward, m)")
    ax.set_title(f"Top-down  {os.path.basename(args.data)}  {args.frame}")
    ax.set_aspect("equal"); ax.invert_xaxis(); ax.grid(alpha=0.2)
    td = os.path.join(args.out, f"topdown_{args.frame}.png")
    fig.savefig(td, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("top-down:", td)

    # perspective 3D -- the demo shot
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111, projection="3d")
    sub = np.random.RandomState(0).permutation(len(xyz))[:60000]
    ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2], s=0.5, c=c[sub])
    ax.set_xlabel("x fwd"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
    ax.set_zlim(-2, 8); ax.view_init(elev=18, azim=-70)
    ax.set_title(f"3D reconstruction  {args.frame}")
    pv = os.path.join(args.out, f"perspective_{args.frame}.png")
    fig.savefig(pv, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("perspective:", pv)


if __name__ == "__main__":
    main()
