"""Render an orbiting fly-around of a point cloud (.ply) to an H.264 clip.

Intended as the slide-opening "3D reconstruction" hero shot: the camera circles
the accumulated street cloud so the depth/structure reads at a glance. Works on
any cloud written by depth_to_pointcloud.py / accumulate_pointcloud.py (and later
the crisp GT-depth re-collect), so the same shot upgrades for free when data does.
"""
import os, sys, argparse, subprocess, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

FFMPEG = "E:/miniforge3/envs/lss/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def read_ply(path):
    with open(path, "r") as f:
        line = ""
        n = 0
        while not line.startswith("end_header"):
            line = f.readline()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        data = np.loadtxt(f, max_rows=n)
    return data[:, :3], data[:, 3:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="../results/pointcloud/accum_carla_dualcam_t01_temporal_model_0.ply")
    ap.add_argument("--out", default="../results/RECON_spin_town01.mp4")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--elev", type=float, default=42.0)
    ap.add_argument("--points", type=int, default=60000, help="render subsample")
    ap.add_argument("--zmax", type=float, default=12.0, help="drop points above this height (sky/far)")
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    keep = xyz[:, 2] < args.zmax
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > args.points:
        idx = np.random.RandomState(0).permutation(len(xyz))[:args.points]
        xyz, rgb = xyz[idx], rgb[idx]
    c = np.clip(rgb / 255.0, 0, 1)
    # centre + symmetric limits so the cloud stays framed while spinning
    mid = xyz.mean(0)
    span = np.percentile(np.abs(xyz - mid), 96, axis=0)
    r = float(max(span[0], span[1]))
    print(f"{len(xyz)} pts rendered | centre {mid.round(1)} | radius {r:.1f} m")

    tmp = args.out.replace(".mp4", "_raw.mp4")
    writer = None
    for k in range(args.frames):
        azim = -90 + 360.0 * k / args.frames
        fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black"); ax.set_facecolor("black")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c=c, depthshade=False)
        ax.set_xlim(mid[0] - r, mid[0] + r); ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(-2, args.zmax)
        ax.view_init(elev=args.elev, azim=azim); ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        cv2.putText(img, "Dual-cam 3D reconstruction (accumulated over drive)", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(img)
        if k % 12 == 0:
            print(f"{k}/{args.frames}", flush=True)
    writer.release()

    subprocess.run([FFMPEG, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", args.out], check=True, capture_output=True)
    os.remove(tmp)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
