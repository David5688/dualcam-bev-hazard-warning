# -*- coding: utf-8 -*-
"""每城產一張「收集路線規劃圖」:道路網 + 所有 spawn 點 + 建議的分段起點(farthest-point
spread,涵蓋全城)。只讀地圖、不開相機,安全。收集後可再把 poses 實際軌跡疊上去。

用法(CARLA server 要開著):
  python route_plan.py --host 127.0.0.1 --port 2000
輸出:results/route_plans/<town>.png
"""
import carla, argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (城, 角色, 計畫幀數, 分段數)
PLAN = [
    ("Town01",   "TEST (held-out, unfiltered, SINGLE continuous run)", 2700, 1),
    ("Town07",   "VAL (held-out, unfiltered, SINGLE continuous run)",  2700, 1),
    ("Town10HD", "TRAIN (filtered)", 2500, 5),
    ("Town02",   "TRAIN (filtered)", 2500, 5),
    ("Town04",   "TRAIN (filtered)", 2500, 5),
    ("Town05",   "TRAIN (filtered)", 2500, 5),
    ("Town06",   "TRAIN (filtered)", 2500, 5),
]

def farthest_points(pts, k):
    """greedy farthest-point sampling -> k 個彼此分散的 index。"""
    k = min(k, len(pts))
    chosen = [int(np.argmax(pts[:, 0] + pts[:, 1]))]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--only", default=None, help="逗號分隔,只重產這些城(如 Town01,Town07)")
    args = ap.parse_args()
    plan = PLAN if not args.only else [p for p in PLAN if p[0] in args.only.split(",")]
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "route_plans")
    os.makedirs(outdir, exist_ok=True)
    client = carla.Client(args.host, args.port); client.set_timeout(300.0)

    for town, role, frames, nseg in plan:
        try:
            world = client.load_world(town)
            m = world.get_map()
            wps = [w for w in m.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
            road = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
            sps = m.get_spawn_points()
            spawn = np.array([[s.location.x, s.location.y] for s in sps])
            seg_idx = farthest_points(spawn, nseg)

            fig, ax = plt.subplots(figsize=(9, 9))
            ax.scatter(road[:, 0], road[:, 1], s=1.2, c="#c9c9c9", label="drivable lanes")
            ax.scatter(spawn[:, 0], spawn[:, 1], s=10, c="#7fb0d8", alpha=0.6, label=f"spawn points ({len(sps)})")
            for j, idx in enumerate(seg_idx):
                x, y = spawn[idx]
                ax.scatter([x], [y], s=320, marker="*", c="#e03b2f", edgecolor="black", zorder=5)
                ax.annotate(f"s{j}", (x, y), color="black", fontsize=12, fontweight="bold",
                            xytext=(6, 6), textcoords="offset points", zorder=6)
            ax.set_aspect("equal"); ax.invert_yaxis()   # top-down, approx aerial orientation
            lane_km = len(wps) * 2.0 / 1000.0
            ax.set_title(f"{town}  |  {role}\nplan {frames} frames · {nseg} segments (red star = segment start) · lanes ~{lane_km:.1f} km",
                         fontsize=11)
            ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.2)
            out = os.path.join(outdir, f"{town}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"{town}: 道路點 {len(wps)}, spawn {len(sps)}, 車道 {lane_km:.1f}km -> {out}")
        except Exception as e:
            print(f"{town}: 失敗 {e}")
    print("完成,圖在:", os.path.abspath(outdir))

if __name__ == "__main__":
    main()
