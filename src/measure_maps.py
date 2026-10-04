# -*- coding: utf-8 -*-
"""量測各 CARLA 城市:實體範圍 + 可行駛車道總長 + 全走完可收幀數。
只載入地圖 + 查路網,不開任何相機(避開 camera-streaming 崩潰),很安全。

用法:
  # 先開 server: G:\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000
  python measure_maps.py --host 127.0.0.1 --port 2000
輸出:終端表格 + map_size_report.csv
"""
import carla, argparse, csv, os

TOWNS = ["Town01", "Town02", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
SPEEDS = [5, 8, 10]      # m/s(市區低速)
FPS = 10
WP_SPACING = 1.0         # 每 1m 取一個 waypoint → 車道總長 ≈ waypoint 數 × 1m

def measure(world):
    m = world.get_map()
    sps = m.get_spawn_points()
    # 車道總長:generate_waypoints 對「所有 lane」每 spacing 取一點 → 數量×spacing ≈ 總長
    wps = m.generate_waypoints(WP_SPACING)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    L_m = len(driving) * WP_SPACING
    # 實體範圍(用 spawn points 的 bbox 估)
    xs = [s.location.x for s in sps]; ys = [s.location.y for s in sps]
    ext_x = (max(xs) - min(xs)) if xs else 0.0
    ext_y = (max(ys) - min(ys)) if ys else 0.0
    return len(sps), ext_x, ext_y, L_m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port); client.set_timeout(120.0)
    rows = []
    hdr = f"{'town':10} {'spawns':>6} {'範圍 m':>14} {'車道km':>7} " + " ".join(f"@{v}m/s".rjust(9) for v in SPEEDS)
    print(hdr); print("-" * len(hdr))
    for t in TOWNS:
        try:
            world = client.load_world(t)
            n, ex, ey, L = measure(world)
            frames = {v: (L / v) * FPS for v in SPEEDS}
            print(f"{t:10} {n:6d} {ex:6.0f}x{ey:<6.0f} {L/1000:6.2f} " +
                  " ".join(f"{frames[v]:9.0f}" for v in SPEEDS))
            row = {"town": t, "spawns": n, "extent_x_m": round(ex), "extent_y_m": round(ey),
                   "lane_len_km": round(L/1000, 2)}
            for v in SPEEDS: row[f"frames_full_{v}mps"] = round(frames[v])
            rows.append(row)
        except Exception as e:
            print(f"{t:10} 失敗: {e}")
    out = os.path.join(os.path.dirname(__file__), "..", "map_size_report.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n已存:", os.path.abspath(out))
    print("解讀:'車道km' = 可行駛車道總長;'@Vm/s' = 以該速度全走完一次需要幾幀(fps=10)。")
    print("對照計畫 2,500 幀/城:< 全走完 → 只收部分;> 全走完 → 會重複繞。")

if __name__ == "__main__":
    main()
