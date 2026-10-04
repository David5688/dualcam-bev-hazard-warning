"""Hazard-warning layer: detect obstacles per frame, track them, and raise
front/rear collision warnings from time-to-collision (TTC).

Pipeline (detection source is pluggable, same as accumulate_pointcloud.py):
  --source gt   : obstacles = clusters of vehicle/pedestrian pixels in bev_gt
  --source pred : obstacles = clusters in the 9set model's predicted BEV (the
                  real deployed system output)

Design:
  * Detect obstacle centroids in the ego BEV -> metric ego (x fwd, y lateral).
    Range to a hazard = hypot(x, y) directly (ego sits at the origin).
  * Track in WORLD coordinates (via poses.json) so a static obstacle stays put
    and association is stable while the ego turns.
  * TTC = range / range_rate, where range_rate = -(d range / dt) computed from
    each track's ego-range history. This captures closing from BOTH ego and
    obstacle motion -- exactly what a collision warning needs.
  * Risk: DANGER if (closing and TTC < t_danger) or range < r_danger;
          CAUTION if (closing and TTC < t_caution) or range < r_caution; else SAFE.
  * Direction: front if ego x > 0 else rear -> front FCW vs rear-collision warning.

Renders a BEV hazard view (ego arrow, distance rings, colour-coded obstacles with
TTC labels, active-warning banner) to an H.264 clip, and logs warnings to CSV.
"""
import os, sys, cv2, json, math, argparse, numpy as np, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_prof_style import fov_visible_mask

# BEV <-> ego metric (matches eval_temporal.py / calibration bev block)
BEV = 240; RES = 0.25; X0 = -30.0; Y0 = -30.0
def _find_ffmpeg():
    """Cross-platform ffmpeg (Windows dev + Jetson/Linux deploy). imageio_ffmpeg
    bundles the right binary per OS; fall back to a system ffmpeg on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = _find_ffmpeg()


def px_to_ego(col, row):
    return (X0 + (BEV - 1 - row) * RES, Y0 + col * RES)     # (x fwd, y lateral)


def detect_obstacles(label, min_area=8):
    """Cluster vehicle(1)/pedestrian(2) pixels -> list of (x_ego, y_ego, cls, area)."""
    out = []
    for cls in (1, 2):
        mask = (label == cls).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            col, row = cent[i]
            x, y = px_to_ego(col, row)
            out.append((x, y, cls, int(stats[i, cv2.CC_STAT_AREA])))
    return out


def world_of(x, y, ex, ey, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    return (ex + c * x - s * y, ey + s * x + c * y)


def vel_of(whist, window=0.6):
    """(vx, vy) in world units per second from a track's recent path, or (0, 0).

    Used to dead-reckon a track that is coasting.  A short window on purpose: this
    is an extrapolation of the last thing the car actually did, not a smoothed
    estimate of what it usually does.
    """
    if not whist or len(whist) < 2:
        return 0.0, 0.0
    t_last = whist[-1][0]
    win = [p for p in whist if t_last - p[0] <= window]
    if len(win) < 2:
        win = whist[-2:]
    dt = win[-1][0] - win[0][0]
    if dt <= 1e-6:
        return 0.0, 0.0
    return (win[-1][1] - win[0][1]) / dt, (win[-1][2] - win[0][2]) / dt


def ego_of(wx, wy, ex, ey, yaw_deg):
    """Inverse of world_of: a world point back into the ego frame (x fwd, y lat).

    Needed by anything that anchors a detection in the world and then has to ask
    "where is that now, from where I am now" -- a coasted track, or a pedestrian
    box that was detected two frames ago while the car kept moving.
    """
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    dx, dy = wx - ex, wy - ey
    return (c * dx + s * dy, -s * dx + c * dy)


class Tracker:
    """Nearest-neighbour association in world frame; keeps ego-range history for TTC.

    A track that goes unmatched on a frame is COASTED rather than hidden: it keeps
    its last detection for up to max_miss frames and active() still reports it,
    flagged tr["coasted"] = True.

    Why (2026-09-11): active() used to yield only tracks matched on the current
    frame, so one dropped detection made the obstacle vanish from the warning
    layer for that frame -- the side fell to SAFE, the event log wrote a CLEAR,
    and the next frame wrote a RAISE again.  Measured on
    results/DEMO_free_20260909_212801_events.csv: the same pedestrian raised and
    cleared five times in 1.2 s, and on DEMO_rear_approach_20260909_213914 the
    rear warning flickered at 0.1 s intervals.  max_miss was already there to
    bridge exactly that gap -- it just never reached the half of the system that
    warns.  Pass coast=False for the old behaviour.

    A coasted track's ego-frame position goes stale as the ego drives on (2 m over
    three frames at 25 km/h), so update() re-projects it from the track's world
    position when the caller passes the current ego pose.  hist is left alone:
    coasting must not feed invented measurements into the TTC fit.
    """
    def __init__(self, gate=4.0, max_miss=3, coast=True, whist_len=40):
        self.gate = gate; self.max_miss = max_miss; self.coast = coast
        # whist is a SECOND, longer history in world coordinates, for anything that
        # needs to see a manoeuvre rather than a closing rate -- trajectory_predict
        # reads it to spot a weave.  hist stays at six samples because ttc_of fits a
        # line through it and the measured TTC accuracy (metrics/ttc_eval.csv) is
        # for that window; lengthening it would silently change a validated number.
        self.whist_len = whist_len
        self.tracks = {}; self.next_id = 0

    def update(self, dets_world, t, ego=None):
        # dets_world: list of dict(wx,wy,x,y,cls,range)
        # ego: (ex, ey, yaw_deg) -- optional, only used to refresh coasted tracks
        for tr in self.tracks.values():
            tr["matched"] = False
        for d in dets_world:
            best, bd = None, self.gate
            for tid, tr in self.tracks.items():
                if tr["matched"]:
                    continue
                dist = math.hypot(tr["wx"] - d["wx"], tr["wy"] - d["wy"])
                if dist < bd:
                    bd, best = dist, tid
            if best is None:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"wx": d["wx"], "wy": d["wy"], "cls": d["cls"],
                                    "hist": [], "whist": [], "miss": 0}
            else:
                tid = best
            tr = self.tracks[tid]
            tr["wx"], tr["wy"] = d["wx"], d["wy"]
            tr["hist"].append((t, d["range"], d["x"], d["y"]))
            tr["hist"] = tr["hist"][-6:]
            tr["whist"].append((t, d["wx"], d["wy"]))
            tr["whist"] = tr["whist"][-self.whist_len:]
            tr["matched"] = True; tr["miss"] = 0; tr["cur"] = d
            tr["coasted"] = False
            # where and when it was last actually seen, for dead reckoning
            tr["wx_seen"], tr["wy_seen"], tr["t_seen"] = d["wx"], d["wy"], t
        dead = []
        for tid, tr in self.tracks.items():
            if not tr["matched"]:
                tr["miss"] += 1
                if self.coast and tr.get("cur") is not None:
                    tr["coasted"] = True
                    # Dead-reckon rather than freeze.  Freezing was tolerable while
                    # max_miss was three frames; over the a second the demo now
                    # allows, a car at town speed travels seven metres, and a
                    # stationary ghost of it sitting in the road is worse than no
                    # track at all.
                    vx, vy = vel_of(tr["whist"])
                    dt = t - tr.get("t_seen", t)
                    tr["wx"] = tr.get("wx_seen", tr["wx"]) + vx * dt
                    tr["wy"] = tr.get("wy_seen", tr["wy"]) + vy * dt
                    if ego is not None:
                        x, y = ego_of(tr["wx"], tr["wy"], ego[0], ego[1], ego[2])
                        d = dict(tr["cur"])
                        d["x"], d["y"], d["range"] = x, y, math.hypot(x, y)
                        tr["cur"] = d
                else:
                    tr.pop("cur", None)
                if tr["miss"] > self.max_miss:
                    dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def active(self):
        """Every track with a current detection -- matched this frame, or coasting."""
        for tid, tr in self.tracks.items():
            if tr.get("cur") is not None:
                yield tid, tr


def ttc_of(hist):
    """range_rate from a short history (least-squares slope); TTC if closing."""
    if len(hist) < 2:
        return math.inf, 0.0
    ts = np.array([h[0] for h in hist]); rs = np.array([h[1] for h in hist])
    A = np.polyfit(ts, rs, 1)                    # slope = d range / dt
    rate = -A[0]                                 # closing positive
    rng = rs[-1]
    return (rng / rate if rate > 1e-3 else math.inf), rate


def risk_of(rng, ttc, t_danger=2.0, t_caution=4.0, r_danger=5.0, r_caution=10.0):
    """Risk level for an obstacle. TTC (time-to-collision) is the PRIMARY metric --
    it is speed-aware, matching production forward-collision warning (Euro NCAP FCW
    warns around TTC 2.4-2.7 s). Range is a secondary proximity floor: too close is
    dangerous even when slow. Definition (satisfy either the TTC or the range test):
        DANGER  : closing with TTC < 2.0 s,  OR range < 5 m
        CAUTION : closing with TTC < 4.0 s,  OR range < 10 m
        SAFE    : otherwise (TTC >= 4 s / not closing, and range >= 10 m)
    """
    if rng < r_danger or ttc < t_danger:
        return "DANGER", (0, 0, 230)
    if rng < r_caution or ttc < t_caution:
        return "CAUTION", (0, 200, 230)
    return "SAFE", (0, 200, 0)


LEVEL_ORDER = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
LEVEL_COLOUR = {"SAFE": (0, 200, 0), "CAUTION": (0, 200, 230), "DANGER": (0, 0, 230)}


class LevelHold(object):
    """Hysteresis on a per-track warning level: escalate now, de-escalate slowly.

    risk_of() is memoryless, so a track sitting near a threshold retriggers on
    every frame it crosses back and forth -- TTC comes from a least-squares fit
    over six range samples and moves by tenths of a second frame to frame, which
    is more than enough to cross 2.0 s repeatedly during one genuine approach.

    An escalation is passed through immediately: a warning must never be late.
    A drop has to hold for fall_frames consecutive frames before it is believed.
    Production FCW does the same thing, and for the same reason -- a warning that
    blinks reads as a broken warning, and a driver stops trusting it.

    Keyed by track id, so two obstacles never share a state.  Call prune() with
    the ids still alive or the table grows for the length of the run.

    rise_frames raises the bar in the other direction: how many consecutive frames
    a HIGHER level must be asked for before it is shown.  It defaults to 1, which
    is the "a warning must never be late" rule above and is what the measured TTC
    path uses -- verified not to delay a single real warning.

    A PREDICTED hazard is a weaker claim than a measured one and wants rise_frames
    above 1.  Measured: with rise_frames=1 a car overtaking one lane over raised a
    CAUTION in 2 of 3 seeds, off a single noisy frame whose fitted lateral speed
    was invented; the manoeuvre never lasted more than 2 frames, while a real lane
    change held for 7.  Asking a prediction to repeat itself three times costs
    0.3 s and removes the whole class.
    """

    def __init__(self, fall_frames=5, rise_frames=1):
        self.fall = max(int(fall_frames), 1)
        self.rise = max(int(rise_frames), 1)
        self.state = {}

    def apply(self, key, level):
        """The level to actually show for this track, given what it showed before."""
        held, below, cand, above = self.state.get(key, ("SAFE", 0, "SAFE", 0))
        if LEVEL_ORDER[level] > LEVEL_ORDER[held]:
            above = above + 1 if level == cand else 1
            cand = level
            if above >= self.rise:
                held, below, above = level, 0, 0
        elif LEVEL_ORDER[level] == LEVEL_ORDER[held]:
            below, cand, above = 0, level, 0
        else:
            cand, above = level, 0
            below += 1
            if below >= self.fall:
                held, below = level, 0
        self.state[key] = (held, below, cand, above)
        return held

    def prune(self, keys):
        keys = set(keys)
        for k in [k for k in self.state if k not in keys]:
            del self.state[k]


def render(frame_dets, ego_rgb_f, ego_rgb_r, size=560):
    """BEV hazard panel: ego arrow at centre, rings, obstacles colour-coded + TTC."""
    img = np.full((size, size, 3), 18, np.uint8)
    cx = cy = size // 2
    m2px = size / 60.0                            # +-30 m
    for r in (10, 20, 30):
        cv2.circle(img, (cx, cy), int(r * m2px), (60, 60, 60), 1)
        cv2.putText(img, f"{r}m", (cx + 4, cy - int(r * m2px) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    # ego arrow (forward = up)
    cv2.arrowedLine(img, (cx, cy + 14), (cx, cy - 16), (255, 255, 255), 2, tipLength=0.4)
    banner = {"front": "SAFE", "rear": "SAFE"}
    bcol = {"front": (0, 200, 0), "rear": (0, 200, 0)}
    order = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    for (x, y, cls, rng, ttc, level, col) in frame_dets:
        px = int(cx + y * m2px); py = int(cy - x * m2px)      # y right, x up
        rad = 9 if cls == 1 else 6
        cv2.circle(img, (px, py), rad, col, -1)
        cv2.circle(img, (px, py), rad, (240, 240, 240), 1)
        lbl = f"{rng:.0f}m" + (f" {ttc:.1f}s" if ttc < 9 else "")
        cv2.putText(img, lbl, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        d = "front" if x >= 0 else "rear"
        if order[level] > order[banner[d]]:
            banner[d] = level; bcol[d] = col
    # warning banners
    for i, d in enumerate(("front", "rear")):
        y0 = 6 + i * 34
        txt = f"{'FRONT' if d=='front' else 'REAR '}: {banner[d]}"
        cv2.rectangle(img, (6, y0), (232, y0 + 28), bcol[d], -1)
        cv2.putText(img, txt, (14, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/carla_dualcam_t01_temporal")
    ap.add_argument("--source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fps", type=float, default=10.0, help="capture fps -> dt = step/fps")
    ap.add_argument("--t_danger", type=float, default=2.0)
    ap.add_argument("--t_caution", type=float, default=4.0)
    ap.add_argument("--r_danger", type=float, default=5.0)
    ap.add_argument("--r_caution", type=float, default=10.0)
    ap.add_argument("--trust_blind", action="store_true",
                    help="trust dynamic (vehicle/ped) predictions in the blind zone too. "
                         "Default OFF: the cameras can't see there, so those detections are "
                         "hallucination (variant D). Only affects --source pred.")
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_9set_depth_bv3_best.pt")
    ap.add_argument("--out", default="../results/HAZARD_town01.mp4")
    ap.add_argument("--out_csv", default="../metrics/hazard_town01.csv")
    args = ap.parse_args()

    poses = json.load(open(f"{args.data}/poses.json"))
    dt = args.step / args.fps
    frames = [f"{i:06d}.png" for i in range(args.start, args.start + args.count * args.step, args.step)]
    frames = [f for f in frames if f in poses and os.path.exists(f"{args.data}/bev_gt/{f}")]

    predictor = None
    if args.source == "pred":
        import torch
        from dualcam_bev_lss import DualCamBEVLSS
        meta = torch.load(args.model, map_location="cpu", weights_only=False)
        ds = meta.get("lss_downsample", 8); pool = meta.get("lss_pool", 0.5)
        mdl = DualCamBEVLSS.from_calib_file(f"{args.data}/calibration.json", num_classes=6,
                                            bev_h=240, bev_w=240, img_h=224, img_w=400,
                                            downsample=ds, pool_res=pool, pretrained=False)
        mdl.load_state_dict(meta.get("model_state_dict", meta), strict=True)
        mdl = mdl.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        def predict(n):
            def rd(p):
                im = cv2.resize(cv2.imread(p), (400, 224))[:, :, ::-1].copy().astype(np.float32) / 255.0
                return torch.from_numpy(im).permute(2, 0, 1)
            x = torch.stack([rd(f"{args.data}/front_rgb/{n}"), rd(f"{args.data}/rear_rgb/{n}")], 0).unsqueeze(0)
            x = x.to(next(mdl.parameters()).device)
            with torch.no_grad():
                return torch.argmax(mdl(x)[0], 0).cpu().numpy().astype(np.uint8)
        predictor = predict

    tracker = Tracker()
    writer = None; log = []
    # variant D: dynamic predictions in the blind zone are hallucination -> drop them
    blind = ~fov_visible_mask(BEV, BEV)
    for k, n in enumerate(frames):
        ex, ey, yaw = poses[n]
        label = cv2.imread(f"{args.data}/bev_gt/{n}", 0) if args.source == "gt" else predictor(n)
        if args.source == "pred" and not args.trust_blind:
            label = label.copy()
            label[blind & np.isin(label, (1, 2))] = 0
        dets = detect_obstacles(label)
        dets_world = []
        for (x, y, cls, area) in dets:
            wx, wy = world_of(x, y, ex, ey, yaw)
            dets_world.append({"wx": wx, "wy": wy, "x": x, "y": y, "cls": cls,
                               "range": math.hypot(x, y)})
        tracker.update(dets_world, k * dt, ego=(ex, ey, yaw))

        frame_dets = []
        for tid, tr in tracker.active():
            d = tr["cur"]; rng = d["range"]
            ttc, rate = ttc_of(tr["hist"])
            level, col = risk_of(rng, ttc, args.t_danger, args.t_caution, args.r_danger, args.r_caution)
            frame_dets.append((d["x"], d["y"], d["cls"], rng, ttc, level, col))
            if level != "SAFE":
                log.append((n, tid, "veh" if d["cls"] == 1 else "ped",
                            "front" if d["x"] >= 0 else "rear", f"{rng:.1f}", f"{ttc:.2f}",
                            f"{rate:.2f}", level))

        bev = render(frame_dets, None, None)
        front = cv2.resize(cv2.imread(f"{args.data}/front_rgb/{n}"), (498, 280))
        rear = cv2.resize(cv2.imread(f"{args.data}/rear_rgb/{n}"), (498, 280))
        cv2.putText(front, "FRONT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rear, "REAR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rgbcol = np.vstack([front, rear])
        canvas = np.hstack([rgbcol, bev])
        cap = np.full((30, canvas.shape[1], 3), 18, np.uint8)
        cv2.putText(cap, f"Hazard warning  {os.path.basename(args.data)}  frame {n}  source={args.source}  dt={dt:.2f}s",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        canvas = np.vstack([canvas, cap])
        if writer is None:
            h, w = canvas.shape[:2]
            tmp = args.out.replace(".mp4", "_raw.mp4")
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(canvas)
    writer.release()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("frame,track,class,dir,range_m,ttc_s,closing_mps,level\n")
        for row in log:
            f.write(",".join(map(str, row)) + "\n")

    tmp = args.out.replace(".mp4", "_raw.mp4")
    subprocess.run([FFMPEG, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", args.out], check=True, capture_output=True)
    os.remove(tmp)
    n_danger = sum(1 for r in log if r[-1] == "DANGER")
    n_caution = sum(1 for r in log if r[-1] == "CAUTION")
    print(f"{len(frames)} frames | warnings: {n_danger} DANGER, {n_caution} CAUTION events")
    print("video:", args.out, "| log:", args.out_csv)


if __name__ == "__main__":
    main()
