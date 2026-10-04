# -*- coding: utf-8 -*-
"""Measure how accurate the TTC and the risk levels actually are.

The hazard demo shows a time-to-collision per obstacle and colours it SAFE /
CAUTION / DANGER, but nothing in this project has ever checked those numbers.
They come from a chain -- BEV segmentation -> connected components -> nearest
neighbour tracking -> least-squares range rate -- and every link adds error.

CARLA knows the truth, so this compares the two:

  predicted   what the system reports: obstacle position from the BEV model,
              range rate from the tracker's own history
  ground truth  the actual NPC actor's relative position and velocity

For each frame every predicted obstacle is matched to the nearest real vehicle
(within --gate metres) and the pair contributes to:

  * range error        how far off the BEV places the object
  * TTC error          only where both TTCs are finite (both closing)
  * risk-level matrix  the number that actually matters for a warning system:
                       how often the level shown equals the level the truth implies

Detection is reported separately: a warning system that never fires has a perfect
TTC error.  Ground truth is split into the region the cameras can see and the
blind zone, because missing what the cameras cannot see is geometry, not a bug.

  python eval_ttc.py --town Town10HD --frames 600
"""
import os
import sys
import math
import time
import argparse
from queue import Queue
from collections import Counter

import cv2
import numpy as np
import torch
import carla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask
from carla_dualcam_collect_gt import spawn_npc_vehicles, BEV_H, BEV_W
from live_demo import calib_dict, to_bgr, FRONT_TF, REAR_TF

LEVELS = ["SAFE", "CAUTION", "DANGER"]


def gt_ttc(ego, actor):
    """Range and TTC of a real actor, from CARLA's own state.

    Range rate is the closing component of the relative velocity, i.e. the
    derivative of |p_rel| -- the same quantity the tracker estimates by fitting a
    line through its range history.
    """
    et = ego.get_transform()
    dp = actor.get_transform().location - et.location
    px, py = dp.x, dp.y
    rng = math.hypot(px, py)
    if rng < 1e-6:
        return 0.0, 0.0
    ve, va = ego.get_velocity(), actor.get_velocity()
    vx, vy = va.x - ve.x, va.y - ve.y
    rate = -(px * vx + py * vy) / rng          # positive = closing
    ttc = rng / rate if rate > 1e-3 else math.inf
    return rng, ttc


def to_ego_frame(ego, actor):
    """Actor position in the ego's own x-forward / y-right frame."""
    et = ego.get_transform()
    dp = actor.get_transform().location - et.location
    a = math.radians(et.rotation.yaw)
    c, s = math.cos(a), math.sin(a)
    return (c * dp.x + s * dp.y, -s * dp.x + c * dp.y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None)
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_v4_best.pt")
    ap.add_argument("--img_h", type=int, default=448)
    ap.add_argument("--img_w", type=int, default=800)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--near_ego", type=float, default=80.0,
                    help="spawn NPCs within this radius so encounters actually happen. "
                         "Spread over the whole map they rarely meet: a 60-frame probe on "
                         "Town10HD with 20 vehicles never had one closer than 60 m, so "
                         "there was nothing to measure TTC against.")
    ap.add_argument("--gate", type=float, default=4.0,
                    help="metres within which a prediction counts as the same object")
    ap.add_argument("--out_csv", default="../metrics/ttc_eval.csv")
    ap.add_argument("--seed", type=int, default=None,
                    help="pin the ego spawn point, the NPC spawns and the traffic "
                         "manager, so two runs measure the SAME drive. Without it the "
                         "scene is different every time and a before/after comparison "
                         "of a tracker change is meaningless.")
    ap.add_argument("--coast", type=int, default=1,
                    help="1 = report a track on frames the detector missed it (the "
                         "deployed behaviour since 2026-09-11), 0 = the old behaviour "
                         "where a missed frame hid the obstacle. Only here so the two "
                         "can be measured against each other.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if args.town and args.town.lower().replace("_opt", "") not in world.get_map().name.lower():
        client.load_world(args.town)
        time.sleep(5.0)
        world = client.get_world()

    settings = world.get_settings()
    old = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    for a in (list(world.get_actors().filter("sensor.*"))
              + list(world.get_actors().filter("vehicle.*"))):
        try:
            a.destroy()
        except Exception:
            pass

    bp = world.get_blueprint_library()
    ego, sensors, tm = None, [], None
    try:
        import random
        if args.seed is not None:
            random.seed(args.seed)
        for sp in random.sample(world.get_map().get_spawn_points(), 50):
            ego = world.try_spawn_actor(bp.filter("vehicle.tesla.model3")[0], sp)
            if ego:
                break
        npcs, tm, _ = spawn_npc_vehicles(world, client, bp, ego, args.vehicles,
                                         near_ego=args.near_ego, seed=args.seed)
        ego.set_autopilot(True, tm.get_port())
        print(f"npc vehicles spawned: {len(npcs)}")

        rgb = bp.find("sensor.camera.rgb")
        rgb.set_attribute("image_size_x", str(args.img_w))
        rgb.set_attribute("image_size_y", str(args.img_h))
        rgb.set_attribute("fov", str(args.fov))
        qf, qr = Queue(), Queue()
        cf = world.spawn_actor(rgb, FRONT_TF, attach_to=ego)
        cr = world.spawn_actor(rgb, REAR_TF, attach_to=ego)
        cf.listen(qf.put); cr.listen(qr.put)
        sensors += [cf, cr]

        meta = torch.load(args.model, map_location="cpu", weights_only=False)
        model = DualCamBEVLSS(calib_dict(args.img_w, args.img_h, args.fov),
                              num_classes=6, bev_h=BEV_H, bev_w=BEV_W,
                              img_h=args.img_h, img_w=args.img_w,
                              downsample=meta.get("lss_downsample", 8),
                              pool_res=meta.get("lss_pool", 0.5),
                              pretrained=False).to(device).eval()
        model.load_state_dict(meta.get("model_state_dict", meta), strict=True)

        blind = ~fov_visible_mask(BEV_H, BEV_W)
        vis = fov_visible_mask(BEV_H, BEV_W)
        tracker = HZ.Tracker(coast=bool(args.coast))

        rng_err, ttc_err, rows = [], [], []
        conf = Counter()
        n_gt_vis = n_gt_blind = n_matched = n_false = 0

        for k in range(args.frames):
            world.tick()
            f = to_bgr(qf.get(timeout=5.0))
            r = to_bgr(qr.get(timeout=5.0))
            x = torch.stack([
                torch.from_numpy(f[:, :, ::-1].copy().astype(np.float32) / 255).permute(2, 0, 1),
                torch.from_numpy(r[:, :, ::-1].copy().astype(np.float32) / 255).permute(2, 0, 1),
            ], 0).unsqueeze(0).to(device)
            with torch.no_grad():
                lab = torch.argmax(model(x)[0], 0).cpu().numpy().astype(np.uint8)
            lab[blind & (lab == 1)] = 0
            lab[lab == 2] = 0

            et = ego.get_transform()
            ex, ey, yaw = et.location.x, et.location.y, et.rotation.yaw
            dets = []
            for (px, py, cls, area) in HZ.detect_obstacles(lab):
                wx, wy = HZ.world_of(px, py, ex, ey, yaw)
                dets.append({"wx": wx, "wy": wy, "x": px, "y": py, "cls": 1,
                             "range": math.hypot(px, py)})
            tracker.update(dets, k / float(args.fps), ego=(ex, ey, yaw))

            # ground truth: real vehicles inside the BEV window
            gts = []
            for v in npcs:
                gx, gy = to_ego_frame(ego, v)
                if abs(gx) > 30 or abs(gy) > 30:
                    continue
                row = int(BEV_H - 1 - (gx + 30.0) / 0.25)
                col = int((gy + 30.0) / 0.25)
                seen = (0 <= row < BEV_H and 0 <= col < BEV_W and vis[row, col])
                g_rng, g_ttc = gt_ttc(ego, v)
                gts.append({"x": gx, "y": gy, "rng": g_rng, "ttc": g_ttc, "vis": seen,
                            "used": False})
                if seen:
                    n_gt_vis += 1
                else:
                    n_gt_blind += 1

            for tid, tr in tracker.active():
                d = tr["cur"]
                p_ttc, rate = HZ.ttc_of(tr["hist"])
                p_lvl, _ = HZ.risk_of(d["range"], p_ttc)
                best, bd = None, args.gate
                for g in gts:
                    if g["used"]:
                        continue
                    dd = math.hypot(g["x"] - d["x"], g["y"] - d["y"])
                    if dd < bd:
                        best, bd = g, dd
                if best is None:
                    n_false += 1
                    continue
                best["used"] = True
                n_matched += 1
                g_lvl, _ = HZ.risk_of(best["rng"], best["ttc"])
                conf[(g_lvl, p_lvl)] += 1
                rng_err.append(d["range"] - best["rng"])
                if math.isfinite(p_ttc) and math.isfinite(best["ttc"]):
                    ttc_err.append(p_ttc - best["ttc"])
                rows.append([k, round(best["rng"], 2), round(d["range"], 2),
                             round(best["ttc"], 2) if math.isfinite(best["ttc"]) else "",
                             round(p_ttc, 2) if math.isfinite(p_ttc) else "",
                             g_lvl, p_lvl, int(best["vis"])])

            if k % 20 == 0:
                near = sorted(math.hypot(*to_ego_frame(ego, v)) for v in npcs)[:3]
                print(f"  {k}/{args.frames}  matched {n_matched}  false {n_false}  "
                      f"nearest NPC {[round(d,1) for d in near]} m", flush=True)

        # ---- report ----
        print()
        print("=" * 70)
        print(f"frames {args.frames}   matched {n_matched}   unmatched predictions {n_false}")
        print(f"ground-truth vehicle appearances: visible {n_gt_vis}, blind zone {n_gt_blind}")
        if n_gt_vis:
            print(f"  matched / visible GT = {100.0*n_matched/n_gt_vis:.1f}%  "
                  f"(objects the cameras could see and the system found)")
        if rng_err:
            e = np.array(rng_err)
            print(f"\nrange error   mean {e.mean():+.2f} m   MAE {np.abs(e).mean():.2f} m   "
                  f"p50 {np.median(np.abs(e)):.2f}  p90 {np.percentile(np.abs(e),90):.2f}")
        if ttc_err:
            e = np.array(ttc_err)
            print(f"TTC error     mean {e.mean():+.2f} s   MAE {np.abs(e).mean():.2f} s   "
                  f"p50 {np.median(np.abs(e)):.2f}  p90 {np.percentile(np.abs(e),90):.2f}   "
                  f"(n={len(e)}, both closing)")
        tot = sum(conf.values())
        if tot:
            print(f"\nrisk level, ground truth (rows) vs system (cols), n={tot}")
            print("            " + "".join(f"{c:>10s}" for c in LEVELS))
            agree = 0
            for g in LEVELS:
                line = f"{g:>10s}  "
                for p in LEVELS:
                    line += f"{conf[(g,p)]:10d}"
                    if g == p:
                        agree += conf[(g, p)]
                print(line)
            print(f"\nlevel agreement {100.0*agree/tot:.1f}%")
            miss = sum(conf[(g, p)] for g in ("DANGER", "CAUTION") for p in ("SAFE",))
            over = sum(conf[(g, p)] for g in ("SAFE",) for p in ("CAUTION", "DANGER"))
            print(f"  under-warned (truth says warn, system says SAFE) {miss} "
                  f"= {100.0*miss/tot:.1f}%")
            print(f"  over-warned  (truth says SAFE, system warns)     {over} "
                  f"= {100.0*over/tot:.1f}%")

        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "gt_range", "pred_range", "gt_ttc", "pred_ttc",
                        "gt_level", "pred_level", "gt_visible"])
            w.writerows(rows)
        print("\nwrote", args.out_csv)

    finally:
        client.set_timeout(20.0)
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(old)
        except Exception:
            pass
        for s in sensors:
            try:
                s.destroy()
            except Exception:
                pass
        try:
            ego.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
