# -*- coding: utf-8 -*-
"""Live hazard demo: drive in CARLA and watch the system work in real time.

Same panels as results/HAZARD_FULL_*.mp4, but nothing is pre-recorded -- the ego
drives on autopilot while the BEV model, the tracker and the TTC risk logic run
frame by frame in a window.

  python live_demo.py --town Town01                       # vehicles only, most stable
  python live_demo.py --town Town01 --peds                # + image-detected pedestrians
  python live_demo.py --town Town01 --gt                  # + live ground truth panel
  python live_demo.py --town Town01 --record ../results/live.mp4

  # a staged hazard that repeats, recorded, with the warnings logged to CSV:
  python live_demo.py --town Town10HD --scenario rear_approach --record auto

Keys:  q 離開   g 切換 GT   h 切換危險警示   p 暫停   c 存一張全解析度截圖

Two design choices worth knowing:

1. Cameras capture natively at the model's input size (default 800x448) instead of
   1280x720-then-resize.  The intrinsics come out identical -- fx = (W/2)/tan(fov/2)
   is 400 for W=800, and scaling 1280's 640 by 800/1280 is also 400 -- so the LSS
   projection is unchanged while the sensor bandwidth drops ~2.5x.

2. Depth cameras are OFF unless --peds.  資料收集規劃.md §13.1 pinned the server's
   first-tick crashes on the depth sensors: with --depth the collector produced
   frames in 4 of 13 attempts, without it 60/60 with the server still alive
   afterwards.  A live demo that dies in front of an audience is worse than one
   without pedestrian boxes, so the stable configuration is the default.
"""
import os
import sys
import time
import math
import argparse
import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import torch
import carla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard_warning as HZ
import demo_scenarios as SC
import demo_events as EV
import trajectory_predict as TP
from hazard_bev import bev_panel
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import fov_visible_mask, render_bev_prof
from live_recon import LiveRecon, backproject
from depth_to_pointcloud import depth_png_to_metres
from carla_dualcam_collect_gt import (
    spawn_npc_vehicles, spawn_walkers, make_bev_gt, get_intrinsic,
    build_lane_points, BEV_H, BEV_W,
)

# Refuse to start a recording with less than this much room on the target drive.
MIN_FREE_GB = 5.0

FRONT_TF = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(yaw=0))
REAR_TF = carla.Transform(carla.Location(x=-1.5, z=1.6), carla.Rotation(yaw=180))


def to_bgr(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    return a.reshape((image.height, image.width, 4))[:, :, :3].copy()


def calib_dict(w, h, fov):
    """The calibration the model needs, built from the live rig rather than a file."""
    K = get_intrinsic(w, h, fov)
    return {
        "image": {"width": w, "height": h, "fov": fov},
        "bev": {"x_min": -30.0, "x_max": 30.0, "y_min": -30.0, "y_max": 30.0,
                "resolution": 0.25, "height": BEV_H, "width": BEV_W},
        "cameras": {
            "front": {"K": K.tolist(),
                      "T_ego_cam": tf_matrix(FRONT_TF).tolist()},
            "rear": {"K": K.tolist(),
                     "T_ego_cam": tf_matrix(REAR_TF).tolist()},
        },
    }


def tf_matrix(tf):
    """4x4 ego->camera transform for a carla.Transform, matching the collector."""
    yaw = math.radians(tf.rotation.yaw)
    pitch = math.radians(tf.rotation.pitch)
    roll = math.radians(tf.rotation.roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tf.location.x, tf.location.y, tf.location.z]
    return T


_SCREEN = None


def screen_size():
    """Usable screen size in pixels, or a safe guess if it cannot be read."""
    global _SCREEN
    if _SCREEN is None:
        try:
            import ctypes
            u = ctypes.windll.user32
            u.SetProcessDPIAware()
            _SCREEN = (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
        except Exception:
            _SCREEN = (1920, 1080)
    return _SCREEN


def fit_to_screen(img, max_width=0, fullscreen=False):
    """Scale a composed frame so the whole window is visible.

    Windowed: only ever shrinks, and leaves room for the window frame and the
    taskbar, which is why the margins are not zero.

    Fullscreen: scales up OR down to fill the screen and letterboxes the rest
    in black, so the panels keep their aspect and sit centred.  The panels are
    composed at --panel pixels, so on a 1920x1080 projector a three-column
    frame is enlarged ~1.25x; that is a display-time resize, the recording and
    the c-key snapshot stay at full composed resolution.
    """
    h, w = img.shape[:2]
    if fullscreen:
        sw, sh = screen_size()
        scale = min(sw / float(w), sh / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=(
                cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA))
        if (nw, nh) == (sw, sh):
            return img
        out = np.zeros((sh, sw, 3), np.uint8)
        y0, x0 = (sh - nh) // 2, (sw - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = img
        return out
    if max_width > 0:
        limit_w, limit_h = max_width, 10 ** 6
    else:
        sw, sh = screen_size()
        limit_w, limit_h = sw - 40, sh - 140
    scale = min(1.0, limit_w / float(w), limit_h / float(h))
    if scale >= 0.999:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def set_window_mode(win, fullscreen):
    """(Re)create the demo window as fullscreen or as a plain autosized window.

    OpenCV cannot switch a window between WINDOW_AUTOSIZE and WINDOW_NORMAL
    once it exists, and fullscreen needs WINDOW_NORMAL, so the window is torn
    down and rebuilt.  The next imshow fills it; the 'window is gone -> quit'
    check in the loop runs after that imshow, so the rebuild is invisible to it.
    """
    try:
        cv2.destroyWindow(win)
    except cv2.error:
        pass
    if fullscreen:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)


def ego_status(ego, kmh, front_m, front_kmh):
    """Why is the ego not moving?  Returns (label, BGR colour).

    Speed alone cannot tell a red light from a jam from a traffic-manager sulk, and
    on this project all three happen.  The signals that separate them:

      is_at_traffic_light + Red   -> a light, and it will clear on its own
      full brake, something close  -> yielding to the vehicle (or walker) ahead
      full brake, nothing close    -> unexplained. Check for orphaned actors first:
                                      dead vehicles from killed runs sit on the road
                                      and the ego yields to them, which is what every
                                      "the traffic manager parked it" case on
                                      2026-08-09 turned out to be. A clean world runs
                                      at 26 km/h with 5-7 of 10 NPCs moving.
    """
    if kmh > 0.5:
        return "MOVING", (120, 220, 120)
    try:
        if ego.is_at_traffic_light() and str(ego.get_traffic_light_state()).endswith("Red"):
            return "RED LIGHT", (90, 190, 240)
    except Exception:
        pass
    c = ego.get_control()
    if front_m is not None and front_m < 12.0 and front_kmh is not None and front_kmh < 2.0:
        return f"YIELDING ({front_m:.0f} m)", (90, 190, 240)
    if c.brake > 0.5:
        return "STUCK (braking, nothing ahead)", (80, 120, 240)
    if c.throttle > 0.2:
        return "STUCK (throttle, not moving)", (80, 120, 240)
    return "STOPPED", (160, 160, 160)


def nearest_ahead(ego, npcs):
    """Distance and speed of the closest NPC vehicle in front of the ego."""
    try:
        et = ego.get_transform()
        fwd = et.get_forward_vector()
    except Exception:
        return None, None
    best = None
    for v in npcs:
        try:
            dl = v.get_transform().location - et.location
            if (dl.x * fwd.x + dl.y * fwd.y) <= 0:
                continue
            d = math.sqrt(dl.x * dl.x + dl.y * dl.y)
            if best is None or d < best[0]:
                vv = v.get_velocity()
                best = (d, 3.6 * math.sqrt(vv.x * vv.x + vv.y * vv.y))
        except Exception:
            pass
    return best if best else (None, None)


WIN = "DualCam live hazard demo"

_ORDER = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}


def worst_per_side(obstacles):
    """Highest risk level in front of and behind the ego.

    Same split hazard_bev.bev_panel uses for its FRONT:/REAR: banners (x >= 0 is
    ahead), so the sound always agrees with what the panel shows.
    """
    front = rear = "SAFE"
    for o in obstacles:
        x, y, cls, level, col, ttc = o[:6]
        if x >= 0:
            if _ORDER[level] > _ORDER[front]:
                front = level
        else:
            if _ORDER[level] > _ORDER[rear]:
                rear = level
    return front, rear


class Alerter(object):
    """Audible warning, on its own thread.

    winsound.Beep blocks for the full duration of the tone, so calling it from the
    render loop would stall the simulation -- a 300 ms DANGER tone at 10 fps eats
    three frames.  The loop only publishes the current levels; this thread decides
    what to play and when to repeat.

    Front and rear use different pitches.  The whole point of the rig is that it
    watches behind as well as ahead, and a listener can tell a high beep from a low
    one without looking at the screen.
    """
    FRONT_HZ, REAR_HZ = 1500, 700

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.muted = False
        self.front = self.rear = "SAFE"
        self._stop = False
        self._thread = None
        if enabled:
            import threading
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def set(self, front, rear):
        self.front, self.rear = front, rear

    def toggle_mute(self):
        """Silence without tearing down the thread, so it can come back instantly."""
        self.muted = not self.muted
        return self.muted

    def stop(self):
        self._stop = True

    def _run(self):
        import winsound
        last = 0.0
        while not self._stop:
            if self.muted:
                time.sleep(0.05)
                continue
            now = time.time()
            # DANGER repeats fast and insistently; CAUTION is an occasional reminder.
            # Front wins when both sides are shouting -- it is the one you steer into.
            for side, lvl, hz in (("front", self.front, self.FRONT_HZ),
                                  ("rear", self.rear, self.REAR_HZ)):
                if lvl == "DANGER":
                    gap, dur = 0.45, 130
                elif lvl == "CAUTION":
                    gap, dur = 1.4, 70
                else:
                    continue
                if now - last >= gap:
                    try:
                        winsound.Beep(hz, dur)
                    except Exception:
                        self.enabled = False
                        return
                    last = time.time()
                break
            time.sleep(0.05)


def run_stem(scenario):
    """../results/DEMO_<scenario>_<timestamp>, shared by every file a run writes.

    One stem for the video, the event CSV and the summary means the three are
    still obviously the same run a week later, which matters once there are a
    dozen of them from rehearsals.
    """
    tag = scenario if scenario and scenario != "none" else "free"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "results",
                                         "DEMO_%s_%s" % (tag, ts)))


def titled(img, label, colour=(230, 230, 230)):
    bar = np.full((28, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


class PedWorker(object):
    """Run the pedestrian detector on its own thread and CUDA stream.

    The detector is the slowest stage by a wide margin -- measured 2026-10-02
    at ~120-150 ms per front+rear pair, against ~15 ms for the BEV model --
    and running it inline every --ped_every frames made every third frame
    three times longer than the others.  Here the main loop hands a frame
    pair over and carries on; the boxes come back one or two frames later
    together with the ego pose of the frame they were detected in, which is
    exactly what ped_pose already recorded for the inline path, so the
    tracker sees the same (slightly stale) data it always did.  A frame
    offered while the previous one is still in the detector is dropped, so
    the detector can never fall behind.

    torch releases the GIL inside CUDA kernels and the detector runs on a
    separate stream, so the BEV forward pass on the main thread is not
    queued behind it.  --ped_sync restores the inline path.
    """

    def __init__(self, calib, device, score):
        import threading
        import pedestrian_detect as PD
        self.PD, self.calib, self.device, self.score = PD, calib, device, score
        PD.detector(device)                   # load the weights now, not mid-demo
        self._in = Queue(maxsize=1)
        self._out = Queue()
        self.busy = False
        self._stream = (torch.cuda.Stream() if str(device).startswith("cuda")
                        else None)
        self._th = threading.Thread(target=self._run, name="ped-detector",
                                    daemon=True)
        self._th.start()

    def submit(self, f_img, r_img, d_f, d_r, pose):
        """Offer a frame pair.  False if the detector is still on the last one."""
        if self.busy:
            return False
        self.busy = True
        self._in.put((f_img, r_img, d_f, d_r, pose))
        return True

    def result(self):
        """(peds, pose, seconds) of the latest finished detection, or None."""
        got = None
        while True:
            try:
                got = self._out.get_nowait()
            except Empty:
                return got

    def _run(self):
        while True:
            job = self._in.get()
            if job is None:
                return
            f_img, r_img, d_f, d_r, pose = job
            t0 = time.time()
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        peds = self.PD.pedestrians_from_arrays(
                            {"front": f_img, "rear": r_img},
                            {"front": d_f, "rear": d_r},
                            self.calib, self.device, self.score)
                    self._stream.synchronize()
                else:
                    peds = self.PD.pedestrians_from_arrays(
                        {"front": f_img, "rear": r_img},
                        {"front": d_f, "rear": d_r},
                        self.calib, self.device, self.score)
            except Exception as exc:
                print("!! pedestrian detector failed: %s" % exc)
                peds = []
            self._out.put((peds, pose, time.time() - t0))
            self.busy = False

    def close(self):
        try:
            self._in.put_nowait(None)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None, help="load this map first (omit = use current)")
    ap.add_argument("--load_wait", type=int, default=600,
                    help="seconds to wait for a map load to finish (first visit is slow)")
    ap.add_argument("--model", default="../checkpoints/dualcam_bev_lss_v4_best.pt")
    ap.add_argument("--img_h", type=int, default=448)
    ap.add_argument("--img_w", type=int, default=800)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=10)
    # Demo defaults, deliberately NOT the collection defaults.  Collection wanted
    # every frame to contain actors, so it used 40 vehicles / 60 walkers all spawned
    # within 150 m of the ego -- which gridlocks small maps: that setting is what
    # left Town05's first collection with 70.9% duplicate frames, the ego parked in
    # a jam for whole segments.  A demo needs the car to actually drive, so the
    # density is lower and the actors are spread over the map.
    ap.add_argument("--vehicles", type=int, default=20)
    # ⚠️ The measurement below was taken in a world polluted with ~31 orphaned
    # vehicles from killed runs, so it does NOT isolate walkers -- the ego was
    # yielding to dead cars as much as to pedestrians.  A clean world with 15
    # walkers and --ignore_walkers drives at 26 km/h.  Re-measure before treating
    # walkers as the cause of anything.
    # Walkers default to 0 because they stop the demo dead.  CARLA's traffic manager
    # yields to a pedestrian indefinitely, and the walker controllers send them
    # across roads, so the ego parks behind one and never moves again.  Measured on
    # Town01, 12 vehicles, 150 frames: with 10 walkers the ego held 0.0 km/h the
    # whole run; with 0 walkers it drove at 19-26 km/h throughout.  (This is also
    # the likeliest real cause of the Town05 collection gridlock, which ran 60
    # walkers concentrated within 150 m of the ego.)
    ap.add_argument("--walkers", type=int, default=0,
                    help="pedestrians. Non-zero looks richer but tends to freeze the ego; "
                         "use with --manual so you can drive around them.")
    ap.add_argument("--near_ego", type=float, default=0.0,
                    help="concentrate spawns within this radius of the ego (0 = whole map). "
                         "Non-zero jams small towns.")
    ap.add_argument("--peds", action="store_true",
                    help="detect pedestrians in the camera image and place them via depth "
                         "(adds two depth sensors -- see the crash note in the docstring)")
    ap.add_argument("--ped_every", type=int, default=3,
                    help="run the person detector every Nth frame (it is the slowest stage)")
    ap.add_argument("--ped_sync", action="store_true",
                    help="run the person detector inline on the main loop, the way it "
                         "was before 2026-10-02, instead of on its own thread. The "
                         "threaded path hands boxes back one or two frames later; "
                         "this is the A/B switch if that ever looks wrong.")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--no_predict", dest="predict", action="store_false",
                    help="turn off the trajectory predictor and warn on the radial "
                         "TTC alone. Worth having as a switch because it is the "
                         "direct A/B: a car weaving behind you at a steady distance "
                         "moves its range by 0.1 m, so TTC stays infinite and the "
                         "old logic says SAFE for the whole manoeuvre.")
    ap.add_argument("--predict_horizon", type=float, default=TP.HORIZON,
                    help="how many seconds ahead the trajectory predictor projects")
    ap.add_argument("--watch_arm", type=float, default=1.5,
                    help="seconds of detected weaving before a vehicle is put on "
                         "the dangerous-vehicle watch list. Distance plays no part: "
                         "a car weaving at the 30 m edge of the BEV cannot reach you "
                         "yet, and that is exactly when you want to know about it.")
    ap.add_argument("--track_miss", type=float, default=1.0,
                    help="seconds a track survives without a detection before it is "
                         "retired. The library default is 0.3 s, which is fine at "
                         "close range and too short at 25 m, where the BEV drops a "
                         "car often enough that its id kept being retired and the "
                         "same vehicle came back a stranger. The track is dead-"
                         "reckoned meanwhile, not frozen.")
    ap.add_argument("--watch_debug", action="store_true",
                    help="trace watch-list decisions: ghosts kept, adoptions tried, "
                         "releases. For working out why a flagged vehicle stopped "
                         "being flagged.")
    ap.add_argument("--watch_release", type=float, default=0.0,
                    help="seconds of GENUINE clean driving before a flagged vehicle "
                         "comes off the list. DEFAULT 0 = never: once a driver has "
                         "been caught weaving, this system does not forget within "
                         "the span of a demo, which is the behaviour asked for and "
                         "also what a human driver does. Set a positive number to "
                         "let vehicles earn their way off. Either way only frames "
                         "where the vehicle was moving enough to judge, and whose "
                         "measured SDLP is below WATCH_CLEAR_SDLP, count towards "
                         "it -- a car stopped at a light is not driving well, it "
                         "is not driving, and a car too far away to measure is not "
                         "evidence of anything.")
    ap.add_argument("--predict_hold", type=float, default=2.0,
                    help="seconds a PREDICTED warning is held after the prediction "
                         "stops asking for it. Much longer than --warn_hold because "
                         "a weave is one continuous piece of behaviour: the measured "
                         "swing dips below the detector threshold twice per cycle, "
                         "and a 0.5 s hold turned one minute of weaving into 19 "
                         "separate rear warnings.")
    ap.add_argument("--warn_hold", type=float, default=0.5,
                    help="seconds a warning must stay lower before it is allowed to "
                         "drop (escalation is always immediate). 0 = the old memoryless "
                         "behaviour, which flickered: the same pedestrian raised and "
                         "cleared five times in 1.2 s on 2026-09-09 because TTC, fitted "
                         "over six range samples, crosses a threshold back and forth "
                         "during one genuine approach.")
    ap.add_argument("--gt", action="store_true", help="also draw a live ground-truth panel")
    ap.add_argument("--gt_every", type=int, default=2,
                    help="rebuild the GT raster every Nth frame (it queries the map per cell)")
    ap.add_argument("--recon", action="store_true",
                    help="add a live 3D reconstruction panel. See --recon_source for where "
                         "its depth comes from")
    ap.add_argument("--recon_every", type=int, default=2,
                    help="fold a new frame into the cloud every Nth frame")
    ap.add_argument("--recon_render_every", type=int, default=1,
                    help="redraw the reconstruction panel every Nth frame and show the "
                         "last drawing in between. The render is ~15 ms of CPU per "
                         "frame at the 260k-point cap; 2 halves that for a panel that "
                         "updates at 5 Hz instead of 10.")
    ap.add_argument("--recon_stride", type=int, default=4,
                    help="pixel subsampling when back-projecting (4 turns a 800x448 frame "
                         "into ~22k points instead of 358k)")
    ap.add_argument("--recon_source", default="model", choices=["model", "depth"],
                    help="where the depth for the reconstruction comes from. 'model' uses "
                         "the network's own depth head -- RGB only, no depth sensor. "
                         "'depth' uses CARLA's depth cameras, i.e. perfect ground truth, "
                         "for the comparison the thesis needs: measured on test_town01 the "
                         "model's depth is off by a median 0.19 m at confidence >0.3, which "
                         "is already finer than the 0.35 m voxel the cloud is quantised to.")
    ap.add_argument("--recon_cams", default="front", choices=["front", "rear", "both"],
                    help="which camera feeds the reconstruction. Front only by default: "
                         "the chase view looks forward, so rear points land behind the "
                         "virtual camera and mostly add clutter.")
    ap.add_argument("--recon_splat", type=int, default=6,
                    help="largest splat radius, in pixels. The renderer already sizes each "
                         "splat to half the projected voxel spacing; this only caps how big "
                         "the very near field is allowed to get. 3 was enough while the cloud "
                         "was RGB-only and sparse, but a depth camera fills the near ground "
                         "densely and the 0.35 m quantisation then projects to ~20 px gaps "
                         "under the car, which a 3 px splat leaves as a checkerboard.")
    ap.add_argument("--recon_conf", type=float, default=0.3,
                    help="drop pixels whose peak depth-bin probability is below this. The "
                         "depth head is a coarse 56x100 grid upsampled to full resolution, "
                         "and its unconfident pixels are what smear into radial streaks once "
                         "accumulated (the same finding accumulate_pointcloud.py records).")
    ap.add_argument("--recon_sky_conf", type=float, default=0.7,
                    help="confidence required ABOVE the horizon. Sky has no depth, but the "
                         "head still has to emit one: measured on Town10HD it hands the sky "
                         "~25 m -- squarely inside the kept range -- at a median confidence "
                         "of 0.11, and 31%% of it still clears a 0.3 threshold. Ground pixels "
                         "sit at confidence 1.00, so demanding more above the horizon costs "
                         "almost nothing real and removes the white speckle.")
    ap.add_argument("--recon_voxel", type=float, default=0.35)
    ap.add_argument("--recon_max", type=int, default=260000)
    # Staged hazards.  See demo_scenarios.py for why: on autopilot a warning is
    # a coincidence, and the rear-facing half of the rig -- the point of the
    # project -- is the half least likely to get one.
    ap.add_argument("--scenario", default="none", choices=SC.CHOICES,
                    help="stage a repeatable hazard instead of waiting for one: "
                         "rear_approach (a car closes from behind), front_brake "
                         "(the lead vehicle stops hard), cut_in (a car changes "
                         "into your lane), reverse_ped (someone crosses behind "
                         "you). The ego is spawned somewhere with enough clear "
                         "lane to stage it, and it re-arms until you quit.")
    ap.add_argument("--scenario_once", dest="scenario_loop", action="store_false",
                    help="stage the hazard once instead of repeating it")
    ap.add_argument("--seed", type=int, default=None,
                    help="pin the random source and the traffic manager, so the "
                         "same scenario plays out the same way twice. Worth "
                         "setting once a run looks good on stage.")
    ap.add_argument("--record", default=None,
                    help="also write the window to this mp4. 'auto' names it "
                         "../results/DEMO_<scenario>_<timestamp>.mp4 -- use that "
                         "one live, so a demo that crashes still leaves a video "
                         "you can play instead.")
    ap.add_argument("--record_secs", type=float, default=180.0,
                    help="stop RECORDING after this many seconds; the demo itself "
                         "carries on. 0 = no cap, which is what filled the drive "
                         "on a run left going overnight (2026-09-08, 21.5 h, "
                         "65.5 GB) -- every recording made after that came out as "
                         "an empty 44-byte file until space was freed -- and what "
                         "wrote 40.0 GB on a later one (2026-09-11, 274,787 "
                         "frames, 7.6 simulated hours, 12.3 h of wall clock).")
    ap.add_argument("--events", default="auto",
                    help="warning-event CSV plus a summary json ('auto' = beside "
                         "the recording, 'none' = do not log). One row per "
                         "warning transition; the numbers a slide needs.")
    ap.add_argument("--panel", type=int, default=520)
    ap.add_argument("--max_width", type=int, default=0,
                    help="shrink the window to at most this many pixels wide "
                         "(0 = fit the screen automatically). The panel count is "
                         "not fixed -- turning on the reconstruction or the ground "
                         "truth adds a column each -- so the composed image is "
                         "scaled at display time rather than asking for --panel to "
                         "be retuned every time. Recording is unaffected and stays "
                         "at full resolution.")
    ap.add_argument("--fullscreen", action="store_true",
                    help="open the window fullscreen on the primary screen, scaled "
                         "to fill it with black bars for the leftover aspect. The f "
                         "key toggles it live either way.")
    ap.add_argument("--manual", action="store_true",
                    help="drive it yourself with WASD instead of the traffic manager")
    ap.add_argument("--wheel", action="store_true",
                    help="steer with a physical wheel instead of WASD. Needs "
                         "wheel_config.ini beside live_demo.py -- run "
                         "calibrate_wheel.py once to write it. Implies --manual; "
                         "Tab still switches back to autopilot, and the keyboard "
                         "keys keep working alongside it.")
    ap.add_argument("--wheel_config", default=None,
                    help="use this wheel_config.ini instead of the one beside "
                         "live_demo.py")
    ap.add_argument("--ignore_lights", action="store_true",
                    help="let the ego run red lights. A CARLA light cycle is 10-20 s, which "
                         "on a 20 s demo means the car is parked for most of it -- fine for "
                         "data collection, dull to watch. Ego only; the NPCs still obey.")
    ap.add_argument("--ignore_lights_all", action="store_true",
                    help="every vehicle ignores lights, not just the ego -- use this one, "
                         "since the ego mostly waits behind stopped NPCs, not at the light "
                         "itself. Traffic gets chaotic at junctions; it is a demo setting.")
    ap.add_argument("--ignore_walkers", action="store_true",
                    help="the ego does not brake for pedestrians. This is how you get both a "
                         "moving car AND pedestrians on screen: CARLA's traffic manager "
                         "yields to a walker indefinitely, so a demo with walkers ends with "
                         "the ego at 0 km/h on full brake under a green light (measured on "
                         "Town10HD: drove 22-29 km/h for 8 s, then stopped for good). The "
                         "pedestrians still appear, are still detected, and still raise "
                         "warnings -- the car just keeps going, which is honest for a system "
                         "that warns rather than steers.")
    ap.add_argument("--unstick", type=int, default=50,
                    help="if the ego sits below 0.5 km/h for this many frames on autopilot, "
                         "teleport it to another spawn point (0 = never). A safety net, not "
                         "a fix. NOTE: the long stalls this was written for turned out to be "
                         "orphaned actors from killed runs -- 31 dead vehicles parked on the "
                         "road -- and a clean world (see --keep_world) drives fine without "
                         "it. Left in because a genuine traffic-manager deadlock is still "
                         "possible and a frozen demo is worse than a jump cut.")
    ap.add_argument("--speed", type=float, default=None,
                    help="ego speed as a percentage difference from the limit "
                         "(negative = faster, e.g. -30 for 30%% over)")
    ap.add_argument("--profile", type=int, default=0,
                    help="every N frames, print where the time went and how many NPC "
                         "vehicles are actually moving (0 = off)")
    ap.add_argument("--keep_world", dest="clean_world", action="store_false",
                    help="do NOT wipe actors left over from previous runs at startup. "
                         "Only pass this if another client is deliberately sharing the "
                         "world -- otherwise the leftovers become road blocks.")
    ap.add_argument("--no_sound", action="store_true",
                    help="silence the audible warning. It is on by default with a window "
                         "and always off headless -- a background smoke test that beeps at "
                         "the room is not helpful.")
    ap.add_argument("--no_display", action="store_true", help="headless (for smoke tests)")
    ap.add_argument("--max_frames", type=int, default=0, help="stop after N frames (0 = forever)")
    args = ap.parse_args()

    stem = run_stem(args.scenario)
    if args.record == "auto":
        args.record = stem + ".mp4"
    if args.events == "auto":
        args.events = (os.path.splitext(args.record)[0] if args.record else stem) + "_events.csv"
    elif args.events == "none":
        args.events = None
    # Always run with a seed, and always say which one.  A demo is the one place
    # where "it looked great in rehearsal" has to be reproducible on demand, and
    # the old behaviour -- unseeded unless asked -- meant a good run could not be
    # got back afterwards because nothing recorded what it had been.  An
    # unseeded run now picks one, prints it, and stores it in the summary json,
    # so pinning a rehearsal is a copy-and-paste rather than a re-roll.
    if args.seed is None:
        args.seed = int(time.time()) & 0x7FFFFFFF
        print("no --seed given; this run is seed %d "
              "(pass --seed %d to play it again)" % (args.seed, args.seed))
    else:
        print("seed %d" % args.seed)
    import random as _rnd
    _rnd.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    torch.manual_seed(args.seed)
    scenario = SC.make_scenario(args.scenario, fps=args.fps,
                                loop=args.scenario_loop, seed=args.seed)
    if scenario is not None and scenario.needs_peds and not args.peds:
        # Staging a pedestrian while the only thing that can see one is switched
        # off produces a silent demo, which looks exactly like a broken system.
        args.peds = True
        print("scenario %s needs the pedestrian detector -- enabling --peds"
              % scenario.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    world = client.get_world()
    want = args.town.lower().replace("_opt", "") if args.town else None
    if want and want not in world.get_map().name.lower():
        # load_world has a hard 60 s wait that set_timeout does not extend, and a
        # first visit to a map blows through it -- but the engine keeps loading in
        # the background after the call gives up.  So fire it, swallow the timeout,
        # and poll until the server reports the map (same trick as prewarm_map.py).
        print(f"loading {args.town} (first visit builds a cache, be patient) ...")
        t0 = time.time()
        try:
            client.load_world(args.town)
        except RuntimeError as exc:
            print(f"  load_world returned '{exc}' -- polling instead")
        while time.time() - t0 < args.load_wait:
            try:
                w = client.get_world()
                if want in w.get_map().name.lower():
                    break
            except RuntimeError:
                pass
            time.sleep(2.0)
        else:
            raise SystemExit(f"server never reached {args.town} within {args.load_wait}s")
        world = client.get_world()
        print(f"  on {world.get_map().name} after {time.time() - t0:.0f}s")

    settings = world.get_settings()
    old_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    # Sweep out actors left behind by earlier runs.  A script killed before its
    # cleanup -- a timeout, a Ctrl-C, a crashed client -- leaves its vehicles in the
    # world with no autopilot, and they then sit on the road as obstacles forever.
    # Measured on 2026-08-09: a demo asking for 10 vehicles found 41 in the world
    # with none of them moving, and spent its time yielding to the dead ones.
    if args.clean_world:
        stale = (list(world.get_actors().filter("sensor.*"))
                 + list(world.get_actors().filter("controller.ai.walker"))
                 + list(world.get_actors().filter("walker.pedestrian.*"))
                 + list(world.get_actors().filter("vehicle.*")))
        for a in stale:
            try:
                if a.type_id.startswith("controller"):
                    a.stop()
                a.destroy()
            except Exception:
                pass
        if stale:
            print(f"cleared {len(stale)} actors left over from earlier runs")

    bp_lib = world.get_blueprint_library()
    ego = None
    sensors = []
    tm = None
    events = None          # the finally block writes the summary; it must exist
    try:
        spawn_points = world.get_map().get_spawn_points()
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        import random
        # A staged hazard needs room -- a rear approach set up inside a junction
        # just deadlocks the traffic manager -- so the scenario gets to say which
        # spawn points it can work with, best first.
        candidates = (scenario.pick_spawn_points(world, spawn_points) if scenario
                      else random.sample(spawn_points, len(spawn_points)))
        for sp in candidates:
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("cannot spawn ego")

        npcs, tm, _ = spawn_npc_vehicles(world, client, bp_lib, ego, args.vehicles,
                                         near_ego=args.near_ego, seed=args.seed)
        wheel = None
        if args.wheel:
            import wheel_input
            try:
                wheel = wheel_input.WheelInput(args.wheel_config)
                print("wheel: %s" % wheel.describe())
            except wheel_input.WheelNotConfigured as exc:
                # Not fatal.  Losing the wheel should cost the presenter the
                # wheel, not the demo -- WASD still drives.
                print("!! wheel not available (%s) -- falling back to WASD" % exc)
                wheel = None
            else:
                args.manual = True

        autopilot = not args.manual
        ego.set_autopilot(autopilot, tm.get_port())
        if args.ignore_lights or args.ignore_lights_all:
            tm.ignore_lights_percentage(ego, 100.0)
            # Letting only the ego jump lights does not keep it moving: it still
            # queues behind NPCs that are waiting at one, which is exactly what a
            # 200-frame trace on Town01 showed -- ego stopped 6 m behind a stationary
            # car with more of them ahead.  --ignore_lights_all frees the whole queue.
            if args.ignore_lights_all:
                for v in npcs:
                    try:
                        tm.ignore_lights_percentage(v, 100.0)
                    except Exception:
                        pass
            print("ignoring traffic lights:", "all vehicles" if args.ignore_lights_all else "ego only")
        if args.ignore_walkers:
            try:
                tm.ignore_walkers_percentage(ego, 100.0)
                print("ego ignores pedestrians (they still appear and still warn)")
            except AttributeError:
                print("WARNING: this CARLA build has no ignore_walkers_percentage; "
                      "use --manual to drive around pedestrians instead")
        if args.speed is not None:
            tm.vehicle_percentage_speed_difference(ego, args.speed)
        walkers, controllers = spawn_walkers(world, bp_lib, args.walkers,
                                             ego=ego, near_ego=args.near_ego)
        print(f"npc vehicles: {len(npcs)}   walkers: {len(walkers)}")

        if scenario is not None:
            scenario.setup(world, bp_lib, ego, tm)
            if scenario.wants_reverse:
                # reverse_ped is about what the driver cannot see while backing
                # up, so the demo starts in the manoeuvre rather than asking the
                # presenter to remember three keys under the lights.
                autopilot = False
                ego.set_autopilot(False, tm.get_port())

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(args.img_w))
        rgb_bp.set_attribute("image_size_y", str(args.img_h))
        rgb_bp.set_attribute("fov", str(args.fov))
        q_f, q_r = Queue(), Queue()
        cam_f = world.spawn_actor(rgb_bp, FRONT_TF, attach_to=ego)
        cam_r = world.spawn_actor(rgb_bp, REAR_TF, attach_to=ego)
        cam_f.listen(q_f.put); cam_r.listen(q_r.put)
        sensors += [cam_f, cam_r]

        q_df = q_dr = None
        need_depth_cams = args.peds or (args.recon and args.recon_source == "depth")
        if need_depth_cams:
            d_bp = bp_lib.find("sensor.camera.depth")
            d_bp.set_attribute("image_size_x", str(args.img_w))
            d_bp.set_attribute("image_size_y", str(args.img_h))
            d_bp.set_attribute("fov", str(args.fov))
            q_df, q_dr = Queue(), Queue()
            d_f = world.spawn_actor(d_bp, FRONT_TF, attach_to=ego)
            d_r = world.spawn_actor(d_bp, REAR_TF, attach_to=ego)
            d_f.listen(q_df.put); d_r.listen(q_dr.put)
            sensors += [d_f, d_r]

        # model
        meta = torch.load(args.model, map_location="cpu", weights_only=False)
        ck_h, ck_w = meta.get("img_h"), meta.get("img_w")
        if ck_h and ck_w and (ck_h, ck_w) != (args.img_h, args.img_w):
            print(f"WARNING: checkpoint trained at {ck_w}x{ck_h}, running at "
                  f"{args.img_w}x{args.img_h} -- the LSS projection will be wrong")
        # The constructor takes the calibration dict directly; from_calib_file is
        # only a json loader around it.  Here the rig is live, so the dict is built
        # from the sensors we just spawned rather than read from a dataset.
        model = DualCamBEVLSS(
            calib_dict(args.img_w, args.img_h, args.fov),
            num_classes=6, bev_h=BEV_H, bev_w=BEV_W,
            img_h=args.img_h, img_w=args.img_w,
            downsample=meta.get("lss_downsample", 8),
            pool_res=meta.get("lss_pool", 0.5), pretrained=False,
        ).to(device).eval()
        model.load_state_dict(meta.get("model_state_dict", meta), strict=True)
        print(f"model: {os.path.basename(args.model)}  input {args.img_w}x{args.img_h}")
        if not args.no_display:
            # Keys go to the OpenCV window, not the terminal.  Pressing q with the
            # console focused does nothing, which reads as "q doesn't work".
            print("keys go to the VIDEO WINDOW (click it first): "
                  "q/Esc quit  g GT  h hazard  m mute  p pause  c snapshot  "
                  "f fullscreen  Tab auto/manual  WASD drive")
            set_window_mode(WIN, args.fullscreen)

        lane_pts = build_lane_points(world.get_map()) if args.gt else None
        recon = LiveRecon(voxel=args.recon_voxel, max_points=args.recon_max,
                          size=args.panel,
                          max_radius=args.recon_splat) if args.recon else None
        calib = calib_dict(args.img_w, args.img_h, args.fov)
        ped_worker = None
        if args.peds and not args.ped_sync:
            ped_worker = PedWorker(calib, device, args.score)
            print("pedestrian detector: on its own thread (--ped_sync for inline)")
        # Mixed precision for the BEV model.  Measured on 20 test_town01 frames
        # (2026-10-02): 17.6 -> 13.2 ms, BEV labels agree on 99.98% of cells
        # and the depth-head argmax on 99.8%.
        use_amp = str(device).startswith("cuda")

        blind = ~fov_visible_mask(BEV_H, BEV_W)
        tracker = HZ.Tracker(whist_len=max(64, int(6.0 * args.fps)),
                             max_miss=max(3, int(args.track_miss * args.fps)))
        hold = HZ.LevelHold(fall_frames=max(1, round(args.warn_hold * args.fps)))
        # The predicted hazard gets its own hysteresis, and it is asymmetric in
        # both directions relative to the measured one.
        #
        # It must repeat itself before escalating (rise): a prediction three
        # seconds out is a weaker claim than a measured closing rate, and one
        # noisy frame is enough to invent a lateral velocity that is not there.
        #
        # It also holds far longer before clearing (fall): what it reports is a
        # driver's behaviour, and a driver does not stop being erratic for the
        # half second their car happens to be crossing the centre of its lane.
        # Measured on the first live rear_weave run -- a genuinely weaving car at
        # 23 m produced 19 rear CAUTIONs in 60 s, raising and clearing every 0.5 s
        # as the measured swing crossed the detector threshold twice per cycle.
        # The weave is one event lasting a minute, and it should be logged as one.
        phold = HZ.LevelHold(fall_frames=max(1, round(args.predict_hold * args.fps)),
                             rise_frames=max(1, round(0.3 * args.fps)))
        # Which vehicles have been caught driving erratically.  Latched and
        # distance-free on purpose: a car weaving at 28 m cannot hit you yet, and
        # knowing about it before it is 8 m back is the whole point.
        # NB: release must NOT be clamped up to 1 -- WatchList reads 0 as "never
        # release", and max(1, ...) turned the default into "release after one
        # clean frame", the exact opposite.
        watch = TP.WatchList(arm=max(1, round(args.watch_arm * args.fps)),
                             release=int(round(args.watch_release * args.fps)))
        if args.watch_debug:
            watch.debug = lambda m: print("   [watch] " + m, flush=True)
        alerter = Alerter(enabled=not args.no_sound and not args.no_display)
        if alerter.enabled:
            print("audible warning on: front = high beep, rear = low beep "
                  "(--no_sound to silence)")
        events = EV.EventLog(args.events, fps=args.fps, scenario=args.scenario,
                             meta={"town": world.get_map().name,
                                   "model": os.path.basename(args.model),
                                   "vehicles": args.vehicles,
                                   "walkers": args.walkers,
                                   "peds": bool(args.peds),
                                   "predict": bool(args.predict),
                                   "seed": args.seed}) if args.events else None
        if events is not None:
            print("logging warnings to", args.events)
        writer = None
        record_path = args.record      # args.record is cleared if it cannot start
        rec_frames = 0
        record_done = False
        record_max = int(args.record_secs * args.fps) if args.record_secs > 0 else 0
        shots = 0
        show_gt, show_hazard, paused = args.gt, True, False
        gt_cache = np.zeros((BEV_H, BEV_W), np.uint8)
        peds = []
        ped_pose = (0.0, 0.0, 0.0)   # ego pose when peds was last refreshed
        weave_amp = 0.0              # highest SDLP of a flagged vehicle, metres
        n_flagged = 0                # vehicles currently on the watch list
        _was_flagged = set()         # for the watch-list transition log
        _adopt_log = set()           # tracks that inherited a flag by re-identification
        flagged_frames = 0           # frames with at least one vehicle on the list
        first_flag_t = None
        t_prev = time.time()
        fps_ema = None

        # Manual driving state.  cv2.waitKey reports key PRESSES only -- there is no
        # "key is still held" and no key-up -- so throttle and steering are ramped up
        # while their key keeps arriving and decayed on any frame it does not.  That
        # gives usable control without pulling in pygame just for key state.
        ctl = carla.VehicleControl()
        steer, throttle, brake = 0.0, 0.0, 0.0
        reverse = bool(scenario is not None and scenario.wants_reverse)
        wheel_rev_held = False      # edge detect for the wheel's reverse button
        wheel_warned = False        # say "no report yet" once, not every frame
        # compose time and ego speed are measured after the profile print, so the
        # line reports the previous frame's values for those two.
        t_tick = t_read = t_haz = t_comp = t_recon = t_ped = 0.0
        recon_img = None             # last rendered reconstruction panel
        kmh_last = 0.0
        stuck_frames = 0
        front_m = front_kmh = None
        status, status_col = "MOVING", (120, 220, 120)

        k = 0
        while True:
            if not paused:
                if scenario is not None:
                    # Before the tick, so a staged brake or lane change lands on
                    # the frame the cameras are about to see.
                    scenario.tick(k, world, ego, tm)
                if not autopilot:
                    ctl.steer = float(np.clip(steer, -1.0, 1.0))
                    ctl.throttle = float(np.clip(throttle, 0.0, 1.0))
                    ctl.brake = float(np.clip(brake, 0.0, 1.0))
                    ctl.reverse = reverse
                    ego.apply_control(ctl)
                _t = time.time()
                world.tick()
                t_tick = time.time() - _t
                _t = time.time()
                f_img = to_bgr(q_f.get(timeout=5.0))
                r_img = to_bgr(q_r.get(timeout=5.0))
                d_f_img = to_bgr(q_df.get(timeout=5.0)) if need_depth_cams else None
                d_r_img = to_bgr(q_dr.get(timeout=5.0)) if need_depth_cams else None
                t_read = time.time() - _t

                t0 = time.time()
                x = torch.stack([
                    torch.from_numpy(f_img[:, :, ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1),
                    torch.from_numpy(r_img[:, :, ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1),
                ], 0).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=use_amp):
                    if recon is not None:
                        out, depth_logits = model(x, return_depth=True)
                        prob = torch.softmax(depth_logits.float(), dim=1)
                        centers = (model.d0 + (torch.arange(model.D, device=device) + 0.5)
                                   * model.dd).float()
                        exp_d = (prob * centers[None, :, None, None]).sum(1)
                        pk = prob.max(1).values                  # (N, fH, fW)
                        if args.recon_conf > 0:
                            exp_d = exp_d.masked_fill(pk < args.recon_conf, 0.0)
                        if args.recon_sky_conf > args.recon_conf:
                            # The cameras have no pitch, so the horizon lands on the
                            # principal point -- the middle row of the feature grid.
                            horizon = pk.shape[1] // 2
                            sky = torch.zeros_like(pk, dtype=torch.bool)
                            sky[:, :horizon, :] = True
                            exp_d = exp_d.masked_fill(sky & (pk < args.recon_sky_conf), 0.0)
                        exp_d = torch.nn.functional.interpolate(
                            exp_d[:, None], size=(args.img_h, args.img_w),
                            mode="nearest")[:, 0].cpu().numpy()
                    else:
                        out, exp_d = model(x), None
                    lab = torch.argmax(out[0] if isinstance(out, tuple) else out[0],
                                       0).cpu().numpy().astype(np.uint8)
                t_model = time.time() - t0

                lab[blind & (lab == 1)] = 0          # variant D: blind-zone vehicles are noise
                lab[lab == 2] = 0                    # pedestrians come from the image detector

                tf = ego.get_transform()
                ex, ey = tf.location.x, tf.location.y
                yaw = tf.rotation.yaw

                t0 = time.time()
                if ped_worker is not None:
                    got = ped_worker.result()
                    if got is not None:
                        # Boxes from a frame one or two ticks back, with the pose
                        # of THAT frame -- see the note on ped_pose below.
                        peds, ped_pose, t_ped = got
                    if k % args.ped_every == 0:
                        ped_worker.submit(f_img, r_img, d_f_img, d_r_img,
                                          (ex, ey, yaw))
                elif args.peds and k % args.ped_every == 0:
                    import pedestrian_detect as PD
                    peds = PD.pedestrians_from_arrays(
                        {"front": f_img, "rear": r_img},
                        {"front": d_f_img, "rear": d_r_img},
                        calib_dict(args.img_w, args.img_h, args.fov), device, args.score)
                    # Where the car was when it saw them.  The detector runs every
                    # ped_every frames but its boxes are reused on the frames in
                    # between, and converting those stale ego-frame points with the
                    # CURRENT pose drags every pedestrian along with the car -- 2 m
                    # over three frames at 25 km/h.  That is what put 31.31 -> 45.47
                    # -> 41.66 m into the 2026-09-09 event log inside 0.3 s, and the
                    # tracker fitted a TTC through it.
                    ped_pose = (ex, ey, yaw)
                    t_ped = time.time() - t0

                _t = time.time()
                dets = []
                for (x_, y_, cls, area) in HZ.detect_obstacles(lab):
                    wx, wy = HZ.world_of(x_, y_, ex, ey, yaw)
                    dets.append({"wx": wx, "wy": wy, "x": x_, "y": y_, "cls": 1,
                                 "range": math.hypot(x_, y_)})
                for (x_, y_, conf, cam, box) in peds:
                    wx, wy = HZ.world_of(x_, y_, *ped_pose)
                    px, py = HZ.ego_of(wx, wy, ex, ey, yaw)
                    dets.append({"wx": wx, "wy": wy, "x": px, "y": py, "cls": 2,
                                 "range": math.hypot(px, py)})
                tracker.update(dets, k / float(args.fps), ego=(ex, ey, yaw))

                # Always computed, even with the overlay hidden: h used to empty
                # this list, which silenced the beeps and wrote a CLEAR for every
                # live warning into the event log.  h is a drawing switch.
                obstacles = []
                live_ids = []
                weave_amp_now = 0.0
                track_info = {}      # tid -> (x, y, sdlp, rev) for the watch log
                t_now = k / float(args.fps)
                _ev = ego.get_velocity()
                ego_state = (ex, ey, yaw, _ev.x, _ev.y)
                for tid, tr in tracker.active():
                    d = tr["cur"]
                    live_ids.append(tid)
                    ttc, rate = HZ.ttc_of(tr["hist"])
                    level, _ = HZ.risk_of(d["range"], ttc, 2.0, 4.0, 5.0, 10.0)
                    if d["cls"] == 2 and len(tr["hist"]) < 2 and level != "SAFE":
                        # One frame of confirmation before a brand-new pedestrian
                        # track may warn: with no history its TTC is infinite, so
                        # the level would come from range alone.
                        level = "SAFE"
                    level = hold.apply(tid, level)
                    why, path, flagged = "", None, False
                    # Vehicles only.  Pedestrian tracks come from the image
                    # detector every ped_every frames, which is neither dense
                    # enough nor steady enough to fit a manoeuvre to.
                    if args.predict and d["cls"] == 1:
                        a = TP.assess(tr["whist"], ego_state,
                                      horizon=args.predict_horizon)
                        if tid not in watch.state:
                            # A track id we have not seen before is not the same
                            # thing as a car we have not seen before.  A junction,
                            # or one gap in detection longer than the tracker's
                            # max_miss, retires the id and the same vehicle comes
                            # back as a stranger -- which is how a flagged weaver
                            # was a plain SAFE square thirty seconds later in the
                            # 2026-09-17 recording, with the driver still visibly
                            # behind us.  Check it against the vehicles we have
                            # lost before treating it as new.
                            _adopted = watch.adopt(tid, tr["wx"], tr["wy"], t_now)
                            if _adopted:
                                _adopt_log.add(tid)
                                print("[%6.1fs] track %d re-identified as a vehicle "
                                      "already on the watch list" % (t_now, tid),
                                      flush=True)
                        flagged = watch.update(tid, a["weaving"], a["judgeable"],
                                               a["sdlp"], a["state"], t_now)
                        plevel, preason = TP.predict_level(a, flagged)
                        plevel = phold.apply(tid, plevel)
                        path = a["path"]
                        if HZ.LEVEL_ORDER[plevel] > HZ.LEVEL_ORDER[level]:
                            # The predictor saw something the range never will.
                            level, why = plevel, preason
                        track_info[tid] = (d["x"], d["y"], a["sdlp"], a["rev"])
                        if flagged and a["judgeable"] and a["sdlp"] > weave_amp_now:
                            weave_amp_now = a["sdlp"]
                    obstacles.append((d["x"], d["y"], d["cls"], level,
                                      HZ.LEVEL_COLOUR[level], ttc, False, path, why,
                                      flagged))
                hold.prune(live_ids)
                phold.prune(live_ids)
                watch.prune(live_ids, t_now)
                _now_flagged = set(i for i in live_ids if watch.flagged(i))
                for _t in sorted(_now_flagged - _was_flagged):
                    # The range must come from the flagged track itself.  This used
                    # to report the first obstacle in the list, whichever that was.
                    _x, _y, _sd, _rv = track_info.get(_t, (0.0, 0.0, 0.0, 0))
                    print("[%6.1fs] FLAGGED track %d via %s: %s %.1f m, "
                          "SDLP %.2f m, %d reversals"
                          % (t_now, _t,
                             "RE-ID" if _t in _adopt_log else "weave evidence",
                             "REAR " if _x < 0 else "FRONT",
                             math.hypot(_x, _y), _sd, _rv), flush=True)
                for _t in sorted(_was_flagged - _now_flagged):
                    print("[%6.1fs] track %d left the watch list (released or lost)"
                          % (t_now, _t), flush=True)
                _was_flagged = _now_flagged
                n_flagged = watch.count()
                if n_flagged:
                    flagged_frames += 1
                    if first_flag_t is None:
                        first_flag_t = t_now
                # Hold the last swing we could actually measure.  A flagged car
                # stopped at a light reports ~0 m, and blanking the number then
                # makes a live warning look like a stale one.
                if not n_flagged:
                    weave_amp = 0.0
                elif weave_amp_now > 0:
                    weave_amp = weave_amp_now
                t_haz = time.time() - _t

                if show_gt and k % args.gt_every == 0:
                    gt_cache = make_bev_gt(world, ego, lane_pts)

                if recon is not None and k % args.recon_every == 0:
                    _tr = time.time()
                    _cams = (("front", f_img, 0), ("rear", r_img, 1))
                    if args.recon_cams != "both":
                        _cams = tuple(c for c in _cams if c[0] == args.recon_cams)
                    for cam, img_bgr, di in _cams:
                        if args.recon_source == "depth":
                            src = depth_png_to_metres(d_f_img if cam == "front" else d_r_img)
                        else:
                            src = exp_d[di]
                        pts, col = backproject(
                            src, img_bgr, calib["cameras"][cam]["K"],
                            calib["cameras"][cam]["T_ego_cam"],
                            dmin=2.5, dmax=35.0, stride=args.recon_stride)
                        recon.add(pts, col / 255.0, ex, ey, yaw)
                    t_recon = time.time() - _tr

                lvl_front, lvl_rear = worst_per_side(obstacles)
                alerter.set(lvl_front, lvl_rear)
                if events is not None:
                    events.update(k, obstacles, kmh_last)

                if k % 5 == 0:
                    front_m, front_kmh = nearest_ahead(ego, npcs)
                status, status_col = ego_status(ego, kmh_last, front_m, front_kmh)

                # Only a genuine stall counts towards the unstick timer.  Without
                # this a long red light would teleport the car away from a junction
                # it was legitimately waiting at.
                if args.unstick and autopilot and status != "RED LIGHT":
                    if kmh_last < 0.5:
                        stuck_frames += 1
                        if stuck_frames >= args.unstick:
                            # With a scenario running, come down somewhere it can
                            # still be staged rather than at a random junction.
                            sp = random.choice(candidates[:20] if scenario else spawn_points)
                            ego.set_transform(sp)
                            if scenario is not None:
                                # The staged actors are back where the ego used to
                                # be, which is now hundreds of metres away.
                                scenario.cleanup()
                                scenario.setup(world, bp_lib, ego, tm)
                            # The tracker keeps obstacles in world coordinates, so a
                            # teleport would look like every object moving hundreds of
                            # metres in one frame -- and TTC would go wild.  Start clean.
                            tracker = HZ.Tracker(
                                whist_len=max(64, int(6.0 * args.fps)),
                                max_miss=max(3, int(args.track_miss * args.fps)))
                            hold = HZ.LevelHold(
                                fall_frames=max(1, round(args.warn_hold * args.fps)))
                            phold = HZ.LevelHold(
                                fall_frames=max(1, round(args.predict_hold * args.fps)),
                                rise_frames=max(1, round(0.3 * args.fps)))
                            watch = TP.WatchList(
                                arm=max(1, round(args.watch_arm * args.fps)),
                                release=int(round(args.watch_release * args.fps)))
                            n_flagged = 0
                            if recon is not None:
                                recon.reset()
                            stuck_frames = 0
                            print(f"[{k:5d}] ego stuck for {args.unstick} frames -- "
                                  f"moved to a new spawn point", flush=True)
                    else:
                        stuck_frames = 0
                elif status == "RED LIGHT":
                    stuck_frames = 0

                if args.profile and k % args.profile == 0:
                    # "is the traffic actually moving" -- a stopped ego tells you
                    # nothing on its own; it matters whether it is alone in being
                    # stuck or the whole town is.
                    speeds = []
                    for v in npcs:
                        try:
                            vel = v.get_velocity()
                            speeds.append(3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
                        except Exception:
                            pass
                    speeds = np.array(speeds) if speeds else np.zeros(1)
                    moving = int((speeds > 1.0).sum())
                    # Why is the ego not moving?  Guessing from speed alone confused
                    # walkers, red lights and being physically blocked.  The control
                    # the autopilot is actually issuing separates them: brake ~1 with
                    # no light means it is yielding to something (a walker or a car),
                    # throttle > 0 at 0 km/h means it is stuck against geometry.
                    c = ego.get_control()
                    try:
                        at_light = ego.is_at_traffic_light()
                        tl = ego.get_traffic_light_state()
                        light = f"{'AT' if at_light else '--'}:{str(tl).split('.')[-1][:3]}"
                    except Exception:
                        light = "?"
                    # What is in front of it?  A green light plus full brake means the
                    # autopilot is yielding, and the only way to tell what to is to
                    # look for the nearest actor ahead and whether that one is moving.
                    et = ego.get_transform()
                    fwd = et.get_forward_vector()
                    near = []
                    for v in npcs:
                        try:
                            dl = v.get_transform().location - et.location
                            d = math.sqrt(dl.x**2 + dl.y**2)
                            ahead = (dl.x * fwd.x + dl.y * fwd.y) > 0
                            vv = v.get_velocity()
                            near.append((d, ahead, 3.6 * math.sqrt(vv.x**2 + vv.y**2)))
                        except Exception:
                            pass
                    aheads = sorted([n for n in near if n[1]])
                    front = (f"{aheads[0][0]:5.1f} m @{aheads[0][2]:4.1f} km/h"
                             if aheads else "   none")
                    print(f"[{k:5d}] front-vehicle {front} | ", end="")
                    print(f"[{k:5d}] tick {t_tick*1000:6.1f}  read {t_read*1000:6.1f}  "
                          f"model {t_model*1000:6.1f}  hazard {t_haz*1000:5.1f}  "
                          f"compose {t_comp*1000:5.1f} ms | "
                          f"ego {kmh_last:5.1f} km/h thr {c.throttle:.2f} brk {c.brake:.2f} "
                          f"light {light} | NPC moving {moving}/{len(speeds)} "
                          f"median {np.median(speeds):5.1f} max {speeds.max():5.1f} km/h",
                          flush=True)

                k += 1

            # ---- compose ----
            _tc = time.time()
            P = args.panel
            cam_w = int(P * 0.95)
            cam_h = int(cam_w * args.img_h / args.img_w)
            fv = cv2.resize(f_img, (cam_w, cam_h))
            rv = cv2.resize(r_img, (cam_w, cam_h))
            for (x_, y_, conf, cam, box) in peds:
                img = fv if cam == "front" else rv
                sx, sy = cam_w / args.img_w, cam_h / args.img_h
                x1, y1, x2, y2, _ = box
                cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                              (0, 140, 255), 2)
            cv2.putText(fv, "FRONT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(rv, "REAR", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cols = [titled(np.vstack([fv, rv]), "CAMERA (front / rear)")]
            cols.append(titled(cv2.resize(bev_panel(lab, obstacles if show_hazard else []),
                                          (P, P)), "PREDICTION + HAZARD"))
            if recon is not None:
                _rt = ("3D RECONSTRUCTION (from RGB)" if args.recon_source == "model"
                       else "3D RECONSTRUCTION (depth camera)")
                if recon_img is None or k % max(1, args.recon_render_every) == 0:
                    recon_img = recon.render(ex, ey, yaw)
                cols.append(titled(cv2.resize(recon_img, (P, P)), _rt))
            if show_gt:
                cols.append(titled(cv2.resize(render_bev_prof(gt_cache, out_size=P, dim_blind=blind),
                                              (P, P)), "GROUND TRUTH"))

            hmax = max(c.shape[0] for c in cols)
            cols = [np.vstack([c, np.full((hmax - c.shape[0], c.shape[1], 3), 24, np.uint8)])
                    if c.shape[0] < hmax else c for c in cols]
            canvas = np.hstack(cols)

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
            bar = np.full((30, canvas.shape[1], 3), 18, np.uint8)
            v = ego.get_velocity()
            kmh = 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            drive = ((f"AUTO {status}" if autopilot else
                     f"MANUAL thr {throttle:.2f} brk {brake:.2f} str {steer:+.2f}")
                     + (" [R]" if reverse else ""))
            info = (f"{fps_ema:5.1f} FPS   model {t_model*1000:5.1f} ms" +
                    (f"   ped {t_ped*1000:5.1f} ms" if args.peds else "") +
                    (f"   recon {t_recon*1000:5.1f} ms" if args.recon else "") +
                    f"   {kmh:5.1f} km/h   {drive}   tracks {len(obstacles)}   frame {k}" +
                    (f"   DANGEROUS VEHICLE x{n_flagged} (SDLP {weave_amp:.2f} m)"
                     if n_flagged else "") +
                    ("   [PAUSED]" if paused else ""))
            cv2.putText(bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # "● LIVE" at the right end of the bar.  The blink is driven by the frame
            # counter rather than the wall clock so that a recording blinks at the
            # same rate it did on screen -- roughly 1 Hz at 10 fps.  It goes solid
            # while paused, because a blinking LIVE on a frozen picture is a lie.
            bw = bar.shape[1]
            lit = paused or (k // max(args.fps // 2, 1)) % 2 == 0
            cv2.circle(bar, (bw - 78, 15), 7, (60, 60, 235) if lit else (55, 55, 75), -1,
                       cv2.LINE_AA)
            cv2.putText(bar, "LIVE", (bw - 62, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (245, 245, 245), 2, cv2.LINE_AA)

            # Sound state, left of the LIVE badge.  Worth showing: a muted demo that
            # never beeps looks identical to one whose warnings are not firing.
            if alerter.enabled:
                snd, sc = ("MUTED", (110, 110, 110)) if alerter.muted else ("SOUND", (150, 220, 150))
            else:
                snd, sc = ("NO SND", (90, 90, 90))
            cv2.putText(bar, snd, (bw - 152, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1,
                        cv2.LINE_AA)

            if scenario is not None:
                # Its own strip rather than more text in the status bar: the
                # audience needs to know what they are about to be shown, and
                # that line is already at the width where nobody reads it.
                sb = np.full((26, canvas.shape[1], 3), 44, np.uint8)
                cv2.putText(sb, "%s   |   %s   |   staged x%d"
                            % (scenario.banner or scenario.name, scenario.hint,
                               scenario.armed),
                            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (80, 210, 250), 1, cv2.LINE_AA)
                canvas = np.vstack([sb, canvas])

            canvas = np.vstack([bar, canvas])
            t_comp = time.time() - _tc
            kmh_last = kmh

            if args.record and not record_done:
                if writer is None:
                    free_gb = 0.0
                    try:
                        import shutil
                        free_gb = shutil.disk_usage(
                            os.path.dirname(os.path.abspath(args.record))).free / 2**30
                    except Exception:
                        free_gb = -1.0
                    if 0 <= free_gb < MIN_FREE_GB:
                        # The 44-byte mp4 files from 2026-09-09 were written on a
                        # full disk: OpenCV opens the container, every write then
                        # fails silently, and the demo looks fine until you go to
                        # play the recording back.  Say so now instead.
                        print("!! only %.1f GB free where the recording would go -- "
                              "NOT recording (need %.0f GB). The demo carries on."
                              % (free_gb, MIN_FREE_GB))
                        args.record = None
                    else:
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"FMP4"),
                            args.fps, (canvas.shape[1], canvas.shape[0]))
                        if not writer.isOpened():
                            # write() on a writer that never opened is a silent
                            # no-op, so without this check a failed recording is
                            # indistinguishable from a good one until afterwards.
                            print("!! could not open the recording %s -- "
                                  "the demo carries on without it" % args.record)
                            writer = None
                            args.record = None
                        else:
                            print("recording to %s (capped at %.0f s)"
                                  % (args.record, args.record_secs))
                if writer is not None:
                    writer.write(canvas)
                    rec_frames += 1
                    if record_max and rec_frames >= record_max:
                        # Cap the RECORDING, not the demo.  An uncapped --record
                        # auto is what produced a 1.42 GB file from one rehearsal
                        # and a 13.6-hour run that filled the drive; stopping the
                        # demo instead would mean the presenter loses the floor
                        # at a fixed time, which is worse.
                        writer.release()
                        writer = None
                        record_done = True
                        print("recording capped at %.0f s (%d frames): %s -- "
                              "the demo carries on" % (args.record_secs, rec_frames,
                                                       args.record))

            if not args.no_display:
                cv2.imshow(WIN, fit_to_screen(canvas, args.max_width,
                                              fullscreen=args.fullscreen))
                key = cv2.waitKey(1) & 0xFF
                # Clicking the window's X does not stop anything by itself -- the next
                # imshow just builds the window again, so the demo looks unclosable.
                # Treat "the window is gone" as a quit.
                try:
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        print("window closed -- shutting down")
                        break
                except cv2.error:
                    break
                if key in (ord("q"), 27):          # q or Esc
                    break
                if key == ord("g"):
                    show_gt = not show_gt
                    if show_gt and lane_pts is None:
                        lane_pts = build_lane_points(world.get_map())
                if key in (ord("m"), ord("M")):
                    if alerter.enabled:
                        print("sound:", "muted" if alerter.toggle_mute() else "on")
                if key == ord("h"):
                    show_hazard = not show_hazard
                if key in (ord("f"), ord("F")):
                    args.fullscreen = not args.fullscreen
                    set_window_mode(WIN, args.fullscreen)
                    cv2.imshow(WIN, fit_to_screen(canvas, args.max_width,
                                                  fullscreen=args.fullscreen))
                if key in (ord("c"), ord("C")):
                    # A still of the whole window, full resolution, for the
                    # slides -- taking it here beats cropping a screenshot of a
                    # window that has already been shrunk to fit the screen.
                    shot = "%s_shot%02d.png" % (stem, shots)
                    d = os.path.dirname(shot)
                    if d and not os.path.isdir(d):
                        os.makedirs(d, exist_ok=True)
                    cv2.imwrite(shot, canvas)
                    shots += 1
                    print("saved", shot)
                if key == ord("p"):
                    paused = not paused
                if key == 9:                                  # Tab
                    autopilot = not autopilot
                    ego.set_autopilot(autopilot, tm.get_port())
                    steer = throttle = brake = 0.0
                    print("autopilot:", autopilot)

                # driving keys -- ramp while held, decay when not
                if key in (ord("w"), ord("W")):
                    throttle, brake = min(throttle + 0.35, 1.0), 0.0
                elif key in (ord("s"), ord("S")):
                    brake, throttle = min(brake + 0.5, 1.0), 0.0
                else:
                    throttle *= 0.85
                    brake *= 0.6
                if key in (ord("a"), ord("A")):
                    steer = max(steer - 0.25, -1.0)
                elif key in (ord("d"), ord("D")):
                    steer = min(steer + 0.25, 1.0)
                else:
                    steer *= 0.7                              # self-centring
                if key == ord(" "):
                    brake, throttle = 1.0, 0.0
                if key in (ord("r"), ord("R")):
                    reverse = not reverse
                    print("reverse:", reverse)

                # The wheel reports where the controls ARE, so it replaces the
                # ramp-and-decay above outright rather than adding to it.  That
                # loop exists only because cv2's waitKey reports key-down and
                # never key-up; an axis has no such problem.
                if wheel is not None and not autopilot:
                    w = wheel.poll()
                    if w.live:
                        steer, throttle, brake = w.steer, w.throttle, w.brake
                        if wheel.handbrake_button and w.pressed(wheel.handbrake_button):
                            brake, throttle = 1.0, 0.0
                        if wheel.reverse_button:
                            # Toggle on the press, not once per frame the
                            # button happens to be held.
                            down = w.pressed(wheel.reverse_button)
                            if down and not wheel_rev_held:
                                reverse = not reverse
                                print("reverse:", reverse)
                            wheel_rev_held = down
                    elif not wheel_warned:
                        print("wheel has not reported yet -- turn it slightly")
                        wheel_warned = True

            if args.max_frames and k >= args.max_frames:
                break

        if writer is not None:
            writer.release()
            writer = None
        if rec_frames:
            print("recorded %d frames (%.0f s): %s"
                  % (rec_frames, rec_frames / float(args.fps), record_path))

    finally:
        try:
            alerter.stop()
        except Exception:
            pass
        try:
            if events is not None:
                try:
                    events.meta["watch_flagged_s"] = round(
                        flagged_frames / float(args.fps), 1)
                    events.meta["watch_first_flag_s"] = first_flag_t
                except Exception:
                    pass
                events.close(json_path=os.path.splitext(args.events)[0]
                             .replace("_events", "") + "_summary.json")
        except Exception as exc:
            print("could not write the event summary:", exc)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if ped_worker is not None:
                ped_worker.close()
        except Exception:
            pass
        client.set_timeout(20.0)
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(old_settings)
        except Exception:
            pass
        # The staged actors are deliberately NOT destroyed here.  Destroying a
        # traffic-manager vehicle on the way out crashes this CARLA build's
        # client outright -- 0xC0000409, no traceback, measured both before and
        # after restoring the world settings on 2026-09-08 -- which is the same
        # reason the NPC vehicles above are left behind.  The clean_world sweep
        # at the next startup collects all of them.
        for s in sensors:
            try:
                s.destroy()
            except Exception:
                pass
        try:
            ego.destroy()
        except Exception:
            pass
        print("cleaned up")


if __name__ == "__main__":
    main()
