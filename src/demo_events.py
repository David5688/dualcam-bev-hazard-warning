# -*- coding: utf-8 -*-
"""Warning-event log and end-of-run summary for live_demo.py.

The demo already shows a warning; what it could not do was say afterwards how
many there were.  "The system warns you about cars behind" is an assertion; "a
4 min 12 s drive raised 14 rear warnings, the earliest 3.4 s before closest
approach, with no warning while the road was clear for 71% of the run" is a
result -- and it is the kind of line a slide needs.

Two files come out of a run:

  <stem>_events.csv     one row per warning transition, per side
  <stem>_summary.json   the aggregate, also printed to the console at exit

Only transitions are logged, not frames.  A DANGER that holds for four seconds
is one event with a duration, not forty rows; that is what makes the CSV
readable, and counting held frames separately is what the warn ratio is for.
"""
import io
import json
import math
import os
import time


ORDER = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
LEVELS = ("CAUTION", "DANGER")
CLASS_NAME = {1: "vehicle", 2: "pedestrian"}


def worst_on_side(obstacles, front):
    """The obstacle driving the warning on one side, or None.

    Highest level wins; ties are broken by the shorter TTC, then by range, so
    the logged row is about the thing the driver would actually hit first.
    Matches the front/rear split hazard_bev.bev_panel draws its banners with
    (x >= 0 is ahead), so the CSV and the picture never disagree.
    """
    best = None
    for o in obstacles:
        x, y, cls, level, col, ttc = o[:6]
        why = o[8] if len(o) > 8 else ""
        if (x >= 0) != front:
            continue
        if level == "SAFE":
            continue
        rng = math.hypot(x, y)
        t = ttc if (ttc is not None and math.isfinite(ttc)) else float("inf")
        key = (ORDER[level], -t, -rng)
        if best is None or key > best[0]:
            best = (key, {"level": level, "cls": int(cls), "range": rng, "ttc": t,
                          "why": why})
    return best[1] if best else None


class _SideState(object):
    """Warning state for one side of the vehicle."""

    def __init__(self, side):
        self.side = side
        self.level = "SAFE"
        self.since_t = 0.0
        self.since_k = 0
        self.peak = None          # worst detail seen during the current warning
        self.counts = {"CAUTION": 0, "DANGER": 0}
        self.warn_frames = 0
        self.min_ttc = float("inf")
        self.min_range = float("inf")


class EventLog(object):
    """Collects warning transitions and the numbers that summarise a run."""

    # reason is appended, not inserted: every column a slide or a script already
    # reads keeps its position.  It is "" for a warning the radial TTC raised and
    # one of INTRUDE / LANE / WEAVE when the trajectory predictor is what saw it
    # -- the only way to tell, afterwards, which half of the system earned its
    # place on a given run.
    HEADER = ("t_s,frame,side,event,level,class,range_m,ttc_s,ego_kmh,"
              "held_s,scenario,reason\n")

    def __init__(self, path, fps=10, scenario="none", meta=None):
        self.path = path
        self.fps = max(int(fps), 1)
        self.scenario = scenario or "none"
        self.meta = dict(meta or {})
        self.sides = {"front": _SideState("front"), "rear": _SideState("rear")}
        self.rows = []
        self.frames = 0
        self.t_start = time.time()
        self._fh = None
        if path:
            d = os.path.dirname(os.path.abspath(path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            self._fh = io.open(path, "w", encoding="utf-8", newline="")
            self._fh.write(self.HEADER)

    # -- per frame -----------------------------------------------------------
    def update(self, k, obstacles, kmh):
        """Fold one frame in.  Returns nothing; call it every frame."""
        self.frames = max(self.frames, k + 1)
        t = k / float(self.fps)
        for front in (True, False):
            side = "front" if front else "rear"
            st = self.sides[side]
            cur = worst_on_side(obstacles, front)
            level = cur["level"] if cur else "SAFE"

            if cur:
                st.warn_frames += 1
                if cur["ttc"] < st.min_ttc:
                    st.min_ttc = cur["ttc"]
                if cur["range"] < st.min_range:
                    st.min_range = cur["range"]
                if st.peak is None or ORDER[cur["level"]] > ORDER[st.peak["level"]] \
                        or cur["ttc"] < st.peak["ttc"]:
                    st.peak = dict(cur)

            if ORDER[level] > ORDER[st.level]:
                # SAFE -> CAUTION -> DANGER: a new warning, or an escalation.
                st.counts[level] += 1
                self._row(t, k, side, "RAISE" if st.level == "SAFE" else "UPGRADE",
                          cur, kmh, 0.0)
                if st.level == "SAFE":
                    st.since_t, st.since_k = t, k
                st.level = level
            elif ORDER[level] < ORDER[st.level]:
                held = t - st.since_t
                if level == "SAFE":
                    self._row(t, k, side, "CLEAR", st.peak, kmh, held)
                    st.peak = None
                else:
                    self._row(t, k, side, "DOWNGRADE", cur, kmh, held)
                st.level = level

    def _row(self, t, k, side, event, det, kmh, held):
        det = det or {}
        ttc = det.get("ttc", float("inf"))
        row = {
            "t_s": round(t, 2), "frame": int(k), "side": side, "event": event,
            "level": det.get("level", "SAFE"),
            "class": CLASS_NAME.get(det.get("cls"), ""),
            "range_m": round(det.get("range", float("nan")), 2)
            if det.get("range") is not None else "",
            "ttc_s": round(ttc, 2) if math.isfinite(ttc) else "",
            "ego_kmh": round(kmh, 1), "held_s": round(held, 2),
            "scenario": self.scenario, "reason": det.get("why", "") or "",
        }
        self.rows.append(row)
        if self._fh:
            self._fh.write("%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                row["t_s"], row["frame"], row["side"], row["event"], row["level"],
                row["class"], row["range_m"], row["ttc_s"], row["ego_kmh"],
                row["held_s"], row["scenario"], row["reason"]))
            self._fh.flush()          # a demo that crashes should still leave its log

    # -- end of run ----------------------------------------------------------
    def summary(self):
        wall = time.time() - self.t_start
        frames = max(self.frames, 1)
        out = {
            "scenario": self.scenario,
            "frames": self.frames,
            "sim_seconds": round(self.frames / float(self.fps), 1),
            "wall_seconds": round(wall, 1),
            "mean_fps": round(self.frames / wall, 2) if wall > 0 else 0.0,
            "events": len([r for r in self.rows if r["event"] in ("RAISE", "UPGRADE")]),
            "sides": {},
        }
        out.update(self.meta)
        for side, st in self.sides.items():
            out["sides"][side] = {
                "caution": st.counts["CAUTION"],
                "danger": st.counts["DANGER"],
                "warn_frames": st.warn_frames,
                "warn_ratio": round(st.warn_frames / float(frames), 3),
                "min_ttc_s": round(st.min_ttc, 2) if math.isfinite(st.min_ttc) else None,
                "min_range_m": round(st.min_range, 2) if math.isfinite(st.min_range) else None,
            }
        return out

    def close(self, json_path=None, quiet=False):
        if self._fh:
            self._fh.close()
            self._fh = None
        s = self.summary()
        if json_path:
            d = os.path.dirname(os.path.abspath(json_path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with io.open(json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(s, indent=2, ensure_ascii=False))
        if not quiet:
            self.report(s, json_path)
        return s

    def report(self, s=None, json_path=None):
        s = s or self.summary()
        line = "-" * 62
        print("")
        print(line)
        print("  RUN SUMMARY   scenario: %s" % s["scenario"])
        print(line)
        print("  %d frames / %.1f s simulated in %.1f s wall  (%.2f FPS)"
              % (s["frames"], s["sim_seconds"], s["wall_seconds"], s["mean_fps"]))
        print("")
        print("  %-6s %8s %8s %10s %10s %10s"
              % ("side", "caution", "danger", "warn %", "min TTC", "min range"))
        for side in ("front", "rear"):
            d = s["sides"][side]
            print("  %-6s %8d %8d %9.1f%% %9s %9s"
                  % (side, d["caution"], d["danger"], 100.0 * d["warn_ratio"],
                     ("%.2f s" % d["min_ttc_s"]) if d["min_ttc_s"] is not None else "-",
                     ("%.1f m" % d["min_range_m"]) if d["min_range_m"] is not None else "-"))
        print("")
        if self.path:
            print("  events : %s" % self.path)
        if json_path:
            print("  summary: %s" % json_path)
        print(line)
