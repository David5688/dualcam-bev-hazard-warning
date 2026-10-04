# -*- coding: utf-8 -*-
"""Audit the v4 collection and write dataset_provenance.md (規劃 §9).

Reports, per dataset: frame count, per-prefix breakdown (filtered pass vs the
negatives pass), manifest stats, actor config, weather-free provenance, and the
class content actually present in the BEV ground truth.

The point of the content stats is to check the two things the plan is explicit
about and that a manifest cannot tell you:

  * negatives -- §4 wants 15-25% low/empty-actor frames.  The manifest's
    kept_negative only counts frames the *filter* rejected and kept anyway, so
    it undercounts once a second unfiltered pass is added.  Here we count actual
    empty/low frames from the GT rasters.
  * pedestrians -- §14's open question.  Measured as GT pixels per frame.

Usage:
  python v4_report.py [--data ..\\data\\v4] [--stride 10] [--out ..\\data\\v4\\dataset_provenance.md]

--stride subsamples frames for the pixel statistics (counts are always exact).
"""
import os
import re
import glob
import json
import argparse
from collections import Counter, OrderedDict

import cv2
import numpy as np

CLASSES = ["background", "vehicle", "pedestrian", "road", "sidewalk", "lane_marking"]
VEHICLE, PED = 1, 2

# A car footprint is ~128 px at 0.25 m/cell (see collector selftest [2]), so
# "< 260 px" is a deliberately generous stand-in for "fewer than two vehicles".
TWO_VEHICLES_PX = 260


def scan_dataset(root, stride):
    bev = sorted(glob.glob(os.path.join(root, "bev_gt", "*.png")))
    if not bev:
        return None

    prefixes = Counter()
    for f in bev:
        m = re.match(r"^([a-z]+\d+_)", os.path.basename(f))
        prefixes[m.group(1) if m else "(none)"] += 1

    sub = bev[::stride] if stride > 1 else bev
    totals = np.zeros(len(CLASSES), dtype=np.int64)
    n_empty_veh = 0      # no vehicle pixels at all
    n_low_veh = 0        # fewer pixels than two average cars
    n_has_ped = 0
    for f in sub:
        a = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        c = np.bincount(a.ravel(), minlength=len(CLASSES))[:len(CLASSES)]
        totals += c
        if c[VEHICLE] == 0:
            n_empty_veh += 1
        if c[VEHICLE] < TWO_VEHICLES_PX:
            n_low_veh += 1
        if c[PED] > 0:
            n_has_ped += 1

    n = max(len(sub), 1)
    stats = OrderedDict()
    stats["frames"] = len(bev)
    stats["prefixes"] = dict(prefixes)
    stats["sampled"] = len(sub)
    stats["px_per_frame"] = {name: round(float(totals[i]) / n, 1)
                             for i, name in enumerate(CLASSES)}
    stats["pct_no_vehicle"] = round(100.0 * n_empty_veh / n, 1)
    stats["pct_low_vehicle"] = round(100.0 * n_low_veh / n, 1)
    stats["pct_any_pedestrian"] = round(100.0 * n_has_ped / n, 1)

    # sibling directories must agree with bev_gt or the set is not usable
    sib = OrderedDict()
    for d in ("front_rgb", "rear_rgb", "front_depth", "rear_depth", "bev_gt_color"):
        p = os.path.join(root, d)
        sib[d] = len(glob.glob(os.path.join(p, "*.png"))) if os.path.isdir(p) else 0
    stats["subdirs"] = sib
    stats["subdirs_consistent"] = all(v == len(bev) for k, v in sib.items()
                                      if k != "bev_gt_color" or v > 0)

    mpath = os.path.join(root, "manifest.json")
    if os.path.isfile(mpath):
        with open(mpath, encoding="utf-8") as fh:
            m = json.load(fh)
        # a manifest written mid-collection has null sections, so never chain .get
        cap = m.get("capture") or {}
        stats["manifest"] = {
            "complete": m.get("complete"),
            "stats": m.get("stats"),
            "actors": {k: v for k, v in (m.get("actors") or {}).items()
                       if k != "ego_spawn"},
            "capture": {k: cap.get(k) for k in ("cam_w", "cam_h", "fov", "fps", "depth")},
            "content_filter": (m.get("content_filter") or {}).get("enabled"),
            "carla": m.get("carla_version"),
            "digest": ((m.get("integrity") or {}).get("digest") or "")[:16],
        }
    else:
        stats["manifest"] = None
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("..", "data", "v4"))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(args.data, "dataset_provenance.md")
    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isdir(d))

    rows, details = [], []
    grand = 0
    for d in dirs:
        s = scan_dataset(d, args.stride)
        if s is None:
            continue
        name = os.path.basename(d)
        grand += s["frames"]
        man = s["manifest"] or {}
        act = man.get("actors", {}) or {}
        mst = man.get("stats", {}) or {}
        rows.append((name, s["frames"],
                     "on" if man.get("content_filter") else "off",
                     act.get("near_ego", "?"),
                     mst.get("keep_rate", "-"),
                     s["pct_low_vehicle"], s["px_per_frame"]["pedestrian"],
                     "ok" if s["subdirs_consistent"] else "MISMATCH"))
        details.append((name, s))

    lines = []
    lines.append("# v4 dataset provenance\n")
    lines.append("Generated by `src/v4_report.py`. Frame counts are exact; "
                 "pixel statistics are sampled every %d frames.\n" % args.stride)
    lines.append("\n## Summary\n")
    lines.append("| dataset | frames | filter | near_ego | keep_rate | "
                 "low-vehicle frames | ped px/frame | subdirs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %d | %s | %s | %s | %.1f%% | %.1f | %s |" %
                     (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    lines.append("\n**Total frames: %d**\n" % grand)
    lines.append("\n`low-vehicle frames` = frames whose GT holds fewer vehicle pixels "
                 "than two average cars (%d px). This is the honest read on §4's "
                 "negative-example requirement (15-25%%); `keep_rate` and the manifest's "
                 "`kept_negative` only describe the filtered pass.\n" % TWO_VEHICLES_PX)

    lines.append("\n## Per dataset\n")
    for name, s in details:
        lines.append("\n### %s\n" % name)
        lines.append("- frames: **%d** %s" % (s["frames"], s["prefixes"]))
        lines.append("- subdirs: %s%s" % (s["subdirs"],
                     "" if s["subdirs_consistent"] else "  **<-- MISMATCH**"))
        lines.append("- px/frame: %s" % s["px_per_frame"])
        lines.append("- no vehicle at all: %.1f%% of frames; fewer than two cars: %.1f%%; "
                     "any pedestrian: %.1f%%"
                     % (s["pct_no_vehicle"], s["pct_low_vehicle"], s["pct_any_pedestrian"]))
        m = s["manifest"]
        if m is None:
            lines.append("- manifest: **missing**")
        else:
            lines.append("- manifest.complete: %s (note: this flag tolerates missing "
                         "image files -- trust the subdir counts above)" % m["complete"])
            lines.append("- manifest.stats: %s" % m["stats"])
            lines.append("- actors: %s" % m["actors"])
            lines.append("- capture: %s" % m["capture"])
            lines.append("- CARLA: %s, digest %s" % (m["carla"], m["digest"]))

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
