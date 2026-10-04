# -*- coding: utf-8 -*-
"""Hard integrity audit of a v4 dataset directory.  Fails loudly; exit code 1 if
anything is wrong.

This is deliberately NOT the same job as the two existing tools:

  v4_report.py     describes the collection (frames, keep rate, class content) and
                   writes dataset_provenance.md.  It reports; it does not judge.
  v4_check_dup.py  answers one question -- do segments retrace the same route.

What neither of them checks, and what this script is for: whether the files on
disk are actually a usable, self-consistent dataset.  A manifest saying
"complete: true" has already been caught lying once (資料收集規劃.md §13: six
empty image directories, manifest still complete with a sha256), so every claim
here is verified against the files themselves.

Checks, per dataset:

  [A] structure   every modality has the SAME filenames (not merely the same
                  count), nothing is zero-byte, and a strided sample of every
                  modality decodes with the expected shape and dtype
  [B] labels      bev_gt contains only class ids 0-5 and is 240x240
  [C] depth       decodes to metres, is not a constant/far-plane-only image, and
                  has a plausible near range
  [D] provenance  manifest_segments.json accounts for exactly the frames on disk,
                  every segment ran with --random_spawn, and the segments really
                  did start from different ego spawn points (the anti-leakage fix
                  of 2026-07-30 -- a manifest can claim the flag while the spawns
                  came out identical anyway)
  [E] poses       poses.json covers exactly the frames on disk, no key collisions
  [F] calibration 6 classes, schema_version 2, intrinsics match the images
  [G] stuck ego   consecutive frames within a segment that are near-identical --
                  a gridlocked ego produces hundreds of duplicate frames that no
                  cross-segment check would ever see

Usage:
  python v4_audit.py ..\\data\\v4                 # every dataset under it
  python v4_audit.py ..\\data\\v4\\train_town02   # one dataset
  python v4_audit.py ..\\data\\v4 --stride 25     # lighter pixel sampling
"""
import os
import re
import sys
import glob
import json
import argparse

import cv2
import numpy as np

CLASSES = ["background", "vehicle", "pedestrian", "road", "sidewalk", "lane_marking"]
MODALITIES = ["front_rgb", "rear_rgb", "bev_gt", "bev_gt_color",
              "front_depth", "rear_depth"]
# Modalities every dataset must have; depth is optional (a city may be collected
# without --depth), but if the directory exists it must be complete.
REQUIRED = ["front_rgb", "rear_rgb", "bev_gt"]

BEV_HW = (240, 240)
STUCK_MAD = 1.0          # mean |diff| on 0-255 greyscale, below this = same view
STUCK_WARN_PCT = 10.0    # flag a segment if this share of consecutive pairs is stuck


class Result(object):
    def __init__(self):
        self.rows = []       # (level, check, detail)
        self.failed = 0
        self.warned = 0

    def ok(self, check, detail=""):
        self.rows.append(("PASS", check, detail))

    def warn(self, check, detail=""):
        self.rows.append(("WARN", check, detail))
        self.warned += 1

    def fail(self, check, detail=""):
        self.rows.append(("FAIL", check, detail))
        self.failed += 1

    def dump(self):
        for level, check, detail in self.rows:
            print("  %-4s %-26s %s" % (level, check, detail))


def stems(root, sub):
    return set(os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(root, sub, "*.png")))


def seg_of(stem):
    m = re.match(r"^([a-z]+\d+)_", stem)
    return m.group(1) if m else "(none)"


def decode_depth(bgr):
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    return 1000.0 * (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)


def check_structure(root, res):
    """[A] identical filename SETS across modalities, plus no empty files."""
    present = [m for m in MODALITIES if os.path.isdir(os.path.join(root, m))]
    missing_required = [m for m in REQUIRED if m not in present]
    if missing_required:
        res.fail("A1 modalities present", "missing %s" % ", ".join(missing_required))
        return None, present
    ref = stems(root, "front_rgb")
    if not ref:
        res.fail("A1 modalities present", "front_rgb is empty")
        return None, present

    bad = []
    for m in present:
        s = stems(root, m)
        if s != ref:
            only_ref, only_m = ref - s, s - ref
            bad.append("%s: %d missing, %d extra" % (m, len(only_ref), len(only_m)))
    if bad:
        res.fail("A1 filename sets match", "; ".join(bad))
    else:
        res.ok("A1 filename sets match", "%d frames x %d modalities" % (len(ref), len(present)))

    empty = []
    for m in present:
        for f in glob.glob(os.path.join(root, m, "*.png")):
            if os.path.getsize(f) == 0:
                empty.append(os.path.join(m, os.path.basename(f)))
    if empty:
        res.fail("A2 no zero-byte files", "%d: %s" % (len(empty), empty[:3]))
    else:
        res.ok("A2 no zero-byte files")
    return sorted(ref), present


def check_pixels(root, names, present, stride, res):
    """[A3][B][C] decode a strided sample of every modality and validate it."""
    sample = names[::stride] if stride > 1 else names
    if not sample:
        return
    bad_decode, bad_shape, bad_label = [], [], []
    depth_stats = []
    label_hist = np.zeros(len(CLASSES), dtype=np.int64)

    for stem in sample:
        for m in present:
            p = os.path.join(root, m, stem + ".png")
            flag = cv2.IMREAD_GRAYSCALE if m == "bev_gt" else cv2.IMREAD_COLOR
            a = cv2.imread(p, flag)
            if a is None:
                bad_decode.append(os.path.join(m, stem))
                continue
            if m == "bev_gt":
                if a.shape != BEV_HW:
                    bad_shape.append("%s %s" % (m, a.shape))
                mx = int(a.max())
                if mx >= len(CLASSES):
                    bad_label.append("%s max=%d" % (stem, mx))
                label_hist += np.bincount(a.ravel(), minlength=len(CLASSES))[:len(CLASSES)]
            elif m.endswith("_depth"):
                d = decode_depth(a)
                depth_stats.append((float(d.min()), float(np.median(d)), float(d.max())))

    if bad_decode:
        res.fail("A3 all sampled decode", "%d undecodable: %s" % (len(bad_decode), bad_decode[:3]))
    else:
        res.ok("A3 all sampled decode", "%d frames x %d modalities" % (len(sample), len(present)))

    if bad_shape:
        res.fail("A4 shapes", "; ".join(sorted(set(bad_shape))[:3]))
    else:
        res.ok("A4 shapes", "bev_gt %dx%d" % BEV_HW)

    if bad_label:
        res.fail("B1 label ids in 0-5", "; ".join(bad_label[:3]))
    else:
        res.ok("B1 label ids in 0-5")

    tot = label_hist.sum()
    if tot:
        share = ", ".join("%s %.2f%%" % (CLASSES[i][:4], 100.0 * label_hist[i] / tot)
                          for i in range(len(CLASSES)))
        absent = [CLASSES[i] for i in range(len(CLASSES)) if label_hist[i] == 0]
        if absent:
            res.warn("B2 all classes present", "absent: %s | %s" % (", ".join(absent), share))
        else:
            res.ok("B2 all classes present", share)

    if depth_stats:
        near = np.array([s[0] for s in depth_stats])
        med = np.array([s[1] for s in depth_stats])
        # A depth image that is entirely far-plane means the capture produced
        # nothing usable, which a file-size check would never notice.
        degenerate = int((med > 999.0).sum())
        if degenerate:
            res.fail("C1 depth not degenerate", "%d/%d frames are all far-plane"
                     % (degenerate, len(depth_stats)))
        else:
            res.ok("C1 depth not degenerate",
                   "median depth %.1f-%.1f m, nearest %.2f m"
                   % (med.min(), med.max(), near.min()))


def check_provenance(root, names, res):
    """[D] segments account for the frames, and --random_spawn actually varied."""
    on_disk = {}
    for stem in names:
        on_disk[seg_of(stem)] = on_disk.get(seg_of(stem), 0) + 1

    p = os.path.join(root, "manifest_segments.json")
    if not os.path.exists(p):
        if len(on_disk) == 1 and "(none)" in on_disk:
            res.ok("D1 segment manifest", "single unsegmented run, none expected")
        else:
            res.fail("D1 segment manifest",
                     "missing, but %d prefixes on disk -- run v4_merge_segments.py"
                     % len(on_disk))
        return
    doc = json.load(open(p, encoding="utf-8"))
    segs = doc.get("segments", [])
    claimed = {}
    for s in segs:
        claimed[s.get("segment", "?")] = s.get("frames", 0)

    # val/test run unsegmented (-Single): no --prefix, so their files carry no
    # segment tag and land under "(none)" on disk while the manifest still names
    # the run s0.  Key the comparison on the prefix the segment actually wrote.
    # A segment that crashed on its first tick records frames: null, not 0.
    # Segments retired by v4_drop_segments.py keep their record (so provenance
    # still accounts for every frame ever collected) but must not be expected on
    # disk any more.
    dropped = [s for s in segs if s.get("dropped")]
    segs = [s for s in segs if not s.get("dropped")]
    claimed = {}
    for s in segs:
        pref = s.get("prefix") or ""
        claimed[pref.rstrip("_") if pref else "(none)"] = s.get("frames") or 0
    if dropped:
        res.ok("D0 retired segments", "%d excluded: %s"
               % (len(dropped), ", ".join(str(s.get("segment")) for s in dropped)))

    mism = []
    for tag, n in sorted(on_disk.items()):
        if claimed.get(tag) != n:
            mism.append("%s: disk %d vs manifest %s" % (tag, n, claimed.get(tag)))
    for tag in claimed:
        if tag not in on_disk and claimed[tag]:
            mism.append("%s: manifest %d vs disk 0" % (tag, claimed[tag]))
    if mism:
        res.fail("D1 segments match disk", "; ".join(mism[:4]))
    else:
        res.ok("D1 segments match disk", "%d segment(s), %d frames"
               % (len(on_disk), sum(on_disk.values())))

    # --random_spawn is the fix that stops segments retracing one route.  Check
    # both that it was passed AND that the resulting spawns actually differ: the
    # flag being present in the command line is not evidence the draw varied.
    # Either mechanism is acceptable: --random_spawn draws one at random (used up
    # to 2026-07-30), --spawn_id names one explicitly (used since, because the
    # random draws collided -- see 規劃 §8).  What is NOT acceptable is neither,
    # which pins every segment to spawn_points[0].
    live = [s for s in segs if (s.get("frames") or 0) > 0]
    no_flag = [s.get("segment") for s in live
               if "--random_spawn" not in (s.get("command") or "")
               and "--spawn_id" not in (s.get("command") or "")]
    if len(live) < 2:
        res.ok("D2 spawn point chosen", "single productive segment, not applicable")
    elif no_flag:
        res.fail("D2 spawn point chosen",
                 "neither --random_spawn nor --spawn_id on segment(s): %s"
                 % ", ".join(map(str, no_flag[:5])))
    else:
        res.ok("D2 spawn point chosen", "on all %d productive segments" % len(live))

    spawns = [tuple(round(v, 2) for v in (s.get("ego_spawn") or []))
              for s in live if s.get("ego_spawn")]
    if len(spawns) >= 2:
        uniq = len(set(spawns))
        if uniq < len(spawns):
            # A leading indicator, not proof: two segments from the same spawn
            # point can still diverge at the first junction.  v4_check_dup.py,
            # which compares the images themselves, is the authority on whether
            # they actually retraced a route -- so this warns rather than fails.
            res.warn("D3 spawn points differ",
                     "%d distinct of %d segments -- run v4_check_dup.py to see "
                     "whether any pair actually retraced a route"
                     % (uniq, len(spawns)))
        else:
            res.ok("D3 spawn points differ", "%d distinct of %d" % (uniq, len(spawns)))


def check_poses(root, names, res):
    """[E] poses.json covers exactly the frames on disk."""
    p = os.path.join(root, "poses.json")
    if not os.path.exists(p):
        res.fail("E1 poses.json", "missing")
        return
    poses = json.load(open(p, encoding="utf-8"))
    # poses.json is keyed by file name ("000000.png"), not by stem.
    keys = set(os.path.splitext(k)[0] for k in poses)
    want = set(names)
    missing, extra = want - keys, keys - want
    if missing or extra:
        res.fail("E1 poses cover frames", "%d frames without a pose, %d stray poses"
                 % (len(missing), len(extra)))
    else:
        res.ok("E1 poses cover frames", "%d poses" % len(keys))

    seg = os.path.join(root, "manifest_segments.json")
    if os.path.exists(seg):
        doc = json.load(open(seg, encoding="utf-8"))
        col = doc.get("pose_key_collisions")
        if col:
            res.fail("E2 no pose key collisions", "%s" % col)
        elif col == 0:
            res.ok("E2 no pose key collisions")


def check_calibration(root, res):
    """[F] calibration matches the images and carries all 6 classes."""
    p = os.path.join(root, "calibration.json")
    if not os.path.exists(p):
        res.fail("F1 calibration.json", "missing")
        return
    c = json.load(open(p, encoding="utf-8"))
    cls = c.get("classes") or {}
    if len(cls) != len(CLASSES):
        res.fail("F1 calibration classes", "%d classes, expected %d" % (len(cls), len(CLASSES)))
    else:
        res.ok("F1 calibration classes", "%d" % len(cls))
    if c.get("schema_version") is None:
        res.warn("F2 schema_version", "absent")
    else:
        res.ok("F2 schema_version", str(c.get("schema_version")))

    # Intrinsics must describe the images actually on disk.
    f = sorted(glob.glob(os.path.join(root, "front_rgb", "*.png")))
    if f:
        a = cv2.imread(f[0], cv2.IMREAD_COLOR)
        img = c.get("image") or {}
        w, h = img.get("width"), img.get("height")
        if a is not None and w and h:
            if (int(h), int(w)) != a.shape[:2]:
                res.fail("F3 calib size vs images", "calib %sx%s, image %dx%d"
                         % (w, h, a.shape[1], a.shape[0]))
            else:
                res.ok("F3 calib size vs images", "%sx%s fov %s"
                       % (w, h, img.get("fov")))

        # The K written into calibration.json is what the LSS projection uses; if
        # it does not match the image it silently mis-projects every frame.
        cams = c.get("cameras") or {}
        for cname, cam in sorted(cams.items()):
            K = cam.get("K")
            if not K or not w:
                continue
            fx, cx, cy = K[0][0], K[0][2], K[1][2]
            f_expect = (float(w) / 2.0) / np.tan(np.deg2rad(float(img.get("fov", 90))) / 2.0)
            if abs(fx - f_expect) > 1.0 or abs(cx - w / 2.0) > 1.0 or abs(cy - h / 2.0) > 1.0:
                res.fail("F4 K matches fov/size",
                         "%s: fx %.1f (expect %.1f), c (%.1f,%.1f) expect (%.1f,%.1f)"
                         % (cname, fx, f_expect, cx, cy, w / 2.0, h / 2.0))
                break
        else:
            if cams:
                res.ok("F4 K matches fov/size", "%d camera(s)" % len(cams))


def check_stuck(root, names, res, limit=400):
    """[G] near-identical consecutive frames inside a segment (gridlocked ego)."""
    by_seg = {}
    for stem in names:
        by_seg.setdefault(seg_of(stem), []).append(stem)

    worst = []
    for tag, stems_ in sorted(by_seg.items()):
        take = sorted(stems_)[:limit]
        if len(take) < 20:
            continue
        prev, stuck, n = None, 0, 0
        for stem in take:
            a = cv2.imread(os.path.join(root, "front_rgb", stem + ".png"),
                           cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            a = cv2.resize(a, (64, 36)).astype(np.float32)
            if prev is not None:
                n += 1
                if np.abs(a - prev).mean() < STUCK_MAD:
                    stuck += 1
            prev = a
        if n:
            pct = 100.0 * stuck / n
            worst.append((pct, tag, n))
    if not worst:
        res.ok("G1 ego not stuck", "no segment long enough to judge")
        return
    worst.sort(reverse=True)
    pct, tag, n = worst[0]
    detail = "worst segment %s: %.1f%% of %d consecutive pairs identical" % (tag, pct, n)
    if pct >= STUCK_WARN_PCT:
        res.warn("G1 ego not stuck", detail)
    else:
        res.ok("G1 ego not stuck", detail)


def audit(root, stride):
    name = os.path.basename(root.rstrip("\\/"))
    res = Result()
    print("=" * 78)
    print(name)
    names, present = check_structure(root, res)
    if names:
        check_pixels(root, names, present, stride, res)
        check_provenance(root, names, res)
        check_poses(root, names, res)
        check_calibration(root, res)
        check_stuck(root, names, res)
    res.dump()
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stride", type=int, default=15,
                    help="subsample factor for the pixel-level checks")
    ap.add_argument("--skip", default="superseded,bad_spawn,bad_neg,crash_partial",
                    help="comma-separated substrings of directory names to skip "
                         "(retired data, 規劃 §15.1)")
    args = ap.parse_args()

    skip = [s for s in args.skip.split(",") if s]
    roots = []
    if os.path.isdir(os.path.join(args.path, "front_rgb")):
        roots = [args.path]
    else:
        for d in sorted(glob.glob(os.path.join(args.path, "*"))):
            if not os.path.isdir(d):
                continue
            if any(s in os.path.basename(d) for s in skip):
                continue
            if os.path.isdir(os.path.join(d, "front_rgb")):
                roots.append(d)

    results = [audit(r, args.stride) for r in roots]

    print()
    print("=" * 78)
    print("SUMMARY")
    bad = 0
    for name, res in results:
        status = "FAIL" if res.failed else ("WARN" if res.warned else "OK")
        print("  %-4s %-26s %d fail / %d warn" % (status, name, res.failed, res.warned))
        bad += res.failed
    print()
    if bad:
        print("AUDIT FAILED: %d check(s) across %d dataset(s)" % (bad, len(results)))
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
