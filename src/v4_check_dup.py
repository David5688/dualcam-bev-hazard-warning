# -*- coding: utf-8 -*-
"""Detect near-duplicate frames ACROSS segments of a segmented collection.

Why this exists: split_contiguous.py only guards against *adjacent* frames
leaking between splits. On 2026-07-30 the collection driver was found to start
every segment from spawn_points[0] (the collector's default when --random_spawn
is absent), so segments retraced the same route: in Town02, 84% of the s1_/s3_
frames were near-identical to s0_ at the same index. Those duplicates sit far
apart in frame order, sail through the contiguous split, and land in different
splits -- the same leakage that inflated this project's early numbers, entering
through a different door.

The fix was --random_spawn. This script is the regression test for it: run it on
any segmented dataset and it reports, for every pair of segments, the fraction of
frames that look like the same view.

  python v4_check_dup.py ..\\data\\v4\\train_town02
  python v4_check_dup.py --all ..\\data\\v4 --thresh 10

Similarity is a mean absolute difference of 64x36 greyscale front-camera frames
on a 0-255 scale; below --thresh the two views are treated as the same place.
Frames are compared at matching indices (segments that retrace a route stay in
step because each starts at the same point and drives at similar speed), and
also against a random sample of the other segment to catch offset overlap.
"""
import os
import re
import sys
import glob
import random
import argparse

import cv2
import numpy as np


def load_sigs(files, size=(64, 36)):
    out = []
    for f in files:
        a = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        out.append(cv2.resize(a, size).astype(np.float32))
    return out


def segments_of(root):
    files = sorted(glob.glob(os.path.join(root, "front_rgb", "*.png")))
    segs = {}
    for f in files:
        m = re.match(r"^([a-z]+\d+_)", os.path.basename(f))
        segs.setdefault(m.group(1) if m else "(none)", []).append(f)
    return segs


def check(root, thresh, limit, sample, seed=0, verbose=True):
    name = os.path.basename(root.rstrip("\\/"))
    segs = segments_of(root)
    if len(segs) < 2:
        if verbose:
            print("%-24s only %d segment -- nothing to compare"
                  % (name, len(segs)))
        return []

    rng = random.Random(seed)
    sigs = {}
    for tag, files in segs.items():
        take = files[:limit]
        sigs[tag] = load_sigs(take)

    findings = []
    tags = sorted(sigs)
    if verbose:
        print("%s  segments: %s" % (name, {t: len(segs[t]) for t in tags}))
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = sigs[tags[i]], sigs[tags[j]]
            n = min(len(a), len(b))
            if n == 0:
                continue
            aligned = np.array([np.abs(a[k] - b[k]).mean() for k in range(n)])
            pct_aligned = 100.0 * float((aligned < thresh).mean())

            # offset overlap: does a random frame of one segment have a close
            # match anywhere in the other?
            idx = rng.sample(range(len(a)), min(sample, len(a)))
            hits = 0
            for k in idx:
                d = min(np.abs(a[k] - b[m]).mean() for m in range(len(b)))
                hits += (d < thresh)
            pct_any = 100.0 * hits / max(len(idx), 1)

            flag = ""
            if pct_aligned >= 20 or pct_any >= 20:
                flag = "   <-- DUPLICATE ROUTE"
                findings.append((name, tags[i], tags[j], pct_aligned, pct_any))
            if verbose:
                print("  %s vs %s : %5.1f%% same-index, %5.1f%% best-match "
                      "(n=%d)%s" % (tags[i], tags[j], pct_aligned, pct_any,
                                    n, flag))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true",
                    help="treat path as a parent directory of datasets")
    ap.add_argument("--thresh", type=float, default=3.0,
                    help="mean |diff| below which two views count as the same. "
                         "Was 10.0 until 2026-07-31, which produced false positives: "
                         "Town05's s16_/s26_ were flagged as a duplicate route at "
                         "72.5%% best-match although their closest matching frames sat "
                         "at MAD 10.55 (median; minimum 5.33) -- different spawn points, "
                         "different routes, merely similar-looking streets. Genuinely "
                         "retraced segments look nothing like that: neg2_/s19_, which "
                         "started from the identical spawn point, matched at median MAD "
                         "1.13 with 60%% of frames under 1.5. 3.0 sits between the two "
                         "populations, above the 1.5 the dedup tool calls identical.")
    ap.add_argument("--limit", type=int, default=300,
                    help="frames per segment to load")
    ap.add_argument("--sample", type=int, default=40,
                    help="frames per pair for the offset-overlap probe")
    args = ap.parse_args()

    roots = []
    if args.all:
        roots = [d for d in sorted(glob.glob(os.path.join(args.path, "*")))
                 if os.path.isdir(d)]
    else:
        roots = [args.path]

    all_findings = []
    for r in roots:
        if not os.path.isdir(os.path.join(r, "front_rgb")):
            continue
        all_findings += check(r, args.thresh, args.limit, args.sample)
        print()

    if all_findings:
        print("DUPLICATE ROUTES FOUND in %d segment pair(s):" % len(all_findings))
        for f in all_findings:
            print("  %s  %s vs %s  %.1f%% same-index, %.1f%% best-match"
                  % (f[0], f[1], f[2], f[3], f[4]))
        print("\nSegments retracing one route leak across split_contiguous. "
              "Check that the collector ran with --random_spawn.")
        return 1
    print("no duplicate routes detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
