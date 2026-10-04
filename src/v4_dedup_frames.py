# -*- coding: utf-8 -*-
"""Build a training frame list with near-duplicate frames removed.

Why: CARLA's traffic manager gridlocks the ego (規劃 §5 records that >30 vehicles
does it; the v4 spec settled on 40), and while it sits still the collector keeps
saving frames.  Measured by v4_audit.py check G1 on 2026-07-30:

    train_town02 / s1_      51.9% of consecutive frames identical to the previous
    train_town10hd / neg0_  22.3%
    train_town06 / neg0_    19.3%

With the v4 city-level hold-out (train = 5 cities, val = Town07, test = Town01)
these duplicates are NOT a leakage problem -- no training-city frame can reach
val or test.  They are a data-quality problem: a junction the ego was stuck at
gets learned hundreds of times, quietly reweighting the training distribution
toward whatever was in front of a parked car.

So this does not touch the collected data.  It emits a frame list, and training
reads that list instead of the directory.

Never run this on Town07 / Town01: val and test are deliberately unfiltered and
continuous (規劃 §3), both to keep the metrics on a natural distribution and
because the hazard/TTC demo needs evenly spaced frames.

  python v4_dedup_frames.py ..\\data\\v4\\train_town02 --dry_run
  python v4_dedup_frames.py ..\\data\\v4\\train_town02 --out train_dedup.txt
  python v4_dedup_frames.py ..\\data\\v4 --all --out train_dedup.txt
"""
import os
import re
import sys
import glob
import argparse

import cv2
import numpy as np

VAL_TEST_HINTS = ("val_", "test_")


def seg_of(name):
    m = re.match(r"^([a-z]+\d+_)", name)
    return m.group(1) if m else "(none)"


def dedup(root, mad, size=(64, 36)):
    files = sorted(os.path.basename(f) for f in
                   glob.glob(os.path.join(root, "front_rgb", "*.png")))
    by_seg = {}
    for f in files:
        by_seg.setdefault(seg_of(f), []).append(f)

    kept_all, stats = [], []
    for tag in sorted(by_seg):
        seq = by_seg[tag]
        last = None
        kept = []
        for f in seq:
            a = cv2.imread(os.path.join(root, "front_rgb", f), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            sig = cv2.resize(a, size).astype(np.float32)
            # Compare against the last KEPT frame, not the previous frame: while
            # the ego is stopped every frame differs from its neighbour by ~0, so
            # comparing pairwise would keep a long stalled run intact.
            if last is None or np.abs(sig - last).mean() >= mad:
                kept.append(f)
                last = sig
        kept_all += kept
        stats.append((tag, len(seq), len(kept)))
    return sorted(kept_all), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true",
                    help="treat path as a parent dir; every train_* dataset under it")
    ap.add_argument("--mad", type=float, default=1.5,
                    help="mean |diff| on 0-255 greyscale below which a frame counts "
                         "as a duplicate of the last kept one. 1.5 is the valley of "
                         "a measured bimodal distribution, not a guess: over 3,188 "
                         "consecutive pairs from four training cities the stalled "
                         "mode occupies 0.50-1.25 (547 pairs), the 1.50-1.75 bin is "
                         "nearly empty (12), and real motion runs from ~1.75 up past "
                         "6 (1,814 pairs). A threshold of 1.0 sits on the shoulder of "
                         "the stalled peak and leaves its 1.00-1.25 tail behind.")
    ap.add_argument("--out", default="train_dedup.txt",
                    help="written inside each dataset directory")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    roots = []
    if args.all:
        for d in sorted(glob.glob(os.path.join(args.path, "*"))):
            base = os.path.basename(d)
            if not os.path.isdir(os.path.join(d, "front_rgb")):
                continue
            if base.startswith(VAL_TEST_HINTS):
                print("skipping %s (val/test stay continuous, 規劃 §3)" % base)
                continue
            if not base.startswith("train_"):
                continue
            roots.append(d)
    else:
        roots = [args.path]

    grand_in = grand_out = 0
    for root in roots:
        name = os.path.basename(root.rstrip("\\/"))
        kept, stats = dedup(root, args.mad)
        n_in = sum(s[1] for s in stats)
        print("=" * 66)
        print("%s : %d -> %d frames (%.1f%% dropped)"
              % (name, n_in, len(kept),
                 100.0 * (n_in - len(kept)) / max(n_in, 1)))
        for tag, a, b in stats:
            if a and (a - b) / float(a) >= 0.05:
                print("    %-8s %4d -> %4d  (%.0f%% dropped)"
                      % (tag, a, b, 100.0 * (a - b) / a))
        grand_in += n_in
        grand_out += len(kept)

        if not args.dry_run:
            p = os.path.join(root, args.out)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(kept) + "\n")
            print("    wrote %s" % p)

    print("=" * 66)
    print("TOTAL %d -> %d frames (%.1f%% dropped, mad<%.2f)"
          % (grand_in, grand_out,
             100.0 * (grand_in - grand_out) / max(grand_in, 1), args.mad))
    if args.dry_run:
        print("(dry run, no list written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
