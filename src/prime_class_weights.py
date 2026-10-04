# -*- coding: utf-8 -*-
"""Fill the class-weight cache that train_dualcam_bev_fpn.py reads.

The trainer computes the label histogram single-threaded through the full
Dataset, which decodes far more than the labels and took 35 minutes on the v4
training set before the first epoch could start.  The histogram only needs the
BEV label PNGs, so this reads those directly across a process pool -- about two
minutes for the same 11,429 frames.

The cache key is produced by the trainer's own _weights_cache_key(), against
datasets built exactly as the trainer builds them, so a cache written here is
accepted only by a run with the identical frame list.

  python prime_class_weights.py --towns ../data/v4/train_town10hd,... \\
      --split_train train_v4.txt
"""
import os
import sys
import json
import argparse
import multiprocessing as mp

import cv2
import numpy as np

from dualcam_bev_dataset import DualCamBEVDataset
from train_dualcam_bev_fpn import _weights_cache_key


def hist_of(path, num_classes=6):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(a.ravel(), minlength=num_classes)[:num_classes].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", required=True)
    ap.add_argument("--split_train", default="train_v4.txt")
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--out", default="../metrics/class_weights_cache.json")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    town_dirs = [t.strip() for t in args.towns.split(",") if t.strip()]
    sets = [DualCamBEVDataset(d, split=args.split_train) for d in town_dirs]

    paths = []
    for s in sets:
        for name in s.files:
            paths.append(os.path.join(s.root, "bev_gt", name))
    print("%d label files across %d towns" % (len(paths), len(sets)))

    with mp.Pool(args.procs) as pool:
        hists = pool.map(hist_of, paths, chunksize=64)
    hist = np.sum(hists, axis=0)

    # Build the same ConcatDataset-like object the trainer keys on.
    class _Concat(object):
        datasets = sets
    key = _weights_cache_key(_Concat(), args.num_classes)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "hist": hist.astype(float).tolist()}, fh)

    freq = hist / hist.sum()
    print("pixel ratio:", [round(float(x), 6) for x in freq])
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
