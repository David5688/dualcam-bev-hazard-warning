# -*- coding: utf-8 -*-
"""Measure whether the v4 content filter biased the model toward busy scenes.

The training cities were collected with `--filter` (a frame is kept only if the
camera can see >= 2 vehicles or >= 2 pedestrians, plus a fraction of the rejected
frames as negatives), while val/test were collected unfiltered so the metrics sit
on a natural distribution (規劃 §3/§4).  The filter's bite varied a lot by city:
94.4% of ticks kept on Town10HD, but only 50.3% on Town06.

So the model trained on a distribution with more vehicles per frame than it meets
at test time.  That cannot inflate the reported numbers -- the test city is
unfiltered -- but it can make the model worse in one specific way: having learned
that there is almost always a car about, it may invent one when the road is
empty.  That shows up as precision loss on empty frames, not as recall loss.

This script splits the test frames by how much vehicle there actually is in the
ground truth and reports vehicle IoU / precision / recall per bucket, plus the
one number that measures hallucination directly: predicted vehicle pixels per
frame on frames whose GT contains no vehicle at all (where IoU is undefined).

  python v4_eval_filter_bias.py --data ..\\data\\v4\\test_town01 \\
      --split test_v4.txt --model ..\\checkpoints\\dualcam_bev_lss_v4_best.pt \\
      --img_h 448 --img_w 800
"""
import os
import csv
import argparse

import cv2
import numpy as np
import torch

from dualcam_bev_lss import DualCamBEVLSS

VEHICLE = 1
# A car footprint is ~128 px at 0.25 m/cell, so 260 px is about two cars -- the
# same yardstick v4_report.py uses for "low vehicle content".
TWO_VEHICLES_PX = 260


def read_img(path, img_h, img_w):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test_v4.txt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--img_h", type=int, default=448)
    ap.add_argument("--img_w", type=int, default=800)
    ap.add_argument("--out_csv", default="../metrics/filter_bias.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    front_dir = os.path.join(args.data, "front_rgb")
    rear_dir = os.path.join(args.data, "rear_rgb")
    gt_dir = os.path.join(args.data, "bev_gt")
    with open(os.path.join(args.data, args.split), encoding="utf-8") as fh:
        files = [l.strip() for l in fh if l.strip()]

    sample = cv2.imread(os.path.join(gt_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    bev_h, bev_w = sample.shape

    meta = torch.load(args.model, map_location="cpu", weights_only=False)
    ds = meta.get("lss_downsample", 8) if isinstance(meta, dict) else 8
    pool = meta.get("lss_pool", 0.5) if isinstance(meta, dict) else 0.5
    model = DualCamBEVLSS.from_calib_file(
        os.path.join(args.data, "calibration.json"),
        num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w,
        img_h=args.img_h, img_w=args.img_w,
        downsample=ds, pool_res=pool, pretrained=False).to(device)
    state = meta["model_state_dict"] if isinstance(meta, dict) and "model_state_dict" in meta else meta
    model.load_state_dict(state, strict=True)
    model.eval()

    buckets = ["empty", "low", "rich"]
    stat = {b: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "pred_px": 0} for b in buckets}

    with torch.no_grad():
        for name in files:
            gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE).astype(np.int64)
            gt_v = (gt == VEHICLE)
            n_gt = int(gt_v.sum())
            b = "empty" if n_gt == 0 else ("low" if n_gt < TWO_VEHICLES_PX else "rich")

            imgs = torch.stack([read_img(os.path.join(front_dir, name), args.img_h, args.img_w),
                                read_img(os.path.join(rear_dir, name), args.img_h, args.img_w)],
                               dim=0).unsqueeze(0).to(device)
            pred = torch.argmax(model(imgs)[0], dim=0).cpu().numpy().astype(np.int64)
            pred_v = (pred == VEHICLE)

            s = stat[b]
            s["n"] += 1
            s["pred_px"] += int(pred_v.sum())
            s["tp"] += int((pred_v & gt_v).sum())
            s["fp"] += int((pred_v & ~gt_v).sum())
            s["fn"] += int((~pred_v & gt_v).sum())

    rows = []
    print("=" * 78)
    print("vehicle, split by how much vehicle the GROUND TRUTH actually contains")
    print("%-6s %6s %10s %10s %10s %14s" % ("bucket", "frames", "IoU", "precision",
                                            "recall", "pred px/frame"))
    for b in buckets:
        s = stat[b]
        if not s["n"]:
            continue
        denom = s["tp"] + s["fp"] + s["fn"]
        iou = s["tp"] / denom if denom else float("nan")
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else float("nan")
        ppf = s["pred_px"] / s["n"]
        label = {"empty": "empty ", "low": "low   ", "rich": "rich  "}[b]
        print("%-6s %6d %10.4f %10.4f %10.4f %14.1f"
              % (label, s["n"], iou, prec, rec, ppf))
        rows.append([b, s["n"], iou, prec, rec, ppf])

    e = stat["empty"]
    print("-" * 78)
    print("empty = GT has no vehicle at all -> IoU/precision/recall are undefined "
          "(0/0);\n        the number that matters there is pred px/frame: every one "
          "of those\n        pixels is a hallucinated vehicle. %d frames, %.1f px each."
          % (e["n"], (e["pred_px"] / e["n"]) if e["n"] else 0.0))
    print("low   = GT has some vehicle but less than two car footprints (<%d px)"
          % TWO_VEHICLES_PX)
    print("rich  = GT has at least two car footprints -- the distribution the "
          "filter selected for")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "frames", "iou", "precision", "recall", "pred_px_per_frame"])
        w.writerows(rows)
    print("wrote %s" % args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
