#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import random
import argparse
import numpy as np


def get_class_pixels(label_path, num_classes=5):
    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        raise RuntimeError(f"Cannot read {label_path}")

    hist = np.bincount(
        gt.reshape(-1),
        minlength=num_classes
    ).astype(np.int64)

    return hist


def assign_bin(hist):
    vehicle = hist[1]
    pedestrian = hist[2]

    has_vehicle = vehicle > 0
    has_ped = pedestrian > 0

    if has_vehicle and has_ped:
        return "vehicle_pedestrian"
    elif has_ped:
        return "pedestrian_only"
    elif has_vehicle:
        return "vehicle_only"
    else:
        return "background_road_only"


def split_group(files, train_ratio, val_ratio):
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return train, val, test


def write_split(path, files):
    with open(path, "w") as f:
        for x in files:
            f.write(x + "\n")


def summarize_split(name, files, stats):
    total_hist = None

    for f in files:
        hist = stats[f]["hist"]
        if total_hist is None:
            total_hist = hist.copy()
        else:
            total_hist += hist

    if total_hist is None:
        total_hist = np.zeros(5, dtype=np.int64)

    total = total_hist.sum()
    ratio = total_hist / max(total, 1)

    print()
    print("=" * 60)
    print(name)
    print("samples:", len(files))
    print("pixel histogram:", total_hist.tolist())
    print("pixel ratio:")
    print(f"  background : {ratio[0]:.6f}")
    print(f"  vehicle    : {ratio[1]:.6f}")
    print(f"  pedestrian : {ratio[2]:.6f}")
    print(f"  road       : {ratio[3]:.6f}")
    print(f"  sidewalk   : {ratio[4]:.6f}")

    n_vehicle = sum(stats[f]["hist"][1] > 0 for f in files)
    n_ped = sum(stats[f]["hist"][2] > 0 for f in files)
    n_both = sum(
        (stats[f]["hist"][1] > 0 and stats[f]["hist"][2] > 0)
        for f in files
    )

    print("frames with vehicle   :", n_vehicle)
    print("frames with pedestrian:", n_ped)
    print("frames with both      :", n_both)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="carla_dualcam_gt_out")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=5)

    args = parser.parse_args()

    random.seed(args.seed)

    front_dir = os.path.join(args.data, "front_rgb")
    rear_dir = os.path.join(args.data, "rear_rgb")
    gt_dir = os.path.join(args.data, "bev_gt")

    files = sorted([
        os.path.basename(x)
        for x in glob.glob(os.path.join(front_dir, "*.png"))
    ])

    if len(files) == 0:
        raise RuntimeError(f"No images found in {front_dir}")

    valid_files = []
    stats = {}

    print("Scanning labels...")

    for f in files:
        front_path = os.path.join(front_dir, f)
        rear_path = os.path.join(rear_dir, f)
        gt_path = os.path.join(gt_dir, f)

        if not os.path.exists(front_path):
            continue
        if not os.path.exists(rear_path):
            continue
        if not os.path.exists(gt_path):
            continue

        hist = get_class_pixels(gt_path, args.num_classes)
        group = assign_bin(hist)

        stats[f] = {
            "hist": hist,
            "group": group,
        }

        valid_files.append(f)

    groups = {
        "vehicle_pedestrian": [],
        "pedestrian_only": [],
        "vehicle_only": [],
        "background_road_only": [],
    }

    for f in valid_files:
        groups[stats[f]["group"]].append(f)

    print()
    print("Group summary:")
    for k, v in groups.items():
        print(f"{k:24s}: {len(v)}")

    train_files = []
    val_files = []
    test_files = []

    for group_name, group_files in groups.items():
        tr, va, te = split_group(
            group_files,
            args.train,
            args.val
        )

        train_files.extend(tr)
        val_files.extend(va)
        test_files.extend(te)

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    write_split(os.path.join(args.data, "train.txt"), train_files)
    write_split(os.path.join(args.data, "val.txt"), val_files)
    write_split(os.path.join(args.data, "test.txt"), test_files)

    print()
    print("Saved:")
    print(os.path.join(args.data, "train.txt"), len(train_files))
    print(os.path.join(args.data, "val.txt"), len(val_files))
    print(os.path.join(args.data, "test.txt"), len(test_files))

    summarize_split("TRAIN", train_files, stats)
    summarize_split("VAL", val_files, stats)
    summarize_split("TEST", test_files, stats)

    print()
    print("Done.")


if __name__ == "__main__":
    main()