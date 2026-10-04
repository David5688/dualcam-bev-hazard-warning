# -*- coding: utf-8 -*-
"""Write the v4 city-level split files, and print the training command to match.

The v4 split is by CITY (規劃 §1, confirmed 2026-07-30):

    train  Town10HD, Town02, Town04, Town05, Town06   -- every frame, no internal val
    val    Town07                                     -- whole city
    test   Town01                                     -- whole city

That is why this is not split_contiguous.py.  That script cuts one continuous
drive into train/val/test blocks with a gap, which is the right tool when train
and test come from the same drive.  Here they never do: the hold-out is a city
the model has never seen, so there is nothing to cut and no gap to leave.

Training cities use the de-duplicated frame list from v4_dedup_frames.py when one
is present (a gridlocked ego can otherwise contribute hundreds of copies of one
junction).  val/test use every frame: they must stay continuous and unfiltered so
the metrics sit on a natural distribution and the hazard/TTC demo has evenly
spaced frames (規劃 §3).

  python v4_build_splits.py ..\\data\\v4
  python v4_build_splits.py ..\\data\\v4 --no_dedup      # ignore the dedup lists
"""
import os
import sys
import glob
import argparse

TRAIN_CITIES = ["train_town10hd", "train_town02", "train_town04",
                "train_town05", "train_town06"]
VAL_CITY = "val_town07"
TEST_CITY = "test_town01"

DEDUP_LIST = "train_dedup.txt"


def frames_of(root):
    return sorted(os.path.basename(f) for f in
                  glob.glob(os.path.join(root, "front_rgb", "*.png")))


def write_list(path, names, dry):
    if dry:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(names) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="the data/v4 directory")
    ap.add_argument("--train_name", default="train_v4.txt")
    ap.add_argument("--val_name", default="val_v4.txt")
    ap.add_argument("--test_name", default="test_v4.txt")
    ap.add_argument("--no_dedup", action="store_true",
                    help="use every frame of the training cities, ignoring "
                         "train_dedup.txt")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    missing = []
    train_dirs, total_train = [], 0
    print("TRAIN cities")
    for city in TRAIN_CITIES:
        root = os.path.join(args.data, city)
        if not os.path.isdir(os.path.join(root, "front_rgb")):
            missing.append(city)
            print("  %-18s MISSING" % city)
            continue
        allf = frames_of(root)
        ded = os.path.join(root, DEDUP_LIST)
        if not args.no_dedup and os.path.exists(ded):
            names = [l.strip() for l in open(ded, encoding="utf-8") if l.strip()]
            src = "dedup"
        else:
            names = allf
            src = "all"
        # A dedup list written before frames were retired can name files that no
        # longer exist; keep only what is on disk.
        on_disk = set(allf)
        names = [n for n in names if n in on_disk]
        write_list(os.path.join(root, args.train_name), names, args.dry_run)
        train_dirs.append(root)
        total_train += len(names)
        print("  %-18s %5d frames (%s, %d on disk)" % (city, len(names), src, len(allf)))

    print("VAL / TEST cities (every frame, continuous)")
    holdout = []
    for city, out in ((VAL_CITY, args.val_name), (TEST_CITY, args.test_name)):
        root = os.path.join(args.data, city)
        if not os.path.isdir(os.path.join(root, "front_rgb")):
            missing.append(city)
            print("  %-18s MISSING" % city)
            continue
        names = frames_of(root)
        write_list(os.path.join(root, out), names, args.dry_run)
        holdout.append((city, root, len(names)))
        print("  %-18s %5d frames" % (city, len(names)))

    print()
    print("train %d / val %d / test %d"
          % (total_train,
             holdout[0][2] if len(holdout) > 0 else 0,
             holdout[1][2] if len(holdout) > 1 else 0))
    if missing:
        print("INCOMPLETE -- missing: %s" % ", ".join(missing))

    if not args.dry_run and train_dirs and len(holdout) >= 1:
        print()
        print("training command:")
        print("  python train_dualcam_bev_fpn.py --arch lss --num_classes 6 ^")
        print("    --img_h 448 --img_w 800 --epochs 80 --batch 4 ^")
        print("    --dice 1.0 --cosine --augment --depth_weight 0.1 ^")
        print("    --boost_vehicle 3.0 --boost_lane 2.5 ^")
        print("    --towns %s ^" % ",".join(train_dirs))
        print("    --val_towns %s ^" % holdout[0][1])
        print("    --split_train %s --split_val %s ^" % (args.train_name, args.val_name))
        print("    --save ..\\checkpoints\\dualcam_bev_lss_v4_best.pt ^")
        print("    --metrics_csv ..\\metrics\\metrics_v4.csv")
        print()
        print("  NOTE --img_h/--img_w: the v4 images are 1280x720, four times the "
              "pixels of the 640x360 the old recipes used. 800x448 keeps the aspect "
              "ratio and was the resolution of the fov120hr experiment; it needs "
              "~4x the memory of the 400x224 default, so check it fits before "
              "starting an 80-epoch run.")
    if args.dry_run:
        print("(dry run, nothing written)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
