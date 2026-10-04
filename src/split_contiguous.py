"""Contiguous, gap-separated train/val/test split for continuous driving sequences.

The frames in these datasets come from a single continuous CARLA drive, so
adjacent frames are near-duplicates. A random shuffle (see split_dualcam_dataset.py)
puts near-duplicates of test frames into train, which inflates every metric.
This script instead cuts the sequence into contiguous blocks and drops `gap`
frames between them, so no test frame has a train neighbour.

Default arguments reproduce the split that produced the project's only
leak-free numbers (train 0-1049, val 1100-1249, test 1300-1499).
"""
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root containing front_rgb/")
    ap.add_argument("--train", type=int, default=1050)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--gap", type=int, default=50, help="frames discarded between blocks")
    ap.add_argument("--dry_run", action="store_true", help="print plan, write nothing")
    ap.add_argument("--out_dir", default=None, help="defaults to --data")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(os.path.join(args.data, "front_rgb"))
                   if f.endswith(".png"))
    n = len(files)

    need = args.train + args.gap + args.val + args.gap + args.test
    if need > n:
        raise SystemExit(f"need {need} frames but dataset has {n}")

    a = 0
    b = a + args.train
    c = b + args.gap
    d = c + args.val
    e = d + args.gap
    f = e + args.test

    blocks = {
        "train.txt": files[a:b],
        "val.txt": files[c:d],
        "test.txt": files[e:f],
    }

    print(f"dataset: {args.data}  ({n} frames)")
    print(f"train  [{a:5d}, {b-1:5d}]  {len(blocks['train.txt']):4d}")
    print(f"  gap  [{b:5d}, {c-1:5d}]  {args.gap:4d}  (discarded)")
    print(f"val    [{c:5d}, {d-1:5d}]  {len(blocks['val.txt']):4d}")
    print(f"  gap  [{d:5d}, {e-1:5d}]  {args.gap:4d}  (discarded)")
    print(f"test   [{e:5d}, {f-1:5d}]  {len(blocks['test.txt']):4d}")
    if f < n:
        print(f"unused [{f:5d}, {n-1:5d}]  {n-f:4d}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    out_dir = args.out_dir or args.data
    for name, items in blocks.items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as fh:
            for x in items:
                fh.write(x + "\n")
        print("wrote:", path)


if __name__ == "__main__":
    main()
