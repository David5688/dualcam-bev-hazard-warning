# -*- coding: utf-8 -*-
"""Retire whole segments from a v4 dataset, keeping the dataset self-consistent.

Used on 2026-07-30 for Town02, where v4_check_dup.py found three pairs of
segments that retraced the same route despite --random_spawn (the draw collided:
18 distinct spawn points across 22 segments).  Cross-segment duplicates sit far
apart in frame order, sail through split_contiguous -- which only guards against
ADJACENT frames -- and land in different splits.  That is the leakage that
inflated this project's early numbers, so the smaller segment of each pair goes.

Files are MOVED, never deleted: everything lands under a retired directory with a
PROVENANCE_NOTE.md, per 規劃 §15.1.  poses.json is pruned to match (a stale pose
for a frame that no longer exists makes v4_audit.py's E1 fail, correctly), and
manifest_segments.json keeps the dropped segments with a `dropped` reason so the
provenance still accounts for every frame ever collected.

  python v4_drop_segments.py ..\\data\\v4\\train_town02 r5_ r32_ r21_ \\
      --reason "duplicate route (v4_check_dup: r1_/r5_ 58%, r28_/r32_ 64%, r15_/r21_ 35%)"
  python v4_drop_segments.py ... --dry_run
"""
import os
import re
import sys
import glob
import json
import shutil
import argparse
import datetime

MODALITIES = ["front_rgb", "rear_rgb", "bev_gt", "bev_gt_color",
              "front_depth", "rear_depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("prefixes", nargs="+",
                    help="segment prefixes to retire, e.g. r5_ r32_ r21_")
    ap.add_argument("--reason", default="retired")
    ap.add_argument("--retired_root", default=None,
                    help="default: <parent of dataset>/retired/<dataset>_dropped")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    name = os.path.basename(ds)
    prefixes = [p if p.endswith("_") else p + "_" for p in args.prefixes]
    retired = args.retired_root or os.path.join(
        os.path.dirname(ds), "retired", name + "_dropped")

    # 1. collect the files belonging to those prefixes
    moves = []
    for m in MODALITIES:
        d = os.path.join(ds, m)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            base = os.path.basename(f)
            if any(base.startswith(p) for p in prefixes):
                moves.append((m, base, f))

    per_prefix = {}
    for m, base, _ in moves:
        if m == "bev_gt":
            p = re.match(r"^([a-z]+\d+_)", base)
            per_prefix[p.group(1)] = per_prefix.get(p.group(1), 0) + 1

    print("dataset : %s" % ds)
    print("retiring: %s" % ", ".join(prefixes))
    for p in sorted(per_prefix):
        print("  %-8s %d frames x %d modalities" % (p, per_prefix[p], len(MODALITIES)))
    print("  total  : %d files" % len(moves))
    if not moves:
        print("nothing to do")
        return 1

    before = len(glob.glob(os.path.join(ds, "bev_gt", "*.png")))
    dropped = sum(per_prefix.values())
    print("  frames : %d -> %d" % (before, before - dropped))

    if args.dry_run:
        print("\n(dry run, nothing moved)")
        return 0

    # 2. move the files
    for m, base, src in moves:
        dst_dir = os.path.join(retired, m)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        shutil.move(src, os.path.join(dst_dir, base))
    print("moved %d files -> %s" % (len(moves), retired))

    # 3. prune poses.json (backup first -- it cannot be rebuilt from anything else)
    pp = os.path.join(ds, "poses.json")
    if os.path.exists(pp):
        poses = json.load(open(pp, encoding="utf-8"))
        shutil.copy2(pp, pp + ".bak")
        kept = {k: v for k, v in poses.items()
                if not any(k.startswith(p) for p in prefixes)}
        moved_poses = {k: v for k, v in poses.items() if k not in kept}
        json.dump(kept, open(pp, "w", encoding="utf-8"))
        json.dump(moved_poses,
                  open(os.path.join(retired, "poses.json"), "w", encoding="utf-8"))
        print("poses.json: %d -> %d (backup at poses.json.bak)" % (len(poses), len(kept)))

    # 4. annotate manifest_segments.json rather than deleting the record
    mp = os.path.join(ds, "manifest_segments.json")
    if os.path.exists(mp):
        doc = json.load(open(mp, encoding="utf-8"))
        n = 0
        for s in doc.get("segments", []):
            if (s.get("prefix") or "") in prefixes:
                s["dropped"] = {"reason": args.reason,
                                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                                "moved_to": retired}
                n += 1
        doc["frames_on_disk"] = before - dropped
        json.dump(doc, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("manifest_segments.json: marked %d segment(s) dropped" % n)

    # 5. provenance note
    note = os.path.join(retired, "PROVENANCE_NOTE.md")
    with open(note, "a", encoding="utf-8") as fh:
        fh.write("# %s -- retired segments\n\n" % name)
        fh.write("- 日期: %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        fh.write("- 來源資料集: `%s`\n" % ds)
        fh.write("- 退役段: %s\n" % ", ".join("`%s` (%d 幀)" % (p, per_prefix.get(p, 0))
                                              for p in sorted(prefixes)))
        fh.write("- 原因: %s\n" % args.reason)
        fh.write("- 影像已移到本目錄(未刪除);對應的 poses 存在本目錄的 `poses.json`;\n"
                 "  來源的 `manifest_segments.json` 保留這些段的紀錄並標記 `dropped`。\n")
        fh.write("- 不要拿本目錄的資料做訓練或評估。\n\n")
    print("wrote %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
