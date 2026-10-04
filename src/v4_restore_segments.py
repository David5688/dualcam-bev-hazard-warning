# -*- coding: utf-8 -*-
"""Put segments retired by v4_drop_segments.py back into their dataset.

Needed because a retirement decision can turn out to be wrong.  On 2026-07-30
Town02's r21_ was dropped on a v4_check_dup.py "duplicate route" flag produced
with the old threshold of MAD < 10.  That threshold was measured the next day to
generate false positives -- Town05's s16_/s26_ were flagged the same way while
their closest matching frames sat at MAD 10.55, i.e. not duplicates at all -- and
the tool now defaults to 3.0.  Frames were moved rather than deleted precisely so
that a call like that could be taken back.

Reverses everything the drop did: the image files, the poses entries, and the
`dropped` marker in manifest_segments.json.

  python v4_restore_segments.py ..\\data\\v4\\train_town02 r21_
  python v4_restore_segments.py ..\\data\\v4\\train_town02 r21_ --dry_run
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
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--retired_root", default=None,
                    help="default: <parent of dataset>/retired/<dataset>_dropped")
    ap.add_argument("--reason", default="restored")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    name = os.path.basename(ds)
    prefixes = [p if p.endswith("_") else p + "_" for p in args.prefixes]
    retired = args.retired_root or os.path.join(
        os.path.dirname(ds), "retired", name + "_dropped")

    if not os.path.isdir(retired):
        print("no retired directory at %s" % retired)
        return 1

    moves = []
    for m in MODALITIES:
        d = os.path.join(retired, m)
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

    before = len(glob.glob(os.path.join(ds, "bev_gt", "*.png")))
    restored = sum(per_prefix.values())
    print("dataset  : %s" % ds)
    print("restoring: %s" % ", ".join(prefixes))
    for p in sorted(per_prefix):
        print("  %-8s %d frames x %d modalities" % (p, per_prefix[p], len(MODALITIES)))
    print("  frames : %d -> %d" % (before, before + restored))
    if not moves:
        print("nothing to restore")
        return 1
    if args.dry_run:
        print("\n(dry run, nothing moved)")
        return 0

    # 1. images back
    for m, base, src in moves:
        dst_dir = os.path.join(ds, m)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        dst = os.path.join(dst_dir, base)
        if os.path.exists(dst):
            print("REFUSING: %s already exists in the dataset" % dst)
            return 1
        shutil.move(src, dst)
    print("moved %d files back" % len(moves))

    # 2. poses back
    rp = os.path.join(retired, "poses.json")
    dp = os.path.join(ds, "poses.json")
    if os.path.exists(rp) and os.path.exists(dp):
        retired_poses = json.load(open(rp, encoding="utf-8"))
        poses = json.load(open(dp, encoding="utf-8"))
        shutil.copy2(dp, dp + ".bak")
        back = {k: v for k, v in retired_poses.items()
                if any(k.startswith(p) for p in prefixes)}
        poses.update(back)
        keep_retired = {k: v for k, v in retired_poses.items() if k not in back}
        json.dump(poses, open(dp, "w", encoding="utf-8"))
        json.dump(keep_retired, open(rp, "w", encoding="utf-8"))
        print("poses.json: +%d -> %d entries" % (len(back), len(poses)))

    # 3. clear the dropped marker
    mp = os.path.join(ds, "manifest_segments.json")
    if os.path.exists(mp):
        doc = json.load(open(mp, encoding="utf-8"))
        n = 0
        for s in doc.get("segments", []):
            if (s.get("prefix") or "") in prefixes and s.get("dropped"):
                s["restored"] = {"reason": args.reason,
                                 "when": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "previously_dropped": s.pop("dropped")}
                n += 1
        doc["frames_on_disk"] = before + restored
        json.dump(doc, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("manifest_segments.json: cleared dropped marker on %d segment(s)" % n)

    note = os.path.join(retired, "PROVENANCE_NOTE.md")
    with open(note, "a", encoding="utf-8") as fh:
        fh.write("\n## %s 還原\n\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        fh.write("- 還原段:%s\n" % ", ".join("`%s` (%d 幀)" % (p, per_prefix.get(p, 0))
                                             for p in sorted(prefixes)))
        fh.write("- 原因:%s\n" % args.reason)
    print("appended to %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
