# -*- coding: utf-8 -*-
"""Stitch per-segment poses/manifest snapshots back into one record.

carla_dualcam_collect_gt.py builds `poses` and `manifest` from scratch on every
run and writes the whole file, so a segmented collection (§8) leaves only the
last segment's records behind.  collect_v4_driver.ps1 therefore snapshots
poses_<prefix>.json / manifest_<prefix>.json after each segment; this script
merges them into:

  poses.json            every frame of every segment, keyed by file name
  manifest_segments.json  the per-segment manifests plus a combined tally

Frame names carry the segment prefix, so keys never collide across segments.

Usage:
  python v4_merge_segments.py ..\\data\\v4\\train_town04
  python v4_merge_segments.py --all ..\\data\\v4
"""
import os
import re
import sys
import glob
import json
import argparse


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def merge_one(root, verbose=True):
    pose_files = sorted(glob.glob(os.path.join(root, "poses_*.json")))
    man_files = sorted(glob.glob(os.path.join(root, "manifest_*.json")))
    if not pose_files and not man_files:
        if verbose:
            print("%s: no segment snapshots, nothing to merge" % root)
        return None

    merged_poses = {}
    dupes = 0
    for p in pose_files:
        for k, v in _load(p).items():
            if k in merged_poses:
                dupes += 1
            merged_poses[k] = v

    segments = []
    tally = {"ticks": 0, "saved": 0, "kept_content": 0,
             "kept_negative": 0, "skipped": 0}
    for m in man_files:
        d = _load(m)
        seg = re.sub(r"^manifest_|\.json$", "", os.path.basename(m))
        st = d.get("stats") or {}
        segments.append({
            "segment": seg,
            "prefix": (d.get("capture") or {}).get("prefix"),
            "complete": d.get("complete"),
            "frames": (d.get("frames") or {}).get("count"),
            "stats": st,
            "actors": {k: v for k, v in (d.get("actors") or {}).items()
                       if k != "ego_spawn"},
            "ego_spawn": (d.get("actors") or {}).get("ego_spawn"),
            "command": d.get("command"),
            "digest": ((d.get("integrity") or {}).get("digest") or "")[:16],
        })
        for k in tally:
            if isinstance(st.get(k), int):
                tally[k] += st[k]

    tally["keep_rate"] = (round(tally["saved"] / tally["ticks"], 4)
                          if tally["ticks"] else None)

    n_bev = len(glob.glob(os.path.join(root, "bev_gt", "*.png")))
    out = {
        "note": ("Merged from per-segment snapshots. The collector overwrites "
                 "poses.json/manifest.json on every run, so the top-level "
                 "manifest.json describes only the LAST segment."),
        "segments": segments,
        "combined": tally,
        "frames_on_disk": n_bev,
        "poses_merged": len(merged_poses),
        "pose_key_collisions": dupes,
    }

    if merged_poses:
        _write(os.path.join(root, "poses.json"), merged_poses)
    _write(os.path.join(root, "manifest_segments.json"), out)

    if verbose:
        name = os.path.basename(root.rstrip("\\/"))
        flag = ""
        if merged_poses and len(merged_poses) != n_bev:
            flag = "   <-- poses (%d) != frames (%d)" % (len(merged_poses), n_bev)
        if dupes:
            flag += "   <-- %d duplicate pose keys" % dupes
        print("%-20s segments=%d  frames=%d  poses=%d  keep_rate=%s%s"
              % (name, len(segments), n_bev, len(merged_poses),
                 tally["keep_rate"], flag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true",
                    help="treat path as a parent directory of datasets")
    args = ap.parse_args()

    if args.all:
        for d in sorted(glob.glob(os.path.join(args.path, "*"))):
            if os.path.isdir(d):
                merge_one(d)
    else:
        merge_one(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
