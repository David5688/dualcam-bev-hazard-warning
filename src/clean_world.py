# -*- coding: utf-8 -*-
"""Destroy every vehicle, walker and sensor left in a CARLA world.

Scripts that are killed before their cleanup runs -- a tool timeout, Ctrl-C, a
crashed server client -- leave their actors behind.  Those actors have no
autopilot, so they sit on the road forever as obstacles.  On 2026-08-09 a live
demo launched with --vehicles 10 found 43 vehicles in the world with none of them
moving: the other 33 were orphans from earlier runs, and they were what the ego
kept yielding to.

Run this with nothing else connected, then start the demo fresh.

  python clean_world.py            # report and destroy
  python clean_world.py --dry_run  # report only
"""
import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    actors = world.get_actors()

    groups = {
        "sensor": list(actors.filter("sensor.*")),
        "vehicle": list(actors.filter("vehicle.*")),
        "walker": list(actors.filter("walker.pedestrian.*")),
        "controller": list(actors.filter("controller.ai.walker")),
    }
    print("map:", world.get_map().name)
    for k, v in groups.items():
        print(f"  {k:11s} {len(v)}")

    if args.dry_run:
        print("\n(dry run, nothing destroyed)")
        return 0

    # Sensors first: destroying a vehicle with sensors attached can leave the
    # sensors orphaned and still streaming.  Controllers before their walkers.
    n = 0
    for key in ("sensor", "controller", "walker", "vehicle"):
        for a in groups[key]:
            try:
                if key == "controller":
                    a.stop()
                a.destroy()
                n += 1
            except Exception:
                pass
    print(f"\ndestroyed {n} actors")

    left = world.get_actors()
    print("remaining: vehicles %d, walkers %d, sensors %d"
          % (len(list(left.filter("vehicle.*"))),
             len(list(left.filter("walker.pedestrian.*"))),
             len(list(left.filter("sensor.*")))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
