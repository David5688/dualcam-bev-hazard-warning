# -*- coding: utf-8 -*-
"""Watch a running live_demo from a second, read-only CARLA client.

live_demo owns the simulation: it is the one that set synchronous mode and the one
that calls world.tick().  This script never ticks and never spawns -- it only reads
actor state -- so it can run alongside without disturbing the demo.

Prints one line per sample: the ego's speed, the control its autopilot is issuing,
the traffic light it is at, the nearest vehicle ahead, and how much of the traffic
is moving.  That is the set of signals that separates "waiting at a red light" from
"yielding to something" from "the traffic manager has parked it".

  python watch_ego.py --seconds 30
"""
import math
import time
import argparse

import carla


def speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--every", type=float, default=1.0)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    print("map:", world.get_map().name)

    t_end = time.time() + args.seconds
    while time.time() < t_end:
        actors = world.get_actors()
        vehicles = list(actors.filter("vehicle.*"))
        walkers = list(actors.filter("walker.pedestrian.*"))

        # The ego is the vehicle carrying the rig: it is the one with cameras
        # attached to it.  Nothing else in the demo has sensors.
        cams = list(actors.filter("sensor.camera.*"))
        parents = {c.parent.id for c in cams if c.parent is not None}
        ego = next((v for v in vehicles if v.id in parents), None)
        if ego is None:
            print("no ego with attached cameras found -- is the demo running?")
            time.sleep(args.every)
            continue

        kmh = speed_kmh(ego)
        c = ego.get_control()
        try:
            at = ego.is_at_traffic_light()
            tl = str(ego.get_traffic_light_state()).split(".")[-1]
        except Exception:
            at, tl = False, "?"

        et = ego.get_transform()
        fwd = et.get_forward_vector()
        best = None
        for v in vehicles:
            if v.id == ego.id:
                continue
            dl = v.get_transform().location - et.location
            if (dl.x * fwd.x + dl.y * fwd.y) <= 0:
                continue
            d = math.sqrt(dl.x * dl.x + dl.y * dl.y)
            if best is None or d < best[0]:
                best = (d, speed_kmh(v))

        # nearest walker in any direction -- a pedestrian beside the bonnet is enough
        # to make the traffic manager hold station
        wd = None
        for w in walkers:
            dl = w.get_transform().location - et.location
            d = math.sqrt(dl.x * dl.x + dl.y * dl.y)
            if wd is None or d < wd:
                wd = d

        others = [v for v in vehicles if v.id != ego.id]
        moving = sum(1 for v in others if speed_kmh(v) > 1.0)

        if kmh > 0.5:
            why = "MOVING"
        elif at and tl == "Red":
            why = "RED LIGHT"
        elif best and best[0] < 12 and best[1] < 2:
            why = f"YIELDING to vehicle {best[0]:.1f} m"
        elif wd is not None and wd < 12:
            why = f"WALKER {wd:.1f} m away"
        elif c.brake > 0.5:
            why = "STUCK: braking, nothing ahead"
        elif c.throttle > 0.2:
            why = "STUCK: throttle but not moving"
        else:
            why = "STOPPED"

        print("%s  %5.1f km/h  thr %.2f brk %.2f  light %s:%-6s  front %s  walker %s  "
              "traffic %d/%d moving  | %s"
              % (time.strftime("%H:%M:%S"), kmh, c.throttle, c.brake,
                 "AT" if at else "--", tl,
                 f"{best[0]:5.1f} m @{best[1]:4.1f}" if best else "  none",
                 f"{wd:5.1f} m" if wd is not None else "  none",
                 moving, len(others), why), flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
