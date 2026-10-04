# -*- coding: utf-8 -*-
"""Wait for a freshly started CARLA server to be genuinely ready.

Deliberately does NOT call load_world. get_world() succeeding is a false
positive -- the server answers RPC well before the map is usable -- but the fix
is not to load the map here: a load in the ready check plus the load inside the
collector is a DOUBLE load, and that pair is what repeatedly crushed the server
(2026-07-12). The single load belongs to the collector. Here we only prove the
server is alive and its current map is fully built, by requiring several
consecutive stable non-empty spawn-point reads.

Usage:
  python wait_carla_ready.py [--streak 3] [--timeout 300]
Exit code 0 = READY, 1 = timed out.
"""
import sys
import time
import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--streak", type=int, default=3,
                    help="consecutive stable non-empty spawn-point reads required")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    deadline = time.time() + args.timeout
    streak = 0
    last = None

    while time.time() < deadline:
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(20.0)
            version = client.get_server_version()
            world = client.get_world()
            carla_map = world.get_map()
            spawns = len(carla_map.get_spawn_points())
            state = (carla_map.name, spawns)
            print(f"server={version} map={state[0]} spawn={spawns}", flush=True)

            # stable = same map AND same spawn count as last time, and non-empty
            if spawns > 0 and state == last:
                streak += 1
            elif spawns > 0:
                streak = 1
            else:
                streak = 0
            last = state

            if streak >= args.streak:
                print("READY", flush=True)
                return 0
        except Exception as e:
            streak = 0
            last = None
            print(f"not ready: {str(e)[:110]}", flush=True)
        time.sleep(args.interval)

    print("TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
