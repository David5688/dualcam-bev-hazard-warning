"""Get the server onto a map before collecting, however long it takes.

client.load_world() has a hard 60 s wait that client.set_timeout() does NOT
extend. A map being loaded for the first time (the engine builds its cache on
the first visit) can easily blow through that -- Town06 does. But the load does
not abort when the call times out: the engine carries on in the background.

So: fire the load, swallow the timeout, then poll get_world() until the server
reports the map we asked for. The collector then finds the server already there
and skips its own load_world -- which also avoids the double-load that has
repeatedly crashed this box.
"""
import argparse
import sys
import time

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", required=True)
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for the map")
    args = ap.parse_args()

    want = args.town.lower().replace("_opt", "")
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)

    try:
        current = client.get_world().get_map().name
    except Exception as e:
        print(f"cannot reach server: {e}")
        return 1

    if want in current.lower():
        print(f"already on {current}")
        return 0

    print(f"on {current}, loading {args.town} (first visit builds a cache, be patient)")
    t0 = time.time()
    try:
        client.load_world(args.town)
        print(f"load_world returned after {time.time() - t0:.0f}s")
    except Exception as e:
        # expected for a cold map: the call times out but the engine keeps loading
        print(f"load_world timed out after {time.time() - t0:.0f}s -- polling instead")

    while time.time() - t0 < args.wait:
        time.sleep(10)
        try:
            c = carla.Client(args.host, args.port)
            c.set_timeout(30.0)
            name = c.get_world().get_map().name
            if want in name.lower():
                print(f"server is on {name} after {time.time() - t0:.0f}s")
                return 0
            print(f"  ...still on {name} ({time.time() - t0:.0f}s)")
        except Exception:
            print(f"  ...server busy ({time.time() - t0:.0f}s)")

    print(f"gave up after {args.wait}s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
