import sys, time
import carla

HOST, PORT = "127.0.0.1", 2000
TOWN = "Town01"
DEADLINE = time.time() + 180  # 3 min to boot + load

def try_once():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    ver = client.get_server_version()
    world = client.get_world()
    cur = world.get_map().name
    if TOWN.lower().replace("_opt", "") not in cur.lower():
        print(f"loading {TOWN} (current={cur}) ...", flush=True)
        world = client.load_world(TOWN)
        time.sleep(3.0)
    sp = world.get_map().get_spawn_points()
    return ver, world.get_map().name, len(sp)

ok_streak = 0
last_err = None
while time.time() < DEADLINE:
    try:
        ver, mp, n = try_once()
        print(f"server={ver} map={mp} spawn_points={n}", flush=True)
        if n > 0:
            ok_streak += 1
            if ok_streak >= 2:
                print("READY", flush=True)
                sys.exit(0)
        else:
            ok_streak = 0
    except Exception as e:
        ok_streak = 0
        last_err = str(e)[:120]
        print(f"not ready: {last_err}", flush=True)
    time.sleep(5)

print(f"TIMEOUT last_err={last_err}", flush=True)
sys.exit(1)
