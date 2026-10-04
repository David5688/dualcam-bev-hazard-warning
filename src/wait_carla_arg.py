import sys, time
import carla

HOST, PORT = "127.0.0.1", 2000
TOWN = sys.argv[1] if len(sys.argv) > 1 else "Town01"
DEADLINE = time.time() + 200

def try_once():
    client = carla.Client(HOST, PORT); client.set_timeout(20.0)
    ver = client.get_server_version()
    world = client.get_world()
    cur = world.get_map().name
    if TOWN.lower().replace("_opt", "") not in cur.lower():
        world = client.load_world(TOWN); time.sleep(3.0)
    sp = world.get_map().get_spawn_points()
    return ver, world.get_map().name, len(sp)

ok = 0
while time.time() < DEADLINE:
    try:
        ver, mp, n = try_once()
        print(f"server={ver} map={mp} spawn={n}", flush=True)
        if n > 0:
            ok += 1
            if ok >= 2:
                print("READY", flush=True); sys.exit(0)
        else:
            ok = 0
    except Exception as e:
        ok = 0; print(f"not ready: {str(e)[:90]}", flush=True)
    time.sleep(5)
print("TIMEOUT", flush=True); sys.exit(1)
