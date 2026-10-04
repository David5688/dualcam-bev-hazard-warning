import os
import sys
import cv2
import time
import json
import carla
import random
import hashlib
import argparse
import datetime
import numpy as np
from queue import Queue

IMG_W, IMG_H = 1280, 720
FPS = 10

BEV_X_MIN, BEV_X_MAX = -30.0, 30.0
BEV_Y_MIN, BEV_Y_MAX = -30.0, 30.0
BEV_RES = 0.25
BEV_H = int((BEV_X_MAX - BEV_X_MIN) / BEV_RES)
BEV_W = int((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES)

# both cameras sit on the ego x axis, this far fore/aft of the ego origin
CAM_OFFSET_X = 1.5
# depth span the LSS model lifts over; the visible-region test uses the same
# bounds so "visible" here means exactly what it means in fov_recall_ceiling.py
# and eval_visible_region.py
VIS_D_MIN, VIS_D_MAX = 2.0, 42.0

CALIB_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 1


COLOR = {
    0: (0, 0, 0),
    1: (0, 0, 255),       # vehicle
    2: (255, 0, 255),     # pedestrian
    3: (80, 80, 80),      # road
    4: (0, 255, 0),       # sidewalk
    5: (0, 255, 255),     # lane marking (yellow)
}

def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def depth_png_to_metres(bgr):
    """Decode CARLA's raw depth encoding (stored as BGR PNG) to metres.

    CARLA packs depth as normalized = (R + G*256 + B*256^2) / (256^3 - 1),
    metres = 1000 * normalized. Far plane is 1000 m.
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return 1000.0 * normalized

def safe_destroy(actor):
    try:
        if actor is not None and actor.is_alive:
            actor.destroy()
    except Exception:
        pass

def write_json_atomic(path, payload):
    """Write via a temp file + replace, so a crash mid-write cannot leave a
    truncated manifest/poses file behind on a long collection run."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def dataset_digest(out_dir, names, subdirs):
    """sha256 over '<subdir>/<name>:<bytes>' for every saved file.

    A content hash of ~20 000 PNGs would cost minutes; name+size catches the
    failure that actually happens here (a run that died part-way leaving a
    truncated or missing file) and is cheap enough to run at the end of a
    collection.
    """
    h = hashlib.sha256()
    for n in names:
        for sub in subdirs:
            try:
                size = os.path.getsize(os.path.join(out_dir, sub, n))
            except OSError:
                size = -1
            h.update(("%s/%s:%d\n" % (sub, n, size)).encode("utf-8"))
    return h.hexdigest()


def colorize(label):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, c in COLOR.items():
        out[label == k] = c
    return out

def get_intrinsic(width, height, fov):
    focal = width / (2.0 * np.tan(np.radians(fov) / 2.0))
    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K


def carla_transform_to_matrix(transform):
    loc = transform.location
    rot = transform.rotation

    cy = np.cos(np.radians(rot.yaw))
    sy = np.sin(np.radians(rot.yaw))
    cp = np.cos(np.radians(rot.pitch))
    sp = np.sin(np.radians(rot.pitch))
    cr = np.cos(np.radians(rot.roll))
    sr = np.sin(np.radians(rot.roll))

    R_yaw = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ], dtype=np.float32)

    R_pitch = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ], dtype=np.float32)

    R_roll = np.array([
        [1, 0,  0],
        [0, cr, -sr],
        [0, sr,  cr]
    ], dtype=np.float32)

    R = R_yaw @ R_pitch @ R_roll

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [loc.x, loc.y, loc.z]

    return T

def world_to_ego_xy(actor_loc, ego_tf):
    ego_loc = ego_tf.location
    dx = actor_loc.x - ego_loc.x
    dy = actor_loc.y - ego_loc.y

    yaw = np.radians(ego_tf.rotation.yaw)
    c = np.cos(-yaw)
    s = np.sin(-yaw)

    x = c * dx - s * dy
    y = s * dx + c * dy

    return x, y

def ego_xy_to_world_loc(x, y, ego_tf):
    yaw = np.radians(ego_tf.rotation.yaw)
    c = np.cos(yaw)
    s = np.sin(yaw)

    dx = c * x - s * y
    dy = s * x + c * y

    wx = ego_tf.location.x + dx
    wy = ego_tf.location.y + dy

    return carla.Location(x=wx, y=wy, z=ego_tf.location.z)

def ego_xy_to_bev_px(xy):
    """Ego-frame metres -> continuous BEV raster (col, row).

    Same axes as draw_map_bev: x (ego forward) runs up the rows, y (ego right)
    runs along the columns. The result is in PIXEL-CENTRE coordinates -- integer
    value k means "the centre of cell k" -- because that is what cv2's polygon
    fill rule expects. draw_map_bev's `x = X_MIN + (BEV_H-1-row)*RES` gives a
    cell's lower EDGE, hence the half-cell shift here; without it every box
    rasterises one row and one column too wide.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    col = (xy[:, 1] - BEV_Y_MIN) / BEV_RES - 0.5
    row = (BEV_H - 1) - (xy[:, 0] - BEV_X_MIN) / BEV_RES + 0.5
    return np.stack([col, row], axis=1)


def fill_convex_polygon(bev, poly_px, cls_id):
    """Paint every cell whose CENTRE lies inside the convex polygon.

    cv2.fillPoly is deliberately not used here: it paints every cell the polygon
    merely touches (rounding each edge outwards), which turns a 4.0 x 2.0 m car
    into 17 x 9 = 153 cells instead of its true 16 x 8 = 128 -- a ~20% inflation
    of every vehicle in the ground truth, and worse in relative terms for small
    actors. The polygon is convex (it is a convex hull), so the exact test is
    just "same side of every edge", vectorised over the candidate cells.

    Returns the number of cells painted.
    """
    c0 = max(int(np.floor(poly_px[:, 0].min())), 0)
    c1 = min(int(np.ceil(poly_px[:, 0].max())), BEV_W - 1)
    r0 = max(int(np.floor(poly_px[:, 1].min())), 0)
    r1 = min(int(np.ceil(poly_px[:, 1].max())), BEV_H - 1)
    if c1 < c0 or r1 < r0:
        return 0

    cc, rr = np.meshgrid(np.arange(c0, c1 + 1, dtype=np.float64),
                         np.arange(r0, r1 + 1, dtype=np.float64))
    inside = np.ones(cc.shape, dtype=bool)

    x = poly_px[:, 0]
    y = poly_px[:, 1]
    # signed area fixes the winding, so the test works for either orientation
    twice_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    sgn = 1.0 if twice_area >= 0 else -1.0

    for i in range(poly_px.shape[0]):
        x0, y0 = poly_px[i]
        x1, y1 = poly_px[(i + 1) % poly_px.shape[0]]
        cross = (x1 - x0) * (rr - y0) - (y1 - y0) * (cc - x0)
        inside &= (sgn * cross >= -1e-9)      # cells exactly on an edge count

    if not inside.any():
        return 0
    bev[r0:r1 + 1, c0:c1 + 1][inside] = cls_id
    return int(inside.sum())


def actor_footprint_ego(actor, ego_tf):
    """Ego-frame (x, y) footprint corners of an actor's REAL bounding box.

    Built from CARLA's own ``bounding_box.get_world_vertices``, so the box's
    local offset (``bounding_box.location``) and the actor's pitch/roll are all
    honoured. The previous version rebuilt the box from ``extent`` + yaw about
    the actor origin, which shifted every vehicle whose box is not centred on
    that origin, and it clamped the size up to 0.8 x 0.6 m -- inflating a
    pedestrian to about three times its true footprint.

    Returns the convex hull of the 8 projected vertices (the top-down
    footprint), or None if the actor has no usable box.
    """
    tf = actor.get_transform()
    bbox = actor.bounding_box
    if bbox is None:
        return None

    try:
        verts = bbox.get_world_vertices(tf)
        pts = np.array([world_to_ego_xy(v, ego_tf) for v in verts], dtype=np.float32)
    except Exception:
        # older API: rebuild from extent + local offset + yaw (still honours
        # bounding_box.location, which the old code ignored)
        ext, off = bbox.extent, bbox.location
        yaw = np.radians(tf.rotation.yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([
            [off.x + ext.x, off.y + ext.y],
            [off.x + ext.x, off.y - ext.y],
            [off.x - ext.x, off.y - ext.y],
            [off.x - ext.x, off.y + ext.y],
        ], dtype=np.float32)
        wx = tf.location.x + c * local[:, 0] - s * local[:, 1]
        wy = tf.location.y + s * local[:, 0] + c * local[:, 1]
        pts = np.array([
            world_to_ego_xy(carla.Location(x=float(a), y=float(b), z=0.0), ego_tf)
            for a, b in zip(wx, wy)
        ], dtype=np.float32)

    if pts.shape[0] < 3:
        return None
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return hull.reshape(-1, 2).astype(np.float32)


def draw_actor_box(bev, actor, ego_tf, cls_id, footprint=None):
    """Rasterize an actor's real bounding-box footprint into the BEV.

    An actor is drawn whenever ANY PART of its box falls inside the grid -- not
    only when its centre does. The old centre-only test dropped a whole bus or
    truck the moment its centre crossed the 30 m boundary, even while its tail
    was still several metres inside the view, which under-counted vehicles right
    at the edge. The fill clips to the canvas, so we only need a cheap reject
    for actors that cannot possibly overlap.
    """
    fp = footprint if footprint is not None else actor_footprint_ego(actor, ego_tf)
    if fp is None or fp.shape[0] < 3:
        return

    # cheap reject: the whole footprint sits off one side of the grid
    if (fp[:, 0].max() < BEV_X_MIN or fp[:, 0].min() > BEV_X_MAX or
            fp[:, 1].max() < BEV_Y_MIN or fp[:, 1].min() > BEV_Y_MAX):
        return

    px = ego_xy_to_bev_px(fp)
    painted = fill_convex_polygon(bev, px, cls_id)

    # rasterisation guarantee: a real pedestrian footprint is only ~0.4 m across
    # = under two cells, so its polygon can miss every cell centre and vanish.
    # Rather than inflate every box (the old 0.8 x 0.6 m clamp, which is what
    # made pedestrians too fat), paint the single centre cell in that case.
    if painted == 0:
        cc = int(np.floor(float(px[:, 0].mean()) + 0.5))
        rr = int(np.floor(float(px[:, 1].mean()) + 0.5))
        if 0 <= cc < BEV_W and 0 <= rr < BEV_H:
            bev[rr, cc] = cls_id

def visible_xy(x, y, fov_deg=90.0, d_min=VIS_D_MIN, d_max=VIS_D_MAX,
               cam_x=CAM_OFFSET_X):
    """Is the ego-frame point (x, y) inside either camera's wedge?

    Same convention as fov_recall_ceiling.py / eval_visible_region.py (front cam
    at +cam_x looking +x, rear cam at -cam_x looking -x, both clipped to the LSS
    depth span), generalised from their hard-coded 90 deg to any FOV: at 90 deg
    tan(45) == 1, which reduces exactly to their |lateral| <= |along-axis| test.
    Works on scalars or arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.tan(np.radians(float(fov_deg)) / 2.0)
    # tan(45 deg) evaluates to 0.9999999999999999, so a point sitting exactly on
    # the 90 deg diagonal (|lateral| == |along-axis|, which happens on a great
    # many cells of a 0.25 m grid) would fall outside by one ULP. EPS is far
    # below the cell size, so it only recovers the exact-boundary case.
    eps = 1e-9

    fx, fy = x - cam_x, y
    fd = np.hypot(fx, fy)
    front = (fx > 0) & (np.abs(fy) <= t * fx + eps) & (fd >= d_min) & (fd <= d_max)

    rx, ry = x + cam_x, y
    rd = np.hypot(rx, ry)
    rear = (rx < 0) & (np.abs(ry) <= t * (-rx) + eps) & (rd >= d_min) & (rd <= d_max)

    return front | rear


def filter_decision(n_veh, n_ped, min_veh, min_ped, keep_empty_ratio, rng=random):
    """'content' | 'negative' | 'skip' for one frame's visible-actor counts.

    Below-threshold frames are not simply dropped: keeping a slice of them
    preserves negatives, without which the model learns "there is always a car"
    and hallucinates vehicles in empty scenes.
    """
    if n_veh >= min_veh or n_ped >= min_ped:
        return "content"
    if rng.random() < keep_empty_ratio:
        return "negative"
    return "skip"


def count_visible_actors(world, ego, fov_deg=90.0, d_min=VIS_D_MIN, d_max=VIS_D_MAX):
    """(vehicles, pedestrians) whose footprint reaches into the camera wedges.

    This is the content-filter metric: "is there anything in this frame the model
    could possibly learn from", judged on the region the cameras actually see
    rather than on the whole +-30 m BEV (half of which is a blind side wedge).
    An actor counts if ANY footprint corner or its centre is visible. There is
    no occlusion test -- a vehicle hidden behind a building still counts -- so
    this is an upper bound on what is genuinely photographed.
    """
    ego_tf = ego.get_transform()
    reject = d_max + 10.0
    counts = {"vehicle": 0, "pedestrian": 0}
    all_actors = world.get_actors()      # one RPC, filtered twice below

    for key, pattern in (("vehicle", "vehicle.*"), ("pedestrian", "walker.pedestrian.*")):
        for actor in all_actors.filter(pattern):
            if actor.id == ego.id:
                continue
            try:
                cx, cy = world_to_ego_xy(actor.get_transform().location, ego_tf)
                if abs(cx) > reject or abs(cy) > reject:      # cheap reject first
                    continue
                fp = actor_footprint_ego(actor, ego_tf)
                if fp is None:
                    continue
                xs = np.append(fp[:, 0], fp[:, 0].mean())
                ys = np.append(fp[:, 1], fp[:, 1].mean())
                if bool(visible_xy(xs, ys, fov_deg, d_min, d_max).any()):
                    counts[key] += 1
            except Exception:
                continue

    return counts["vehicle"], counts["pedestrian"]


def build_lane_points(carla_map, spacing=0.5):
    """Precompute (world_x, world_y) of all lane-marking samples once (map is static)."""
    pts = []
    for wp in carla_map.generate_waypoints(spacing):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        loc = wp.transform.location
        fwd = wp.transform.get_forward_vector()
        rx, ry = -fwd.y, fwd.x          # unit world-space right vector
        hw = wp.lane_width / 2.0

        rm = wp.right_lane_marking
        if rm is not None and rm.type != carla.LaneMarkingType.NONE:
            pts.append((loc.x + rx * hw, loc.y + ry * hw))

        lm = wp.left_lane_marking
        if lm is not None and lm.type != carla.LaneMarkingType.NONE:
            pts.append((loc.x - rx * hw, loc.y - ry * hw))

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def draw_lane_markings(bev, lane_pts, ego_tf, cls_id=5):
    """Vectorized: transform cached world lane points into ego BEV and paint cls_id."""
    if lane_pts.shape[0] == 0:
        return

    ego_loc = ego_tf.location
    yaw = np.radians(ego_tf.rotation.yaw)
    c = np.cos(-yaw)
    s = np.sin(-yaw)

    dx = lane_pts[:, 0] - ego_loc.x
    dy = lane_pts[:, 1] - ego_loc.y

    x = c * dx - s * dy
    y = s * dx + c * dy

    m = (x >= BEV_X_MIN) & (x < BEV_X_MAX) & (y >= BEV_Y_MIN) & (y < BEV_Y_MAX)
    x = x[m]
    y = y[m]

    bx = ((x - BEV_X_MIN) / BEV_RES).astype(np.int32)
    by = ((y - BEV_Y_MIN) / BEV_RES).astype(np.int32)

    row = BEV_H - 1 - bx
    col = by

    v = (row >= 0) & (row < BEV_H) & (col >= 0) & (col < BEV_W)
    bev[row[v], col[v]] = cls_id


def make_bev_gt(world, ego, lane_pts=None):
    bev = np.zeros((BEV_H, BEV_W), dtype=np.uint8)

    draw_map_bev(world, ego, bev, step=2)

    ego_tf = ego.get_transform()

    if lane_pts is not None:
        draw_lane_markings(bev, lane_pts, ego_tf, cls_id=5)

    actors = world.get_actors()

    vehicles = actors.filter("vehicle.*")
    walkers = actors.filter("walker.pedestrian.*")

    for v in vehicles:
        if v.id == ego.id:
            continue
        try:
            draw_actor_box(bev, v, ego_tf, 1)
        except Exception:
            pass

    for w in walkers:
        try:
            draw_actor_box(bev, w, ego_tf, 2)
        except Exception:
            pass

    return bev

def draw_map_bev(world, ego, bev, step=2):
    carla_map = world.get_map()
    ego_tf = ego.get_transform()

    for row in range(0, BEV_H, step):
        for col in range(0, BEV_W, step):

            x = BEV_X_MIN + (BEV_H - 1 - row) * BEV_RES
            y = BEV_Y_MIN + col * BEV_RES

            loc = ego_xy_to_world_loc(x, y, ego_tf)

            wp_road = carla_map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=carla.LaneType.Driving
            )

            if wp_road is not None:
                bev[row:row+step, col:col+step] = 3
                continue

            wp_sidewalk = carla_map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=carla.LaneType.Sidewalk
            )

            if wp_sidewalk is not None:
                bev[row:row+step, col:col+step] = 4

VEHICLE_CATEGORIES = ("car", "large", "two_wheel")
LARGE_ID_KEYWORDS = ("truck", "bus", "carlacola", "firetruck", "ambulance",
                     "sprinter", "hgv", "cybertruck", "volkswagen.t2")


def vehicle_category(bp):
    """car / large (truck+bus+van) / two_wheel (motorcycle+bicycle).

    The collector used to hard-filter to number_of_wheels == 4, which threw away
    every motorcycle and bicycle and was the main reason the traffic looked so
    uniform. Categories let us spawn a deliberate mix instead. CARLA >= 0.9.12
    exposes base_type on the blueprint; the wheel count and id keywords are
    fallbacks for anything that does not.
    """
    if bp.has_attribute("base_type"):
        base = str(bp.get_attribute("base_type").as_str()).lower()
        if base in ("truck", "van", "bus"):
            return "large"
        if base in ("motorcycle", "bicycle"):
            return "two_wheel"
        if base == "car":
            return "car"

    if bp.has_attribute("number_of_wheels"):
        try:
            if int(bp.get_attribute("number_of_wheels")) == 2:
                return "two_wheel"
        except (TypeError, ValueError):
            pass

    bid = str(bp.id).lower()
    if any(k in bid for k in LARGE_ID_KEYWORDS):
        return "large"
    return "car"


def parse_vehicle_mix(text):
    """'60,20,20' -> normalised weights for (car, large, two_wheel)."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != len(VEHICLE_CATEGORIES):
        raise ValueError(f"--vehicle_mix needs {len(VEHICLE_CATEGORIES)} comma-separated "
                         f"numbers (car,large,two_wheel), got {text!r}")
    vals = [float(p) for p in parts]
    if any(v < 0 for v in vals):
        raise ValueError("--vehicle_mix values must be >= 0")
    total = sum(vals)
    if total <= 0:
        raise ValueError("--vehicle_mix must not be all zeros")
    return tuple(v / total for v in vals)


def pick_vehicle_category(mix, available, rng=random):
    """Draw a category by weight, ignoring categories with no blueprint."""
    weights = [w if available.get(c) else 0.0 for c, w in zip(VEHICLE_CATEGORIES, mix)]
    if sum(weights) <= 0:
        return None
    return rng.choices(VEHICLE_CATEGORIES, weights=weights, k=1)[0]


def spawn_npc_vehicles(world, client, bp_lib, ego, num_vehicles,
                       mix=(0.6, 0.2, 0.2), near_ego=0.0, seed=None):
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(10.0)
    if seed is not None:
        # only pin the traffic manager when the caller asked for a reproducible
        # run; the paper datasets deliberately leave the random source free
        tm.set_random_device_seed(int(seed))

    by_cat = {c: [] for c in VEHICLE_CATEGORIES}
    for bp in bp_lib.filter("vehicle.*"):
        by_cat[vehicle_category(bp)].append(bp)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    if near_ego > 0:
        # bias traffic towards the ego so filtered collection finds content
        # sooner; points outside the radius stay as a tail rather than being
        # dropped, so we can still reach num_vehicles on a big map
        ego_loc = ego.get_location()
        spawn_points.sort(key=lambda sp: 0 if sp.location.distance(ego_loc) <= near_ego else 1)

    npcs = []
    spawned = {c: 0 for c in VEHICLE_CATEGORIES}

    for sp in spawn_points:
        if len(npcs) >= num_vehicles:
            break

        if sp.location.distance(ego.get_location()) < 10.0:
            continue

        cat = pick_vehicle_category(mix, by_cat)
        if cat is None:
            break

        bp = random.choice(by_cat[cat])
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        npc = world.try_spawn_actor(bp, sp)

        if npc is None and cat != "car" and by_cat["car"]:
            # buses and trucks often do not fit where a car would; fall back so
            # a failed large draw does not silently cost us a vehicle slot
            cat = "car"
            bp = random.choice(by_cat["car"])
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            npc = world.try_spawn_actor(bp, sp)

        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
            spawned[cat] += 1

    return npcs, tm, spawned

def spawn_walkers(world, bp_lib, num_walkers, ego=None, near_ego=0.0):
    walkers = []
    controllers = []

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    def nav_location():
        """A navigable spawn point, preferring ones near the ego when asked."""
        if near_ego > 0 and ego is not None:
            ego_loc = ego.get_location()
            for _ in range(20):
                loc = world.get_random_location_from_navigation()
                if loc is not None and loc.distance(ego_loc) <= near_ego:
                    return loc
        return world.get_random_location_from_navigation()

    for _ in range(num_walkers):
        loc = nav_location()
        if loc is None:
            continue

        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            continue

        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

        walkers.append(walker)
        controllers.append(controller)

    world.tick()

    for c in controllers:
        try:
            c.start()
            loc = nav_location()
            if loc is not None:
                c.go_to_location(loc)
            c.set_max_speed(random.uniform(0.8, 1.8))
        except Exception:
            pass

    return walkers, controllers

class _StubActor:
    """Minimal stand-in for a carla.Actor, for the offline self-test.

    The bounding box and transform are REAL carla objects, so the geometry path
    exercised here (including bounding_box.get_world_vertices) is the same code
    that runs against a live server.
    """

    def __init__(self, x, y, yaw, extent, bbox_offset=(0.0, 0.0, 0.0), actor_id=1):
        self._tf = carla.Transform(carla.Location(x=float(x), y=float(y), z=0.0),
                                   carla.Rotation(yaw=float(yaw)))
        self.bounding_box = carla.BoundingBox(
            carla.Location(x=float(bbox_offset[0]), y=float(bbox_offset[1]),
                           z=float(bbox_offset[2])),
            carla.Vector3D(float(extent[0]), float(extent[1]), float(extent[2])))
        self.id = actor_id

    def get_transform(self):
        return self._tf


class _StubBlueprint:
    def __init__(self, bp_id, attrs=None):
        self.id = bp_id
        self._attrs = dict(attrs or {})

    def has_attribute(self, key):
        return key in self._attrs

    def get_attribute(self, key):
        return _StubAttribute(self._attrs[key])


class _StubAttribute:
    def __init__(self, value):
        self._v = value

    def as_str(self):
        return str(self._v)

    def __int__(self):
        return int(self._v)

    def __str__(self):
        return str(self._v)


def _drawn_cells(actor, cls_id=1, ego_tf=None):
    """Rasterize one actor alone and return (pixel count, ego-frame centroid)."""
    ego_tf = ego_tf or carla.Transform(carla.Location(), carla.Rotation())
    bev = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    draw_actor_box(bev, actor, ego_tf, cls_id)
    rows, cols = np.nonzero(bev == cls_id)
    if rows.size == 0:
        return 0, None, bev
    # cell CENTRES (draw_map_bev's formula gives the lower edge)
    x = BEV_X_MIN + (BEV_H - 1 - rows) * BEV_RES + BEV_RES / 2.0
    y = BEV_Y_MIN + cols * BEV_RES + BEV_RES / 2.0
    return int(rows.size), (float(x.mean()), float(y.mean())), bev


def run_selftest():
    """Offline checks: no CARLA server, no dataset touched (only a temp dir)."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))

    print("[1] visible region matches fov_recall_ceiling.py / eval_visible_region.py")
    rows = np.arange(BEV_H)[:, None]
    cols = np.arange(BEV_W)[None, :]
    X = np.broadcast_to(BEV_X_MIN + (BEV_H - 1 - rows) * BEV_RES, (BEV_H, BEV_W))
    Y = np.broadcast_to(BEV_Y_MIN + cols * BEV_RES, (BEV_H, BEV_W))
    fx, fy = X - 1.5, Y
    ref_front = (fx > 0) & (np.abs(fx) >= np.abs(fy)) & (np.hypot(fx, fy) >= 2.0) & (np.hypot(fx, fy) <= 42.0)
    rx, ry = X + 1.5, Y
    ref_rear = (rx < 0) & (np.abs(rx) >= np.abs(ry)) & (np.hypot(rx, ry) >= 2.0) & (np.hypot(rx, ry) <= 42.0)
    ref = ref_front | ref_rear
    mine = visible_xy(X, Y, 90.0, 2.0, 42.0)
    check("90 deg mask identical cell-for-cell", np.array_equal(ref, mine),
          f"visible = {ref.mean():.1%} of the grid")
    wide = visible_xy(X, Y, 120.0, 2.0, 42.0)
    check("120 deg strictly contains 90 deg", bool((wide | ref == wide).all()) and wide.sum() > ref.sum(),
          f"{ref.mean():.1%} -> {wide.mean():.1%}")
    check("blind side wedge really is blind", not bool(visible_xy(0.0, 20.0, 90.0)),
          "point 20 m to the left is not visible at 90 deg")

    print("[2] bounding box uses the real corners (bbox.location honoured)")
    # box centre sits 1 m ahead of the actor origin -- the old extent+yaw rebuild
    # ignored this and drew every such vehicle 1 m out of place
    car = _StubActor(x=10.0, y=0.0, yaw=0.0, extent=(2.0, 1.0, 0.75), bbox_offset=(1.0, 0.0, 0.75))
    n, centroid, _ = _drawn_cells(car)
    check("offset box lands at 11 m not 10 m", n > 0 and abs(centroid[0] - 11.0) < 0.2,
          f"centroid x = {centroid[0]:.2f} m, {n} px")
    check("lateral centroid stays on axis", abs(centroid[1]) < 0.15, f"y = {centroid[1]:.3f} m")
    expected = (4.0 * 2.0) / (BEV_RES ** 2)
    check("area matches the true footprint", abs(n - expected) / expected < 0.10,
          f"{n} px vs {expected:.0f} px expected")

    print("[3] rotation and the ego frame")
    turned = _StubActor(x=10.0, y=0.0, yaw=90.0, extent=(2.0, 1.0, 0.75))
    _, _, bev_t = _drawn_cells(turned)
    r, c = np.nonzero(bev_t == 1)
    span_x = (r.max() - r.min() + 1) * BEV_RES
    span_y = (c.max() - c.min() + 1) * BEV_RES
    check("yaw 90 swaps length and width", abs(span_x - 2.0) < 0.3 and abs(span_y - 4.0) < 0.3,
          f"x span {span_x:.2f} m, y span {span_y:.2f} m")
    right = _StubActor(x=0.0, y=8.0, yaw=0.0, extent=(2.0, 1.0, 0.75))
    _, cen_r, _ = _drawn_cells(right)
    check("+y is the ego's right, not its front", abs(cen_r[1] - 8.0) < 0.2 and abs(cen_r[0]) < 0.2,
          f"centroid = ({cen_r[0]:.2f}, {cen_r[1]:.2f})")

    print("[4] a bus straddling the 30 m boundary is still drawn (2026-07-14 fix)")
    bus = _StubActor(x=31.0, y=0.0, yaw=0.0, extent=(6.0, 1.3, 1.6))
    n_bus, cen_bus, _ = _drawn_cells(bus)
    check("tail inside the grid still rasterises", n_bus > 0, f"{n_bus} px")
    check("only the in-range part is drawn", cen_bus is not None and cen_bus[0] < BEV_X_MAX,
          f"centroid x = {cen_bus[0]:.2f} m")
    gone = _StubActor(x=45.0, y=0.0, yaw=0.0, extent=(6.0, 1.3, 1.6))
    check("a bus fully outside draws nothing", _drawn_cells(gone)[0] == 0)

    print("[5] pedestrians keep their true size (no 0.8 x 0.6 m inflation)")
    ped = _StubActor(x=8.0, y=2.0, yaw=0.0, extent=(0.18, 0.18, 0.93), actor_id=2)
    n_ped, cen_ped, _ = _drawn_cells(ped, cls_id=2)
    old_floor_px = (0.8 * 0.6) / (BEV_RES ** 2)      # what the old clamp produced
    true_px = (0.36 * 0.36) / (BEV_RES ** 2)
    check("smaller than the old inflated footprint", 0 < n_ped < old_floor_px,
          f"{n_ped} px vs old {old_floor_px:.0f} px (true {true_px:.1f} px)")
    check("still located correctly", abs(cen_ped[0] - 8.0) < 0.4 and abs(cen_ped[1] - 2.0) < 0.4,
          f"centroid = ({cen_ped[0]:.2f}, {cen_ped[1]:.2f})")
    tiny = _StubActor(x=5.0, y=0.0, yaw=0.0, extent=(0.05, 0.05, 0.5), actor_id=3)
    n_tiny, cen_tiny, _ = _drawn_cells(tiny, cls_id=2)
    check("sub-cell actor still gets >= 1 px", n_tiny >= 1, f"{n_tiny} px")
    check("...placed at its own centre", cen_tiny is not None and abs(cen_tiny[0] - 5.0) < 0.3)

    print("[6] vehicle categories and the spawn mix")
    cases = [
        (_StubBlueprint("vehicle.tesla.model3", {"base_type": "car", "number_of_wheels": 4}), "car"),
        (_StubBlueprint("vehicle.carlamotors.firetruck", {"base_type": "truck", "number_of_wheels": 4}), "large"),
        (_StubBlueprint("vehicle.volkswagen.t2", {"base_type": "van", "number_of_wheels": 4}), "large"),
        (_StubBlueprint("vehicle.mitsubishi.fusorosa", {"base_type": "bus", "number_of_wheels": 4}), "large"),
        (_StubBlueprint("vehicle.yamaha.yzf", {"base_type": "motorcycle", "number_of_wheels": 2}), "two_wheel"),
        (_StubBlueprint("vehicle.bh.crossbike", {"base_type": "bicycle", "number_of_wheels": 2}), "two_wheel"),
        (_StubBlueprint("vehicle.gazelle.omafiets", {"number_of_wheels": 2}), "two_wheel"),
        (_StubBlueprint("vehicle.carlamotors.carlacola", {"number_of_wheels": 4}), "large"),
        (_StubBlueprint("vehicle.audi.a2", {"number_of_wheels": 4}), "car"),
    ]
    bad = [(bp.id, vehicle_category(bp), want) for bp, want in cases if vehicle_category(bp) != want]
    check("every blueprint lands in the right category", not bad, str(bad))
    check("mix parses and normalises", parse_vehicle_mix("60,20,20") == (0.6, 0.2, 0.2))
    for bad_mix in ("60,20", "60,20,20,10", "0,0,0", "-1,50,50", "a,b,c"):
        try:
            parse_vehicle_mix(bad_mix)
            check(f"rejects {bad_mix!r}", False)
        except ValueError:
            pass
    rng = random.Random(0)
    avail = {c: ["bp"] for c in VEHICLE_CATEGORIES}
    draws = [pick_vehicle_category((0.6, 0.2, 0.2), avail, rng) for _ in range(40000)]
    frac = {c: draws.count(c) / len(draws) for c in VEHICLE_CATEGORIES}
    check("40k draws reproduce 60/20/20",
          all(abs(frac[c] - w) < 0.01 for c, w in zip(VEHICLE_CATEGORIES, (0.6, 0.2, 0.2))),
          "car %.3f large %.3f two_wheel %.3f" % (frac["car"], frac["large"], frac["two_wheel"]))
    avail_no2 = {"car": ["bp"], "large": ["bp"], "two_wheel": []}
    draws2 = [pick_vehicle_category((0.6, 0.2, 0.2), avail_no2, rng) for _ in range(2000)]
    check("empty category renormalises away", "two_wheel" not in draws2,
          "car %.2f" % (draws2.count("car") / len(draws2)))

    print("[7] content filter decisions")
    check("2 vehicles pass", filter_decision(2, 0, 2, 2, 0.0) == "content")
    check("2 pedestrians pass on their own", filter_decision(0, 2, 2, 2, 0.0) == "content")
    check("1 and 1 is below threshold", filter_decision(1, 1, 2, 2, 0.0) == "skip")
    check("keep_empty_ratio 1.0 keeps everything", filter_decision(0, 0, 2, 2, 1.0) == "negative")
    rng = random.Random(1)
    dec = [filter_decision(0, 0, 2, 2, 0.2, rng) for _ in range(20000)]
    neg = dec.count("negative") / len(dec)
    check("negatives land near 20%", abs(neg - 0.2) < 0.02, f"{neg:.3f}")

    print("[8] manifest helpers (temp dir, cleaned up)")
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="dualcam_selftest_")
    try:
        os.makedirs(os.path.join(tmp, "bev_gt"))
        names = []
        for i in range(3):
            n = "%06d.png" % i
            names.append(n)
            with open(os.path.join(tmp, "bev_gt", n), "wb") as f:
                f.write(b"x" * (10 + i))
        p = os.path.join(tmp, "manifest.json")
        write_json_atomic(p, {"a": 1, "b": "路"})
        write_json_atomic(p, {"a": 2, "b": "路"})       # overwrite in place
        with open(p, encoding="utf-8") as f:
            round_trip = json.load(f)
        check("atomic write overwrites and round-trips utf-8",
              round_trip == {"a": 2, "b": "路"} and not os.path.exists(p + ".tmp"))
        d1 = dataset_digest(tmp, names, ["bev_gt"])
        d2 = dataset_digest(tmp, names, ["bev_gt"])
        check("digest is stable", d1 == d2, d1[:16] + "...")
        with open(os.path.join(tmp, "bev_gt", names[1]), "wb") as f:
            f.write(b"x" * 999)                              # simulate a truncated frame
        check("digest catches a changed file size", dataset_digest(tmp, names, ["bev_gt"]) != d1)
        check("digest tolerates a missing file",
              isinstance(dataset_digest(tmp, names + ["nope.png"], ["bev_gt"]), str))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[9] geometry constants agree with the LSS model")
    check("BEV grid is 240 x 240 at 0.25 m", (BEV_H, BEV_W, BEV_RES) == (240, 240, 0.25))
    check("camera offset matches the rig", CAM_OFFSET_X == 1.5)
    check("visible-region depth span matches dbound", (VIS_D_MIN, VIS_D_MAX) == (2.0, 42.0))

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.64.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--out", default="carla_dualcam_gt_out")
    parser.add_argument("--frames", type=int, default=2000,
                        help="how many frames to SAVE. With --filter the run keeps ticking "
                             "past skipped frames until this many are on disk (or --max_ticks).")
    parser.add_argument("--max_ticks", type=int, default=0,
                        help="safety cap on simulated ticks; 0 = frames*8 when filtering, "
                             "frames when not (an unfiltered run saves every tick)")
    parser.add_argument("--vehicles", type=int, default=40,
                        help="NPC vehicle count. Above ~40 the traffic gridlocks and the ego "
                             "stops moving, which wastes the whole session.")
    parser.add_argument("--walkers", type=int, default=60,
                        help="pedestrian count; pedestrian is the weakest class, and walkers "
                             "do not gridlock traffic, so this runs higher than --vehicles")
    parser.add_argument("--vehicle_mix", default="60,20,20",
                        help="spawn ratio car,large(truck+bus+van),two_wheel(motorcycle+bicycle). "
                             "The old collector hard-filtered to 4-wheelers, which is why the "
                             "traffic was so uniform.")
    parser.add_argument("--near_ego", type=float, default=0.0,
                        help="metres: bias NPC vehicle/pedestrian spawning to within this radius "
                             "of the ego (0 = whole map). Raises the content-filter keep rate on "
                             "big maps where traffic otherwise scatters out of sight.")
    parser.add_argument("--filter", action="store_true",
                        help="content filter: only save a frame when the cameras can actually see "
                             "actors (see --min_veh/--min_ped), plus --keep_empty_ratio negatives. "
                             "TRAINING cities only -- val/test/demo must stay unfiltered so they "
                             "are continuous and keep their natural distribution.")
    parser.add_argument("--min_veh", type=int, default=2,
                        help="save when this many vehicles are inside the camera wedges")
    parser.add_argument("--min_ped", type=int, default=2,
                        help="...or this many pedestrians (the two thresholds are OR'd)")
    parser.add_argument("--keep_empty_ratio", type=float, default=0.2,
                        help="fraction of below-threshold frames to keep anyway. Empty frames are "
                             "NEGATIVES: drop them all and the model learns 'there is always a car' "
                             "and hallucinates in empty scenes, hurting precision.")
    parser.add_argument("--vis_dmin", type=float, default=VIS_D_MIN,
                        help="near bound of the visible region used by the filter (LSS dbound)")
    parser.add_argument("--vis_dmax", type=float, default=VIS_D_MAX,
                        help="far bound of the visible region used by the filter (LSS dbound)")
    parser.add_argument("--random_spawn", action="store_true")
    parser.add_argument("--spawn_id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preview", default="dualcam_gt_preview.mp4")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="simulation fps (higher = smoother motion, denser sampling)")
    parser.add_argument("--prefix", default="",
                        help="filename prefix, e.g. Town01_ (lets multiple towns share one dataset dir)")
    parser.add_argument("--cam_w", type=int, default=IMG_W)
    parser.add_argument("--cam_h", type=int, default=IMG_H)
    parser.add_argument("--fov", type=float, default=90.0,
                        help="horizontal FOV of both cameras. Wider (e.g. 120) shrinks the "
                             "left/right blind wedges that make turns fail. The intrinsics K "
                             "written to calibration.json follow this, so the LSS projection "
                             "adapts automatically.")
    parser.add_argument("--depth", action="store_true",
                        help="also capture front/rear depth cameras (for BEVDepth supervision); "
                             "stores CARLA's raw depth encoding as 3-channel PNG, decode with depth_png_to_metres()")
    parser.add_argument("--bev_rgb", action="store_true",
                        help="also capture a top-down bird's-eye RGB camera (30m up, pitch -90, fov 90 "
                             "-> covers +-30m, matching the BEV). REFERENCE/comparison ONLY, never fed to "
                             "the model; saved to bev_rgb/.")
    parser.add_argument("--fov2", type=float, default=None,
                        help="same-scene dual-FOV: attach a SECOND front/rear rig at this FOV, "
                             "photographing the identical ticks; requires --out2. bev_gt is shared "
                             "(FOV-independent), only the RGB differs -> cleanest FOV ablation test set.")
    parser.add_argument("--out2", default=None,
                        help="output dir for the --fov2 rig (gets its own calibration.json + rgb, "
                             "shares this run's bev_gt/bev_gt_color)")
    parser.add_argument("--stuck_abort", type=int, default=0,
                        help="end the segment when the ego has moved less than "
                             "--stuck_move for this many consecutive ticks (0 = off). "
                             "CARLA's autopilot deadlocks at some junctions and behind "
                             "walkers standing in the lane; on Town05 that produced a "
                             "segment of 215 frames in which the ego travelled 12 m and "
                             "84%% of frames were near-duplicates. Leave OFF for val/test, "
                             "which must be one continuous run.")
    parser.add_argument("--stuck_move", type=float, default=0.5,
                        help="metres the ego must cover to count as moving (see --stuck_abort)")
    parser.add_argument("--warmup_secs", type=float, default=0.0,
                        help="after load_world, let the server free-run asynchronously for this "
                             "many seconds before switching to synchronous mode and spawning "
                             "anything. A freshly loaded map is still streaming levels and "
                             "compiling shaders, and on this machine the server dies on the very "
                             "first tick of a just-loaded map in roughly two thirds of Town02 / "
                             "Town05 attempts (RenderThread, 38-50 s after server start). Does "
                             "not touch any captured pixel: nothing is spawned yet.")
    parser.add_argument("--teardown_timeout", type=float, default=30.0,
                        help="RPC timeout (s) used for the teardown phase only, after all frames "
                             "and the manifest are on disk. The server usually dies while "
                             "destroying sensors, and the 240 s collection timeout would "
                             "otherwise be spent waiting on a corpse (~200 s per crashed segment).")
    parser.add_argument("--selftest", action="store_true",
                        help="run the offline geometry/sampling checks and exit (no CARLA server)")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if (args.fov2 is None) != (args.out2 is None):
        parser.error("--fov2 and --out2 must be given together")
    try:
        vehicle_mix = parse_vehicle_mix(args.vehicle_mix)
    except ValueError as e:
        parser.error(str(e))
    if not 0.0 <= args.keep_empty_ratio <= 1.0:
        parser.error("--keep_empty_ratio must be between 0 and 1")

    fps = args.fps
    cam_w = args.cam_w
    cam_h = args.cam_h

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "front_rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "rear_rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "bev_gt"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "bev_gt_color"), exist_ok=True)
    if args.depth:
        os.makedirs(os.path.join(args.out, "front_depth"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "rear_depth"), exist_ok=True)
    if args.bev_rgb:
        os.makedirs(os.path.join(args.out, "bev_rgb"), exist_ok=True)

    if args.fov2 is not None:
        for sub in ("front_rgb", "rear_rgb", "bev_gt", "bev_gt_color"):
            os.makedirs(os.path.join(args.out2, sub), exist_ok=True)
        if args.depth:
            os.makedirs(os.path.join(args.out2, "front_depth"), exist_ok=True)
            os.makedirs(os.path.join(args.out2, "rear_depth"), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(240.0)

    if args.town is not None:
        # load_world has its own hard 60 s wait that client.set_timeout does not
        # extend, and a map being visited for the first time can take longer than
        # that to build its cache. So skip the load when the server already sits
        # on the requested map -- a caller can pre-warm it (see prewarm_map.py).
        # This also avoids the double-load that has repeatedly killed the server.
        current = client.get_world().get_map().name
        if args.town.lower().replace("_opt", "") in current.lower():
            print(f"server already on {current}, skipping load_world")
            world = client.get_world()
        else:
            world = client.load_world(args.town)
    else:
        world = client.get_world()

    # Let the just-loaded map settle before anything is spawned or ticked.  In
    # synchronous mode the server only renders when we tick, so all of a fresh
    # map's level streaming and shader work would otherwise land on the very
    # first tick -- at the same moment four 1280x720 render targets go live.
    # That is exactly where the server dies on Town02 / Town05 (RenderThread,
    # 38-50 s after server start, 0 frames saved).  Free-running for a few
    # seconds first moves that work off the critical tick.  No actor exists yet,
    # so this cannot affect a single captured pixel.
    if args.warmup_secs > 0:
        print(f"warm-up: free-running the server for {args.warmup_secs:.1f}s "
              f"before going synchronous")
        time.sleep(args.warmup_secs)

    settings = world.get_settings()
    old_settings = settings

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()

    if args.spawn_id >= 0:
        ego_spawn = spawn_points[args.spawn_id % len(spawn_points)]
    elif args.random_spawn:
        ego_spawn = random.choice(spawn_points)
    else:
        ego_spawn = spawn_points[0]

    ego = world.try_spawn_actor(ego_bp, ego_spawn)

    if ego is None:
        for sp in random.sample(spawn_points, len(spawn_points)):
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego is not None:
                ego_spawn = sp
                break

    if ego is None:
        raise RuntimeError("cannot spawn ego vehicle")

    print("ego spawn:", ego_spawn.location)

    npc_list, tm, mix_actual = spawn_npc_vehicles(
        world, client, bp_lib, ego, args.vehicles,
        mix=vehicle_mix, near_ego=args.near_ego, seed=args.seed)
    ego.set_autopilot(True, tm.get_port())

    walkers, controllers = spawn_walkers(
        world, bp_lib, args.walkers, ego=ego, near_ego=args.near_ego)

    print("npc vehicles:", len(npc_list),
          "(car %d / large %d / two_wheel %d)" % (mix_actual["car"], mix_actual["large"],
                                                  mix_actual["two_wheel"]))
    print("walkers:", len(walkers))

    fov = args.fov
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(cam_w))
    rgb_bp.set_attribute("image_size_y", str(cam_h))
    rgb_bp.set_attribute("fov", str(fov))

    front_tf = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(yaw=0))
    rear_tf = carla.Transform(carla.Location(x=-1.5, z=1.6), carla.Rotation(yaw=180))

    q_front = Queue()
    q_rear = Queue()

    front_cam = world.spawn_actor(rgb_bp, front_tf, attach_to=ego)
    rear_cam = world.spawn_actor(rgb_bp, rear_tf, attach_to=ego)

    front_cam.listen(q_front.put)
    rear_cam.listen(q_rear.put)

    # depth cameras: identical pose/intrinsics to the RGB cameras, so the depth
    # map is pixel-aligned with the RGB and can supervise the LSS lift step
    front_depth_cam = rear_depth_cam = None
    q_front_depth = Queue()
    q_rear_depth = Queue()
    if args.depth:
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(cam_w))
        depth_bp.set_attribute("image_size_y", str(cam_h))
        depth_bp.set_attribute("fov", str(fov))
        front_depth_cam = world.spawn_actor(depth_bp, front_tf, attach_to=ego)
        rear_depth_cam = world.spawn_actor(depth_bp, rear_tf, attach_to=ego)
        front_depth_cam.listen(q_front_depth.put)
        rear_depth_cam.listen(q_rear_depth.put)

    # top-down bird's-eye RGB (reference/comparison only, NOT a model input).
    # 30 m up, looking straight down, fov 90 -> ground coverage ~= +-30 m (matches BEV).
    bev_rgb_cam = None
    q_bev_rgb = Queue()
    if args.bev_rgb:
        bev_bp = bp_lib.find("sensor.camera.rgb")
        bev_bp.set_attribute("image_size_x", "600")
        bev_bp.set_attribute("image_size_y", "600")
        bev_bp.set_attribute("fov", "90")
        bev_tf = carla.Transform(carla.Location(x=0.0, z=30.0), carla.Rotation(pitch=-90.0))
        bev_rgb_cam = world.spawn_actor(bev_bp, bev_tf, attach_to=ego)
        bev_rgb_cam.listen(q_bev_rgb.put)

    # same-scene second rig at fov2 (attached to the same ego, same pose -> same ticks)
    front_cam2 = rear_cam2 = front_depth_cam2 = rear_depth_cam2 = None
    q_front2 = Queue()
    q_rear2 = Queue()
    q_front_depth2 = Queue()
    q_rear_depth2 = Queue()
    K2 = None
    if args.fov2 is not None:
        rgb_bp2 = bp_lib.find("sensor.camera.rgb")
        rgb_bp2.set_attribute("image_size_x", str(cam_w))
        rgb_bp2.set_attribute("image_size_y", str(cam_h))
        rgb_bp2.set_attribute("fov", str(args.fov2))
        front_cam2 = world.spawn_actor(rgb_bp2, front_tf, attach_to=ego)
        rear_cam2 = world.spawn_actor(rgb_bp2, rear_tf, attach_to=ego)
        front_cam2.listen(q_front2.put)
        rear_cam2.listen(q_rear2.put)
        if args.depth:
            depth_bp2 = bp_lib.find("sensor.camera.depth")
            depth_bp2.set_attribute("image_size_x", str(cam_w))
            depth_bp2.set_attribute("image_size_y", str(cam_h))
            depth_bp2.set_attribute("fov", str(args.fov2))
            front_depth_cam2 = world.spawn_actor(depth_bp2, front_tf, attach_to=ego)
            rear_depth_cam2 = world.spawn_actor(depth_bp2, rear_tf, attach_to=ego)
            front_depth_cam2.listen(q_front_depth2.put)
            rear_depth_cam2.listen(q_rear_depth2.put)
        K2 = get_intrinsic(cam_w, cam_h, args.fov2)

    K = get_intrinsic(cam_w, cam_h, fov)

    front_tf = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=1.6),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )

    rear_tf = carla.Transform(
        carla.Location(x=-1.5, y=0.0, z=1.6),
        carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)
    )

    front_T_ego_cam = carla_transform_to_matrix(front_tf)
    rear_T_ego_cam = carla_transform_to_matrix(rear_tf)

    try:
        town_name = world.get_map().name
    except Exception:
        town_name = args.town or "unknown"
    try:
        carla_version = {"client": client.get_client_version(),
                         "server": client.get_server_version()}
    except Exception:
        carla_version = {"client": "unknown", "server": "unknown"}

    calib = {
        "schema_version": CALIB_SCHEMA_VERSION,
        "generator": os.path.basename(__file__),
        "carla_version": carla_version,
        "town": town_name,
        "has_depth": bool(args.depth),

        "image": {
            "width": cam_w,
            "height": cam_h,
            "fov": fov
        },

        "bev": {
            "x_min": BEV_X_MIN,
            "x_max": BEV_X_MAX,
            "y_min": BEV_Y_MIN,
            "y_max": BEV_Y_MAX,
            "resolution": BEV_RES,
            "height": BEV_H,
            "width": BEV_W
        },

        "classes": {
            "0": "background",
            "1": "vehicle",
            "2": "pedestrian",
            "3": "road",
            "4": "sidewalk",
            "5": "lane_marking"
        },

        "cameras": {
            "front": {
                "name": "CAM_FRONT",
                "K": K.tolist(),
                "T_ego_cam": front_T_ego_cam.tolist(),
                "translation": [1.5, 0.0, 1.6],
                "rotation_deg": {
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0
                }
            },
            "rear": {
                "name": "CAM_BACK",
                "K": K.tolist(),
                "T_ego_cam": rear_T_ego_cam.tolist(),
                "translation": [-1.5, 0.0, 1.6],
                "rotation_deg": {
                    "pitch": 0.0,
                    "yaw": 180.0,
                    "roll": 0.0
                }
            }
        }
    }

    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    if args.fov2 is not None:
        # same rig geometry (extrinsics identical), only FOV/intrinsics differ
        calib2 = json.loads(json.dumps(calib))
        calib2["image"]["fov"] = args.fov2
        K2list = np.asarray(K2).tolist()
        calib2["cameras"]["front"]["K"] = K2list
        calib2["cameras"]["rear"]["K"] = K2list
        with open(os.path.join(args.out2, "calibration.json"), "w") as f:
            json.dump(calib2, f, indent=2)

    print("building lane-marking points from map...")
    lane_pts = build_lane_points(world.get_map(), spacing=0.5)
    print("lane marking points:", lane_pts.shape[0])

    saved_names = []
    stats = {"ticks": 0, "saved": 0, "kept_content": 0, "kept_negative": 0, "skipped": 0}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": os.path.basename(__file__),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished": None,
        "complete": False,
        "carla_version": carla_version,
        "town": town_name,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "seed": args.seed,
        "args": {k: v for k, v in sorted(vars(args).items())},
        "capture": {
            "cam_w": cam_w, "cam_h": cam_h, "fov": fov, "fps": fps,
            "depth": bool(args.depth), "bev_rgb": bool(args.bev_rgb),
            "fov2": args.fov2, "out2": args.out2,
            "prefix": args.prefix,
            "bev": calib["bev"], "classes": calib["classes"],
        },
        "content_filter": {
            "enabled": bool(args.filter),
            "min_veh": args.min_veh,
            "min_ped": args.min_ped,
            "keep_empty_ratio": args.keep_empty_ratio,
            "visible_region": {"fov_deg": fov, "d_min": args.vis_dmin, "d_max": args.vis_dmax,
                               "cam_offset_x": CAM_OFFSET_X},
        },
        "actors": {
            "ego_spawn": [ego_spawn.location.x, ego_spawn.location.y, ego_spawn.rotation.yaw],
            "vehicles_requested": args.vehicles,
            "vehicles_spawned": len(npc_list),
            "vehicle_mix_requested": dict(zip(VEHICLE_CATEGORIES, vehicle_mix)),
            "vehicle_mix_spawned": mix_actual,
            "walkers_requested": args.walkers,
            "walkers_spawned": len(walkers),
            "near_ego": args.near_ego,
        },
        "stats": stats,
        "frames": {"count": 0, "first": None, "last": None},
        "integrity": None,
    }

    frame_subdirs = ["front_rgb", "rear_rgb", "bev_gt"]
    if args.depth:
        frame_subdirs += ["front_depth", "rear_depth"]
    if args.bev_rgb:
        frame_subdirs += ["bev_rgb"]

    def save_manifest(final=False):
        stats["saved"] = len(saved_names)
        manifest["frames"] = {
            "count": len(saved_names),
            "first": saved_names[0] if saved_names else None,
            "last": saved_names[-1] if saved_names else None,
        }
        denom = stats["ticks"] or 1
        stats["keep_rate"] = round(len(saved_names) / denom, 4)
        if final:
            manifest["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            manifest["complete"] = len(saved_names) >= args.frames
            manifest["integrity"] = {
                "algo": "sha256",
                "over": "<subdir>/<frame>:<bytes> for " + ",".join(frame_subdirs),
                "subdirs": frame_subdirs,
                "digest": dataset_digest(args.out, saved_names, frame_subdirs),
            }
        try:
            write_json_atomic(os.path.join(args.out, "manifest.json"), manifest)
            if args.fov2 is not None:
                # the second rig photographs the same ticks; same provenance,
                # its own file list (and its own FOV in the capture block)
                m2 = json.loads(json.dumps(manifest))
                m2["capture"]["fov"] = args.fov2
                m2["capture"]["same_scene_partner"] = os.path.abspath(args.out)
                if final:
                    subs2 = [s for s in frame_subdirs if s != "bev_rgb"]
                    m2["integrity"] = {
                        "algo": "sha256",
                        "over": "<subdir>/<frame>:<bytes> for " + ",".join(subs2),
                        "subdirs": subs2,
                        "digest": dataset_digest(args.out2, saved_names, subs2),
                    }
                write_json_atomic(os.path.join(args.out2, "manifest.json"), m2)
        except Exception as e:
            print("manifest write failed:", e)

    writer = cv2.VideoWriter(
        os.path.join(args.out, args.preview),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720)
    )

    poses = {}   # per-frame ego pose (world x,y,yaw) for temporal BEV fusion

    max_ticks = args.max_ticks if args.max_ticks > 0 else (
        args.frames * 8 if args.filter else args.frames)
    if args.filter:
        print(f"content filter ON: keep when visible vehicles >= {args.min_veh} or "
              f"pedestrians >= {args.min_ped}, plus {args.keep_empty_ratio:.0%} of the rest "
              f"as negatives; target {args.frames} saved frames, tick cap {max_ticks}")
    else:
        print(f"content filter OFF: saving every tick, {args.frames} frames (continuous)")

    # Stuck-ego detection.  CARLA's autopilot can park the ego permanently at a
    # deadlocked junction or behind a walker standing in the lane, and the
    # collector will happily record the same view for as long as you let it.
    # Measured on Town05 (2026-07-31): one segment recorded 215 frames over 6.6
    # minutes while the ego travelled 12.2 m -- median displacement per frame
    # 0.000 m -- and 84% of those frames were near-duplicates.  Two other segments
    # of the same run drove normally (0.69 and 1.70 m per frame), so this is a
    # property of where the ego happens to end up, not of the traffic density: a
    # probe with the vehicle count halved to 20 came out *worse* (73.2% duplicates
    # vs 70.9%).
    #
    # Aborting the segment is the right response rather than merely skipping the
    # frames: the driver restarts with a different spawn point, which is the only
    # thing that actually gets the ego moving again.
    stuck_ticks = 0
    last_loc = ego.get_transform().location
    stuck_aborted = False

    try:
        while len(saved_names) < args.frames and stats["ticks"] < max_ticks:
            world.tick()
            stats["ticks"] += 1

            # Track the standstill streak unconditionally, even with the abort
            # switched off: "how long does the ego actually stand still" is the
            # number that tells you whether a threshold is sane, and guessing it
            # cost several probe runs on 2026-07-31.
            loc = ego.get_transform().location
            if loc.distance(last_loc) < args.stuck_move:
                stuck_ticks += 1
                stats["max_stuck_streak"] = max(stats.get("max_stuck_streak", 0),
                                                stuck_ticks)
            else:
                stuck_ticks = 0
                last_loc = loc

            if args.stuck_abort > 0 and stuck_ticks >= args.stuck_abort:
                print("STUCK: ego moved < %.2f m for %d consecutive ticks "
                      "(%.1f s of simulation) -- ending this segment at "
                      "%d/%d frames so the next one can start elsewhere"
                      % (args.stuck_move, stuck_ticks,
                         stuck_ticks / float(fps), len(saved_names), args.frames))
                stats["stuck_aborted"] = True
                stuck_aborted = True
                break

            # ALWAYS drain every sensor queue, even for a frame we are about to
            # skip: in synchronous mode each tick pushes exactly one image per
            # sensor, so a queue left unread desynchronises the rig (every later
            # frame would pair stale images) and grows without bound.
            front_data = q_front.get(timeout=5.0)
            rear_data = q_rear.get(timeout=5.0)
            depth_data = ((q_front_depth.get(timeout=5.0), q_rear_depth.get(timeout=5.0))
                          if args.depth else None)
            bev_rgb_data = q_bev_rgb.get(timeout=5.0) if args.bev_rgb else None
            data2 = None
            if args.fov2 is not None:
                data2 = [q_front2.get(timeout=5.0), q_rear2.get(timeout=5.0), None, None]
                if args.depth:
                    data2[2] = q_front_depth2.get(timeout=5.0)
                    data2[3] = q_rear_depth2.get(timeout=5.0)

            if args.filter:
                n_veh, n_ped = count_visible_actors(
                    world, ego, fov_deg=fov, d_min=args.vis_dmin, d_max=args.vis_dmax)
                decision = filter_decision(n_veh, n_ped, args.min_veh, args.min_ped,
                                           args.keep_empty_ratio)
                if decision == "content":
                    stats["kept_content"] += 1
                elif decision == "negative":
                    stats["kept_negative"] += 1
                else:
                    stats["skipped"] += 1
                    continue      # keep driving, decode nothing, write nothing
            else:
                stats["kept_content"] += 1

            front_img = image_to_bgr(front_data)
            rear_img = image_to_bgr(rear_data)

            bev_gt = make_bev_gt(world, ego, lane_pts)
            bev_color = colorize(bev_gt)

            # numbered by SAVED frame, not by tick, so a filtered dataset is
            # still 000000..N-1 with no holes
            name = f"{args.prefix}{len(saved_names):06d}.png"
            _etf = ego.get_transform()
            poses[name] = [_etf.location.x, _etf.location.y, _etf.rotation.yaw]

            cv2.imwrite(os.path.join(args.out, "front_rgb", name), front_img)
            cv2.imwrite(os.path.join(args.out, "rear_rgb", name), rear_img)
            cv2.imwrite(os.path.join(args.out, "bev_gt", name), bev_gt)
            cv2.imwrite(os.path.join(args.out, "bev_gt_color", name), bev_color)

            if args.depth:
                # store CARLA's raw depth encoding losslessly (BGR, decode later)
                fd = image_to_bgr(depth_data[0])
                rd = image_to_bgr(depth_data[1])
                cv2.imwrite(os.path.join(args.out, "front_depth", name), fd)
                cv2.imwrite(os.path.join(args.out, "rear_depth", name), rd)

            if args.bev_rgb:
                bev_rgb_img = image_to_bgr(bev_rgb_data)
                cv2.imwrite(os.path.join(args.out, "bev_rgb", name), bev_rgb_img)

            if args.fov2 is not None:
                # same tick, second FOV rig -> identical scene, shared bev_gt
                front_img2 = image_to_bgr(data2[0])
                rear_img2 = image_to_bgr(data2[1])
                cv2.imwrite(os.path.join(args.out2, "front_rgb", name), front_img2)
                cv2.imwrite(os.path.join(args.out2, "rear_rgb", name), rear_img2)
                cv2.imwrite(os.path.join(args.out2, "bev_gt", name), bev_gt)
                cv2.imwrite(os.path.join(args.out2, "bev_gt_color", name), bev_color)
                if args.depth:
                    fd2 = image_to_bgr(data2[2])
                    rd2 = image_to_bgr(data2[3])
                    cv2.imwrite(os.path.join(args.out2, "front_depth", name), fd2)
                    cv2.imwrite(os.path.join(args.out2, "rear_depth", name), rd2)

            saved_names.append(name)

            front_show = cv2.resize(front_img, (640, 360))
            rear_show = cv2.resize(rear_img, (640, 360))
            bev_show = cv2.resize(bev_color, (360, 360), interpolation=cv2.INTER_NEAREST)

            top = np.hstack([front_show, rear_show])
            bottom = np.zeros((360, 1280, 3), dtype=np.uint8)
            bottom[:, 460:820] = bev_show

            cv2.putText(top, "Front RGB", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(top, "Rear RGB", (660, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(bottom, "CARLA Actor Ground Truth BEV",
                        (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            writer.write(np.vstack([top, bottom]))

            if len(saved_names) % 50 == 1 or len(saved_names) == args.frames:
                print("saved %d/%d (tick %d, keep %.0f%%) unique: %s"
                      % (len(saved_names), args.frames, stats["ticks"],
                         100.0 * len(saved_names) / max(stats["ticks"], 1), np.unique(bev_gt)))
                # incremental checkpoint: a hard crash on a long run still leaves
                # poses.json + manifest.json for every frame written so far
                try:
                    write_json_atomic(os.path.join(args.out, "poses.json"), poses)
                    if args.fov2 is not None:
                        write_json_atomic(os.path.join(args.out2, "poses.json"), poses)
                except Exception:
                    pass
                save_manifest()

        if stats["ticks"] >= max_ticks and len(saved_names) < args.frames:
            print(f"WARNING: hit the tick cap ({max_ticks}) with only {len(saved_names)}/"
                  f"{args.frames} frames saved -- the filter is too strict for this town, "
                  f"or the ego is stuck. Raise --max_ticks / --near_ego, or lower --min_veh.")

    finally:
        writer.release()

        try:
            write_json_atomic(os.path.join(args.out, "poses.json"), poses)
            if args.fov2 is not None:
                write_json_atomic(os.path.join(args.out2, "poses.json"), poses)
        except Exception:
            pass

        save_manifest(final=True)

        # Everything above this line is already on disk, so from here on the only
        # thing at stake is tidiness.  CARLA 0.9.16 reliably dies during actor
        # teardown on this machine (see 資料收集規劃.md §13), and with the 240 s
        # collection timeout still in force every teardown RPC then blocks for the
        # full 240 s against a dead server: a segment that crashed on its first
        # tick took 250 s, of which ~200 s was this wait.  During a segmented grind
        # that is the dominant cost, so drop the timeout before teardown.  A leaked
        # actor costs nothing either way -- the driver kills the server after every
        # segment.
        try:
            client.set_timeout(float(args.teardown_timeout))
        except Exception:
            pass

        # IMPORTANT: leave synchronous mode FIRST, otherwise destroying actors
        # blocks waiting for ticks and the server times out / hangs.
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(old_settings)
        except Exception:
            pass

        safe_destroy(front_cam)
        safe_destroy(rear_cam)
        safe_destroy(front_depth_cam)
        safe_destroy(rear_depth_cam)
        safe_destroy(bev_rgb_cam)
        safe_destroy(front_cam2)
        safe_destroy(rear_cam2)
        safe_destroy(front_depth_cam2)
        safe_destroy(rear_depth_cam2)

        for c in controllers:
            try:
                c.stop()
            except Exception:
                pass

        for c in controllers:
            try:
                c.destroy()
            except Exception:
                pass

        for w in walkers:
            safe_destroy(w)

        try:
            client.apply_batch([
                carla.command.DestroyActor(x) for x in npc_list
            ])
        except Exception:
            pass

        try:
            ego.set_autopilot(False)
        except Exception:
            pass
        safe_destroy(ego)

    print("saved dataset:", args.out)
    print("frames: %d saved / %d ticks (keep %.1f%%)"
          % (len(saved_names), stats["ticks"],
             100.0 * len(saved_names) / max(stats["ticks"], 1)))
    if args.filter:
        print("  by content: %d   negatives kept: %d   skipped: %d"
              % (stats["kept_content"], stats["kept_negative"], stats["skipped"]))
    print("manifest:", os.path.join(args.out, "manifest.json"))
    print("preview:", os.path.join(args.out, args.preview))

if __name__ == "__main__":
    main()
