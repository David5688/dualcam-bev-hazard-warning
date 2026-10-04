# -*- coding: utf-8 -*-
"""Scripted hazard scenarios for live_demo.py.

Why this exists: on autopilot the ego wanders Town10HD and whether a DANGER
warning ever fires is luck.  A 200-frame trace on 2026-08-09 spent most of its
time either at a red light or driving down an empty street -- fine for showing
that the BEV runs live, useless for showing what the system is *for*.  In front
of a panel of judges that is a coin flip, and the rear-facing half of the rig
(the whole point of the project) is the half least likely to come up.

So each scenario here stages one hazard on purpose, and re-arms itself so the
same hazard repeats for as long as the demo runs.  Nothing in the perception
path is touched: the model, the tracker and the TTC logic never learn that the
approaching car was placed there rather than found.

  rear_approach   a car closes fast from behind          -> low-pitch rear alert
  front_brake     the lead vehicle slams on the brakes   -> TTC collapses
  cut_in          a car in the left lane cuts across     -> a track appears close
  reverse_ped     a pedestrian crosses behind the ego    -> rear alert while reversing
  rear_weave      a car behind swerves side to side      -> the case radial TTC cannot see

Each one needs room: staging a rear approach at a junction just deadlocks the
traffic manager.  pick_spawn_points() below re-orders the map's spawn points so
the ego starts somewhere with enough clear lane, which is what makes the
scenario fire within seconds instead of eventually.
"""
import math
import random

import carla


STEP = 5.0                      # waypoint chaining step, metres


def _run_is_clear(wp, distance, forward=True, step=STEP):
    """Follow the lane for `distance` m; return the end waypoint, or None.

    None means the lane ran out, forked, or entered a junction -- all three make
    a staged scenario unreliable, so they are treated the same way.
    """
    cur = wp
    travelled = 0.0
    while travelled < distance:
        nxt = cur.next(step) if forward else cur.previous(step)
        if not nxt:
            return None
        cur = nxt[0]
        if cur.is_junction:
            return None
        travelled += step
    return cur


def _same_direction_left(wp):
    """The lane immediately left of `wp` if a car may legally be driven there.

    get_left_lane() happily returns the oncoming lane, whose lane_id has the
    opposite sign.  Staging a cut-in from oncoming traffic looks like a head-on
    crash, not a lane change, so those are rejected.
    """
    left = wp.get_left_lane()
    if left is None:
        return None
    if left.lane_type != carla.LaneType.Driving:
        return None
    if left.lane_id * wp.lane_id <= 0:
        return None
    return left


class _Base(object):
    """One staged hazard.

    Lifecycle, called from live_demo's main loop:
        pick_spawn_points()  before the ego exists -- orders candidate spawns
        setup()              after the ego and the traffic manager exist
        tick()               once per simulated frame
        cleanup()            in the finally block
    """
    name = "none"
    hint = ""
    # How much clear lane the staging needs, in metres, ahead of and behind the ego.
    need_ahead = 0.0
    need_behind = 0.0
    # reverse_ped wants the demo to start in manual reverse; the rest do not.
    wants_reverse = False
    # live_demo zeroes the BEV's pedestrian class -- people come from the image
    # detector instead -- so a scenario about a pedestrian needs --peds or it
    # stages someone the system is structurally unable to see.  Measured: a
    # reverse_ped run without it logged 0 warnings in 150 frames.
    needs_peds = False

    def __init__(self, fps=10, loop=True, seed=None):
        self.fps = max(int(fps), 1)
        self.loop = loop
        self.rng = random.Random(seed)
        self.actors = []
        self.banner = ""          # short ASCII line for the HUD
        self.armed = 0            # how many times the hazard has been staged

    # -- helpers -------------------------------------------------------------
    def _frames(self, seconds):
        return int(seconds * self.fps)

    def _spawn_vehicle(self, world, bp_lib, transform, filter_="vehicle.audi.tt"):
        bp = bp_lib.filter(filter_)
        bp = bp[0] if bp else self.rng.choice(list(bp_lib.filter("vehicle.*")))
        if bp.has_attribute("color"):
            # A saturated colour so the staged actor is obvious in the camera
            # panel -- the audience should be able to see which car the warning
            # is about without reading the BEV.
            try:
                bp.set_attribute("color", "200,20,20")
            except Exception:
                pass
        tf = carla.Transform(transform.location + carla.Location(z=0.4), transform.rotation)
        v = world.try_spawn_actor(bp, tf)
        if v is not None:
            self.actors.append(v)
        return v

    def _ego_waypoint(self, world, ego):
        return world.get_map().get_waypoint(ego.get_location(),
                                            project_to_road=True,
                                            lane_type=carla.LaneType.Driving)

    def _behind(self, world, ego, gap):
        """Waypoint `gap` m behind the ego, preferring a junction-free run."""
        wp = self._ego_waypoint(world, ego)
        back = _run_is_clear(wp, gap, forward=False)
        if back is None:
            back = wp.previous(gap)
            back = back[0] if back else None
        return back

    # -- interface -----------------------------------------------------------
    def pick_spawn_points(self, world, spawn_points):
        """Spawn points with enough clear lane for this scenario, best first.

        Falls back to the untouched list when the map has nothing suitable, so a
        scenario never stops the demo from starting -- it just may not fire.
        """
        cmap = world.get_map()
        ok = []
        for sp in spawn_points:
            wp = cmap.get_waypoint(sp.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
                continue
            if self.need_ahead and _run_is_clear(wp, self.need_ahead, True) is None:
                continue
            if self.need_behind and _run_is_clear(wp, self.need_behind, False) is None:
                continue
            if not self._extra_ok(wp):
                continue
            ok.append(sp)
        self.rng.shuffle(ok)
        if not ok:
            print("  scenario %s: no spawn point on this map has %.0f m ahead / "
                  "%.0f m behind clear -- starting anywhere, the hazard may take "
                  "longer to stage" % (self.name, self.need_ahead, self.need_behind))
            return list(spawn_points)
        rest = [sp for sp in spawn_points if sp not in ok]
        print("  scenario %s: %d of %d spawn points have the room it needs"
              % (self.name, len(ok), len(spawn_points)))
        return ok + rest

    def _extra_ok(self, wp):
        return True

    def setup(self, world, bp_lib, ego, tm):
        raise NotImplementedError

    def tick(self, k, world, ego, tm):
        pass

    def cleanup(self):
        for a in self.actors:
            try:
                if a.type_id.startswith("controller"):
                    a.stop()
                a.destroy()
            except Exception:
                pass
        self.actors = []


class RearApproach(_Base):
    """A car closes on the ego from behind at roughly twice its speed.

    This is the scenario the project is built around and the one autopilot
    almost never produces: CARLA's traffic manager keeps a 2.5 m gap and will
    not tailgate, so nothing ever approaches from behind fast enough to raise a
    rear warning.  Here the chaser gets a 0.5 m following distance and a large
    negative speed difference (negative = faster than the limit), which closes
    the 45 m gap in about six seconds at town speeds.
    """
    name = "rear_approach"
    hint = "REAR: car closing from behind -- listen for the LOW beep"
    need_ahead = 45.0
    need_behind = 55.0

    GAP = 45.0            # where the chaser is (re-)staged, metres behind the ego
    SPEED_DIFF = -75.0    # 75% over the limit
    EGO_SLOWDOWN = 25.0   # hold the ego 25% under, so the gap actually closes

    def __init__(self, *a, **kw):
        _Base.__init__(self, *a, **kw)
        self.chaser = None

    def setup(self, world, bp_lib, ego, tm):
        back = self._behind(world, ego, self.GAP)
        if back is None:
            print("  scenario rear_approach: no lane behind the ego, nothing staged")
            return
        self.chaser = self._spawn_vehicle(world, bp_lib, back.transform)
        if self.chaser is None:
            print("  scenario rear_approach: the chaser did not fit, nothing staged")
            return
        self.chaser.set_autopilot(True, tm.get_port())
        for fn, arg in ((tm.vehicle_percentage_speed_difference, self.SPEED_DIFF),
                        (tm.distance_to_leading_vehicle, 0.5),
                        (tm.ignore_lights_percentage, 100.0),
                        (tm.ignore_signs_percentage, 100.0)):
            try:
                fn(self.chaser, arg)
            except Exception:
                pass
        try:
            # Left to itself the traffic manager solves a slow car ahead by
            # changing lane, which turns the staged rear approach into an
            # overtake that never gets close enough to warn about.
            tm.auto_lane_change(self.chaser, False)
        except Exception:
            pass
        try:
            tm.vehicle_percentage_speed_difference(ego, self.EGO_SLOWDOWN)
        except Exception:
            pass
        self.armed = 1
        self.banner = "SCENARIO: rear approach"
        print("  scenario rear_approach: chaser staged %.0f m behind, %.0f%% over "
              "the limit" % (self.GAP, abs(self.SPEED_DIFF)))

    def tick(self, k, world, ego, tm):
        if self.chaser is None or not self.loop:
            return
        try:
            el = ego.get_location()
            cl = self.chaser.get_location()
            fwd = ego.get_transform().get_forward_vector()
        except Exception:
            return
        dx, dy = cl.x - el.x, cl.y - el.y
        along = dx * fwd.x + dy * fwd.y              # >0 means the chaser is ahead
        dist = math.hypot(dx, dy)
        # Re-stage once it has overtaken (the pass is over, nothing more to warn
        # about) or once it has been left behind (a red light, a wrong turn).
        if along > 20.0 or dist > 95.0:
            back = self._behind(world, ego, self.GAP)
            if back is None:
                return
            try:
                self.chaser.set_transform(
                    carla.Transform(back.transform.location + carla.Location(z=0.4),
                                    back.transform.rotation))
                self.chaser.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.armed += 1
            except Exception:
                pass


class FrontBrake(_Base):
    """The lead vehicle brakes hard, on a timer.

    The interesting quantity for the write-up is how early the warning fires,
    and that needs a repeatable event: same gap, same deceleration, every time.
    The lead runs on autopilot until the brake, is driven manually through it,
    and is handed back afterwards.
    """
    name = "front_brake"
    hint = "FRONT: lead vehicle brakes hard -- watch TTC collapse"
    need_ahead = 70.0
    need_behind = 10.0

    GAP = 18.0
    ARM_AFTER = 7.0       # seconds of normal driving before the first brake
    HOLD = 4.0            # seconds of full brake
    COOLDOWN = 12.0       # seconds before staging the next one

    def __init__(self, *a, **kw):
        _Base.__init__(self, *a, **kw)
        self.lead = None
        self.state = "idle"
        self.until = 0

    def setup(self, world, bp_lib, ego, tm):
        wp = self._ego_waypoint(world, ego)
        ahead = _run_is_clear(wp, self.GAP, forward=True)
        if ahead is None:
            print("  scenario front_brake: no clear lane ahead, nothing staged")
            return
        self.lead = self._spawn_vehicle(world, bp_lib, ahead.transform)
        if self.lead is None:
            print("  scenario front_brake: the lead vehicle did not fit, nothing staged")
            return
        self.lead.set_autopilot(True, tm.get_port())
        for fn, arg in ((tm.vehicle_percentage_speed_difference, 0.0),
                        (tm.ignore_lights_percentage, 100.0)):
            try:
                fn(self.lead, arg)
            except Exception:
                pass
        try:
            tm.auto_lane_change(self.lead, False)
        except Exception:
            pass
        self.state = "wait"
        self.until = self._frames(self.ARM_AFTER)
        self.banner = "SCENARIO: front brake"
        print("  scenario front_brake: lead staged %.0f m ahead, first brake in "
              "%.0f s" % (self.GAP, self.ARM_AFTER))

    def tick(self, k, world, ego, tm):
        if self.lead is None:
            return
        if self.state == "wait" and k >= self.until:
            try:
                self.lead.set_autopilot(False, tm.get_port())
                self.lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.lead.set_light_state(carla.VehicleLightState.Brake)
            except Exception:
                pass
            self.state = "braking"
            self.until = k + self._frames(self.HOLD)
            self.armed += 1
            self.banner = "SCENARIO: front brake  [BRAKING]"
        elif self.state == "braking" and k >= self.until:
            try:
                self.lead.set_light_state(carla.VehicleLightState.NONE)
                self.lead.set_autopilot(True, tm.get_port())
            except Exception:
                pass
            self.state = "cooldown" if self.loop else "done"
            self.until = k + self._frames(self.COOLDOWN)
            self.banner = "SCENARIO: front brake"
        elif self.state == "cooldown" and k >= self.until:
            # Only re-arm while the lead is still the car in front; after a
            # junction it is usually somewhere else entirely and braking it
            # would warn about nothing.
            self.until = k + self._frames(2.0)
            try:
                dl = self.lead.get_location() - ego.get_location()
                if math.hypot(dl.x, dl.y) < 40.0:
                    self.state = "wait"
            except Exception:
                pass


class CutIn(_Base):
    """A car in the neighbouring lane cuts in front of the ego.

    Staged rather than waited for because the traffic manager's own lane changes
    happen where they are convenient, not where a camera is pointed.
    """
    name = "cut_in"
    hint = "FRONT: vehicle cutting in from the left"
    need_ahead = 60.0
    need_behind = 10.0

    GAP = 14.0
    ARM_AFTER = 5.0
    COOLDOWN = 14.0

    def __init__(self, *a, **kw):
        _Base.__init__(self, *a, **kw)
        self.car = None
        self.until = 0
        self.state = "idle"

    def _extra_ok(self, wp):
        return _same_direction_left(wp) is not None

    def setup(self, world, bp_lib, ego, tm):
        wp = self._ego_waypoint(world, ego)
        ahead = _run_is_clear(wp, self.GAP, forward=True)
        left = _same_direction_left(ahead) if ahead is not None else None
        if left is None:
            print("  scenario cut_in: no same-direction lane to the left, nothing staged")
            return
        self.car = self._spawn_vehicle(world, bp_lib, left.transform)
        if self.car is None:
            print("  scenario cut_in: the cut-in vehicle did not fit, nothing staged")
            return
        self.car.set_autopilot(True, tm.get_port())
        for fn, arg in ((tm.vehicle_percentage_speed_difference, -15.0),
                        (tm.ignore_lights_percentage, 100.0)):
            try:
                fn(self.car, arg)
            except Exception:
                pass
        try:
            tm.auto_lane_change(self.car, False)
        except Exception:
            pass
        self.state = "wait"
        self.until = self._frames(self.ARM_AFTER)
        self.banner = "SCENARIO: cut-in"
        print("  scenario cut_in: vehicle staged in the left lane, cutting in "
              "after %.0f s" % self.ARM_AFTER)

    def tick(self, k, world, ego, tm):
        if self.car is None:
            return
        if self.state == "wait" and k >= self.until:
            try:
                # force_lane_change(actor, direction): True is the vehicle's
                # right, which from the left lane is the ego's lane.
                tm.force_lane_change(self.car, True)
                self.armed += 1
            except Exception:
                pass
            self.state = "cooldown" if self.loop else "done"
            self.until = k + self._frames(self.COOLDOWN)
        elif self.state == "cooldown" and k >= self.until:
            # Put it back in the left lane so the manoeuvre can repeat.
            try:
                tm.force_lane_change(self.car, False)
            except Exception:
                pass
            self.state = "wait"
            self.until = k + self._frames(self.ARM_AFTER)


class ReversePed(_Base):
    """A pedestrian walks across the road behind the ego.

    The rear camera earns its place here: reversing is the one manoeuvre where
    the driver genuinely cannot see, and it is the clearest answer to "why two
    cameras".  The walker is driven directly with WalkerControl instead of an AI
    controller -- the navmesh decides where a controller-driven walker goes, and
    for a demo the crossing has to happen behind the ego, on cue, every time.
    """
    name = "reverse_ped"
    hint = "REAR: pedestrian crossing behind you -- Tab, R, then W to reverse"
    need_ahead = 10.0
    need_behind = 30.0
    wants_reverse = True
    needs_peds = True

    # The gap has to sit inside the warning envelope or the scenario stages a
    # crossing nobody is warned about: risk_of() raises CAUTION below 10 m and
    # DANGER below 5 m, and someone walking ACROSS keeps their range roughly
    # constant, so TTC never fires and range is the only thing that can.  At 11 m
    # -- the first value tried -- a 150-frame run logged nothing at all.
    GAP = 7.0             # metres behind the ego
    SIDE = 4.5            # start this far to one side of the lane centre
    SPEED = 1.3           # m/s, an unhurried walk
    CROSS = 9.0           # metres to cover before restarting

    def __init__(self, *a, **kw):
        _Base.__init__(self, *a, **kw)
        self.walker = None
        self.origin = None
        self.direction = None

    def _stage_points(self, world, ego):
        back = self._behind(world, ego, self.GAP)
        if back is None:
            return None, None
        tf = back.transform
        right = tf.get_right_vector()
        start = tf.location + carla.Location(x=right.x * -self.SIDE,
                                             y=right.y * -self.SIDE, z=1.0)
        return start, right

    def setup(self, world, bp_lib, ego, tm):
        start, right = self._stage_points(world, ego)
        if start is None:
            print("  scenario reverse_ped: no lane behind the ego, nothing staged")
            return
        bp = self.rng.choice(list(bp_lib.filter("walker.pedestrian.*")))
        if bp.has_attribute("is_invincible"):
            # Invincible on purpose: a ragdolled walker stays down and the
            # scenario cannot repeat, and the ego does reverse into them.
            bp.set_attribute("is_invincible", "true")
        yaw = math.degrees(math.atan2(right.y, right.x))
        self.walker = world.try_spawn_actor(
            bp, carla.Transform(start, carla.Rotation(yaw=yaw)))
        if self.walker is None:
            print("  scenario reverse_ped: the walker did not fit, nothing staged")
            return
        self.actors.append(self.walker)
        self.origin = start
        self.direction = right
        self.armed = 1
        self.banner = "SCENARIO: pedestrian behind"
        print("  scenario reverse_ped: walker staged %.0f m behind. "
              "Tab for manual, R for reverse, W to back up." % self.GAP)

    def tick(self, k, world, ego, tm):
        if self.walker is None or self.origin is None:
            return
        try:
            self.walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(self.direction.x, self.direction.y, 0.0),
                speed=self.SPEED))
            here = self.walker.get_location()
        except Exception:
            return
        gone = math.hypot(here.x - self.origin.x, here.y - self.origin.y)
        if gone > self.CROSS and self.loop:
            start, right = self._stage_points(world, ego)
            if start is None:
                return
            yaw = math.degrees(math.atan2(right.y, right.x))
            try:
                self.walker.set_transform(carla.Transform(start, carla.Rotation(yaw=yaw)))
                self.origin = start
                self.direction = right
                self.armed += 1
            except Exception:
                pass


class RearWeave(_Base):
    """A car behind the ego swerves from side to side while following it.

    The scenario the radial TTC cannot see.  hazard_warning.ttc_of fits a line
    through the RANGE history, and a car weaving at a steady following distance
    barely changes its range at all -- replayed at +-2 m and a 3 s period, range
    moved between 15.00 m and 15.10 m, the closing rate stayed inside +-0.21 m/s,
    TTC was infinite throughout and the worst level raised over six seconds of a
    car repeatedly crossing into the ego's lane was SAFE.  trajectory_predict.py
    is what answers this one, and this scenario is what puts it on screen.

    The weaver is driven manually rather than by the traffic manager: the traffic
    manager's whole job is to keep a vehicle centred in its lane, so there is no
    setting that makes it weave.  Steering is a sine wave and the throttle is a
    proportional controller on the speed difference to the ego, which keeps the
    car behind us instead of dropping away or driving into our boot.
    """
    name = "rear_weave"
    hint = "REAR: car behind is weaving -- watch the predicted path, not the range"
    need_ahead = 40.0
    need_behind = 45.0

    GAP = 22.0            # where the weaver is (re-)staged, metres behind the ego
    PERIOD = 4.0          # seconds per full left-right-left cycle
    WEAVE_AMP = 2.5       # metres the aim point is swung either side of the trail.
                          # Commanding the offset rather than the steering angle is
                          # what makes the amplitude mean something: the first
                          # version added a sine straight onto the steering, and the
                          # pure-pursuit term simply corrected it back out -- the
                          # measured swing was 0.9 m peak-to-peak, under the 2.5 m
                          # the detector needs, so nothing ever fired.  Offsetting
                          # the aim point makes the follower CHASE the weave instead
                          # of fighting it, and the amplitude is then whatever this
                          # says it is.
    HOLD_GAP = 16.0       # metres it tries to keep behind the ego
    KP = 0.12             # throttle per m/s of speed error
    LOOKAHEAD = 5.0       # pure-pursuit lookahead along the ego's trail, metres
    K_FOLLOW = 2.0        # steering per radian of heading error
    EGO_SLOWDOWN = 15.0   # hold the ego a little under the limit so it stays catchable

    def __init__(self, *a, **kw):
        _Base.__init__(self, *a, **kw)
        self.weaver = None
        self.t0 = 0
        # Breadcrumbs of where the ego has been.  The weaver steers at these
        # rather than at map waypoints: a weave by definition takes the car off
        # its lane centre, and get_waypoint() then snaps to whichever lane it has
        # strayed into, so a waypoint-following weaver hunts between lanes and
        # eventually commits to the wrong one.  Measured on the first version,
        # which had a sine wave and no following term at all: the gap grew from
        # 22 m to 67 m and the lateral offset reached 40 m -- it simply drove off
        # down another street, and the predictor correctly reported nothing,
        # because by then there was nothing behind us to report.
        self.trail = []

    def setup(self, world, bp_lib, ego, tm):
        back = self._behind(world, ego, self.GAP)
        if back is None:
            print("  scenario rear_weave: no lane behind the ego, nothing staged")
            return
        self.weaver = self._spawn_vehicle(world, bp_lib, back.transform)
        if self.weaver is None:
            print("  scenario rear_weave: the weaver did not fit, nothing staged")
            return
        # No autopilot at all -- tick() drives it.
        try:
            self.weaver.set_autopilot(False, tm.get_port())
        except Exception:
            pass
        try:
            tm.vehicle_percentage_speed_difference(ego, self.EGO_SLOWDOWN)
        except Exception:
            pass
        self.t0 = 0
        self.trail = []
        self.armed = 1
        self.banner = "SCENARIO: rear weave"
        print("  scenario rear_weave: weaver staged %.0f m behind, %.1f s period, "
              "swing +-%.1f m" % (self.GAP, self.PERIOD, self.WEAVE_AMP))

    def _target(self, wl):
        """Point on the ego's trail about LOOKAHEAD m ahead of the weaver."""
        if len(self.trail) < 2:
            return None
        best_i, best_d = 0, 1e9
        for i, (tx, ty) in enumerate(self.trail):
            d = (tx - wl.x) ** 2 + (ty - wl.y) ** 2
            if d < best_d:
                best_d, best_i = d, i
        travelled = 0.0
        for i in range(best_i, len(self.trail) - 1):
            ax, ay = self.trail[i]
            bx, by = self.trail[i + 1]
            travelled += math.hypot(bx - ax, by - ay)
            if travelled >= self.LOOKAHEAD:
                return self.trail[i + 1]
        return self.trail[-1]

    def tick(self, k, world, ego, tm):
        if self.weaver is None:
            return
        try:
            et = ego.get_transform()
            el, ev = et.location, ego.get_velocity()
            wl, wv = self.weaver.get_location(), self.weaver.get_velocity()
            fwd = et.get_forward_vector()
        except Exception:
            return

        # Drop a breadcrumb every time the ego has moved far enough to matter.
        if not self.trail or math.hypot(el.x - self.trail[-1][0],
                                        el.y - self.trail[-1][1]) > 1.0:
            self.trail.append((el.x, el.y))
            self.trail = self.trail[-400:]

        dx, dy = wl.x - el.x, wl.y - el.y
        along = dx * fwd.x + dy * fwd.y        # negative while it is behind us
        dist = math.hypot(dx, dy)

        # Re-stage once it has drawn level, been left behind, or fallen over.
        if along > 6.0 or dist > 70.0:
            if not self.loop:
                return
            back = self._behind(world, ego, self.GAP)
            if back is None:
                return
            try:
                self.weaver.set_transform(
                    carla.Transform(back.transform.location + carla.Location(z=0.4),
                                    back.transform.rotation))
                self.weaver.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.t0 = k
                self.trail = []
                self.armed += 1
            except Exception:
                pass
            return

        ego_speed = math.sqrt(ev.x * ev.x + ev.y * ev.y)
        weaver_speed = math.sqrt(wv.x * wv.x + wv.y * wv.y)
        # Close the gap gently: match the ego's speed, plus a term for how far
        # off HOLD_GAP we are.  Weaving costs distance over the ground, so a
        # weaver that merely matched speed would slowly fall behind.
        want = ego_speed + 0.5 * (dist - self.HOLD_GAP)
        want = max(2.0, min(want, ego_speed + 7.0))
        err = want - weaver_speed
        throttle = max(0.0, min(self.KP * err, 0.85))
        brake = 0.0 if err > -1.0 else min(0.35, -0.06 * err)

        # Pure pursuit down the ego's trail, aimed at a point swung from side to
        # side.  Following keeps it on the road; the swinging aim point is the
        # hazard, and its amplitude is in metres rather than steering units.
        steer = 0.0
        tgt = self._target(wl)
        if tgt is not None:
            ax, ay = tgt[0] - wl.x, tgt[1] - wl.y
            n = math.hypot(ax, ay)
            if n > 0.5:
                phase = 2.0 * math.pi * (k - self.t0) / (self.PERIOD * self.fps)
                off = self.WEAVE_AMP * math.sin(phase)
                ux, uy = ax / n, ay / n
                aim_x = tgt[0] - uy * off          # perpendicular to the approach
                aim_y = tgt[1] + ux * off
                wyaw = math.radians(self.weaver.get_transform().rotation.yaw)
                want_yaw = math.atan2(aim_y - wl.y, aim_x - wl.x)
                herr = math.atan2(math.sin(want_yaw - wyaw), math.cos(want_yaw - wyaw))
                steer = max(-1.0, min(self.K_FOLLOW * herr, 1.0))
        try:
            self.weaver.apply_control(carla.VehicleControl(
                throttle=float(throttle), steer=float(steer), brake=float(brake)))
        except Exception:
            pass


SCENARIOS = {
    "rear_approach": RearApproach,
    "rear_weave": RearWeave,
    "front_brake": FrontBrake,
    "cut_in": CutIn,
    "reverse_ped": ReversePed,
}

CHOICES = ["none"] + sorted(SCENARIOS)


def make_scenario(name, fps=10, loop=True, seed=None):
    if not name or name == "none":
        return None
    if name not in SCENARIOS:
        raise SystemExit("unknown scenario '%s' (choose from %s)"
                         % (name, ", ".join(CHOICES)))
    return SCENARIOS[name](fps=fps, loop=loop, seed=seed)
