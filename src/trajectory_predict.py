# -*- coding: utf-8 -*-
"""Trajectory prediction for the vehicle behind: where is it going, and does that
put it into us?

Why this module exists
----------------------
The TTC in hazard_warning.py is radial: it fits a line through the RANGE history
and warns when range / closing-rate gets short.  That is the right thing for a
car driving straight at you, and it is completely blind to one that swerves.

Measured on a synthetic weave (car holds 15 m behind, matches speed, swings +-2 m
across the lane with a 3 s period -- an amplitude that carries it into the ego's
lane twice):

    range varies between 15.00 m and 15.10 m
    closing rate stays within +-0.21 m/s
    TTC is infinite for the whole manoeuvre
    worst risk level raised over 6 seconds: SAFE

The driver is told nothing while a car crosses into their lane behind them,
twice.  Range cannot see a manoeuvre that is almost entirely sideways.

What this adds
--------------
Two quantities, both from the track's own world-frame history (Tracker.whist),
both computed on the real detector output -- no ground truth anywhere:

  TTI    time-to-intrusion.  Propagate the target's motion RELATIVE to the ego and
         report the first moment its predicted position lands inside a box around
         the ego.  This is the "will it hit us" answer the radial TTC cannot give,
         because it carries the lateral velocity term that TTC drops.

  weave  a behaviour flag.  Fit the target's own mean path over the window, take
         each sample's perpendicular deviation from it, and look for a sustained
         oscillation.  Deliberately ego-independent: it describes the driver
         behind, not the geometry between us, so it survives the ego itself
         changing lane.

The two are reported separately on purpose.  A car weaving hard 40 m back is not
about to hit anyone -- TTI is infinite and should be -- but it is still something
a driver wants to know about, and conflating the two would mean either crying
wolf or staying silent.

Assumptions, stated because they bound where this is honest
-----------------------------------------------------------
* Constant relative velocity over the horizon (default 3 s).  The prediction says
  "if both of you keep doing what you are doing now" -- it does not model the
  other driver changing their mind, and for a weave it deliberately extrapolates
  the CURRENT swing rather than fitting a sinusoid: a 3 s period sampled at 10 Hz
  under ~0.5 m of position noise does not support a reliable frequency fit, and a
  wrong phase would predict a swing away from us at exactly the wrong moment.
* The ego travels straight over the horizon.  At town speeds and a 3 s horizon
  this costs little, and the alternative -- propagating the ego's yaw rate -- adds
  its own error on a curve.  A run on a tight bend should be read with that in mind.
"""
import math


# Ego box, metres.  A Model 3 is 4.69 x 1.85; the margin is what makes an
# "intrusion" mean "close enough to matter" rather than "paint touching paint".
EGO_HALF_L = 2.4
EGO_HALF_W = 0.95
MARGIN = 0.6

# Every default below was chosen by sweeping against 0.5 m of position noise --
# the measured BEV centroid error (metrics/ttc_eval.csv) -- on six replayed
# manoeuvres: two real weaves, a straight close, an overtake one lane over, a
# steady follower, and a single clean lane change.  The criterion was to fire on
# both weaves while staying silent on all four of the others, the lane change
# included: changing lane once is not weaving, and a system that calls it weaving
# would be crying wolf at every normal overtake.
HORIZON = 3.0          # how far ahead to predict, seconds
STEP = 0.1             # prediction sampling, seconds
FIT_WINDOW = 2.0       # seconds of history used for the velocity fit. 1.0 s leaves
                       # the lateral slope so noisy (sigma ~0.55 m/s, i.e. +-1.6 m
                       # over a 3 s horizon, about the width of the corridor) that
                       # a car overtaking one lane over was predicted to intrude in
                       # 1 of 3 seeds. 2.0 s removed that entirely.
WEAVE_WINDOW = 4.0     # seconds of history the weave detector looks at
WEAVE_MIN_SDLP = 0.52  # SDLP-style standard deviation of lateral deviation, metres,
                       # before a track counts as weaving.  THIS is the decision;
                       # peak-to-peak is still computed but only reported.
                       #
                       # Measured on seven replayed manoeuvres x five seeds at 0.5 m
                       # position noise, sweeping each statistic for its own best
                       # threshold, gross (+-2 m) weaves against all clean driving:
                       #
                       #     statistic       threshold   missed   false alarm
                       #     peak-to-peak      1.35 m      1.5%       4.3%
                       #     SDLP              0.52 m      4.3%       0.8%
                       #
                       # SDLP misses more individual frames and raises five times
                       # fewer false alarms, and for THIS design that trade is
                       # strongly favourable: the watch list latches, so missing a
                       # few frames costs nothing once 1.5 s of evidence accumulates,
                       # while a false alarm under the default --watch_release 0 is
                       # permanent for the run.  The noise floor is the other half of
                       # it -- straight driving reads 0.51 m peak-to-peak but only
                       # 0.13 m SDLP, against 3.54 / 1.13 m for a gross weave, so the
                       # signal-to-floor ratio improves from 6.9x to 8.7x.  A single
                       # lane change, the hardest negative, reaches a p90 of 1.50 m
                       # peak-to-peak -- inside weave territory -- but only 0.44 m SDLP.
WEAVE_MIN_AMP = 2.5    # peak-to-peak equivalent, kept for reporting and comparison
WEAVE_MIN_REV = 2      # direction reversals required in the window
SMOOTH = 9             # samples in the moving average used before differencing
LANE_WINDOW = 1.0      # fit window for the LATERAL channel only -- see assess().
MIN_LAT_SPEED = 1.0    # m/s of lateral closing before a lane entry is predicted.
                       # Over a 1.6 s baseline the fitted lateral slope still
                       # carries sigma ~0.26 m/s, so a 3 s extrapolation has a
                       # 1-sigma error near 0.8 m and a 2-sigma one near the full
                       # 1.55 m corridor -- and a 2-sigma draw is not rare, it is
                       # every twentieth frame of every track. Measured: a car
                       # holding a steady 3.5 m one lane over was predicted into
                       # our lane at 2.6 s from a fitted 0.53 m/s that was pure
                       # noise. Demanding ~4 sigma of real lateral speed is what
                       # separates a lane change from a noisy straight line; the
                       # cost is that a very slow drift is not called early.
MIN_FIT_PTS = 8        # samples required before a velocity fit is believed
MIN_FIT_SPAN = 0.8     # absolute floor, seconds, for that span
MIN_FIT_FRAC = 0.8     # ...and it must also cover this much of the fit window.
                       # A partial window is the same problem in miniature: at a
                       # 0.8 s baseline the lateral slope still carries ~0.77 m/s
                       # of noise, enough to swing a 3 s prediction by 2.3 m and
                       # put the car in the next lane inside ours. Requiring most
                       # of the window costs the first ~1.6 s of every track --
                       # during which the radial TTC, which needs only six
                       # samples, is still running. The two cover different
                       # failure modes and it is fair for the slower one to wait.


def _fit_axis(ts, vs):
    """Least-squares line through (t, v); returns (value at t_last, slope)."""
    n = len(ts)
    if n < 2:
        return (vs[-1] if vs else 0.0), 0.0
    mt = sum(ts) / n
    mv = sum(vs) / n
    num = sum((t - mt) * (v - mv) for t, v in zip(ts, vs))
    den = sum((t - mt) ** 2 for t in ts)
    if den < 1e-9:
        return vs[-1], 0.0
    slope = num / den
    return mv + slope * (ts[-1] - mt), slope


def world_state(whist, window=FIT_WINDOW):
    """(wx, wy, vx, vy) for a track, from the last `window` seconds of its path.

    Fitting a line rather than differencing the last two samples matters here:
    one sample pair at 10 Hz spans ~0.7 m of travel, which is barely more than the
    position noise, so a difference is mostly noise.  Over a second it is not.
    """
    if not whist:
        return None
    t_last = whist[-1][0]
    win = [p for p in whist if t_last - p[0] <= window]
    # A fit needs both enough samples and enough time.  Without this a track two
    # frames old produced a slope from a 0.1 s baseline: 0.5 m of position noise
    # over 0.1 s is 5 m/s of apparent lateral velocity, which extrapolated over a
    # 3 s horizon put a car overtaking one lane over squarely inside our lane.
    # Reporting nothing until the track has been watched for MIN_FIT_SPAN is the
    # honest answer -- the system genuinely does not know yet.
    need = max(MIN_FIT_SPAN, MIN_FIT_FRAC * window)
    if len(win) < MIN_FIT_PTS or (win[-1][0] - win[0][0]) < need:
        return None
    ts = [p[0] for p in win]
    xs = [p[1] for p in win]
    ys = [p[2] for p in win]
    x, vx = _fit_axis(ts, xs)
    y, vy = _fit_axis(ts, ys)
    return x, y, vx, vy


def _to_ego(wx, wy, ex, ey, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    dx, dy = wx - ex, wy - ey
    return (c * dx + s * dy, -s * dx + c * dy)


def _rotate_to_ego(vx, vy, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    return (c * vx + s * vy, -s * vx + c * vy)


def predict_relative(whist, ego, horizon=HORIZON, step=STEP, window=FIT_WINDOW):
    """Predicted path of a track in the ego's body frame.

    ego is (ex, ey, yaw_deg, evx, evy) -- the ego's world position, heading and
    world velocity.  Returns [(t, x, y), ...] with x forward and y left, t
    counted from now, or None when the history is too short to fit.
    """
    st = world_state(whist, window)
    if st is None:
        return None
    wx, wy, vx, vy = st
    ex, ey, yaw, evx, evy = ego
    x0, y0 = _to_ego(wx, wy, ex, ey, yaw)
    # Relative velocity, expressed in the ego frame.  The ego is assumed to hold
    # its heading over the horizon, so the frame does not rotate as we step.
    rvx, rvy = _rotate_to_ego(vx - evx, vy - evy, yaw)
    out = []
    t = 0.0
    while t <= horizon + 1e-9:
        out.append((t, x0 + rvx * t, y0 + rvy * t))
        t += step
    return out


def time_to_intrusion(path, half_l=EGO_HALF_L, half_w=EGO_HALF_W, margin=MARGIN):
    """First predicted time the target is inside the ego's box, or inf.

    This is the number the radial TTC cannot produce: it is satisfied by a target
    that arrives sideways just as readily as by one that arrives head-on.
    """
    if not path:
        return float("inf")
    L, W = half_l + margin, half_w + margin
    for (t, x, y) in path:
        if abs(x) <= L and abs(y) <= W:
            return t
    return float("inf")


def time_to_lane(path, half_w=EGO_HALF_W, margin=MARGIN, rear_only=True):
    """Predicted time until the target CROSSES INTO the ego's lane, or inf.

    Separate from intrusion because a car that swings into your lane 20 m back has
    not hit you and may never -- but it HAS just taken away your escape route, and
    on a rear-facing warning that is the event a driver reacts to.

    Note the word "crosses".  A target already inside the lane returns inf, not 0:
    the car following you in your own lane has not just done anything, and the
    first version of this function -- which only asked "is |y| small" -- fired at
    the first frame for a steady follower, an overtake, everything.  This reports
    an entry, so it needs the target to be outside the lane first.
    """
    if not path or len(path) < 2:
        return float("inf")
    W = half_w + margin
    y0 = path[0][2]
    if abs(y0) <= W:
        return float("inf")
    span = path[-1][0] - path[0][0]
    if span <= 0:
        return float("inf")
    lat_speed = (path[-1][2] - y0) / span
    # Must actually be coming towards the lane, fast enough to be real.
    if abs(lat_speed) < MIN_LAT_SPEED or lat_speed * y0 >= 0:
        return float("inf")
    for (t, x, y) in path:
        if rear_only and x > 0:
            continue
        if abs(y) <= W:
            return t
    return float("inf")


def weave(whist, window=WEAVE_WINDOW, min_sdlp=WEAVE_MIN_SDLP,
          min_rev=WEAVE_MIN_REV, smooth=SMOOTH):
    """Is this track oscillating about its own mean path?

    Returns (weaving, amplitude_m, reversals, judgeable, sdlp_m).

    sdlp_m is the standard deviation of the lateral deviation -- the same quantity
    driving-safety research calls SDLP (Standard Deviation of Lane Position), the
    established measure of how badly a driver is holding their line and the
    primary endpoint in alcohol, fatigue and drug on-road driving studies.

    Two honest differences from the literature definition, which matter if anyone
    who knows it is listening.  It measures the EGO vehicle against the LANE
    CENTRE with centimetre-grade instrumentation, and treats an increase of about
    2.4 cm as meaningful because that is roughly what a blood alcohol
    concentration of 0.5 g/L costs.  This measures ANOTHER vehicle against ITS OWN
    mean path, from a BEV centroid carrying ~0.5 m of noise -- two orders of
    magnitude coarser, so it detects gross weaving, not subtle impairment.  The
    mean-path baseline is deliberate: the ego does not reliably know another car's
    lane centre, and referencing the target's own path is what stops the ego's own
    lane changes reading as someone else weaving.  Call it SDLP-style, not SDLP.

    judgeable is False when the track has not moved far enough over the window to
    say anything about how it is being driven.  The caller must not read False
    weaving as good behaviour in that case: a car stopped at a red light produces
    no lateral deviation at all, and counting that as clean driving is what let a
    weaving car serve out its probation while both cars sat at a light.

    Method: fit a straight line to the target's world path over the window, take
    each sample's signed perpendicular distance from that line, smooth it, then
    measure peak-to-peak and count sign changes.  Referencing the target's OWN
    mean heading rather than the ego's is what keeps this from firing every time
    the ego changes lane -- the deviation is a property of the other driver.
    """
    if not whist:
        return False, 0.0, 0, False, 0.0
    t_last = whist[-1][0]
    win = [p for p in whist if t_last - p[0] <= window]
    if len(win) < max(smooth * 2, 8):
        return False, 0.0, 0, False, 0.0

    xs = [p[1] for p in win]
    ys = [p[2] for p in win]
    n = len(win)
    mx, my = sum(xs) / n, sum(ys) / n
    # Principal direction of the path, from the covariance -- more stable than
    # first-to-last, which one noisy endpoint can swing.
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx + syy < 1.0:
        # Barely moved: a parked car's "deviation" is pure noise, and its lack of
        # weaving says nothing about the driver.
        return False, 0.0, 0, False, 0.0
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)

    dev = [-(x - mx) * uy + (y - my) * ux for x, y in zip(xs, ys)]
    sm = []
    for i in range(len(dev)):
        lo = max(0, i - smooth // 2)
        hi = min(len(dev), i + smooth // 2 + 1)
        sm.append(sum(dev[lo:hi]) / (hi - lo))

    amp = max(sm) - min(sm)
    mean_sm = sum(sm) / len(sm)
    sdlp = math.sqrt(sum((v - mean_sm) ** 2 for v in sm) / len(sm))
    # Count reversals of the smoothed deviation's direction, ignoring the
    # dithering around a turning point that noise alone produces.
    rev = 0
    last_sign = 0
    for i in range(1, len(sm)):
        d = sm[i] - sm[i - 1]
        if abs(d) < 0.02:
            continue
        sign = 1 if d > 0 else -1
        if last_sign and sign != last_sign:
            rev += 1
        last_sign = sign
    return (sdlp >= min_sdlp and rev >= min_rev), amp, rev, True, sdlp


# --- the two tests want different fit windows, and that is not a fudge ---------
#
# time_to_intrusion is dominated by the longitudinal channel: a car has to close
# 20 m to reach us, and 20 m is estimated far better than 1 m of lateral drift.
# A long window suppresses noise and costs almost nothing, so it takes 2.0 s.
#
# time_to_lane is purely lateral, and a lane change takes about 2 s start to
# finish.  Fitting it with a 2.0 s trailing window means the estimate is a blend
# of "before" and "during" for the whole manoeuvre, and the warning arrives after
# the car has already arrived.  Measured -- longest run of consecutive frames the
# lane test fires, 0.5 s (5 frames) of hysteresis downstream, three seeds:
#
#     fit window   min_lat     real lane change     car steady one lane over
#       1.0 s      1.0 m/s          7 frames              2 frames
#       1.5 s      0.8 m/s          5 frames              1 frame
#       2.0 s      0.8 m/s          3 frames  (missed)    0 frames
#
# 1.0 s / 1.0 m/s separates best: the real manoeuvre clears the hysteresis with
# room to spare, the noisy straight line never does.  So the lateral channel gets
# its own shorter window rather than compromising both tests with one number.

def assess(whist, ego, horizon=HORIZON):
    """Everything this module can say about one track.

    Returns a dict:
      tti       predicted seconds until it is inside the ego's box, or inf
      ttl       predicted seconds until it crosses into the ego's lane, or inf
      weaving   bool, is the driver behind oscillating
      amp       peak-to-peak lateral swing over the weave window, metres
      rev       direction reversals counted in that window
      path      the predicted relative path, for drawing, or None
    """
    path = predict_relative(whist, ego, horizon=horizon, window=FIT_WINDOW)
    lat_path = predict_relative(whist, ego, horizon=horizon, window=LANE_WINDOW)
    weaving, amp, rev, judgeable, sdlp = weave(whist)
    st = world_state(whist, window=FIT_WINDOW)
    return {
        "tti": time_to_intrusion(path),
        "ttl": time_to_lane(lat_path),
        "weaving": weaving, "amp": amp, "rev": rev, "judgeable": judgeable,
        "sdlp": sdlp,
        "path": path,
        "state": st,          # (wx, wy, vx, vy) -- what re-identification needs
    }


# Thresholds for turning an assessment into a warning.  Deliberately looser than
# the radial TTC ones: a prediction three seconds out is a weaker claim than a
# measured closing rate, so it escalates later and says why.
T_INTRUDE_DANGER = 2.0
T_INTRUDE_CAUTION = 3.0
T_LANE_CAUTION = 2.0


def predict_level(a, flagged=False):
    """(level, reason) for one assessment.  reason is a short tag for the HUD/CSV.

    A weave on its own is CAUTION, never DANGER: the car behind is being driven
    erratically, which the driver should know, but erratic is not the same as
    about to hit you -- and a system that shouts DANGER at every swerving car
    thirty metres back would be muted within a minute.

    `flagged` is the watch-list answer for this vehicle, and it does two things.
    It keeps the CAUTION up through the moments a weaving car happens to be
    driving straight, and it escalates an intrusion one level early: a vehicle
    already known to be driven erratically gets less benefit of the doubt when it
    starts coming at you than an anonymous one does.  That is the same judgement
    a human driver makes about the car that has been swerving behind them for the
    last twenty seconds.
    """
    if a["tti"] < T_INTRUDE_DANGER:
        return "DANGER", "INTRUDE"
    if a["tti"] < T_INTRUDE_CAUTION:
        return ("DANGER" if flagged else "CAUTION"), ("WEAVE+INTRUDE" if flagged
                                                      else "INTRUDE")
    if a["ttl"] < T_LANE_CAUTION:
        return ("DANGER" if flagged else "CAUTION"), ("WEAVE+LANE" if flagged
                                                      else "LANE")
    if a["weaving"] or flagged:
        return "CAUTION", "WEAVE"
    return "SAFE", ""

# --- the watch list -----------------------------------------------------------

WATCH_ARM = 15         # frames of weaving before a vehicle is put on the list
WATCH_RELEASE = 300    # frames of genuine clean driving before it comes off again
WATCH_CLEAR_SDLP = 0.25  # SDLP below which a vehicle counts as genuinely straight,
                       # and so serves its probation. Between this and
                       # WEAVE_MIN_SDLP (0.52 m) the answer is "cannot tell" and
                       # neither arms nor releases. 0.25 m sits comfortably above the
                       # 0.13 m median that straight driving reads at this noise
                       # level, and well below the weave threshold.
                       #
                       # Without the band, a flagged weaver that drifted out to
                       # 28-30 m -- where the swing stops clearing the detection
                       # threshold, but noise keeps the measurement well above the
                       # floor -- was credited with 30 s of clean driving and taken
                       # off the list, 44 s before it was flagged again. Measured
                       # on a 240 s run, 2026-09-17. Failing to measure a weave is
                       # not evidence of good driving, and it is the same mistake
                       # as counting a car stopped at a light.
GHOST_TTL = 25.0       # seconds a lost flagged vehicle stays re-identifiable.
                       # 10 s was not enough: on a 240 s run a flagged weaver lost
                       # at 103.5 s did not produce a matchable track again until
                       # 118.3 s -- 14.8 s later, past the old TTL -- so it came
                       # back a stranger and had to earn the flag from scratch. A
                       # vehicle that disappears behind a bus for fifteen seconds
                       # is still the same vehicle when it comes out.
GHOST_GATE = 6.0       # metres of slack when matching it again ...
GHOST_GROWTH = 7.0     # ... plus this much per second it has been missing ...
GHOST_GATE_MAX = 25.0  # ... never exceeding this.
                       # 5 m/s of growth, not the 2.5 first tried: the vehicle may
                       # have been STOPPED when its track died -- at a light, which
                       # is exactly when junctions break tracks -- so the ghost
                       # carries zero velocity while the real car drives off at
                       # town speed. Dead reckoning is then wrong by the full
                       # distance travelled, and a gate that grows slower than
                       # traffic moves can never catch up. The cap keeps a stale
                       # ghost from swallowing half the map.


class WatchList(object):
    """Which vehicles have been caught driving erratically, and stay caught.

    A risk level answers "how bad is this situation right now" and is recomputed
    every frame.  That is the wrong shape for "that driver is a menace", which is
    a property of the vehicle rather than of the current geometry: a weaving car
    is straight for a moment twice per cycle, and a per-frame answer drops the
    label exactly then.  So this latches.

    Two things learned from a 70 s recording on 2026-09-17, in which a car that
    had been correctly flagged at 17 m was a plain green SAFE square at 20 m
    thirty seconds later, with the driver visibly still behind us:

    1. STOPPED IS NOT CLEAN.  Both cars sat at a red light.  A stationary car has
       no lateral deviation, weave() correctly returns False, and the first
       version counted every one of those frames towards the vehicle's release --
       so it served out its probation without ever driving straight.  update()
       now takes `judgeable` and holds its state when the answer means "cannot
       tell" rather than "behaving".

    2. A NEW TRACK ID IS NOT A NEW CAR.  Going through a junction, or one gap in
       detection longer than the tracker's max_miss, kills the track; the same
       car comes back as a fresh id with no history, and pruning the old id threw
       the record away with it.  A flagged vehicle now leaves a ghost -- last
       known world position and velocity -- and a track id we have not seen
       before is matched against those ghosts before it is treated as a stranger.

    The re-identification gate grows with the time since the vehicle was lost,
    because so does the uncertainty in where dead reckoning says it should be.
    It is deliberately generous: wrongly adopting a neighbouring car costs one
    over-cautious warning, while missing the re-identification costs exactly the
    failure this was written to fix.
    """

    debug = None             # set to a print-like callable to trace decisions

    def __init__(self, arm=WATCH_ARM, release=WATCH_RELEASE, ghost_ttl=GHOST_TTL,
                 gate=GHOST_GATE, gate_growth=GHOST_GROWTH,
                 clear_sdlp=WATCH_CLEAR_SDLP):
        self.arm = max(int(arm), 1)
        self.release = max(int(release), 0)      # 0 = never release
        self.ghost_ttl = ghost_ttl
        self.gate = gate
        self.gate_growth = gate_growth
        self.clear_sdlp = clear_sdlp
        self.state = {}          # key -> (flagged, weave_frames, clean_frames)
        self.last = {}           # key -> (wx, wy, vx, vy, t)
        self.ghosts = []         # flagged vehicles whose track has died

    def update(self, key, weaving, judgeable=True, sdlp=0.0, state=None, t=0.0):
        """Fold one frame in; returns True while this vehicle is on the list.

        Three outcomes, not two.  A frame either shows a weave, shows genuinely
        straight driving, or shows neither clearly -- and the third case must not
        be filed as the second.
        """
        flagged, wf, cf = self.state.get(key, (False, 0, 0))
        if not judgeable or (not weaving and sdlp > self.clear_sdlp):
            pass                 # cannot tell: neither arm nor release
        elif weaving:
            wf, cf = wf + 1, 0
            if wf >= self.arm:
                flagged = True
        else:
            cf += 1
            if flagged and self.release and cf >= self.release:
                flagged, wf = False, 0
                if self.debug:
                    self.debug("released track %s after %d clean frames" % (key, cf))
            elif not flagged and cf > self.arm:
                wf = 0           # an isolated positive should not accumulate
        self.state[key] = (flagged, wf, cf)
        if state is not None:
            self.last[key] = (state[0], state[1], state[2], state[3], t)
        return flagged

    def adopt(self, key, wx, wy, t):
        """Is this unfamiliar track id a flagged vehicle coming back?

        Call it for any key update() has not seen yet.  Returns True when a ghost
        matched, and the key inherits the flag already armed.
        """
        if key in self.state:
            return self.state[key][0]
        best, best_d = None, None
        for g in self.ghosts:
            dt = t - g[4]
            if dt < 0 or dt > self.ghost_ttl:
                continue
            px, py = g[0] + g[2] * dt, g[1] + g[3] * dt
            d = math.hypot(wx - px, wy - py)
            gate = min(self.gate + self.gate_growth * dt, GHOST_GATE_MAX)
            if d <= gate and (best_d is None or d < best_d):
                best, best_d = g, d
        if best is None:
            if self.debug:
                self.debug("adopt(%s) failed: %d ghosts, nearest %s"
                           % (key, len(self.ghosts),
                              ("%.1f m" % min(
                                  math.hypot(wx - (g[0] + g[2] * (t - g[4])),
                                             wy - (g[1] + g[3] * (t - g[4])))
                                  for g in self.ghosts) if self.ghosts else "-")))
            return False
        self.ghosts.remove(best)
        self.state[key] = (True, self.arm, 0)
        return True

    def flagged(self, key):
        return self.state.get(key, (False, 0, 0))[0]

    def count(self):
        return sum(1 for v in self.state.values() if v[0])

    def prune(self, keys, t=0.0):
        """Drop dead tracks, but keep a ghost of any that were flagged."""
        keys = set(keys)
        for k in [k for k in self.state if k not in keys]:
            if self.state[k][0] and k in self.last:
                self.ghosts.append(self.last[k])
                if self.debug:
                    self.debug("ghost kept for lost flagged track %s at (%.1f, %.1f)"
                               % (k, self.last[k][0], self.last[k][1]))
            elif self.state[k][0] and self.debug:
                self.debug("lost flagged track %s but had NO stored state" % k)
            del self.state[k]
            self.last.pop(k, None)
        self.ghosts = [g for g in self.ghosts if t - g[4] <= self.ghost_ttl]
