"""Map a steering wheel's axes, and write wheel_config.ini for wheel_input.py.

    python calibrate_wheel.py            # guided: does every step, writes the ini
    python calibrate_wheel.py --live     # just show live values, map nothing

Reads the wheel through winmm's joyGetPosEx -- the same multimedia joystick API
Windows exposes to any DirectInput device -- so this needs no pygame and nothing
installed into the lss env.

Why a tool instead of a fixed axis table: the wheel on this machine enumerates as
VID 0E8F (GreenAsia), a generic HID chipset with no vendor driver, so its axis
order is whatever the board happens to report.  Nothing may be assumed; it has to
be measured.

Two measured quirks of this device, both handled below:

  * Axes read a flat 32767 until the device sends its first real HID report.
    That is not a reading, it is "no data yet" -- and it is indistinguishable
    from a genuinely centred axis, so a baseline taken before the wheel is
    touched is worthless.  Every step here therefore looks for MOVEMENT, and
    the idle step happens last, once the device is awake.
  * Only the axes the board actually populates ever change.  This wheel reports
    wNumAxes=4 (X, Y, Z, R) but leaves Z and R parked at the default forever.
"""

import argparse
import ctypes
import os
import sys
import time
from ctypes import wintypes

WINMM = ctypes.WinDLL("winmm")

JOY_RETURNALL = 0x000000FF
AXES = ("X", "Y", "Z", "R", "U", "V")
DEFAULT = 32767          # what an un-reported axis reads; see module docstring
FULL = 65535


class JOYINFOEX(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("dwXpos", wintypes.DWORD), ("dwYpos", wintypes.DWORD),
                ("dwZpos", wintypes.DWORD), ("dwRpos", wintypes.DWORD),
                ("dwUpos", wintypes.DWORD), ("dwVpos", wintypes.DWORD),
                ("dwButtons", wintypes.DWORD), ("dwButtonNumber", wintypes.DWORD),
                ("dwPOV", wintypes.DWORD), ("dwReserved1", wintypes.DWORD),
                ("dwReserved2", wintypes.DWORD)]


class JOYCAPS(ctypes.Structure):
    _fields_ = [("wMid", wintypes.WORD), ("wPid", wintypes.WORD),
                ("szPname", ctypes.c_wchar * 32),
                ("wXmin", wintypes.UINT), ("wXmax", wintypes.UINT),
                ("wYmin", wintypes.UINT), ("wYmax", wintypes.UINT),
                ("wZmin", wintypes.UINT), ("wZmax", wintypes.UINT),
                ("wNumButtons", wintypes.UINT),
                ("wPeriodMin", wintypes.UINT), ("wPeriodMax", wintypes.UINT),
                ("wRmin", wintypes.UINT), ("wRmax", wintypes.UINT),
                ("wUmin", wintypes.UINT), ("wUmax", wintypes.UINT),
                ("wVmin", wintypes.UINT), ("wVmax", wintypes.UINT),
                ("wCaps", wintypes.UINT), ("wMaxAxes", wintypes.UINT),
                ("wNumAxes", wintypes.UINT), ("wMaxButtons", wintypes.UINT),
                ("szRegKey", ctypes.c_wchar * 32),
                ("szOEMVxD", ctypes.c_wchar * 260)]


class DeviceLost(Exception):
    """The wheel stopped answering part-way through a calibration."""

    def __init__(self, dev):
        Exception.__init__(
            self,
            "slot %d stopped answering -- the wheel dropped off the bus.\n"
            "  Check the USB cable and the power adapter, and in Device\n"
            "  Manager turn off 'Allow the computer to turn off this device\n"
            "  to save power' on its USB hub.  Then run this again." % dev)
        self.dev = dev


def read(dev):
    """Current axis values and button mask, or None if the device is gone."""
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNALL
    # joyGetPosEx has no A/W pair -- JOYINFOEX carries no strings.  Only
    # joyGetDevCaps, which does, comes in two flavours.
    if WINMM.joyGetPosEx(dev, ctypes.byref(info)) != 0:
        return None
    return ({"X": info.dwXpos, "Y": info.dwYpos, "Z": info.dwZpos,
             "R": info.dwRpos, "U": info.dwUpos, "V": info.dwVpos},
            info.dwButtons)


def devices():
    """Every joystick slot that answers, with its capabilities."""
    out = []
    for dev in range(16):
        caps = JOYCAPS()
        if WINMM.joyGetDevCapsW(dev, ctypes.byref(caps),
                                ctypes.sizeof(JOYCAPS)) != 0:
            continue
        out.append((dev, caps))
    return out


def pick(want=None):
    """Choose the device to calibrate.  Prefers one that reports real axes."""
    found = devices()
    if not found:
        sys.exit("no joystick device answered -- is the wheel plugged in "
                 "and powered?")
    print("joystick slots that answer:")
    for dev, caps in found:
        print("  [%d] %-28s  VID %04X PID %04X  %d axes  %d buttons"
              % (dev, caps.szPname.strip(), caps.wMid, caps.wPid,
                 caps.wNumAxes, caps.wNumButtons))
    if want is not None:
        return want
    # A slot with zero axes is a keyboard exposing a controller collection,
    # not a wheel.  Take the first slot that claims axes.
    for dev, caps in found:
        if caps.wNumAxes > 0:
            print("\nusing slot %d (%d axes)" % (dev, caps.wNumAxes))
            return dev
    # Falling back to the first slot regardless is how a whole calibration ran
    # against a keyboard, read zeroes five times, and wrote an ini naming
    # device 0.  A wrong device is worse than no device: it produces a file
    # that looks like a result.
    sys.exit("no slot reports any axes -- the wheel is not connected.\n"
             "  The slots above are keyboards or receivers exposing a game\n"
             "  controller collection, not a wheel.  Re-plug the wheel (and\n"
             "  its power adapter, if it has one) and run this again.")


def watch(dev, seconds, label):
    """Sample for `seconds`, drawing live bars.  Returns min/max per axis."""
    lo = {a: None for a in AXES}
    hi = {a: None for a in AXES}
    buttons = 0
    t0 = time.time()
    lines = 0
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        got = read(dev)
        if got is None:
            # The device stopped answering part-way through.  Say so here
            # rather than letting every remaining step report "nothing moved"
            # and blaming the axis map for a USB problem.
            raise DeviceLost(dev)
        vals, btn = got
        buttons |= btn
        for a in AXES:
            v = vals[a]
            lo[a] = v if lo[a] is None else min(lo[a], v)
            hi[a] = v if hi[a] is None else max(hi[a], v)
        if lines:
            sys.stdout.write("\033[%dA" % lines)
        out = ["  %-18s  %4.1f s left" % (label, left)]
        for a in AXES:
            pos = int(vals[a] / FULL * 40)
            bar = " " * pos + "#"
            out.append("   %s %6d  |%-41s|" % (a, vals[a], bar))
        out.append("   buttons 0x%03X" % btn)
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        lines = len(out)
        time.sleep(0.02)
    return lo, hi, buttons


def moved(lo, hi, floor=3000):
    """Axes whose travel during a step was more than noise, widest first."""
    spread = [(a, hi[a] - lo[a]) for a in AXES
              if lo[a] is not None and hi[a] - lo[a] >= floor]
    return sorted(spread, key=lambda kv: -kv[1])


def step(dev, label, prompt, seconds=5.0):
    print("\n" + "-" * 62)
    print("  %s" % prompt)
    for n in (3, 2, 1):
        sys.stdout.write("\r  starting in %d ..." % n)
        sys.stdout.flush()
        time.sleep(1.0)
    sys.stdout.write("\r" + " " * 40 + "\r")
    lo, hi, btn = watch(dev, seconds, label)
    hits = moved(lo, hi)
    if hits:
        for a, span in hits:
            print("   -> %s moved %d..%d (span %d)" % (a, lo[a], hi[a], span))
    else:
        print("   -> nothing moved")
    return lo, hi, btn, hits


def bit_of(mask):
    """Which button a mask stands for, 1-based, or None if it is not exactly one."""
    if mask == 0 or (mask & (mask - 1)) != 0:
        return None                      # nothing, or more than one at once
    return mask.bit_length()


def map_button(dev, name, timeout=25.0):
    """Ask for one button and return its 1-based number.

    Asks twice.  A wheel rim often has two buttons under one thumb, and a
    mis-mapped gear toggle is discovered mid-demo, in reverse, with an
    audience watching.
    """
    picked = None
    for attempt in ("Press the button you want for %s." % name,
                    "Press the SAME button once more to confirm."):
        print("\n  %s" % attempt)
        print("  (waiting -- Ctrl-C to give up)")
        # Wait for a clean release first, so holding the button from the
        # previous step is not read as the next press.
        t0 = time.time()
        while time.time() - t0 < timeout:
            got = read(dev)
            if got is None:
                raise DeviceLost(dev)
            if got[1] == 0:
                break
            time.sleep(0.02)
        mask = 0
        t0 = time.time()
        while time.time() - t0 < timeout:
            got = read(dev)
            if got is None:
                raise DeviceLost(dev)
            if got[1] != 0:
                mask = got[1]
                break
            time.sleep(0.02)
        if mask == 0:
            print("  nothing pressed within %.0f s" % timeout)
            return None
        n = bit_of(mask)
        if n is None:
            print("  more than one button reported at once (0x%03X) -- "
                  "press just the one." % mask)
            return None
        print("  -> button %d (mask 0x%03X)" % (n, mask))
        if picked is None:
            picked = n
        elif picked != n:
            print("  that was button %d, not %d -- nothing written." % (n, picked))
            return None
    return picked


def merge_ini(path, key, value):
    """Set one key in an existing ini, keeping every other line as it is."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = ["[wheel]"]
    done = False
    for i, line in enumerate(lines):
        if line.split("#", 1)[0].strip().startswith(key):
            head = line.split("=", 1)[0]
            if head.strip() == key:
                lines[i] = "%s = %s" % (key, value)
                done = True
                break
    if not done:
        lines.append("%s = %s" % (key, value))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None,
                    help="joystick slot to use (default: first with axes)")
    ap.add_argument("--live", action="store_true",
                    help="just show live values until Ctrl-C, map nothing")
    ap.add_argument("--button", choices=("reverse", "handbrake"),
                    help="map ONE wheel button to this function and merge it "
                         "into the existing wheel_config.ini. Does not touch "
                         "the axis calibration, so it is safe to run on its own.")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="how long each step samples (default 5)")
    ap.add_argument("--out", default="wheel_config.ini",
                    help="where to write the mapping")
    args = ap.parse_args()

    os.system("")          # let Windows terminals honour the ANSI cursor moves
    dev = pick(args.device)

    if args.live:
        print("\nlive values -- Ctrl-C to stop\n")
        try:
            while True:
                watch(dev, 1.0, "live")
        except KeyboardInterrupt:
            print("\nstopped")
        return

    if args.button:
        path = args.out
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(path):
            sys.exit("%s does not exist yet -- run the full calibration first, "
                     "then map buttons onto it." % path)
        n = map_button(dev, args.button)
        if n is None:
            sys.exit("nothing mapped -- %s left as it was" % os.path.basename(path))
        merge_ini(path, "%s_button" % args.button, n)
        print("\nwrote %s_button = %d to %s" % (args.button, n, path))
        return

    print("\nEach step samples for %.0f s after a 3-second countdown." % args.seconds)
    print("HOLD the input for the whole step -- do not pump it.")

    # Collect every step first and work out the mapping afterwards.  The idle
    # reading is what "pressed" is measured against, and it can only be taken
    # once the device is awake -- so it comes last, and nothing can be decided
    # until it is in.
    plan = [("left", "Turn the wheel FULL LEFT and hold it there."),
            ("right", "Now turn the wheel FULL RIGHT and hold it there."),
            ("throttle", "Press the ACCELERATOR all the way down and hold it."),
            ("brake", "Press the BRAKE all the way down and hold it."),
            ("idle", "Let go of everything.  Do not touch the wheel.")]
    seen = {}
    for name, prompt in plan:
        lo, hi, btn, hits = step(dev, name, prompt, args.seconds)
        seen[name] = {"lo": lo, "hi": hi, "buttons": btn, "hits": hits}

    cfg = {"device": dev}
    idle = seen["idle"]

    def rest_of(axis):
        """Where an axis sits untouched.  Mid-range of the idle step."""
        lo, hi = idle["lo"][axis], idle["hi"][axis]
        return (lo + hi) // 2

    # --- steering ---------------------------------------------------------
    steer = None
    for name in ("left", "right"):
        if seen[name]["hits"]:
            steer = seen[name]["hits"][0][0]
            break
    if steer is None:
        print("\n   !! the steering axis never moved -- check the wheel")
    else:
        left = seen["left"]["lo"][steer]
        right = seen["right"]["hi"][steer]
        cfg["steer_axis"] = steer
        cfg["steer_centre"] = rest_of(steer)
        if right > left:
            cfg["steer_min"], cfg["steer_max"] = left, right
            cfg["steer_invert"] = "no"
        else:
            # The board counts the other way.  Record it, do not fight it.
            cfg["steer_min"] = seen["right"]["lo"][steer]
            cfg["steer_max"] = seen["left"]["hi"][steer]
            cfg["steer_invert"] = "yes"

    # --- pedals -----------------------------------------------------------
    # A cheap wheel often puts both pedals on ONE axis: released sits at the
    # centre, accelerator drives it one way, brake the other.  That is not a
    # quirk to route around -- it has to be mapped as three points on one
    # axis, because the two ends are mutually exclusive by construction (you
    # cannot read "both pedals down" from a single number, and the hardware
    # cannot express it either).
    acc = seen["throttle"]["hits"][0][0] if seen["throttle"]["hits"] else None
    brk = seen["brake"]["hits"][0][0] if seen["brake"]["hits"] else None

    def far_end(name, axis):
        """The value furthest from rest during that step."""
        lo, hi = seen[name]["lo"][axis], seen[name]["hi"][axis]
        r = rest_of(axis)
        return lo if abs(lo - r) >= abs(hi - r) else hi

    if acc is not None and acc == brk:
        cfg["pedal_mode"] = "combined"
        cfg["pedal_axis"] = acc
        cfg["pedal_rest"] = rest_of(acc)
        cfg["pedal_throttle_end"] = far_end("throttle", acc)
        cfg["pedal_brake_end"] = far_end("brake", acc)
        print("\n   both pedals share axis %s -- combined mode" % acc)
    else:
        cfg["pedal_mode"] = "separate"
        if acc is not None:
            cfg["throttle_axis"] = acc
            cfg["throttle_rest"] = rest_of(acc)
            cfg["throttle_full"] = far_end("throttle", acc)
        else:
            print("\n   !! the accelerator moved nothing -- pedals plugged in?")
        if brk is not None:
            cfg["brake_axis"] = brk
            cfg["brake_rest"] = rest_of(brk)
            cfg["brake_full"] = far_end("brake", brk)
        else:
            print("   !! the brake moved nothing")

    # --- what never moved at all -----------------------------------------
    dead = [a for a in AXES
            if all(not any(h[0] == a for h in seen[n]["hits"]) for n, _ in plan)]
    if dead:
        cfg["unused_axes"] = ",".join(dead)
        print("   axes that never moved: %s" % ", ".join(dead))

    print("\n" + "=" * 62)
    print("  mapping")
    print("=" * 62)
    for k in sorted(cfg):
        print("  %-20s %s" % (k, cfg[k]))

    if "steer_axis" not in cfg:
        sys.exit("\nnothing was mapped -- not writing %s.\n"
                 "  An ini without a steering axis is not a partial result,\n"
                 "  it is a file that makes wheel_input.py fail later instead\n"
                 "  of here." % args.out)

    # Write next to this script, not next to wherever the shell happened to
    # be.  A calibration that lands in C:\Users\... is one wheel_input.py
    # will not find, and the first run of this tool wrote nothing anywhere.
    path = args.out
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# written by calibrate_wheel.py -- %s\n"
                 % time.strftime("%Y-%m-%d %H:%M"))
        fh.write("[wheel]\n")
        for k in sorted(cfg):
            fh.write("%s = %s\n" % (k, cfg[k]))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    try:
        main()
    except DeviceLost as exc:
        sys.exit("\n%s" % exc)
