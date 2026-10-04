"""Read a steering wheel and hand live_demo.py steer / throttle / brake.

    from wheel_input import WheelInput
    wheel = WheelInput()                  # reads wheel_config.ini beside this file
    s = wheel.poll()
    ctl.steer, ctl.throttle, ctl.brake = s.steer, s.throttle, s.brake

Run it directly to watch the normalised values:

    python wheel_input.py                 # live view, Ctrl-C to stop

The raw reading comes from winmm's joyGetPosEx -- the same call calibrate_wheel.py
uses -- so this adds no dependency to the lss env.  Everything device-specific
lives in wheel_config.ini; there is no axis table in this file on purpose,
because the wheel here is a generic VID 0E8F board whose axis order is not
guaranteed to survive a driver update, let alone a different wheel.

What this replaces.  live_demo.py drives from cv2's waitKey, which reports "a key
is down" and never "a key came up", so the keyboard path has to ramp the throttle
in and decay the steering by 0.7 every frame to imitate an analogue control that
returns to centre.  A wheel reports absolute positions, so none of that applies:
the values below are what the driver is physically doing, this frame.
"""

import ctypes
import os
import sys
import time
from ctypes import wintypes

WINMM = ctypes.WinDLL("winmm")
JOY_RETURNALL = 0x000000FF
AXES = ("X", "Y", "Z", "R", "U", "V")

# An axis that has never been reported reads exactly this, and so does an axis
# sitting honestly in the middle -- the two are indistinguishable from one
# sample.  See _looks_unreported.
UNREPORTED = 32767


class JOYINFOEX(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("dwXpos", wintypes.DWORD), ("dwYpos", wintypes.DWORD),
                ("dwZpos", wintypes.DWORD), ("dwRpos", wintypes.DWORD),
                ("dwUpos", wintypes.DWORD), ("dwVpos", wintypes.DWORD),
                ("dwButtons", wintypes.DWORD), ("dwButtonNumber", wintypes.DWORD),
                ("dwPOV", wintypes.DWORD), ("dwReserved1", wintypes.DWORD),
                ("dwReserved2", wintypes.DWORD)]


class State(object):
    """One frame of driver input, in the units carla.VehicleControl wants."""

    __slots__ = ("steer", "throttle", "brake", "buttons", "live", "raw")

    def __init__(self, steer=0.0, throttle=0.0, brake=0.0, buttons=0,
                 live=False, raw=None):
        self.steer = steer          # -1.0 full left .. +1.0 full right
        self.throttle = throttle    # 0.0 .. 1.0
        self.brake = brake          # 0.0 .. 1.0
        self.buttons = buttons      # bit mask, bit 0 = button 1
        self.live = live            # False when the wheel has said nothing yet
        self.raw = raw or {}

    def pressed(self, n):
        """True while button `n` (1-based, as the calibration prints them) is down."""
        return bool(self.buttons & (1 << (n - 1)))

    def __repr__(self):
        return ("State(steer=%+.2f throttle=%.2f brake=%.2f buttons=0x%03X %s)"
                % (self.steer, self.throttle, self.brake, self.buttons,
                   "live" if self.live else "ASLEEP"))


class WheelNotConfigured(Exception):
    pass


def _read_ini(path):
    cfg = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("["):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def _num(cfg, key, default=None):
    if key not in cfg:
        if default is None:
            raise WheelNotConfigured("wheel_config.ini has no '%s'" % key)
        return default
    return int(cfg[key])


def _span(value, rest, end):
    """0.0 at `rest`, 1.0 at `end`, clamped.  Works whichever way `end` lies."""
    if end == rest:
        return 0.0
    return max(0.0, min(1.0, (value - rest) / float(end - rest)))


class WheelInput(object):
    def __init__(self, config=None, deadzone=0.06, pedal_deadzone=0.08):
        if config is None:
            config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "wheel_config.ini")
        if not os.path.exists(config):
            raise WheelNotConfigured(
                "%s not found -- run calibrate_wheel.py first" % config)
        cfg = _read_ini(config)
        self.config_path = config
        self.device = _num(cfg, "device", 0)
        self.deadzone = deadzone
        self.pedal_deadzone = pedal_deadzone

        self.steer_axis = cfg.get("steer_axis")
        if self.steer_axis is None:
            raise WheelNotConfigured("wheel_config.ini has no steer_axis")
        self.steer_min = _num(cfg, "steer_min")
        self.steer_max = _num(cfg, "steer_max")
        self.steer_centre = _num(cfg, "steer_centre",
                                 (self.steer_min + self.steer_max) // 2)
        self.steer_invert = cfg.get("steer_invert", "no").lower() in ("yes", "true", "1")

        self.mode = cfg.get("pedal_mode", "separate")
        if self.mode == "combined":
            self.pedal_axis = cfg["pedal_axis"]
            self.pedal_rest = _num(cfg, "pedal_rest")
            self.throttle_end = _num(cfg, "pedal_throttle_end")
            self.brake_end = _num(cfg, "pedal_brake_end")
        else:
            self.throttle_axis = cfg.get("throttle_axis")
            self.brake_axis = cfg.get("brake_axis")
            self.throttle_rest = _num(cfg, "throttle_rest", 0)
            self.throttle_full = _num(cfg, "throttle_full", 0)
            self.brake_rest = _num(cfg, "brake_rest", 0)
            self.brake_full = _num(cfg, "brake_full", 0)

        # Optional: a button that latches reverse.  Without it live_demo.py's
        # R key still works, so this stays unset until someone wants it.
        self.reverse_button = int(cfg["reverse_button"]) if "reverse_button" in cfg else None
        self.handbrake_button = int(cfg["handbrake_button"]) if "handbrake_button" in cfg else None

        self._ever_live = False

    # -- raw ---------------------------------------------------------------
    def _raw(self):
        info = JOYINFOEX()
        info.dwSize = ctypes.sizeof(JOYINFOEX)
        info.dwFlags = JOY_RETURNALL
        if WINMM.joyGetPosEx(self.device, ctypes.byref(info)) != 0:
            return None, 0
        return ({"X": info.dwXpos, "Y": info.dwYpos, "Z": info.dwZpos,
                 "R": info.dwRpos, "U": info.dwUpos, "V": info.dwVpos},
                info.dwButtons)

    @staticmethod
    def _looks_unreported(vals):
        """True while the board has sent nothing since the driver loaded.

        Measured on this wheel: every axis reads a flat 32767 and U/V read 0
        until the first real HID report arrives, after which the pedal axis
        drops to its true resting value.  That matters because the pedal axis
        rests at one END of its travel, not the middle -- so the placeholder
        32767 sits mid-travel and would otherwise be read as a pedal held
        halfway down.  A demo that pulls away on its own before anyone
        touches it is worse than one that waits.
        """
        return (all(vals[a] == UNREPORTED for a in ("X", "Y", "Z", "R"))
                and vals["U"] == 0 and vals["V"] == 0)

    # -- normalised --------------------------------------------------------
    def _steer(self, vals):
        raw = vals[self.steer_axis]
        # Map each side against its own end stop.  This wheel's ends are not
        # symmetric about the centre, so one shared scale would make full lock
        # one way fall short of 1.0.
        if raw >= self.steer_centre:
            hi = self.steer_max - self.steer_centre
            out = (raw - self.steer_centre) / float(hi) if hi else 0.0
        else:
            lo = self.steer_centre - self.steer_min
            out = (raw - self.steer_centre) / float(lo) if lo else 0.0
        out = max(-1.0, min(1.0, out))
        if abs(out) < self.deadzone:
            return 0.0
        # Rescale so the travel just outside the deadzone starts from zero
        # instead of jumping to `deadzone`.
        out = (abs(out) - self.deadzone) / (1.0 - self.deadzone) * (1 if out > 0 else -1)
        return -out if self.steer_invert else out

    def _pedals(self, vals):
        if self.mode == "combined":
            raw = vals[self.pedal_axis]
            # One axis cannot express both pedals at once, and neither can the
            # hardware -- whichever side of rest the axis is on is the pedal
            # that is down.
            t_side = self.throttle_end > self.pedal_rest
            on_throttle_side = (raw > self.pedal_rest) == t_side
            if raw == self.pedal_rest:
                return 0.0, 0.0
            if on_throttle_side:
                t = _span(raw, self.pedal_rest, self.throttle_end)
                b = 0.0
            else:
                t = 0.0
                b = _span(raw, self.pedal_rest, self.brake_end)
        else:
            t = (_span(vals[self.throttle_axis], self.throttle_rest,
                       self.throttle_full) if self.throttle_axis else 0.0)
            b = (_span(vals[self.brake_axis], self.brake_rest,
                       self.brake_full) if self.brake_axis else 0.0)
        if t < self.pedal_deadzone:
            t = 0.0
        if b < self.pedal_deadzone:
            b = 0.0
        return t, b

    def poll(self):
        vals, buttons = self._raw()
        if vals is None:
            return State()                      # unplugged mid-run
        if self._looks_unreported(vals) and not self._ever_live:
            return State(buttons=buttons, live=False, raw=vals)
        self._ever_live = True
        t, b = self._pedals(vals)
        return State(steer=self._steer(vals), throttle=t, brake=b,
                     buttons=buttons, live=True, raw=vals)

    def describe(self):
        bits = ["device %d" % self.device,
                "steer %s [%d..%d] centre %d%s"
                % (self.steer_axis, self.steer_min, self.steer_max,
                   self.steer_centre, " inverted" if self.steer_invert else "")]
        if self.mode == "combined":
            bits.append("pedals %s combined: rest %d, throttle %d, brake %d"
                        % (self.pedal_axis, self.pedal_rest,
                           self.throttle_end, self.brake_end))
        else:
            bits.append("throttle %s [%d->%d], brake %s [%d->%d]"
                        % (self.throttle_axis, self.throttle_rest,
                           self.throttle_full, self.brake_axis,
                           self.brake_rest, self.brake_full))
        return "; ".join(bits)


def main():
    try:
        wheel = WheelInput()
    except WheelNotConfigured as exc:
        sys.exit(str(exc))
    print(wheel.describe())
    print("\nlive -- Ctrl-C to stop\n")
    os.system("")
    lines = 0
    try:
        while True:
            s = wheel.poll()
            rows = []
            if not s.live:
                rows.append("  waiting for the wheel's first report "
                            "-- turn it slightly")
            else:
                rows.append("  %s" % ("live" + " " * 40))
            for label, v, lo in (("steer", s.steer, -1.0),
                                 ("throttle", s.throttle, 0.0),
                                 ("brake", s.brake, 0.0)):
                pos = int((v - lo) / (1.0 - lo) * 40)
                rows.append("   %-9s %+5.2f |%-41s|"
                            % (label, v, " " * pos + "#"))
            rows.append("   buttons  0x%03X   raw %s"
                        % (s.buttons,
                           " ".join("%s=%d" % (a, s.raw.get(a, 0))
                                    for a in ("X", "Y", "Z", "R"))))
            if lines:
                sys.stdout.write("\033[%dA" % lines)
            sys.stdout.write("\n".join(rows) + "\n")
            sys.stdout.flush()
            lines = len(rows)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
