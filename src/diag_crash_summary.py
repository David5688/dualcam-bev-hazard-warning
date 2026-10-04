"""Print a one-line summary of a UE4 CrashContext.runtime-xml.

Shipping-build CARLA leaves no server log unless -log is passed, so the crash
dumps in %LOCALAPPDATA%\\CarlaUE4\\Saved\\Crashes are the primary evidence for
which failure mode a run hit.  The two modes seen on 2026-07-30:

    RenderThread 1/2, sec 38-107   -> first-tick crash (Town02 / Town05, 0 frames)
    TaskGraphThreadHP, sec 1499+   -> teardown crash after a full run (data safe)

Usage:  python diag_crash_summary.py <CrashContext.runtime-xml> [...]
        python diag_crash_summary.py --recent 20        # newest dumps
"""
import os
import re
import sys
import glob
import datetime

CRASHDIR = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "CarlaUE4", "Saved", "Crashes")


def _tag(xml, name):
    m = re.search(r"<%s>(.*?)</%s>" % (name, name), xml, re.S)
    return m.group(1).strip() if m else ""


def summarize(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        xml = fh.read()

    thread = "?"
    frames = []
    for block in re.findall(r"<Thread>(.*?)</Thread>", xml, re.S):
        if "<IsCrashed>true</IsCrashed>" in block:
            thread = _tag(block, "ThreadName") or "?"
            stack = _tag(block, "CallStack")
            frames = [ln.strip() for ln in stack.splitlines() if ln.strip()]

    when = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    msg = " ".join(_tag(xml, "ErrorMessage").split())
    print("  crash %s | thread=%s | sec_since_start=%s | %s" % (
        when.strftime("%m-%d %H:%M:%S"), thread,
        _tag(xml, "SecondsSinceStart"), msg[:110]))
    print("  stack_hash=%s | top=%s" % (
        _tag(xml, "PCallStackHash")[:8], " <- ".join(frames[:4])))


def main(argv):
    if not argv or argv[0] == "--recent":
        n = int(argv[1]) if len(argv) > 1 else 10
        dirs = sorted(glob.glob(os.path.join(CRASHDIR, "UE4CC-*")),
                      key=os.path.getmtime)[-n:]
        paths = [os.path.join(d, "CrashContext.runtime-xml") for d in dirs]
    else:
        paths = argv
    for p in paths:
        if os.path.exists(p):
            summarize(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
