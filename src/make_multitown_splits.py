"""Generate leave-one-city-out multi-town splits for LSS training.

Test city = Town01 (held out entirely; uses its own test.txt = all 300 frames).
Train cities = Town10, Town02, Town04, Town05. Each contributes a contiguous
train block + a gap + a small val block (val is for best-epoch selection only;
the real metric is the held-out Town01). Town10 is capped to a subset so it
does not dominate the four training cities.

Writes mt_train.txt / mt_val.txt into each training-city dir (distinct names so
existing train/val/test.txt from earlier experiments are untouched).
"""
import os

BASE = "E:/dualcam_project/data"

# (dir, first_frame, n_use, n_train, gap, n_val)
#
# Five training towns, 4,080 frames, all depth-equipped. Every frame is
# accounted for -- train + gap + val exactly equals what was collected, so
# nothing sits idle.
#
# Town10/04/07 collected the full 1,000. Town02 and Town05 top out around 540:
# their camera streaming dies there however many times we retry, and the big
# maps (Town03/Town06) cannot even spawn a camera on this machine. Scene mix
# still spans city / motorway-loop / rural / two small towns.
TRAIN_CITIES = [
    ("carla_dualcam_t10_1k", 0, 1000, 850, 50, 100),   # city centre
    ("carla_dualcam_t04_1k", 0, 1000, 850, 50, 100),   # motorway loop
    ("carla_dualcam_t07_1k", 0, 1000, 850, 50, 100),   # rural town
    ("carla_dualcam_t02_1k", 0,  541, 455, 26,  60),   # small town
    ("carla_dualcam_t05_1k", 0,  539, 453, 26,  60),   # small city
]


def main():
    total_tr = total_va = 0
    for d, f0, n_use, n_tr, gap, n_va in TRAIN_CITIES:
        root = os.path.join(BASE, d)
        files = sorted(x for x in os.listdir(os.path.join(root, "front_rgb")) if x.endswith(".png"))
        files = files[f0:f0 + n_use]
        assert len(files) == n_use, f"{d}: wanted {n_use} got {len(files)}"
        assert n_tr + gap + n_va <= n_use, f"{d}: {n_tr}+{gap}+{n_va} > {n_use}"

        train = files[:n_tr]
        val = files[n_tr + gap: n_tr + gap + n_va]

        with open(os.path.join(root, "mt_train.txt"), "w") as fh:
            fh.write("".join(x + "\n" for x in train))
        with open(os.path.join(root, "mt_val.txt"), "w") as fh:
            fh.write("".join(x + "\n" for x in val))

        total_tr += len(train); total_va += len(val)
        print(f"{d:26s} train {len(train):4d} [{train[0]}..{train[-1]}]  "
              f"gap {gap}  val {len(val):3d} [{val[0]}..{val[-1]}]")

    print(f"\ntotal: train {total_tr}  val {total_va}  (test = Town01 held out, 300)")


if __name__ == "__main__":
    main()
