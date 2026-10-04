"""Verify the augmented (image, label) pairs stay geometrically consistent.

Trick: the BEV GT is a deterministic function of the world, and the ego is always
at the BEV centre. Rather than eyeball it, we check the two symmetries hold on
real GT arrays, and that class histograms are exactly preserved.
"""
import os, cv2, numpy as np, torch
from dualcam_bev_dataset import DualCamBEVDataset

D = "E:/dualcam_project/data/carla_dualcam_gt_lane"

# 1. augmentation must be off by default (val/test must never be augmented)
ds = DualCamBEVDataset(D, split="test.txt")
assert ds.augment is False
a, _ = ds[0]; b, _ = ds[0]
assert torch.equal(a, b), "non-augmented dataset must be deterministic"
print("ok  augment defaults to off, and is deterministic")

# 2. class histogram is invariant under the geometric augs (pure pixel permutation)
raw = DualCamBEVDataset(D, split="train.txt")
aug = DualCamBEVDataset(D, split="train.txt", augment=True)
h0 = np.bincount(raw[0][1].numpy().ravel(), minlength=6)
for _ in range(12):
    h = np.bincount(aug[0][1].numpy().ravel(), minlength=6)
    assert (h == h0).all(), f"histogram changed: {h} vs {h0}"
print(f"ok  class histogram invariant over 12 draws: {h0.tolist()}")

# 3. the label really is being transformed (not silently left alone)
labels = set()
for _ in range(40):
    labels.add(aug[0][1].numpy().tobytes())
assert len(labels) >= 3, f"expected ~4 distinct label orientations, got {len(labels)}"
print(f"ok  label takes {len(labels)} distinct orientations over 40 draws (expect up to 4)")

# 4. every augmented label is one of the 4 group elements applied to the original
g = raw[0][1].numpy()
group = {
    "identity": g,
    "mirror": g[:, ::-1],
    "swap": g[::-1, ::-1],
    "swap+mirror": g[::-1, :],
}
seen = {}
for _ in range(60):
    la = aug[0][1].numpy()
    hit = [k for k, v in group.items() if np.array_equal(la, v)]
    assert hit, "augmented label is not in the rig symmetry group!"
    seen[hit[0]] = seen.get(hit[0], 0) + 1
print(f"ok  all 60 augmented labels lie in the symmetry group: {seen}")

# 5. image/label correspondence: under 'swap' the front image must equal the
#    original rear image (up to photometric jitter, so compare structure not values)
raw_imgs, _ = raw[0]
front_raw, rear_raw = raw_imgs[0], raw_imgs[1]
n_swap = n_noswap = 0
for _ in range(40):
    ai, al = aug[0]
    is_swapped = np.array_equal(al.numpy(), group["swap"]) or np.array_equal(al.numpy(), group["swap+mirror"])
    is_mirrored = np.array_equal(al.numpy(), group["mirror"]) or np.array_equal(al.numpy(), group["swap+mirror"])
    src = rear_raw if is_swapped else front_raw          # which raw image became 'front'
    cand = torch.flip(src, dims=[2]) if is_mirrored else src
    # photometric jitter is monotone-ish per channel; compare edge maps instead of pixels
    e1 = np.abs(np.diff(ai[0].mean(0).numpy(), axis=1)) > 0.05
    e2 = np.abs(np.diff(cand.mean(0).numpy(), axis=1)) > 0.05
    agree = (e1 == e2).mean()
    assert agree > 0.90, f"front image does not match the label's orientation (edge agreement {agree:.2f})"
    n_swap += is_swapped; n_noswap += not is_swapped
print(f"ok  front image orientation always matches the label ({n_swap} swapped / {n_noswap} not)")

print("\naugmentation is geometrically consistent")
