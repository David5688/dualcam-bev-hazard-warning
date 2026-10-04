"""Cheap pre-training gate for fov120: verify LSS builds the 120-deg frustum
correctly from calibration.json, and that depth loads. Run before the long train.
"""
import os, sys, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS
from train_dualcam_bev_fpn import DualCamBEVDataset

DS = "E:/dualcam_project/data/carla_dualcam_t10_fov120"
CALIB = os.path.join(DS, "calibration.json")

print("=== 1. LSS 從 fov120 calibration 建幾何 ===")
m = DualCamBEVLSS.from_calib_file(CALIB, num_classes=6, bev_h=240, bev_w=240,
                                  img_h=224, img_w=400)
print(f"pooling 覆蓋: {m.n_valid}/{m.n_points} = {m.n_valid/m.n_points:.1%}  (90°約31%,120°應更高)")
assert m.n_valid > 0, "沒有任何 frustum 點落在 BEV 格內!"

print("\n=== 2. depth 資料載入 ===")
ds = DualCamBEVDataset(DS, img_size=(224, 400), split="train.txt", with_depth=True)
sample = ds[0]
print(f"train.txt 樣本數: {len(ds)}")
print(f"sample 回傳 {len(sample)} 項 (含 depth 應為 3): "
      + ", ".join(f"{tuple(x.shape)}" for x in sample))
assert len(sample) == 3, "with_depth=True 卻沒回傳 depth!"
print(f"label 類別值: {sorted(set(sample[1].unique().tolist()))}  (應含 0-5)")
print("\nGATE PASSED ✅")
