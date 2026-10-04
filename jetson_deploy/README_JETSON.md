# Jetson Orin Nano 部署包

雙鏡頭危險警示系統的**推論**部分,搬到 Jetson Orin Nano 用。整包 ~512MB,直接複製過去。

---

## 這包裡有什麼

```
jetson_deploy/
├── code/            12 支推論程式(完整相依,不含訓練/CARLA 蒐集)
├── checkpoints/     dualcam_bev_lss_retrain_lane_best.pt   (車輛 BEV 主模型,建議用)
│                    dualcam_bev_lss_9set_depth_bv3_best.pt (備援,跨城較保守)
├── weights/         fasterrcnn_resnet50_fpn_coco-258fb6c6.pth  (行人模型,160MB)
├── test_data/       150 幀測試資料(前後 RGB+深度+GT+空拍+pose+calibration.json)
└── README_JETSON.md
```
> 主模型用**重訓版**(Town10/Town01 皆優於 9set:車輛 recall 0.82、警示 F1 0.63)。

---

## ⚠️ 第一件事:PyTorch 要用 Jetson 版(不能用桌機的)

桌機的 `torch 2.11.0+cu128` 是 x86,**Jetson(ARM)完全不能用**。Jetson 要裝**對應 JetPack 版本的 PyTorch**:

```bash
# 先看 JetPack 版本
sudo apt-cache show nvidia-jetpack | grep Version

# 裝 NVIDIA 提供的 Jetson PyTorch wheel(依你的 JetPack 版本挑)
# 官方說明:https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
# 或用 pip index:
#   pip install --no-cache torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126  (JetPack 6 範例)
```

其他套件(Jetson 上 pip 裝):
```bash
pip install "numpy<2" opencv-python matplotlib pillow imageio-ffmpeg
# ffmpeg:Jetson(ARM)上 imageio-ffmpeg 可能沒內建 binary → 裝系統版當退路:
sudo apt-get install -y ffmpeg
```
(程式會先找 imageio-ffmpeg 的 binary,找不到自動用系統 `ffmpeg`。)

---

## 行人模型權重放哪

torchvision 會去 `~/.cache/torch/hub/checkpoints/` 找 Faster R-CNN 權重。把 weights 裡那個檔放過去:
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cp weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ~/.cache/torch/hub/checkpoints/
```
(這樣 Jetson 沒網路也能載入。)

---

## 怎麼跑測試(先驗證 pipeline 在 Jetson 跑得起來)

```bash
cd jetson_deploy/code
# 車+人整合危險警示,跑 test_data 的 150 幀(輸出影片)
python hazard_full.py --data ../test_data --model ../checkpoints/dualcam_bev_lss_retrain_lane_best.pt --start 0 --count 150
```
能跑出 `HAZARD_FULL...mp4`(4 欄:相機/空拍/偵測/GT)就代表**推論管線在 Jetson 上通了**。

單獨測各模組:
```bash
python hazard_bev.py --data ../test_data --source pred --start 0 --count 50   # BEV+警示
python depth_to_pointcloud.py --data ../test_data --frame 000050              # 三維重建
```

---

## ⚠️ 務實的預期與注意

1. **先求「跑得起來」,不要求即時**。Jetson Orin Nano 小,兩個模型(BEV + Faster R-CNN)每幀跑會慢。先確認離線跑通,即時是之後的事(要 TensorRT、YOLO 取代 FRCNN、降解析度)。
2. **calibration.json 是 CARLA 的**。用 test_data(CARLA 錄的)測沒問題;**接真實相機時,內外參要重新標定**(棋盤格),不能用這個。
3. **深度**:test_data 有 CARLA GT 深度。真實環境沒有 GT 深度 → 要嘛用 ZED 的深度、要嘛用模型自己預測的深度(見主專案「下一步」)。
4. Linux 路徑無中文問題(桌機 cv2 中文路徑的坑在 Jetson 不會遇到)。
5. 用的是**已驗證的 9set 模型**(不是還在訓練的 retrain 版)。

---

## 記憶體不夠時

Orin Nano 8GB 同時載兩個模型可能吃緊:
- 降輸入解析度、`--score 0.7`(行人偵測較少候選)
- 或分兩階段:先只跑 BEV(車),再單獨跑行人偵測

---

## Review 問題逐條回應(2026-07-25)

| # | Review 指出的問題 | 狀態 |
|---|---|---|
| 1 | 是 CARLA 離線 demo,不是即時 ROS2 | ✅ **正確,by design**。本包定位就是離線驗證推論管線;即時/ROS2 是之後的工程。 |
| 2 | 缺兩個 Python module,主命令無法啟動 | ✅ **已修**。缺的是 `dualcam_bev_fpn_model.py`、`dualcam_bev_fpn_resnet.py`(render_prof_style 會 import),已補齊為 12 支完整相依,主命令實測可跑。 |
| 3 | FFMPEG 路徑寫死 Windows | ✅ **已修**。改用 `imageio_ffmpeg.get_ffmpeg_exe()`(跨平台自動選 binary),找不到再退回 PATH 上的 `ffmpeg`。 |
| 4 | checkpoint 存了 CARLA 幾何 buffer,換真機標定可能失敗 | ⚠️ **真實問題,已提供做法**(見下「接真實相機」)。CARLA demo 不受影響(幾何一致)。 |
| 5 | checkpoint 不能安全用 `weights_only=True` | ⚠️ **已知**:checkpoint 含 numpy 純量,載入用 `torch.load(..., weights_only=False)`(程式已這樣做)。要更安全可另存純 state_dict。 |
| 6 | 盲區續追 / TTC / 不確定度 / 位置外推 尚未實作 | 部分:**TTC ✅ 已實作**(hazard_warning)、**不確定度 ✅ 已實作**(uncertainty_warning.py,已含在包內);**盲區續追、位置外推 ❌ 未做**(future work,需盲區記憶,見主專案)。 |
| 7 | CARLA depth 解不了 D455 的 Z16 | ✅ **已加解碼器** `d455_z16_to_metres()`(在 depth_to_pointcloud.py);CARLA 用 `depth_png_to_metres`,D455 用新的那個。 |
| 8 | 後鏡頭行人深度來源仍是最重要硬體決策 | ⚠️ **正確,列為硬體決策**(見主專案 `硬體與真實驗證_可行性.md`)。 |

## 接真實相機時(問題 #4 的正確做法)

模型的 BEV 幾何(`valid_idx`/`bev_idx`)是**用 calibration.json 在載入時算好的**,而且會被存進 checkpoint。
直接 `load_state_dict(strict=True)` 會用 **CARLA 幾何覆蓋**你的真機幾何。正確載法:

```python
import torch
from dualcam_bev_lss import DualCamBEVLSS
m = DualCamBEVLSS.from_calib_file("真機的_calibration.json", num_classes=6,   # 用真機內外參算幾何
                                  img_h=224, img_w=400, pretrained=False)
sd = torch.load("checkpoints/dualcam_bev_lss_retrain_lane_best.pt", weights_only=False)
sd = sd.get("model_state_dict", sd)
sd = {k: v for k, v in sd.items() if k not in ("valid_idx", "bev_idx")}   # 保留真機幾何
m.load_state_dict(sd, strict=False)   # 只載學到的權重,不覆蓋幾何
```
> ⚠️ 前提:真機要先做**棋盤格標定**產出正確的 calibration.json(內參 K + 前後外參)。CARLA 的不能用。

## D455 深度用法(問題 #7)

```python
import cv2
from depth_to_pointcloud import d455_z16_to_metres
z16 = cv2.imread("d455_depth.png", cv2.IMREAD_UNCHANGED)   # 保留 16-bit
depth_m = d455_z16_to_metres(z16, depth_scale=0.001)        # depth_scale 用相機實際查到的值
```
