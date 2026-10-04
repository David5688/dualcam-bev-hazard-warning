# A Third Eye for Old Cars: Dual-Camera 3D Reconstruction and Blind-Spot Collision Warning System
# 舊車也有第三隻眼 雙鏡頭三維重建與盲區碰撞預警系統

Retrofit environment perception for existing vehicles using only **two depth cameras (front + rear)**:
BEV semantic segmentation → pedestrian detection with depth localisation → 3D street reconstruction →
multi-object tracking with explicit *unknown* regions → TTC / CPA risk → green / yellow / red warning.

只用車前、車後各一顆深度相機，讓既有車輛知道周圍有誰、往哪走、多久會撞上。
系統將影像提升為 BEV 語意網格，結合行人偵測與深度定位、深度點雲三維街景重建、
多目標追蹤與盲區三態（直接觀測／短時推估／未知），以 TTC 與 CPA 推論風險並輸出三級警示。

## Results (held-out test city, 2,700 frames)

| Metric | Value |
|---|---|
| mIoU (6 classes) | 0.5398 |
| Vehicle IoU | 0.4996 |
| Vehicle IoU, camera-visible region | 0.6177 |
| Vehicle recall, camera-visible region | 0.8163 |
| On-vehicle runtime (embedded AI platform, ROS 2) | 10.92 FPS, p95 latency 127.8 ms |

Per-epoch metrics for every experiment are in `metrics/` (CSV). The final BEV model
corresponds to epoch 15 of `metrics/metrics_lss_5city_1k_train.csv`.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Data collection (CARLA), BEV model, training / evaluation, tracking, hazard logic, live demo |
| `metrics/` | Evaluation CSVs (per-class IoU, visible-region comparison, TTC events) |
| `jetson_deploy/` | On-vehicle ROS 2 deployment code and notes |
| `docs/` | Evaluation protocol, real-car guide, demo commands, programme description |

Key entry points: `src/dualcam_bev_lss.py` (model), `src/eval_full.py` and
`src/eval_visible_region.py` (evaluation), `src/hazard_warning.py` (risk levels),
`src/live_demo.py` (integrated live demo with the 3D reconstruction panel).

## Model weights

The final BEV checkpoint (`dualcam_bev_lss_5city1k_depth_best.pt`, ResNet-18 backbone,
18.47M parameters) is published as a GitHub Release asset of this repository.

## Data

Training data (11,429 frames from five CARLA towns, plus one validation town and one
held-out test town of 2,700 frames each) is generated with `src/carla_dualcam_collect_gt.py`
and the `collect_v4_*.ps1` scripts; see `docs/` for the collection protocol.
The dataset itself is not redistributed because of its size.

## License

Code and trained weights: MIT (see `LICENSE`). Third-party components: see `NOTICE`.
