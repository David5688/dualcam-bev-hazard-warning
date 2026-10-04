# 即時 DEMO 指令

最省事的方式:**雙擊桌面「啟動DEMO.bat」**,它會自動關舊 server、開新的、等到 ready、再啟動 demo。
以下是手動版,適合要改參數的時候。

---

## 每次都要先做:啟動 CARLA server

demo 收尾時會把 server 一起帶走(CARLA 0.9.16 固有行為),所以**每跑一次就要重開一次**。

```bat
taskkill /F /IM CarlaUE4-Win64-Shipping.exe
taskkill /F /IM CarlaUE4.exe
start "" "G:\CARLA_0.9.16\CarlaUE4.exe" -quality-level=Low -carla-rpc-port=2000
cd /d E:\dualcam_project\src
E:\miniforge3\envs\lss\python.exe -u wait_carla_ready.py --host 127.0.0.1 --port 2000 --streak 3 --timeout 240
```

等到印出 `READY` 再跑下面的指令。**不等 ready 就開 demo 會連到半啟動的 server 而失敗。**

---

## 主力指令(桌面批次檔跑的就是這一行)

```bat
cd /d E:\dualcam_project\src
E:\miniforge3\envs\lss\python.exe live_demo.py --town Town10HD --vehicles 12 --walkers 20 --ignore_walkers --peds --recon --recon_source depth
```

畫面三欄:前後相機 / BEV 預測 + 危險警示 / 三維重建。約 10 FPS。

---

## 操作按鍵

**按鍵送到影像視窗,先用滑鼠點一下那個視窗。**

| 鍵 | 功能 |
|---|---|
| `q` / `Esc` | 離開(**用這個,不要點視窗右上角的 X**) |
| `m` | 靜音 / 取消靜音 |
| `g` | 疊上即時 ground truth 面板(會變慢) |
| `h` | 危險警示疊圖開關 |
| `p` | 暫停 |
| `Tab` | 自動駕駛 / 手動切換 |
| `W A S D` | 手動駕駛 |
| `R` | 倒車(會觸發後方警示) |

---

## 常用變化

**自己開(一開始就手動)**

```bat
E:\miniforge3\envs\lss\python.exe live_demo.py --town Town10HD --vehicles 12 --walkers 20 --peds --recon --recon_source depth --manual
```

**錄影**(輸出是 mp4v,要轉檔見最後一節)

```bat
E:\miniforge3\envs\lss\python.exe live_demo.py --town Town10HD --vehicles 12 --walkers 20 --ignore_walkers --peds --recon --recon_source depth --max_frames 650 --record E:\dualcam_project\results\demo.mp4
```

`--max_frames 650` ≈ 65 秒。不加就一直跑到你按 q。

**純 RGB 重建(對照組,證明不用深度相機也做得到)**

```bat
E:\miniforge3\envs\lss\python.exe live_demo.py --town Town10HD --vehicles 12 --walkers 20 --ignore_walkers --peds --recon --recon_source model
```

**換地圖**:`--town Town01` / `Town02` / `Town04` / `Town05` / `Town10HD`
(Town03 的 camera streaming 在這台機器必崩,別用)

**跑順一點**:拿掉 `--recon`(重建面板最貴)或 `--peds`(行人偵測 Faster R-CNN)

**車一直卡在路口**:加 `--ignore_lights_all`(所有車都闖紅燈,適合短片)

---

## 幾個容易踩到的點

- **`--ignore_walkers` 一定要加**,除非你就是要看車停下來。CARLA 的 traffic manager 會無限期禮讓行人,結果是「有行人的 demo = 車停著不動」。加了之後行人照樣出現、照樣被偵測、照樣發警告,只是車不會停 —— 對一個「只警示不介入」的系統來說這樣才誠實。
- **不要在 demo 跑的時候執行 `clean_world.py` 或 `eval_ttc.py`**,它們開頭會清空世界,會把你正在跑的 demo 殺掉。
- 想在旁邊看狀態用 `watch_ego.py`,那支是唯讀的,安全。
- demo 結束時 console 會噴一串 `failed to destroy actor` / `streaming client: connection failed` —— **那是正常的**,資料和錄影都已經寫完了。

---

## 錄影轉檔(cv2 寫不出 H.264)

```bat
"E:\miniforge3\envs\lss\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe" -y -i E:\dualcam_project\results\demo.mp4 -c:v libx264 -pix_fmt yuv420p -crf 23 E:\dualcam_project\results\demo_h264.mp4
```

不轉的話 Windows 內建播放器和 PowerPoint 會卡頓。
