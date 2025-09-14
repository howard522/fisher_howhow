# AI 輔助影片播放器 v2（YOLOv8）

> 動態倍速 · 互動時間軸 · 預取解碼 · 效能監控

## TL;DR
- 先用 YOLOv8 抽樣預標記整部片的人員出現區間（AI 區間）。
- 播放時：**有人→慢(可調，預設 1.5×)**，**無人→快(可調，預設 10×)** 自動切速。
- 時間軸支援滑鼠**拖曳建立/調整/搬移**自訂標記區間（使用者區間）。
- **AI 完成前不能播放**；完成後自動輸出 `<檔名>_ai_marks.csv`。
- 內建**多執行緒解碼 + 預取快取**，快進/跳播更順。
- 右下方顯示**CPU/GPU/FPS**，方便調參。

---

## 特色功能
- **AI 預標記（YOLOv8）**：以抽樣 + 批次推論找出 `person` 片段，膨脹/合併/過短濾除。
- **自動變速播放**：有人/無人兩組倍速可即時調整。
- **互動時間軸**：
  - AI 區間以**灰色**顯示；
  - 使用者標記以**綠色**顯示；
  - 拖曳空白建立新區間；拖曳邊緣**縮放**；拖曳中間**搬移**；
  - 動態刻度、網格、游標時間提示（藍線）。
- **效能監控**（可選）：顯示 CPU%、GPU%/MB（NVIDIA, NVML）、平滑 FPS。
- **預取快取**：背景執行緒專責 `VideoCapture`，主執行緒順暢讀幀。
- **可選即時偵測框**：播放中可疊人框（較耗效能）。
- **CSV 匯出**：
  - AI 完成後自動輸出 `*_ai_marks.csv`；
  - 使用者標記可手動匯出。

---

## 安裝
> 建議 Python 3.9+。

```bash
# 必要套件
pip install ultralytics opencv-python PySimpleGUI numpy

# 效能監控（可選）
pip install psutil pynvml

# 解碼更穩定（可選）
pip install av
```

> **註：** 首次使用 YOLOv8 會自動下載 `yolov8n.pt` 權重。若未自動安裝 PyTorch，請依作業系統/顯卡於官方站點安裝對應版本（支援 CPU 或 CUDA）。

---

## 執行
```bash
python ai_assisted_player.py
```
步驟：
1. **開啟影片**：選擇檔案後，程式會先進行 AI 預標記（此時不可播放）。
2. **AI 完成**：
   - 自動輸出 `同名_ai_marks.csv`（欄位：`start,end`，單位秒，至毫秒）。
   - 解鎖播放控制並開始播放。
3. **播放控制**：
   - 有人× / 無人×：即時調整兩種倍速。
   - 偵測框：勾選後在畫面上畫出 `person` 方框（較慢）。
   - 倒退/快進 5 秒、暫停/播放、拖曳時間軸或拉動滑桿。
4. **時間軸互動**：
   - 空白處拖曳建立新區間（橘色預覽，放開成綠色）；
   - 靠左/右邊緣拖曳可縮放；
   - 區段內拖曳可整段搬移；
   - 游標提示顯示目前時間與是否落在 AI 區間。
5. **匯出標記**：
   - 按「匯出標記」輸出使用者標記 CSV（`start,end`）。

---

## CSV 格式
- **AI CSV（自動產生）**：`<影片名>_ai_marks.csv`
- **使用者 CSV（手動匯出）**：自選路徑
- 共同欄位：
  - `start`：區間起點（秒，小數 3 位）
  - `end`：區間終點（秒，小數 3 位）

---

## 效能與精度調校
> 這些參數可在程式碼中調整（`rough_human_intervals_yolo()` 等）。

- **抽樣/批次：**
  - `frame_skip`（預設 15）：抽樣頻率，數字越大越快但可能漏短暫人影。
  - `batch_size`（預設 16）：批次推論大小；視記憶體調整。
- **模型/輸入尺寸：**
  - `model_name`（`yolov8n.pt` 預設）：可換成 s/m/l/x 平衡精度/速度。
  - `imgsz`（640 預設）、`resize_limit`（960 預設）。
- **門檻與後處理：**
  - `conf_thres`、`iou_thres`：偵測門檻。
  - `pad_seconds`、`merge_gap`、`min_on_duration`：區間膨脹/合併/過短濾除。
- **播放時效能：**
  - 關閉「顯示偵測框」可顯著加速。
  - NVIDIA + 正確安裝 CUDA 版 PyTorch → `cuda` 自動啟用。
  - 預取快取大小：建立 `VideoPrefetcher` 時的 `buffer_size`。

---

## 介面說明
- **主畫面**：960×540 影片預覽（比例固定縮放）。
- **時間軸**：
  - 背景網格 + 時間刻度文字；
  - 灰色：AI 區間；綠色：使用者區間；橘色：拖曳中的暫存區間；藍線：游標/播放位置。
- **控制列**：開啟影片、倒退、暫停/播放、快進、標記開始/結束、匯出標記、顯示偵測框、有人× / 無人× 倍速、重設速度。
- **狀態列**：位置滑桿、當前倍速、CPU/GPU/FPS、游標時間與是否在 AI 區間。

---

## 內部架構（概要）
- **YOLOv8 預標記**：抽樣抓幀 → 批次推論 → 只留 `person` → 依時間連段 → 膨脹/合併/濾短。
- **自動變速**：播放時依「目前時間是否落在 AI 區間」選擇倍速（近似以跳幀實作）。
- **時間軸互動**：命中測試（邊/內部）→ 拖曳建立/縮放/搬移 → `merge_intervals()` 去重疊。
- **預取解碼**：後台執行緒獨占 `VideoCapture`，接受 seek 請求並預放幀至佇列，前台快速讀取。
- **效能監控**：psutil 讀 CPU；NVML 讀 GPU 利用率/記憶體；FPS 以指數平滑。

---

## 疑難排解
- **ImportError: ultralytics / torch**：請先安裝；PyTorch 請依作業系統/顯卡於官方指引安裝 CPU 或 CUDA 版本。
- **Cannot open video**：檔案損壞或缺編碼器；嘗試安裝 `av` 套件；或轉檔成 mp4(h264/aac)。
- **GPU 顯示為 `--`**：未安裝 `pynvml` 或未偵測到 NVIDIA GPU/驅動。
- **預標記耗時**：調大 `frame_skip`、調小 `imgsz`、使用更小的 YOLO 權重（如 `yolov8n.pt`）、或關閉即時框。
- **偵測漏判/碎片化**：
  - 降低 `conf_thres`（更寬鬆）；
  - 增加 `pad_seconds` 與 `merge_gap`；
  - 降低 `frame_skip` 或提高 `sample_fps`。

---

## 相容性
- 常見容器：mp4/avi/mov/mkv（取決於系統解碼）。
- Windows / macOS / Linux；GPU 加速需相容 CUDA 與驅動。

---

## 變更摘要
- **v2 基礎**：YOLOv8 預標記、動態倍速、AI 完成前禁播、自動輸出 AI CSV。
- **此版新增**：互動時間軸、CPU/GPU/FPS 顯示、多執行緒預取、時間軸視覺強化。

---

## Credits
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PySimpleGUI](https://www.pysimplegui.org/)
- [PyAV](https://github.com/PyAV-Org/PyAV)（可選）
- [psutil](https://github.com/giampaolo/psutil)、[pynvml](https://github.com/gpuopenanalytics/pynvml)（可選）

---

## License
MIT（可依需求替換）

