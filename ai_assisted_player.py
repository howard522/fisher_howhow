#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ai_assisted_player.py
"""
AI 輔助影片播放器 v2（YOLOv8 強化版）—動態倍速 & 預先輸出CSV + 互動時間軸/效能監控/預取解碼

新增於此版：
1) 時間軸支援滑鼠拖曳建立/調整「使用者標記區間」。
2) 顯示即時 CPU/GPU 使用率與播放 FPS（平滑平均）。
3) 多執行緒解碼 + 預取快取，降低快進/跳播卡頓。
4) 強化時間軸顯示（動態刻度、網格、游標時間提示、AI/使用者標記分色）。

安裝依賴：
  pip install ultralytics opencv-python PySimpleGUI numpy
（建議）pip install psutil pynvml  # 若需效能監控（GPU 需 NVIDIA 驅動）

使用：
  python ai_assisted_player.py
  → 按「開啟影片」→ 程式先做 YOLOv8 預標記（期間不可播放）→ 完成後自動輸出 AI CSV，並可互動編修時間軸
"""
from __future__ import annotations
import os
import csv
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import PySimpleGUI as sg

# 可選：PyAV（較穩定解碼），沒有就退回 OpenCV
try:
    import av
    _HAS_PYAV = True
except Exception:
    av = None  # type: ignore
    _HAS_PYAV = False

# === YOLOv8 ===
try:
    from ultralytics import YOLO
    _ULTRA_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore
    _ULTRA_AVAILABLE = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

# 效能監控（可選）
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# ============================
# 時間區間結構 & 工具
# ============================
@dataclass
class Interval:
    start: float  # 秒
    end: float    # 秒

    def __post_init__(self):
        if self.end < self.start:
            self.start, self.end = self.end, self.start

    def expand(self, pad: float) -> 'Interval':
        return Interval(max(self.start - pad, 0.0), self.end + pad)

    def length(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: 'Interval', merge_gap: float = 0.0) -> bool:
        return not (self.end + merge_gap < other.start or other.end + merge_gap < self.start)

    def merge(self, other: 'Interval') -> 'Interval':
        return Interval(min(self.start, other.start), max(self.end, other.end))


def merge_intervals(intervals: List[Interval], merge_gap: float = 0.4) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x.start)
    merged = [intervals[0]]
    for cur in intervals[1:]:
        last = merged[-1]
        if last.overlaps(cur, merge_gap=merge_gap):
            merged[-1] = last.merge(cur)
        else:
            merged.append(cur)
    return merged


def in_any_interval(t: float, intervals: List[Interval]) -> bool:
    if not intervals:
        return False
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        iv = intervals[mid]
        if iv.start <= t <= iv.end:
            return True
        if t < iv.start:
            hi = mid - 1
        else:
            lo = mid + 1
    return False

# ============================
# YOLOv8 預先標記
# ============================

def _select_device_str() -> str:
    if _HAS_TORCH and torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_yolo(model_name: str = 'yolov8n.pt'):
    if not _ULTRA_AVAILABLE:
        raise RuntimeError('未安裝 ultralytics，請先執行：pip install ultralytics')
    model = YOLO(model_name)
    try:  # 暖機降低首幀延遲
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy, imgsz=640, device=_select_device_str(), verbose=False)
    except Exception:
        pass
    return model


def _batch_has_person(model, frames_bgr: list, conf_thres: float, iou_thres: float, imgsz: int, device: str) -> list:
    if not frames_bgr:
        return []
    try:
        results = model.predict(frames_bgr, imgsz=imgsz, conf=conf_thres, iou=iou_thres, device=device, verbose=False)
    except Exception:
        results = model.predict(frames_bgr, imgsz=imgsz, conf=conf_thres, iou=iou_thres, device='cpu', verbose=False)
    flags = []
    for r in results:
        ok = False
        if getattr(r, 'boxes', None) is not None:
            cls = r.boxes.cls.detach().cpu().numpy() if hasattr(r.boxes, 'cls') else []
            conf = r.boxes.conf.detach().cpu().numpy() if hasattr(r.boxes, 'conf') else []
            for c, p in zip(cls, conf):
                if int(c) == 0 and float(p) >= conf_thres:
                    ok = True
                    break
        flags.append(ok)
    return flags


def _build_intervals_from_flags(
    results_all: list[tuple[int, bool]],
    fps: float,
    pad_seconds: float,
    merge_gap: float,
    min_on_duration: float,
) -> list[Interval]:
    time_segments: list[Interval] = []
    person_on = False
    start_time = None

    for frame_idx, detected in sorted(results_all, key=lambda x: x[0]):
        t = frame_idx / max(fps, 1.0)
        if detected and not person_on:
            start_time = t
            person_on = True
        elif (not detected) and person_on:
            end_t = t
            if start_time is not None:
                time_segments.append(Interval(start_time, end_t))
            person_on = False
            start_time = None

    if person_on and start_time is not None and results_all:
        last_idx = results_all[-1][0]
        end_t = last_idx / max(fps, 1.0)
        time_segments.append(Interval(start_time, end_t))

    time_segments = [iv for iv in time_segments if iv.length() >= max(0.0, min_on_duration)]
    expanded = [iv.expand(pad_seconds) for iv in time_segments]
    merged = merge_intervals(expanded, merge_gap=merge_gap)
    return merged


def rough_human_intervals_yolo(
    video_path: str,
    sample_fps: float = 3.0,
    pad_seconds: float = 0.5,
    merge_gap: float = 0.6,
    model_name: str = 'yolov8n.pt',
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    imgsz: int = 640,
    resize_limit: int = 960,
    return_model: bool = True,
    frame_skip: int | None = 15,
    batch_size: int = 16,
    min_on_duration: float = 0.00,
) -> Tuple[List[Interval], float, int, Optional[object]]:
    fps = 0.0
    total_frames = 0
    duration = 0.0

    frames: list[tuple[int, np.ndarray]] = []

    if _HAS_PYAV:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        if video_stream.average_rate is not None and float(video_stream.average_rate) > 0:
            fps = float(video_stream.average_rate)
        else:
            fps = 30.0
        frame_index = 0
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame_skip is None:
                    frame_skip_est = max(1, int(round(fps / max(sample_fps, 1e-6))))
                    use_this = (frame_index % frame_skip_est == 0)
                else:
                    use_this = (frame_index % max(1, frame_skip) == 0)
                if use_this:
                    img = frame.to_ndarray(format='bgr24')
                    h, w = img.shape[:2]
                    if max(h, w) > resize_limit:
                        scale = resize_limit / float(max(h, w))
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    frames.append((frame_index, img))
                frame_index += 1
        total_frames = frame_index
        duration = total_frames / max(fps, 1.0)
        container.close()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / max(fps, 1.0)
        if frame_skip is None:
            step = max(1, int(round(fps / max(sample_fps, 1e-6))))
        else:
            step = max(1, int(frame_skip))
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > resize_limit:
                scale = resize_limit / float(max(h, w))
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frames.append((frame_idx, frame))
            frame_idx += step
        cap.release()

    model = load_yolo(model_name)
    device = _select_device_str()

    results_all: list[tuple[int, bool]] = []
    for i in range(0, len(frames), max(1, batch_size)):
        chunk = frames[i:i + max(1, batch_size)]
        idxs = [idx for idx, _ in chunk]
        imgs = [img for _, img in chunk]
        flags = _batch_has_person(model, imgs, conf_thres, iou_thres, imgsz, device)
        results_all.extend(list(zip(idxs, flags)))

    intervals = _build_intervals_from_flags(
        results_all=results_all,
        fps=fps,
        pad_seconds=pad_seconds,
        merge_gap=merge_gap,
        min_on_duration=min_on_duration,
    )

    return intervals, duration, int(round(fps)), (model if return_model else None)

# ============================
# 視覺化：時間軸圖 + 互動
# ============================

def _fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h:d}:{m:02d}:{int(s):02d}"
    return f"{int(m):02d}:{int(s):02d}"


def _dynamic_tick_step(duration: float) -> float:
    candidates = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
    target_ticks = 10
    for step in candidates:
        if duration / step <= target_ticks:
            return step
    return candidates[-1]


def draw_timeline(graph: sg.Graph, width: int, height: int,
                  ai_intervals: List[Interval], user_intervals: List[Interval], duration: float,
                  cursor_t: Optional[float] = None,
                  temp_interval: Optional[Interval] = None):
    graph.erase()
    if duration <= 0:
        return

    # 背景
    graph.draw_rectangle((0, 0), (width, height), fill_color="#111827", line_color=None)

    # 網格 & 刻度
    step = _dynamic_tick_step(duration)
    t = 0.0
    while t <= duration + 1e-6:
        x = int(width * (t / duration))
        graph.draw_line((x, 0), (x, height), color="#1f2937")
        label = _fmt_time(t)
        graph.draw_text(label, (x + 2, height - 2), color="#9ca3af", font=(None, 9),
                        text_location=sg.TEXT_LOCATION_TOP_LEFT)
        t += step

    # AI 區間
    for iv in ai_intervals:
        x0 = int(width * (iv.start / duration))
        x1 = int(width * (iv.end / duration))
        graph.draw_rectangle((x0, 0), (max(x1, x0 + 1), height), fill_color="#4b5563", line_color=None)

    # 使用者區間
    for iv in user_intervals:
        x0 = int(width * (iv.start / duration))
        x1 = int(width * (iv.end / duration))
        graph.draw_rectangle((x0, 0), (max(x1, x0 + 1), height), fill_color="#10b981", line_color=None)
        # 邊界把手
        graph.draw_line((x0, 0), (x0, height), color="#34d399")
        graph.draw_line((x1, 0), (x1, height), color="#34d399")

    # 臨時拖曳中的區間（建立中）
    if temp_interval is not None:
        x0 = int(width * (temp_interval.start / duration))
        x1 = int(width * (temp_interval.end / duration))
        graph.draw_rectangle((x0, 0), (max(x1, x0 + 1), height), fill_color="#f59e0b", line_color="#fbbf24")

    # 游標位置
    if cursor_t is not None:
        x_pos = int(width * (cursor_t / max(duration, 1e-6)))
        graph.draw_line((x_pos, 0), (x_pos, height), color="#93c5fd")

# ============================
# 多執行緒解碼 + 預取
# ============================

class VideoPrefetcher:
    """單一後台執行緒獨占 VideoCapture，負責 seek 與連續讀幀並預放入佇列。
    主執行緒透過 request_index() 通知欲播放位置，再以 read_next() 取回幀。
    """
    def __init__(self, path: str, buffer_size: int = 120):
        self.path = path
        self.buffer_size = buffer_size
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError('無法開啟影片（Prefetcher）: ' + path)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._q: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=buffer_size)
        self._seek_lock = threading.Lock()
        self._seek_idx: Optional[int] = 0  # 初始從 0 開始
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _drain_queue(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _run(self):
        cur_idx = 0
        while not self._stop:
            with self._seek_lock:
                seek_to = self._seek_idx
                self._seek_idx = None
            if seek_to is not None:
                cur_idx = max(0, min(self.total_frames - 1, int(seek_to)))
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, cur_idx)
                self._drain_queue()

            if self._q.full():
                time.sleep(0.002)
                continue

            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            cur_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            try:
                self._q.put_nowait((cur_idx, frame))
            except queue.Full:
                pass

    def request_index(self, frame_idx: int):
        with self._seek_lock:
            self._seek_idx = frame_idx
        self._drain_queue()

    def read_next(self, timeout: float = 0.02) -> Optional[tuple[int, np.ndarray]]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stop = True
        try:
            self._thread.join(timeout=0.2)
        except Exception:
            pass
        if self._cap is not None:
            self._cap.release()

# ============================
# 主播放器
# ============================

def run_player():
    sg.theme('DarkGrey13')

    speed_options = [round(0.25 * i, 2) for i in range(1, 81)]  # 0.25 ~ 20.0

    video_image = sg.Image(key='-FRAME-', size=(960, 540))
    timeline = sg.Graph(canvas_size=(960, 36), graph_bottom_left=(0, 0), graph_top_right=(960, 36), key='-TL-')

    controls = [
        sg.Button('開啟影片', key='-OPEN-', size=(10,1)),
        sg.Button('⏪ 倒退5秒', key='-SEEK_BACK-', size=(12,1)),
        sg.Button('⏯ 暫停/播放', key='-PAUSE-', size=(12,1)),
        sg.Button('⏩ 快進5秒', key='-SEEK_FWD-', size=(12,1)),
        sg.Button('🔖 標記開始', key='-MARK_START-', size=(12,1)),
        sg.Button('🏁 標記結束', key='-MARK_END-', size=(12,1)),
        sg.Button('💾 匯出標記', key='-EXPORT-', size=(12,1)),
        sg.Checkbox('顯示偵測框（較慢）', default=False, key='-SHOW_DET-'),
        sg.Text('有人×'), sg.Spin(values=speed_options, initial_value=1.5, key='-SPD_IN-', size=(5,1)),
        sg.Text('無人×'), sg.Spin(values=speed_options, initial_value=10.0, key='-SPD_OUT-', size=(5,1)),
        sg.Button('重設速度', key='-SPD_RESET-'),
    ]

    status_bar = [
        sg.Text('位置(秒):'),
        sg.Slider(range=(0, 100), default_value=0, resolution=0.01, orientation='h', size=(40, 10), key='-SLIDER-', enable_events=True),
        sg.Text('×速度:'), sg.Text('1.0', key='-SPEEDTXT-'), sg.Text('  |  '),
        sg.Text('CPU: --%  GPU: --%  FPS: --', key='-STATS-'), sg.Text('  |  '),
        sg.Text('游標: --:--  AI: -', key='-CURINFO-')
    ]

    layout = [
        [video_image],
        [timeline],
        controls,
        status_bar,
    ]

    window = sg.Window('AI 輔助影片播放器 v2 (YOLOv8)', layout, finalize=True, return_keyboard_events=True)

    # 綁定時間軸滑鼠事件
    tl_widget = window['-TL-'].Widget
    tl_widget.bind('<Button-1>', ' MOUSEDOWN')
    tl_widget.bind('<B1-Motion>', ' MOUSEDRAG')
    tl_widget.bind('<ButtonRelease-1>', ' MOUSERELEASE')
    tl_widget.bind('<Motion>', ' MOUSEMOVE')

    prefetcher: Optional[VideoPrefetcher] = None
    video_path = None
    fps = 30.0
    duration = 0.0
    total_frames = 0
    current_frame_idx = 0
    paused = True

    # 狀態：AI 是否完成；AI CSV 是否已輸出
    ai_ready = False
    ai_csv_written = False

    # 預先標記資料
    ai_intervals: List[Interval] = []

    # 使用者手動標記
    mark_start_time: Optional[float] = None
    user_marks: List[Interval] = []

    # 時間軸拖曳狀態
    tl_dragging = False
    tl_mode: Optional[str] = None  # 'create' | 'resize-l' | 'resize-r' | 'move'
    tl_active_idx: Optional[int] = None
    tl_drag_start_t: float = 0.0
    tl_tmp_interval: Optional[Interval] = None
    EDGE_PX = 6

    # YOLO 模型（供可選的即時畫框使用）
    yolo_model = None

    # FPS 計算（平滑）
    last_draw_ts = None
    smooth_fps = 0.0

    # NVML 初始化（可選）
    nvml_handle = None
    if _HAS_NVML:
        try:
            nvmlInit()
            if nvmlDeviceGetCount() > 0:
                nvml_handle = nvmlDeviceGetHandleByIndex(0)
        except Exception:
            nvml_handle = None

    def shutdown_nvml():
        if _HAS_NVML:
            try:
                nvmlShutdown()
            except Exception:
                pass

    def set_controls_enabled(enabled: bool):
        for k in ['-SEEK_BACK-','-PAUSE-','-SEEK_FWD-','-MARK_START-','-MARK_END-','-EXPORT-','-SLIDER-']:
            try:
                window[k].update(disabled=not enabled)
            except Exception:
                pass

    def set_position_seconds(sec: float):
        nonlocal current_frame_idx
        if duration <= 0:
            return
        sec = max(0.0, min(max(0.0, duration - 1e-3), sec))
        current_frame_idx = int(sec * fps)
        if prefetcher is not None:
            prefetcher.request_index(current_frame_idx)

    def get_position_seconds() -> float:
        return current_frame_idx / float(max(fps, 1))

    def _safe_float(v, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def compute_speed_factor(cur_t: float) -> float:
        spd_in = _safe_float(values.get('-SPD_IN-', 1.5), 1.5)
        spd_out = _safe_float(values.get('-SPD_OUT-', 10.0), 10.0)
        return spd_in if in_any_interval(cur_t, ai_intervals) else spd_out

    def overlay_speed_text(frame: np.ndarray, speed: float) -> np.ndarray:
        overlay = frame.copy()
        text = f"x{speed:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(overlay, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), thickness=-1)
        cv2.putText(overlay, text, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return overlay

    def overlay_detections_if_enabled(frame_bgr: np.ndarray) -> np.ndarray:
        if not values.get('-SHOW_DET-', False):
            return frame_bgr
        if yolo_model is None:
            return frame_bgr
        device = _select_device_str()
        try:
            results = yolo_model.predict(frame_bgr, imgsz=640, conf=0.25, iou=0.45, device=device, verbose=False)
        except Exception:
            results = yolo_model.predict(frame_bgr, imgsz=640, conf=0.25, iou=0.45, device='cpu', verbose=False)
        if not results:
            return frame_bgr
        r = results[0]
        if r.boxes is None:
            return frame_bgr
        img = frame_bgr.copy()
        xyxy = r.boxes.xyxy.detach().cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
        cls = r.boxes.cls.detach().cpu().numpy() if hasattr(r.boxes, 'cls') else []
        conf = r.boxes.conf.detach().cpu().numpy() if hasattr(r.boxes, 'conf') else []
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            if int(c) != 0:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"person {float(p):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 200, 255), -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def _write_ai_csv(csv_path: str, intervals: List[Interval]):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end'])
            for iv in intervals:
                writer.writerow([f"{iv.start:.3f}", f"{iv.end:.3f}"])

    def _update_stats(cur_t: float):
        nonlocal last_draw_ts, smooth_fps
        now = time.perf_counter()
        if last_draw_ts is not None:
            inst_fps = 1.0 / max(1e-6, now - last_draw_ts)
            # 指數平滑
            alpha = 0.2
            smooth_fps = (1 - alpha) * smooth_fps + alpha * inst_fps if smooth_fps > 0 else inst_fps
        last_draw_ts = now

        # CPU
        cpu_txt = '--%'
        if _HAS_PSUTIL:
            try:
                cpu_txt = f"{psutil.cpu_percent(interval=None):.0f}%"
            except Exception:
                pass
        # GPU（NVML）
        gpu_txt = '--%'
        if nvml_handle is not None:
            try:
                util = nvmlDeviceGetUtilizationRates(nvml_handle)
                mem = nvmlDeviceGetMemoryInfo(nvml_handle)
                gpu_txt = f"{util.gpu}%/{mem.used//(1024*1024)}MB"
            except Exception:
                pass
        window['-STATS-'].update(f"CPU: {cpu_txt}  GPU: {gpu_txt}  FPS: {smooth_fps:.1f}")

        in_ai = '是' if in_any_interval(cur_t, ai_intervals) else '否'
        window['-CURINFO-'].update(f"游標: {_fmt_time(cur_t)}  AI: {in_ai}")

    def _redraw_timeline(cursor_t: Optional[float] = None):
        draw_timeline(window['-TL-'], 960, 36, ai_intervals, user_marks, duration, cursor_t, tl_tmp_interval)

    def _x_to_time(x: int) -> float:
        return max(0.0, min(duration, (x / 960.0) * max(duration, 1e-6)))

    def _hit_test_user_interval(t: float) -> Tuple[Optional[int], Optional[str]]:
        # 回傳 (index, 模式)：靠左/靠右/內部；None 表示沒命中
        if not user_marks:
            return None, None
        # 以像素閾值判斷左右邊緣
        for idx, iv in enumerate(user_marks):
            x0 = int(960 * (iv.start / max(duration, 1e-6)))
            x1 = int(960 * (iv.end / max(duration, 1e-6)))
            tx = int(960 * (t / max(duration, 1e-6)))
            if abs(tx - x0) <= EDGE_PX:
                return idx, 'resize-l'
            if abs(tx - x1) <= EDGE_PX:
                return idx, 'resize-r'
            if x0 + EDGE_PX < tx < x1 - EDGE_PX:
                return idx, 'move'
        return None, None

    def draw_frame_from_prefetch():
        nonlocal current_frame_idx
        if prefetcher is None:
            return
        item = prefetcher.read_next(timeout=0.02)
        if item is None:
            return
        idx, frame = item
        current_frame_idx = idx

        frame_disp = cv2.resize(frame, (960, 540))
        frame_disp = overlay_detections_if_enabled(frame_disp)

        cur_t = get_position_seconds()
        speed = compute_speed_factor(cur_t)
        frame_disp = overlay_speed_text(frame_disp, speed)

        _redraw_timeline(cur_t)

        imgbytes = cv2.imencode('.png', frame_disp)[1].tobytes()
        window['-FRAME-'].update(data=imgbytes)
        window['-SLIDER-'].update(value=cur_t, range=(0, max(duration, 0.01)))
        window['-SPEEDTXT-'].update(f"{speed:.1f}")
        _update_stats(cur_t)

    # 啟動後，先鎖控制
    set_controls_enabled(False)

    # 事件迴圈
    while True:
        event, values = window.read(timeout=10)
        if event in (sg.WIN_CLOSED, 'Exit', 'ESC:27'):
            break

        if event == '-OPEN-':
            path = sg.popup_get_file('選擇影片檔', file_types=(('Video Files', '*.mp4;*.avi;*.mov;*.mkv'), ('All Files', '*.*')))
            if path and os.path.isfile(path):
                if prefetcher is not None:
                    prefetcher.close()
                    prefetcher = None
                prefetcher = VideoPrefetcher(path, buffer_size=180)
                video_path = path
                fps = prefetcher.fps
                total_frames = prefetcher.total_frames
                duration = total_frames / max(fps, 1.0)
                current_frame_idx = 0
                paused = True
                ai_ready = False
                ai_csv_written = False
                ai_intervals = []
                user_marks = []
                set_controls_enabled(False)
                _redraw_timeline(0.0)

                def _bg():
                    return rough_human_intervals_yolo(video_path)
                window.perform_long_operation(_bg, '-AI_DONE-')
                sg.popup_no_wait('正在進行 YOLOv8 預標記… 完成前不可播放', keep_on_top=True)

        elif event == '-AI_DONE-':
            try:
                result = values['-AI_DONE-']
                if isinstance(result, tuple) and len(result) >= 3:
                    ai_intervals = result[0] or []
                    duration_ai = result[1] or duration
                    fps_ai = result[2] or fps
                    duration = duration_ai
                    fps = fps_ai
                    if len(result) >= 4:
                        yolo_model_obj = result[3]
                        if yolo_model_obj is not None:
                            yolo_model = yolo_model_obj  # type: ignore
                else:
                    sg.popup_error('AI 預標記回傳格式錯誤')

                window['-SLIDER-'].update(range=(0, max(duration, 0.01)))
                _redraw_timeline(0.0)
                set_position_seconds(0.0)

                # 自動輸出 AI 標記 CSV
                if video_path and not ai_csv_written:
                    try:
                        ai_csv_path = os.path.splitext(video_path)[0] + '_ai_marks.csv'
                        _write_ai_csv(ai_csv_path, ai_intervals)
                        ai_csv_written = True
                        sg.popup_no_wait(f'已產生 AI 標記 CSV：{ai_csv_path}', keep_on_top=True)
                    except Exception as e:
                        sg.popup_error(f'AI 標記 CSV 產生失敗: {e}')

                ai_ready = True
                set_controls_enabled(True)
                paused = False
            except Exception as e:
                sg.popup_error(f'AI 預標記失敗: {e}')

        elif event == '-SPD_RESET-':
            window['-SPD_IN-'].update(1.5)
            window['-SPD_OUT-'].update(10.0)

        elif event == '-SEEK_BACK-':
            if ai_ready and prefetcher is not None:
                new_idx = max(0, current_frame_idx - int(5.0 * fps))
                prefetcher.request_index(new_idx)
        elif event == '-SEEK_FWD-':
            if ai_ready and prefetcher is not None:
                new_idx = min(total_frames - 1, current_frame_idx + int(5.0 * fps))
                prefetcher.request_index(new_idx)
        elif event == '-PAUSE-':
            if ai_ready:
                paused = not paused
        elif event == '-MARK_START-':
            if ai_ready:
                mark_start_time = get_position_seconds()
                sg.popup_no_wait(f'標記開始: {mark_start_time:.2f}s')
        elif event == '-MARK_END-':
            if ai_ready:
                if mark_start_time is None:
                    sg.popup_no_wait('尚未設定「標記開始」')
                else:
                    end_t = get_position_seconds()
                    user_marks.append(Interval(mark_start_time, end_t))
                    user_marks[:] = merge_intervals(user_marks, merge_gap=0.0)
                    sg.popup_no_wait(f'標記完成: {mark_start_time:.2f}s → {end_t:.2f}s')
                    mark_start_time = None
        elif event == '-EXPORT-':
            if ai_ready:
                if not user_marks:
                    sg.popup_no_wait('目前沒有可匯出的標記。請先用「標記開始 / 標記結束」或在時間軸拖曳建立。')
                else:
                    save_path = sg.popup_get_file('儲存 CSV', save_as=True, default_extension='.csv', file_types=(('CSV', '*.csv'),))
                    if save_path:
                        try:
                            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow(['start', 'end'])
                                for iv in user_marks:
                                    writer.writerow([f"{iv.start:.3f}", f"{iv.end:.3f}"])
                            sg.popup('已匯出標記到：' + save_path)
                        except Exception as e:
                            sg.popup_error(f'匯出失敗: {e}')

        # === 時間軸互動 ===
        if event.startswith('-TL-'):
            # 取得滑鼠在 graph 座標
            mx, my = window['-TL-'].get_mouse_position() or (None, None)
            if mx is not None:
                cur_t = _x_to_time(int(mx))
                _redraw_timeline(cur_t if ai_ready else None)
                in_ai = '是' if in_any_interval(cur_t, ai_intervals) else '否'
                window['-CURINFO-'].update(f"游標: {_fmt_time(cur_t)}  AI: {in_ai}")

            if event.endswith('MOUSEDOWN') and ai_ready:
                if mx is None:
                    continue
                tl_dragging = True
                tl_drag_start_t = _x_to_time(int(mx))
                idx, mode = _hit_test_user_interval(tl_drag_start_t)
                tl_active_idx = idx
                if mode is None:
                    tl_mode = 'create'
                    tl_tmp_interval = Interval(tl_drag_start_t, tl_drag_start_t)
                else:
                    tl_mode = mode

            elif event.endswith('MOUSEDRAG') and ai_ready and tl_dragging:
                if mx is None:
                    continue
                t_now = _x_to_time(int(mx))
                if tl_mode == 'create' and tl_tmp_interval is not None:
                    tl_tmp_interval = Interval(tl_drag_start_t, t_now)
                elif tl_mode in ('resize-l', 'resize-r') and tl_active_idx is not None:
                    iv = user_marks[tl_active_idx]
                    if tl_mode == 'resize-l':
                        user_marks[tl_active_idx] = Interval(min(t_now, iv.end-1e-3), iv.end)
                    else:
                        user_marks[tl_active_idx] = Interval(iv.start, max(t_now, iv.start+1e-3))
                elif tl_mode == 'move' and tl_active_idx is not None:
                    iv = user_marks[tl_active_idx]
                    delta = t_now - tl_drag_start_t
                    new_start = max(0.0, min(duration, iv.start + delta))
                    new_end = max(0.0, min(duration, iv.end + delta))
                    # 防止超界：若超過，對齊界線
                    shift = 0.0
                    if new_start < 0.0:
                        shift = -new_start
                    if new_end > duration:
                        shift = duration - new_end
                    user_marks[tl_active_idx] = Interval(iv.start + delta + shift, iv.end + delta + shift)
                    tl_drag_start_t = t_now  # 持續移動
                _redraw_timeline(get_position_seconds())

            elif event.endswith('MOUSERELEASE') and ai_ready and tl_dragging:
                tl_dragging = False
                if tl_mode == 'create' and tl_tmp_interval is not None:
                    if tl_tmp_interval.length() >= 0.05:
                        user_marks.append(tl_tmp_interval)
                        user_marks[:] = merge_intervals(user_marks, merge_gap=0.0)
                    tl_tmp_interval = None
                tl_mode = None
                tl_active_idx = None
                _redraw_timeline(get_position_seconds())

        elif event == '-SLIDER-':
            if ai_ready and duration > 0:
                set_position_seconds(values['-SLIDER-'])
                paused = True

        # 播放（AI 完成後）
        if ai_ready and not paused and prefetcher is not None and duration > 0:
            cur_t = get_position_seconds()
            speed = compute_speed_factor(cur_t)
            step_frames = max(1, int(round(speed)))
            next_frame = min(current_frame_idx + step_frames, total_frames - 1)
            prefetcher.request_index(next_frame)
            draw_frame_from_prefetch()
            # 節流（粗略），由預取緩衝平衡
            time.sleep(max(0.0, (1.0 / max(fps, 1.0)) / max(speed, 1e-6)))
        else:
            # 暫停時仍更新時間軸游標/效能（較慢頻率）
            if ai_ready:
                _redraw_timeline(get_position_seconds())
                _update_stats(get_position_seconds())

    if prefetcher is not None:
        prefetcher.close()
    shutdown_nvml()
    window.close()


if __name__ == '__main__':
    run_player()
