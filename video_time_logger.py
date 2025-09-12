#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: tools/video_time_logger.py
"""
漁船影片人員活動標記工具 (OpenCV)

功能概述
- 播放單支或多支影片，邊看邊以快捷鍵標記 [開始, 結束] 時間段
- 每段以「MM:SS ~ MM:SS」格式輸出至 .txt（每部影片對應一個 .txt）
- 支援快退/快進、暫停/播放、撤銷、清空、下一部影片

必要套件
- opencv-python (cv2)

使用方式 (CLI)
$ python video_time_logger.py <檔案或資料夾...> \
    --output out_dir \
    --ext mp4 mov avi mkv

範例
$ python video_time_logger.py ./videos --output ./labels
$ python video_time_logger.py a.mp4 b.mp4 --output ./labels
C_125_1713489242_530536

快捷鍵 (視窗聚焦時生效)
  Space / p : 暫停/播放
  i          : 標記區間起點 (IN)
  o          : 以目前時間作為區間終點 (OUT) 並加入清單
  u          : 撤銷 (若有未完成的 IN 先取消，否則移除最後一段)
  r          : 清空所有已標記區間
  h / l      : -1 秒 / +1 秒
  H / L      : -5 秒 / +5 秒
  s          : 立即儲存 .txt
  n          : 儲存並切換到下一部影片
  q          : 儲存後退出

輸出
- 對每個輸入影片 <name> 產生 <name>.txt，內容每行一段：「MM:SS ~ MM:SS」

注意
- 時間以秒數向下取整 (floor) 格式化為 MM:SS，便於與肉眼觀察對齊。
- 若 fps 無法讀取，改用 POS_MSEC 估算時間。
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore


# ------------------------------
# 資料結構與工具函式
# ------------------------------

@dataclass
class Interval:
    start_s: float
    end_s: float

    def as_mmss(self) -> Tuple[str, str]:
        return seconds_to_mmss(self.start_s), seconds_to_mmss(self.end_s)


def seconds_to_mmss(seconds: float) -> str:
    """將秒數向下取整為 MM:SS 字串。
    為了與人工觀察一致，這裡採 floor，避免邊界晃動造成標記不穩定。
    """
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)  # floor
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_video_list(inputs: Sequence[str], exts: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    extset = {"." + e.lower().lstrip(".") for e in exts}
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.suffix.lower() in extset:
                    out.append(q)
        elif p.is_file() and p.suffix.lower() in extset:
            out.append(p)
    return out


# ------------------------------
# 播放與標記主流程
# ------------------------------

class VideoAnnotator:
    def __init__(self, videos: List[Path], out_dir: Path) -> None:
        self.videos = videos
        self.out_dir = out_dir
        ensure_output_dir(out_dir)

    def run(self) -> None:
        if not self.videos:
            print("[錯誤] 找不到任何影片，請確認路徑與副檔名。")
            return
        print(f"[資訊] 待處理影片數量: {len(self.videos)}")
        for idx, video in enumerate(self.videos, start=1):
            print(f"\n[播放 {idx}/{len(self.videos)}] {video}")
            try:
                self._annotate_single(video)
            except KeyboardInterrupt:
                print("\n[中斷] 使用者中止。")
                break

    # ------------- 單支影片標記 -------------
    def _annotate_single(self, video: Path) -> None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[警告] 無法開啟影片: {video}")
            return

        win = "Video Time Logger"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total_frames / fps) if (fps > 0 and total_frames > 0) else None

        intervals: List[Interval] = []
        pending_start: Optional[float] = None
        paused = False

        info = textwrap.dedent(
            """
            [快捷鍵]\nSpace/p: 暫停/播放  |  i: IN  |  o: OUT加段  |  u: 撤銷  |  r: 清空\nh/l: -1s/+1s        |  H/L: -5s/+5s  |  s: 儲存  |  n: 下一部  |  q: 離開
            """
        ).strip()

        def current_time_s() -> float:
            if fps > 0:
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                return max(0.0, frame_idx / fps)
            # 後備: 以毫秒為準
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            return max(0.0, ms / 1000.0)

        def seek_delta(seconds: float) -> None:
            # 關鍵：避免 seek 到負數或超出尾端
            if fps > 0:
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                target = cur + int(seconds * fps)
                target = max(0, target)
                if total_frames > 0:
                    target = min(target, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            else:
                cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                tgt_ms = max(0.0, cur_ms + seconds * 1000.0)
                cap.set(cv2.CAP_PROP_POS_MSEC, tgt_ms)

        def overlay(frame, text_lines: List[str]) -> None:
            # 在左上角疊加資訊，避免頻繁重繪造成閱讀困難
            y = 28
            for line in text_lines:
                cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

        def save_txt() -> Path:
            outf = self.out_dir / f"{video.stem}.txt"
            with outf.open("w", encoding="utf-8") as f:
                for itv in intervals:
                    a, b = itv.as_mmss()
                    f.write(f"{a} ~ {b}\n")
            print(f"[已儲存] {outf}  (共 {len(intervals)} 段)")
            return outf

        # 首幀預讀
        ok, frame = cap.read()
        if not ok:
            print("[警告] 無法讀取影像幀。")
            cap.release()
            cv2.destroyWindow(win)
            return

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    # 到片尾自動暫停，等待使用者操作
                    paused = True
                    # 嘗試停在最後一幀的位置
                    seek_delta(-0.001)
                    ok2, last = cap.read()
                    if ok2:
                        frame = last
                    else:
                        # 若無法讀取最後一幀，保留上一幀
                        pass

            # 畫面資訊
            now_s = current_time_s()
            now_txt = seconds_to_mmss(now_s)
            dur_txt = seconds_to_mmss(duration_s) if duration_s is not None else "??:??"
            lines = [
                f"{video.name}  |  {now_txt} / {dur_txt}  |  段數: {len(intervals)}",
                (f"IN 等待 OUT: {seconds_to_mmss(pending_start)}" if pending_start is not None else "無進行中的 IN"),
                info,
            ]
            disp = frame.copy()
            overlay(disp, lines)
            cv2.imshow(win, disp)

            # waitKey: 播放時用較短間隔，暫停時用較長間隔
            delay = 30 if not paused else 120
            k = cv2.waitKey(delay) & 0xFF

            if k == 0xFF:  # 無按鍵
                continue

            # 控制鍵判斷（同時支援大小寫）
            if k in (ord(' '), ord('p'), ord('P')):
                paused = not paused

            elif k in (ord('i'), ord('I')):
                pending_start = now_s
                paused = True  # 落點更準確

            elif k in (ord('o'), ord('O')):
                if pending_start is not None and now_s > pending_start:
                    intervals.append(Interval(start_s=pending_start, end_s=now_s))
                    pending_start = None
                else:
                    print("[提示] 需要先按 i 設定 IN，且 OUT 必須晚於 IN。")

            elif k in (ord('u'), ord('U')):
                if pending_start is not None:
                    pending_start = None
                elif intervals:
                    intervals.pop()
                else:
                    print("[提示] 無可撤銷項目。")

            elif k in (ord('r'), ord('R')):
                intervals.clear()
                pending_start = None

            elif k in (ord('h'), ord('H')):
                seek_delta(-5.0 if k == ord('H') else -1.0)
                paused = True

            elif k in (ord('l'), ord('L')):
                seek_delta(5.0 if k == ord('L') else 1.0)
                paused = True

            elif k in (ord('s'), ord('S')):
                save_txt()

            elif k in (ord('n'), ord('N')):
                save_txt()
                break  # 下一部

            elif k in (ord('q'), ord('Q')):
                save_txt()
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        # 結束單支影片，釋放資源
        cap.release()
        cv2.destroyWindow(win)


# ------------------------------
# 參數解析與進入點
# ------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="以視覺化播放方式記錄影片中有人員活動的時間區間，輸出 MM:SS ~ MM:SS 至文字檔。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="一個或多個影片檔/資料夾路徑。若為資料夾將遞迴搜尋符合副檔名的影片。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels"),
        help="輸出文字檔資料夾。",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["mp4", "mov", "avi", "mkv", "m4v"],
        help="要搜尋的影片副檔名清單。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    videos = build_video_list(args.inputs, args.ext)
    if not videos:
        print("[錯誤] 找不到影片。請確認 --ext 或輸入路徑。")
        sys.exit(2)
    annotator = VideoAnnotator(videos=videos, out_dir=args.output)
    annotator.run()


if __name__ == "__main__":
    main()
