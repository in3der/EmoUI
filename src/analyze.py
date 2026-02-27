"""
analyze.py
----------
영상 전체를 분석해서 결과를 CSV + 오버레이 영상으로 저장.
H.264 변환 자동 포함 (브라우저 호환).

실행 흐름:
    analyze.py → export_json.py → viewer.html (브라우저)

실행 방법:
    cd EmoUI/src
    python analyze.py --video ./DataSets/PlayVideos/sample.mp4
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import subprocess
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Callable, Optional

from emotion_engine import (
    load_ai_models, process_frame, smooth_va, smooth_emotion,
    reset_smoother, CentroidTracker, CLASS_COLORS
)

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_VIDEO  = str(PROJECT_ROOT / "src" / "DataSets" / "PlayVideos" / "sample.mp4")
DEFAULT_OUTPUT = str(BASE_DIR / "results" / "analysis_result.csv")
DEFAULT_SKIP      = 3
DEFAULT_EMA_ALPHA = 0.25
RESIZE_WIDTH      = 640
RESIZE_HEIGHT     = 360


def hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def draw_overlay(image: np.ndarray, frame_results: list) -> np.ndarray:
    vis = image.copy()
    for r in frame_results:
        x1, y1, x2, y2 = r['coords_orig']
        color = hex_to_bgr(CLASS_COLORS.get(r['emotion'], '#FFFFFF'))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{r['id']} | {r['emotion']}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(vis, f"V:{r['valence']:+.2f} A:{r['arousal']:+.2f}",
                    (x1, min(y2 + 18, image.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(vis, "IVPL Emotion Analysis",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "IVPL Emotion Analysis",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def convert_to_h264(input_path: str, output_path: str) -> bool:
    """mp4v로 저장된 영상을 H.264로 변환. 브라우저 호환."""
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-vcodec', 'libx264', '-acodec', 'aac',
            '-movflags', 'faststart',
            output_path
        ], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("[WARNING] ffmpeg 없음. mp4v 그대로 저장됩니다. seek 기능이 제한될 수 있어요.")
        return False


def analyze_video(video_path: str,
                  output_path: str,
                  hs_model_name: str = "enet_b0_8_va_mtl",
                  skip: int = DEFAULT_SKIP,
                  ema_alpha: float = DEFAULT_EMA_ALPHA,
                  progress_callback: Optional[Callable[[float, str], None]] = None):
    """
    progress_callback(ratio, message): Streamlit 등 외부에서 진행률 받을 때 사용.
    ratio = 0.0 ~ 1.0
    """

    def _progress(ratio: float, msg: str):
        if progress_callback:
            progress_callback(ratio, msg)
        else:
            print(msg)

    _progress(0.0, "모델 로딩 중...")
    base_model, img_processing, hs_recognizer = load_ai_models(hs_model_name)
    tracker = CentroidTracker(maxDisappeared=50)
    reset_smoother()
    _progress(0.05, "모델 로딩 완료")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    _progress(0.05, f"영상 정보: {total_frames:,}프레임 | FPS:{fps:.1f} | {orig_w}x{orig_h} | {duration_sec:.1f}초")

    output_dir   = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem            = Path(output_path).stem
    overlay_raw     = str(output_dir / f"{stem}_overlay_raw.mp4")
    overlay_path    = str(output_dir / f"{stem}_overlay.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(overlay_raw, fourcc, fps, (orig_w, orig_h))

    results      = []
    frame_idx    = 0
    last_results = []

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % skip == 0:
                frame_rgb = cv2.cvtColor(
                    cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT)),
                    cv2.COLOR_BGR2RGB
                )
                rects, frame_faces = process_frame(
                    frame_rgb, img_processing, base_model, hs_recognizer
                )
                objects = tracker.update(rects)

                scale_x = orig_w / RESIZE_WIDTH
                scale_y = orig_h / RESIZE_HEIGHT

                last_results = []
                for objID, cent in objects.items():
                    pid = f"Person_{objID}"

                    closest, min_d = None, 999.0
                    for face in frame_faces:
                        cx = (face['coords'][0] + face['coords'][2]) / 2
                        cy = (face['coords'][1] + face['coords'][3]) / 2
                        d  = float(np.linalg.norm(np.array([cx, cy]) - np.array(cent)))
                        if d < min_d:
                            min_d, closest = d, face

                    if closest is None or min_d >= 60:
                        continue

                    smoothed_emotion = smooth_emotion(pid, closest['pred'])
                    sv, sa = smooth_va(pid, closest['valence'], closest['arousal'], ema_alpha)

                    x1, y1, x2, y2 = closest['coords']
                    coords_orig = (
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y)
                    )

                    last_results.append({
                        'id': pid, 'coords_orig': coords_orig,
                        'emotion': smoothed_emotion,
                        'valence': round(sv, 4), 'arousal': round(sa, 4),
                        'bbox_area': closest['bbox_area'],
                    })
                    results.append({
                        'frame': frame_idx, 'time_sec': round(frame_idx / fps, 2),
                        'id': pid, 'emotion': smoothed_emotion,
                        'valence': round(sv, 4), 'arousal': round(sa, 4),
                        'bbox_area': closest['bbox_area'],
                    })

            writer.write(draw_overlay(image, last_results))

            if frame_idx % 100 == 0:
                ratio = 0.05 + (frame_idx / total_frames) * 0.80
                _progress(ratio, f"분석 중... {frame_idx/total_frames*100:.1f}%")

    finally:
        cap.release()
        writer.release()

    if not results:
        raise RuntimeError("감지된 얼굴이 없습니다. 영상을 확인해주세요.")

    # H.264 변환
    _progress(0.87, "H.264 변환 중...")
    converted = convert_to_h264(overlay_raw, overlay_path)
    if converted:
        os.remove(overlay_raw)
    else:
        # ffmpeg 없으면 raw 파일을 그대로 사용
        os.rename(overlay_raw, overlay_path)

    # CSV 저장
    _progress(0.95, "결과 저장 중...")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    meta_path = Path(output_path).with_suffix('.meta.csv')
    pd.DataFrame([{
        'fps': fps, 'total_frames': total_frames,
        'duration_sec': round(duration_sec, 2),
        'orig_w': orig_w, 'orig_h': orig_h,
        'skip': skip, 'overlay_path': overlay_path,
    }]).to_csv(meta_path, index=False)

    _progress(1.0, f"분석 완료! 총 {len(df):,}개 레코드")

    return {
        'csv_path':     output_path,
        'overlay_path': overlay_path,
        'meta_path':    str(meta_path),
        'total_records': len(df),
        'persons':      sorted(df['id'].unique().tolist()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상 감정 분석 → CSV + 오버레이 영상 저장")
    parser.add_argument("--video",    type=str, default=DEFAULT_VIDEO)
    parser.add_argument("--output",   type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--hs_model", type=str, default="enet_b0_8_va_mtl",
                        choices=["enet_b0_8_va_mtl", "enet_b2_8_va_mtl", "enet_b0_8_best_vgaf"])
    parser.add_argument("--skip",  type=int,   default=DEFAULT_SKIP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_EMA_ALPHA)
    args = parser.parse_args()

    analyze_video(
        video_path=args.video, output_path=args.output,
        hs_model_name=args.hs_model, skip=args.skip, ema_alpha=args.alpha,
    )