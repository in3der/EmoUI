"""
emotion_engine.py
-----------------
감정 분석 핵심 로직 모듈.
"""

import cv2
import numpy as np
from collections import OrderedDict, deque
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_ROOT    = BASE_DIR.parent
BASE_MODEL_PATH = str(PROJECT_ROOT / "models" / "mobilenet_7.h5")

INPUT_SIZE       = (224, 224)
MEAN_BGR         = np.array([103.939, 116.779, 123.68], dtype=np.float32)
MIN_FACE_HEIGHT  = 40    # 리사이즈(640x360) 기준 픽셀. 미만 얼굴 제외.
SMOOTHING_WINDOW = 15

IDX_TO_CLASS = {
    0: 'Anger', 1: 'Disgust', 2: 'Fear',    3: 'Happiness',
    4: 'Neutral', 5: 'Sadness', 6: 'Surprise'
}

CLASS_COLORS = {
    'Anger':     '#FF4B4B',
    'Disgust':   '#9B59B6',
    'Fear':      '#2C3E50',
    'Happiness': '#F1C40F',
    'Neutral':   '#95A5A6',
    'Sadness':   '#3498DB',
    'Surprise':  '#E67E22',
}

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
except ImportError:
    HSEMOTION_AVAILABLE = False
    print("[WARNING] hsemotion-onnx 미설치. pip install hsemotion-onnx 로 설치해주세요.")


def load_ai_models(hs_model_name: str = "enet_b0_8_va_mtl"):
    from facial_analysis import FacialImageProcessing
    from tensorflow.keras.models import load_model

    base_model     = load_model(BASE_MODEL_PATH, compile=False)
    img_processing = FacialImageProcessing(False)

    hs_recognizer = None
    if HSEMOTION_AVAILABLE:
        try:
            hs_recognizer = HSEmotionRecognizer(model_name=hs_model_name)
            print(f"[HSEmotion] {hs_model_name} 로드 완료")
        except Exception as e:
            print(f"[HSEmotion 로드 실패] {e}")
    else:
        print("[WARNING] HSEmotion 없이는 VA 측정이 불가합니다.")

    return base_model, img_processing, hs_recognizer


def measure_va(recognizer, face_crop_rgb: np.ndarray) -> tuple:
    """Valence / Arousal 측정. 실패 시 (0.0, 0.0) 반환."""
    if recognizer is None:
        return 0.0, 0.0
    try:
        face_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
        _, scores = recognizer.predict_emotions(face_bgr, logits=False)
        return (float(np.clip(scores[8], -1.0, 1.0)),
                float(np.clip(scores[9], -1.0, 1.0)))
    except Exception as e:
        print(f"[HSEmotion 추론 오류] {e}")
        return 0.0, 0.0


# 인물 ID별 softmax 확률 버퍼. deque(maxlen)으로 O(1) append/pop.
_emotion_buffer: dict = {}

def smooth_emotion(pid: str, pred: np.ndarray) -> str:
    """최근 SMOOTHING_WINDOW 프레임 softmax 평균의 argmax로 감정 결정."""
    if pid not in _emotion_buffer:
        _emotion_buffer[pid] = deque(maxlen=SMOOTHING_WINDOW)
    _emotion_buffer[pid].append(pred.copy())
    avg_prob = np.mean(_emotion_buffer[pid], axis=0)
    return IDX_TO_CLASS[int(np.argmax(avg_prob))]


_va_smoother: dict = {}

def smooth_va(pid: str, valence: float, arousal: float,
              alpha: float = 0.25) -> tuple:
    """EMA로 Valence / Arousal 스무딩. alpha 높을수록 원신호에 민감."""
    if pid not in _va_smoother:
        _va_smoother[pid] = {'v': valence, 'a': arousal}
    else:
        prev = _va_smoother[pid]
        _va_smoother[pid] = {
            'v': alpha * valence + (1 - alpha) * prev['v'],
            'a': alpha * arousal + (1 - alpha) * prev['a'],
        }
    return _va_smoother[pid]['v'], _va_smoother[pid]['a']


def reset_smoother():
    _va_smoother.clear()
    _emotion_buffer.clear()


class CentroidTracker:
    """프레임 간 얼굴 centroid 추적. maxDisappeared 프레임 미검출 시 ID 삭제."""

    def __init__(self, maxDisappeared: int = 50):
        self.nextObjectID   = 0
        self.objects        = OrderedDict()
        self.disappeared    = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID]     = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects.pop(objectID, None)
        self.disappeared.pop(objectID, None)

    def update(self, rects: list) -> OrderedDict:
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in rects],
            dtype=int)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            oids = list(self.objects.keys())
            octs = list(self.objects.values())
            D    = np.linalg.norm(
                np.array(octs)[:, None] - input_centroids[None], axis=2)

            rows     = D.min(axis=1).argsort()
            cols     = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()

            for r, c in zip(rows, cols):
                if r in used_r or c in used_c:
                    continue
                oid = oids[r]
                self.objects[oid]     = input_centroids[c]
                self.disappeared[oid] = 0
                used_r.add(r)
                used_c.add(c)

            for r in set(range(D.shape[0])) - used_r:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

            for c in set(range(D.shape[1])) - used_c:
                self.register(input_centroids[c])

        return self.objects


def process_frame(frame_rgb: np.ndarray,
                  img_processing,
                  base_model,
                  hs_recognizer) -> tuple:
    """
    단일 프레임 분석.
    rects와 frame_faces는 항상 1:1 대응 (전처리 성공한 얼굴만 rects에 추가).
    bbox_area는 리사이즈(640x360) 기준 픽셀 면적.
    """
    bounding_boxes, _ = img_processing.detect_faces(frame_rgb)

    rects       = []
    face_batch  = []
    face_crops  = []
    face_coords = []

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox.astype(int)[0:4]
        x1, y1 = max(0, x1), max(0, y1)

        if (y2 - y1) < MIN_FACE_HEIGHT:
            continue

        try:
            crop = frame_rgb[y1:y2, x1:x2, :]
            if crop.size == 0:
                continue
            inp = cv2.resize(crop, INPUT_SIZE).astype(np.float32) - MEAN_BGR
        except Exception as e:
            print(f"[얼굴 전처리 오류] {e}")
            continue

        # 전처리 완전 성공 후에만 rects 추가 → face_batch와 1:1 대응 보장
        rects.append((x1, y1, x2, y2))
        face_batch.append(inp)
        face_crops.append(crop)
        face_coords.append((x1, y1, x2, y2))

    frame_faces = []
    if face_batch:
        preds = base_model(np.array(face_batch), training=False).numpy()
        for i, pred in enumerate(preds):
            x1, y1, x2, y2 = face_coords[i]
            va = measure_va(hs_recognizer, face_crops[i])
            frame_faces.append({
                'coords':    face_coords[i],
                'pred':      pred,
                'emotion':   IDX_TO_CLASS[np.argmax(pred)],
                'valence':   va[0],
                'arousal':   va[1],
                'bbox_area': (x2 - x1) * (y2 - y1),
            })

    return rects, frame_faces