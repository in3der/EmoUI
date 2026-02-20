import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
# 현재 파일 기준 루트 설정
BASE_DIR = Path(__file__).resolve().parent   # src 폴더
PROJECT_ROOT = BASE_DIR.parent               # EmoUI 루트

import cv2
import dlib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from collections import OrderedDict
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide", page_title="IVPL Emotion Dashboard")

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
except ImportError:
    HSEMOTION_AVAILABLE = False
    print("[WARNING] hsemotion-onnx not installed. Falling back to AU geometric method.")


## 경로, 상수 설정──────────────────────────────────────────────────────


LANDMARK_MODEL  = PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_7.h5"
DEFAULT_VIDEO   = PROJECT_ROOT / "src" / "DataSets" / "PlayVideos" / "Sample1-009.mp4"

# str 변환 (TensorFlow / dlib 호환)
LANDMARK_MODEL  = str(LANDMARK_MODEL)
BASE_MODEL_PATH = str(BASE_MODEL_PATH)
DEFAULT_VIDEO   = str(DEFAULT_VIDEO)

INPUT_SIZE   = (224, 224)
MEAN_BGR     = np.array([103.939, 116.779, 123.68], dtype=np.float32)
DISPLAY_SKIP = 3   # N 프레임마다 화면/그래프 갱신

IDX_TO_CLASS = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happiness',
                4:'Neutral', 5:'Sadness', 6:'Surprise'}
CLASS_COLORS = {
    'Anger':'#FF4B4B', 'Disgust':'#9B59B6', 'Fear':'#2C3E50',
    'Happiness':'#F1C40F', 'Neutral':'#95A5A6', 'Sadness':'#3498DB',
    'Surprise':'#E67E22',
}
HS_MODELS = ["enet_b0_8_va_mtl", "enet_b2_8_va_mtl", "enet_b0_8_best_vgaf"]



## HSEmotion 모델 로드──────────────────────────────────────────────────────
@st.cache_resource
def load_hsemotion(model_name: str):
    """
    AffectNet 450k 학습 ONNX 모델.
    predict_emotions() → (emotion_str, scores[10])
        scores[0:8] = 감정 확률
        scores[8]   = valence  [-1, 1]
        scores[9]   = arousal  [-1, 1]
    """
    if not HSEMOTION_AVAILABLE:
        return None
    try:
        return HSEmotionRecognizer(model_name=model_name)
    except Exception as e:
        st.warning(f"HSEmotion 로드 실패: {e}")
        return None


## Valence, Arousal 측정 함수 정의───────────────────────────────────────────────────
## 방법1) HSEmotion 기반 / 방법2) AU로 VA 추정────────────────────────────────────────

def measure_va_hsemotion(recognizer, face_crop_rgb: np.ndarray) -> tuple:
    """HSEmotion ONNX 추론으로 Valence/Arousal 측정."""
    try:
        face_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
        _, scores = recognizer.predict_emotions(face_bgr, logits=False)
        return float(np.clip(scores[8], -1.0, 1.0)), \
               float(np.clip(scores[9], -1.0, 1.0))
    except Exception as e:
        print(f"[HSEmotion error] {e}")
        return 0.0, 0.0


def measure_va_au_fallback(face_crop_rgb: np.ndarray, emotion_name: str, lm_detector) -> tuple:
    """
    dlib 68 랜드마크 기반 AU 기하학적 VA 추정.
    ※ hand-crafted 가중치이므로 HSEmotion 실패 시에만 사용.
    """
    try:
        gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        rect  = dlib.rectangle(0, 0, w - 1, h - 1)
        shape = lm_detector(gray, rect)

        if shape.num_parts != 68:
            raise ValueError("landmark count mismatch")

        pts = np.array([[shape.part(i).x, shape.part(i).y]
                        for i in range(68)], dtype=np.float32)
        au  = _extract_au_features(pts)

        valence = float(np.clip(
            au['smile_ratio'] * 4.0
            - au['brow_furrow'] * 2.5
            + au['brow_raise']  * 0.8
            - au['mouth_open']  * 0.5,
            -1.0, 1.0))

        arousal = float(np.clip(
            au['mouth_open']  * 2.5
            + au['eye_open']  * 2.0
            + au['brow_raise']* 1.2
            + au['brow_furrow']* 1.0,
            0.0, 1.0))
        arousal = arousal * 2.0 - 1.0   # [0,1] → [-1,1] 범위 통일

        return valence, arousal

    except Exception:
        defaults = {
            'Happiness': ( 0.65,  0.30), 'Sadness':  (-0.60, -0.40),
            'Anger':     (-0.70,  0.60), 'Fear':     (-0.40,  0.40),
            'Surprise':  ( 0.10,  0.44), 'Disgust':  (-0.50,  0.00),
            'Neutral':   ( 0.05, -0.60),
        }
        return defaults.get(emotion_name, (0.0, -0.4))



## AU feature 추출 (landmark util)──────────────────────────────────────────────────────
def _extract_au_features(pts: np.ndarray) -> dict:
    """dlib 68점 → AU-inspired 기하 피처 (face_width 정규화)."""
    face_width      = np.linalg.norm(pts[16] - pts[0]) + 1e-6
    mouth_center_y  = (pts[51][1] + pts[57][1]) / 2
    corner_avg_y    = (pts[48][1] + pts[54][1]) / 2
    smile_ratio     = (mouth_center_y - corner_avg_y) / face_width   # AU12
    mouth_open      = np.linalg.norm(pts[51] - pts[57]) / face_width  # AU25/26
    brow_raise      = ((pts[37][1] + pts[44][1]) / 2
                       - (pts[19][1] + pts[24][1]) / 2) / face_width  # AU1/2
    inner_brow_dist = np.linalg.norm(pts[21] - pts[22]) / face_width
    brow_furrow     = max(0.0, 0.12 - inner_brow_dist) * 8.0          # AU4
    eye_open        = (np.linalg.norm(pts[37] - pts[41])
                       + np.linalg.norm(pts[44] - pts[46])) / 2 / face_width  # AU5
    return dict(smile_ratio=float(smile_ratio), mouth_open=float(mouth_open),
                brow_raise=float(brow_raise), brow_furrow=float(brow_furrow),
                eye_open=float(eye_open))


## EMA smoother──────────────────────────────────────────────────────
_va_smoother: dict = {}

def smooth_va(pid: str, valence: float, arousal: float, alpha: float) -> tuple:
    """지수 이동 평균으로 VA 시계열 스무딩."""
    if pid not in _va_smoother:
        _va_smoother[pid] = {'v': valence, 'a': arousal}
    else:
        prev = _va_smoother[pid]
        _va_smoother[pid] = {
            'v': alpha * valence + (1 - alpha) * prev['v'],
            'a': alpha * arousal + (1 - alpha) * prev['a'],
        }
    return _va_smoother[pid]['v'], _va_smoother[pid]['a']


## Centroid Tracker──────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
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

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in rects],
            dtype=int)

        if len(self.objects) == 0:
            for c in inputCentroids:
                self.register(c)
        else:
            oids = list(self.objects.keys())
            octs = list(self.objects.values())
            D    = np.linalg.norm(
                np.array(octs)[:, None] - inputCentroids[None], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()

            for r, c in zip(rows, cols):
                if r in used_r or c in used_c:
                    continue
                oid = oids[r]
                self.objects[oid]    = inputCentroids[c]
                self.disappeared[oid] = 0
                used_r.add(r); used_c.add(c)

            for r in set(range(D.shape[0])) - used_r:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

            for c in set(range(D.shape[1])) - used_c:
                self.register(inputCentroids[c])

        return self.objects


## 모델 로드──────────────────────────────────────────────────────
@st.cache_resource
def load_ai_models():
    from facial_analysis import FacialImageProcessing
    lm_det   = dlib.shape_predictor(LANDMARK_MODEL)
    b_model  = load_model(BASE_MODEL_PATH, compile=False)
    img_proc = FacialImageProcessing(False)
    return lm_det, b_model, img_proc

landmark_detector, base_model, imgProcessing = load_ai_models()
tracker = CentroidTracker(maxDisappeared=15)


## 세션 초기화──────────────────────────────────────────────────────
def init_session():
    defaults = {
        'all_data':    [],
        'detected_ids': ["Person_0"],
        'is_running':  False,
        'frame_pos':   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


## UI Setting──────────────────────────────────────────────────────
st.title("🎭 Emotion AI Dashboard")

with st.sidebar:
    st.header("🎮 Control")
    video_path = st.text_input("Video Path:", DEFAULT_VIDEO)

    st.subheader("VA 측정 설정")
    va_backend = st.radio(
        "VA 측정 백엔드",
        ["HSEmotion-ONNX (권장)", "AU Geometric (fallback)"],
        help="HSEmotion: AffectNet 450k 학습 | AU: 랜드마크 규칙 기반, 신뢰성 낮음"
    )
    use_hsemotion = (va_backend == "HSEmotion-ONNX (권장)")

    hs_model_name = st.selectbox(
        "HSEmotion 모델", HS_MODELS,
        help="b0: 빠름(실시간 권장) | b2: 정확 | vgaf: VA 특화(미사용)",
        disabled=not use_hsemotion
    )
    ema_alpha = st.slider(
        "Smoothing (EMA α)", 0.10, 0.50, 0.25, 0.05,
        help="낮을수록 부드러운 곡선, 높을수록 원신호에 민감"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏯️ Start/Pause"):
            st.session_state.is_running = not st.session_state.is_running
    with col2:
        if st.button("🧹 Reset"):
            st.session_state.all_data  = []
            st.session_state.frame_pos = 0
            _va_smoother.clear()
            st.rerun()

    st.divider()
    if use_hsemotion:
        if HSEMOTION_AVAILABLE:
            st.success("✅ HSEmotion-ONNX 사용 가능")
        else:
            st.error("❌ hsemotion-onnx 미설치\n```\npip install hsemotion-onnx\n```")
    else:
        st.warning("⚠️ AU Geometric 사용 중 (테스트 용도)")

# HSEmotion 모델 캐시 로드
hs_recognizer = load_hsemotion(hs_model_name) if use_hsemotion else None
backend_label = f"HSEmotion ({hs_model_name})" if use_hsemotion else "AU Geometric"

col_v, col_g = st.columns([1.3, 1])
with col_g:
    selected_id     = st.selectbox("Monitor Target:",
                                   st.session_state.get('detected_ids', ["Person_0"]))
    timeline_area   = st.empty()
    va_chart_area   = st.empty()
    circumplex_area = st.empty()

with col_v:
    video_area  = st.empty()
    status_text = st.empty()


## 분석 루프──────────────────────────────────────────────────────
if st.session_state.get('is_running') and os.path.exists(video_path):

    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)
    except Exception as e:
        st.error(str(e))
        cap = None

    while cap and cap.isOpened() and st.session_state.get('is_running'):
        ret, image = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        st.session_state.frame_pos = frame_idx

        # 얼굴 감지
        frame_rgb = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = imgProcessing.detect_faces(frame_rgb)

        rects       = []
        face_batch  = []
        face_crops  = []
        face_coords = []

        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox.astype(int)[0:4]
            x1, y1 = max(0, x1), max(0, y1)
            rects.append((x1, y1, x2, y2))
            try:
                crop = frame_rgb[y1:y2, x1:x2, :]
                if crop.size == 0:
                    continue
                inp = cv2.resize(crop, INPUT_SIZE).astype(np.float32) - MEAN_BGR
                face_batch.append(inp)
                face_crops.append(crop)
                face_coords.append((x1, y1, x2, y2))
            except Exception as e:
                print(str(e))

        # 감정 분류 + VA 측정
        frame_faces = []
        if face_batch:
            preds = base_model(np.array(face_batch), training=False).numpy()

            for i, pred in enumerate(preds):
                emotion = IDX_TO_CLASS[np.argmax(pred)]

                if use_hsemotion and hs_recognizer is not None:
                    valence, arousal = measure_va_hsemotion(hs_recognizer, face_crops[i])
                else:
                    valence, arousal = measure_va_au_fallback(
                        face_crops[i], emotion, landmark_detector)

                frame_faces.append({
                    'coords':  face_coords[i],
                    'emotion': emotion,
                    'valence': valence,
                    'arousal': arousal,
                })

        # 트래킹 & 결과 기록
        objects = tracker.update(rects)
        detected = st.session_state.get('detected_ids', [])

        for objID, cent in objects.items():
            pid = f"Person_{objID}"
            if pid not in detected:
                detected.append(pid)
                st.session_state['detected_ids'] = detected

            # 트래킹 ID ↔ 얼굴 데이터 매칭 (centroid 거리 기준)
            closest, min_d = None, 999
            for f in frame_faces:
                cx = (f['coords'][0] + f['coords'][2]) / 2
                cy = (f['coords'][1] + f['coords'][3]) / 2
                d  = np.linalg.norm(np.array([cx, cy]) - np.array(cent))
                if d < min_d:
                    min_d, closest = d, f

            if closest and min_d < 60:
                sv, sa = smooth_va(pid, closest['valence'], closest['arousal'], ema_alpha)
                st.session_state['all_data'].append({
                    "Frame":   frame_idx,
                    "ID":      pid,
                    "Emotion": closest['emotion'],
                    "Valence": sv,
                    "Arousal": sa,
                })

                if frame_idx % DISPLAY_SKIP == 0:
                    c     = closest['coords']
                    color = (255, 255, 0) if pid == selected_id else (0, 255, 0)
                    cv2.rectangle(frame_rgb, (c[0], c[1]), (c[2], c[3]), color, 2)
                    cv2.putText(frame_rgb, pid, (c[0], c[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 화면 갱신
        if frame_idx % DISPLAY_SKIP == 0:
            cv2.putText(frame_rgb, "Intelligent Vision Processing Lab.",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            video_area.image(cv2.resize(frame_rgb, (480, 270)), channels="RGB")
            status_text.text(
                f"Frame: {frame_idx} | GPU 0 (TF classify) | VA: {backend_label} | EMA α={ema_alpha:.2f}"
            )

        # 그래프 갱신
        if frame_idx % 10 == 0 and st.session_state.get('all_data'):
            df   = pd.DataFrame(st.session_state['all_data'])
            t_df = df[df['ID'] == selected_id].tail(60)

            if not t_df.empty:
                # 감정 타임라인
                chart_timeline = alt.Chart(t_df).mark_circle(size=100).encode(
                    x=alt.X('Frame:Q', scale=alt.Scale(zero=False), title=None),
                    y=alt.Y('Emotion:N'),
                    color=alt.Color('Emotion:N',
                        scale=alt.Scale(domain=list(CLASS_COLORS.keys()),
                                        range=list(CLASS_COLORS.values())),
                        legend=None)
                ).properties(height=180)
                timeline_area.altair_chart(chart_timeline, use_container_width=True)

                # Valence / Arousal 연속 회귀 그래프
                va_long = t_df[['Frame', 'Valence', 'Arousal']].melt(
                    id_vars='Frame', var_name='Dimension', value_name='Value')

                chart_va = alt.Chart(va_long).mark_line(
                    strokeWidth=3, interpolate='monotone'
                ).encode(
                    x=alt.X('Frame:Q', scale=alt.Scale(zero=False), title='Frames'),
                    y=alt.Y('Value:Q', scale=alt.Scale(domain=[-1.0, 1.0]),
                            title='Valence / Arousal'),
                    color=alt.Color('Dimension:N',
                        scale=alt.Scale(domain=['Valence', 'Arousal'],
                                        range=['#3498DB', '#E74C3C']),
                        legend=alt.Legend(orient='top-left')),
                    tooltip=['Frame', 'Dimension', alt.Tooltip('Value:Q', format='.3f')]
                ).properties(height=200, title=f"VA Regression — {backend_label}")

                zero_rule = alt.Chart(pd.DataFrame({'y': [0.0]})).mark_rule(
                    strokeDash=[4, 4], color='gray', opacity=0.4
                ).encode(y='y:Q')

                va_chart_area.altair_chart(
                    (chart_va + zero_rule).interactive(), use_container_width=True)

                # Circumplex: VA 평면 (최근 30프레임)
                latest = t_df.tail(30)
                chart_circumplex = alt.Chart(latest).mark_point(
                    size=80, filled=True, opacity=0.7
                ).encode(
                    x=alt.X('Valence:Q', scale=alt.Scale(domain=[-1, 1]),
                             title='Valence (부정 ← → 긍정)'),
                    y=alt.Y('Arousal:Q', scale=alt.Scale(domain=[-1, 1]),
                             title='Arousal (이완 ← → 각성)'),
                    color=alt.Color('Emotion:N',
                        scale=alt.Scale(domain=list(CLASS_COLORS.keys()),
                                        range=list(CLASS_COLORS.values()))),
                    order=alt.Order('Frame:Q')
                ).properties(height=200, title="VA Circumplex (최근 30프레임)")

                h_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
                    color='gray', opacity=0.3).encode(y='y:Q')
                v_rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                    color='gray', opacity=0.3).encode(x='x:Q')

                circumplex_area.altair_chart(
                    (chart_circumplex + h_rule + v_rule).interactive(),
                    use_container_width=True)

    if cap:
        cap.release()