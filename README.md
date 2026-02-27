# EmoUI — 공연 관람 감정 분석 시스템

공연 영상에서 관객의 얼굴 감정을 분석하고, 영상 시나리오 감정선과의 동기화 점수를 측정하는 시스템입니다.

---

## 프로젝트 구조

```
EmoUI/
├── models/
│   ├── mobilenet_7.h5       # 감정 분류 모델 (MobileNet, 7클래스)
│   └── mtcnn.pb             # 얼굴 검출 모델 (MTCNN)
└── src/
    ├── upload.py            # Streamlit 업로드 페이지 (데모 입구)
    ├── analyze.py           # 영상 분석 → CSV + 오버레이 영상 저장
    ├── emotion_engine.py    # 감정 분석 핵심 로직 (단독 실행 안 함)
    ├── export_json.py       # CSV → viewer용 JSON 변환
    ├── synchronizer.py      # 동기화 점수 계산 로직 (단독 실행 안 함)
    ├── viewer.html          # 메인 시각화 뷰어 (영상 + 실시간 그래프)
    ├── scenario.csv         # 영상 시나리오 감정선 정의
    └── results/             # 분석 결과 저장 (실행 후 자동 생성)
```

---

## 실행 흐름

```
upload.py (영상 업로드)
    └── analyze.py          ← AI 연산 (GPU 사용)
        └── emotion_engine.py
    └── export_json.py      ← CSV → JSON 변환
        └── synchronizer.py
            └── viewer.html ← 브라우저에서 결과 확인 (AI 연산 없음)
```

---

## 환경 세팅

### 요구 사항

- CUDA 11.2 이상 (GPU 사용 권장, CPU도 동작하나 속도 느림)

### conda 환경 생성

```bash
conda env create -f fer2026.yaml
conda activate emoui
```

### nginx 설치 (viewer 서빙용, 최초 1회)

```bash
sudo apt install nginx -y
sudo bash -c 'cat > /etc/nginx/sites-available/emoui << EOF
server {
    listen 8888;
    root /절대경로/EmoUI/src;
    index viewer.html;
}
EOF'
sudo ln -s /etc/nginx/sites-available/emoui /etc/nginx/sites-enabled/emoui
sudo nginx -t && sudo systemctl restart nginx
```

> `root` 경로는 본인 환경에 맞게 수정하세요.  
> 예: `/home/username/EmoUI/src`

---

## 실행 방법

### 1. Streamlit 업로드 페이지로 실행 (권장)

```bash
cd EmoUI/src
conda activate emoui
python -m streamlit run upload.py --server.port 8501
```

브라우저에서 `http://서버IP:8501` 접속 후:
1. 분석할 영상(mp4/avi/mov) 업로드
2. 시나리오 CSV 업로드 (선택)
3. 분석 시작 버튼 클릭
4. 완료 후 viewer 링크 클릭 → `http://서버IP:8888/viewer.html`

### 2. 터미널에서 직접 실행

```bash
cd EmoUI/src
conda activate emoui

# 1단계: 영상 분석
python analyze.py --video ./DataSets/PlayVideos/sample.mp4

# 2단계: JSON 변환
python export_json.py

# 3단계: 브라우저에서 viewer.html 접속
# http://서버IP:8888/viewer.html
```

---

## 시나리오 CSV 작성 방법

`src/scenario.csv` 파일을 편집해서 영상의 시간대별 기대 감정선을 정의합니다.

```csv
time_sec,valence,arousal,description
0,0.1,0.0,공연 시작
10,0.3,0.2,등장인물 소개
25,0.7,0.6,신나는 장면
45,0.8,0.8,클라이맥스
54,0.2,-0.1,잠깐의 휴식
61,0.9,0.7,해피엔딩
```

| 컬럼 | 설명 |
|------|------|
| time_sec | 구간 시작 시간 (초) |
| valence | 기대 감정 긍정도 (-1.0 ~ 1.0) |
| arousal | 기대 감정 각성도 (-1.0 ~ 1.0) |
| description | 구간 이름 (viewer에 표시됨) |

---

## analyze.py 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--video` | sample.mp4 | 분석할 영상 경로 |
| `--skip` | 3 | N프레임마다 분석 (클수록 빠르지만 정확도 감소) |
| `--alpha` | 0.25 | EMA 스무딩 강도 (낮을수록 부드러움) |
| `--hs_model` | enet_b0_8_va_mtl | HSEmotion 모델 선택 |

```bash
# 예시: 5프레임마다 분석, 스무딩 강하게
python analyze.py --video ./DataSets/PlayVideos/sample.mp4 --skip 5 --alpha 0.1
```

---

## 주요 기술

| 역할 | 모델/라이브러리 |
|------|----------------|
| 얼굴 검출 | MTCNN |
| 감정 분류 (7클래스) | MobileNet (`mobilenet_7.h5`) |
| Valence/Arousal 측정 | HSEmotion (`enet_b0_8_va_mtl`) |
| 얼굴 추적 | CentroidTracker (centroid 거리 기반) |
| 시각화 | Chart.js + HTML/JS |
| 서버 | nginx (viewer), Streamlit (업로드) |
