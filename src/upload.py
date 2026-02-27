"""
upload.py
---------
Streamlit 업로드 데모 페이지.
영상 + 시나리오 CSV 업로드 → 분석 → viewer.html 링크 제공.

실행 방법:
    cd EmoUI/src
    python -m streamlit run upload.py --server.port 8501
"""

import os
import shutil
import tempfile
import streamlit as st
from pathlib import Path

# analyze, export_json은 같은 디렉토리에 있어야 함
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import analyze_video
from export_json import export

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# nginx로 viewer 서빙 중인 경우 포트 설정
VIEWER_PORT = 8888

st.set_page_config(page_title="IVPL Emotion Analysis", layout="centered")
st.title("🎭 IVPL Emotion Analysis")
st.caption("영상을 업로드하면 감정 분석 후 결과를 확인할 수 있습니다.")

# ── 파일 업로드 ────────────────────────────────────────────────────
st.subheader("1️⃣ 파일 업로드")

col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("분석할 영상 (mp4, avi, mov)", type=["mp4", "avi", "mov"])
with col2:
    scenario_file = st.file_uploader("시나리오 CSV (선택)", type=["csv"])

# ── 설정 ───────────────────────────────────────────────────────────
with st.expander("⚙️ 분석 설정"):
    skip  = st.slider("프레임 건너뛰기 (skip)", 1, 10, 3,
                      help="클수록 빠르지만 정확도 감소. 기본값 3 권장.")
    alpha = st.slider("EMA 스무딩 강도 (alpha)", 0.1, 1.0, 0.25, 0.05,
                      help="낮을수록 부드럽게 변화. 높을수록 즉각 반응.")

# ── 분석 실행 ──────────────────────────────────────────────────────
st.subheader("2️⃣ 분석 실행")

if st.button("▶ 분석 시작", type="primary", disabled=(video_file is None)):
    if video_file is None:
        st.error("영상 파일을 업로드해주세요.")
        st.stop()

    progress_bar = st.progress(0)
    status_text  = st.empty()

    def on_progress(ratio: float, msg: str):
        progress_bar.progress(min(ratio, 1.0))
        status_text.text(msg)

    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(video_file.name).suffix) as tmp_video:
            tmp_video.write(video_file.read())
            tmp_video_path = tmp_video.name

        # 시나리오 저장
        scenario_path = str(BASE_DIR / "scenario.csv")
        if scenario_file:
            with open(scenario_path, 'wb') as f:
                f.write(scenario_file.read())

        output_csv = str(RESULTS_DIR / "analysis_result.csv")

        # 분석 실행
        result = analyze_video(
            video_path=tmp_video_path,
            output_path=output_csv,
            skip=skip,
            ema_alpha=alpha,
            progress_callback=on_progress,
        )

        # JSON 변환
        status_text.text("JSON 변환 중...")
        export(
            result_csv=output_csv,
            scenario_csv=scenario_path,
            output_json=str(RESULTS_DIR / "viewer_data.json"),
        )

        progress_bar.progress(1.0)
        status_text.text("완료!")

        # 임시 파일 삭제
        os.unlink(tmp_video_path)

        # 결과 표시
        st.success(f"✅ 분석 완료! 감지 인물: {', '.join(result['persons'])} | 총 {result['total_records']:,}개 레코드")

        # viewer 링크 — 서버 IP 자동 감지
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()
        except Exception:
            host = "서버IP"

        viewer_url = f"http://{host}:{VIEWER_PORT}/viewer.html"
        st.subheader("3️⃣ 결과 확인")
        st.markdown(f"### 👉 [결과 보기 (viewer)]({viewer_url})")
        st.code(viewer_url)
        st.caption("위 링크를 클릭하거나 주소를 브라우저에 직접 입력하세요.")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        progress_bar.empty()
        status_text.empty()

else:
    if video_file is None:
        st.info("영상 파일을 업로드하면 분석 버튼이 활성화됩니다.")