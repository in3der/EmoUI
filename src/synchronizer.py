"""
synchronizer.py
---------------
영상 시나리오 감정선 ↔ 사람 감정선 동기화 분석 모듈.
"""

import numpy as np
import pandas as pd


def load_scenario(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        for col in ['time_sec', 'valence', 'arousal']:
            if col not in df.columns:
                raise ValueError(f"시나리오 CSV에 '{col}' 컬럼이 없습니다.")
        if 'description' not in df.columns:
            df['description'] = ''
        return df.sort_values('time_sec').reset_index(drop=True)
    except Exception as e:
        print(f"[시나리오 로드 실패] {e}")
        return pd.DataFrame(columns=['time_sec', 'valence', 'arousal', 'description'])


def interpolate_scenario(scenario_df: pd.DataFrame,
                          fps: float,
                          total_frames: int) -> pd.DataFrame:
    """
    시나리오 구간을 선형 보간해 프레임 단위 VA값 생성.
    시나리오 마지막 시간 이후는 마지막 VA값으로 고정 (np.interp 기본 동작).
    """
    if scenario_df.empty:
        return pd.DataFrame(columns=['frame', 'valence', 'arousal'])

    scn        = scenario_df.copy()
    scn['frame'] = (scn['time_sec'] * fps).astype(int)
    all_frames = np.arange(total_frames)

    return pd.DataFrame({
        'frame':   all_frames,
        'valence': np.interp(all_frames, scn['frame'].values, scn['valence'].values),
        'arousal': np.interp(all_frames, scn['frame'].values, scn['arousal'].values),
    })


def aggregate_audience(audience_df: pd.DataFrame) -> pd.DataFrame:
    """
    프레임별 사람 VA를 bbox_area 기반 가중 평균으로 집계.
    bbox_area 없으면 단순 평균 (하위 호환).
    반환: DataFrame (Frame, Valence, Arousal)

    컬럼명 대소문자 혼재 방어: 실제 컬럼 이름을 그대로 사용하고
    반환 시에만 표준 이름(Frame, Valence, Arousal)으로 rename.
    """
    if audience_df.empty:
        return pd.DataFrame(columns=['Frame', 'Valence', 'Arousal'])

    # 실제 컬럼 이름 탐색 (대소문자 무관)
    lower_map = {c.lower(): c for c in audience_df.columns}
    frame_col   = lower_map.get('frame')
    valence_col = lower_map.get('valence')
    arousal_col = lower_map.get('arousal')

    if not all([frame_col, valence_col, arousal_col]):
        return pd.DataFrame(columns=['Frame', 'Valence', 'Arousal'])

    if 'bbox_area' not in audience_df.columns:
        return (audience_df
                .groupby(frame_col)[[valence_col, arousal_col]]
                .mean()
                .reset_index()
                .rename(columns={frame_col: 'Frame',
                                  valence_col: 'Valence',
                                  arousal_col: 'Arousal'}))

    records = []
    for frame_val, group in audience_df.groupby(frame_col):
        w     = group['bbox_area'].values.astype(float)
        w_sum = w.sum()
        if w_sum < 1e-6:
            v = float(group[valence_col].mean())
            a = float(group[arousal_col].mean())
        else:
            v = float(np.dot(group[valence_col].values, w) / w_sum)
            a = float(np.dot(group[arousal_col].values, w) / w_sum)
        records.append({'Frame': frame_val, 'Valence': v, 'Arousal': a})

    return pd.DataFrame(records)


def compute_sync_score(scenario_interp: pd.DataFrame,
                        audience_df: pd.DataFrame,
                        window_size: int = 90) -> float:
    """
    유클리드 거리 기반 동기화 점수.
    VA 공간 최대 거리(2√2)로 정규화 후 [0,1] 반환.
    """
    if scenario_interp.empty or audience_df.empty:
        return 0.0

    aud_agg = aggregate_audience(audience_df)
    if aud_agg.empty or len(aud_agg) < 3:
        return 0.0

    recent    = aud_agg.tail(window_size)
    frame_min = int(recent['Frame'].min())
    frame_max = int(recent['Frame'].max())

    scn_window = scenario_interp[
        (scenario_interp['frame'] >= frame_min) &
        (scenario_interp['frame'] <= frame_max)
    ]
    if scn_window.empty:
        return 0.0

    aud_v = float(recent['Valence'].mean())
    aud_a = float(recent['Arousal'].mean())
    scn_v = float(scn_window['valence'].mean())
    scn_a = float(scn_window['arousal'].mean())

    MAX_DIST = 2.0 * np.sqrt(2.0)
    dist     = np.sqrt((aud_v - scn_v) ** 2 + (aud_a - scn_a) ** 2)
    return float(1.0 - min(dist / MAX_DIST, 1.0))


def compute_section_scores(scenario_df: pd.DataFrame,
                            scenario_interp: pd.DataFrame,
                            audience_df: pd.DataFrame,
                            fps: float,
                            window_size: int = 90) -> pd.DataFrame:
    if scenario_df.empty or audience_df.empty:
        return pd.DataFrame()

    # frame 컬럼명을 루프 밖에서 한 번만 결정
    frame_col = 'Frame' if 'Frame' in audience_df.columns else 'frame'

    records = []
    for i in range(len(scenario_df) - 1):
        row     = scenario_df.iloc[i]
        t_end   = float(scenario_df.iloc[i + 1]['time_sec'])
        f_start = int(row['time_sec'] * fps)
        f_end   = int(t_end * fps)

        section_aud = audience_df[
            (audience_df[frame_col] >= f_start) &
            (audience_df[frame_col] <  f_end)
        ]
        if len(section_aud) < 3:
            continue

        score = compute_sync_score(
            scenario_interp, section_aud,
            window_size=min(window_size, len(section_aud))
        )
        records.append({
            '구간':      str(row['description']),
            '시작(초)':  float(row['time_sec']),
            '종료(초)':  t_end,
            '동기화(%)': int(score * 100),
        })

    return pd.DataFrame(records)


def get_comparison_data(scenario_interp: pd.DataFrame,
                         audience_df: pd.DataFrame) -> pd.DataFrame:
    """
    영상 감정선 + 사람 감정선(가중 평균)을 하나의 DataFrame으로 반환.
    source = '영상' | '사람'
    """
    if scenario_interp.empty or audience_df.empty:
        return pd.DataFrame()

    aud_agg = aggregate_audience(audience_df)
    if aud_agg.empty:
        return pd.DataFrame()

    frame_min = int(aud_agg['Frame'].min())
    frame_max = int(aud_agg['Frame'].max())

    scn_window = scenario_interp[
        (scenario_interp['frame'] >= frame_min) &
        (scenario_interp['frame'] <= frame_max)
    ].copy()

    if scn_window.empty:
        return pd.DataFrame()

    scn_df = pd.DataFrame({
        'frame':   scn_window['frame'].values,
        'source':  '영상',
        'valence': scn_window['valence'].values,
        'arousal': scn_window['arousal'].values,
    })
    aud_df = pd.DataFrame({
        'frame':   aud_agg['Frame'].values,
        'source':  '사람',
        'valence': aud_agg['Valence'].values,
        'arousal': aud_agg['Arousal'].values,
    })

    return pd.concat([scn_df, aud_df], ignore_index=True)