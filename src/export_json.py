"""
export_json.py
--------------
viewer.html에서 사용할 JSON 데이터 파일 생성.
analyze.py 실행 후 실행하면 됨.

실행 방법:
    python export_json.py
    python export_json.py --result results/analysis_result.csv
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def export(result_csv: str = None,
           scenario_csv: str = None,
           output_json: str = None) -> str:
    """
    result_csv, scenario_csv, output_json 을 인자로 받아 동적으로 경로 처리.
    Streamlit 등 외부에서 호출 시 경로 직접 지정 가능.
    반환: 저장된 JSON 경로
    """
    result_csv   = Path(result_csv   or BASE_DIR / "results" / "analysis_result.csv")
    scenario_csv = Path(scenario_csv or BASE_DIR / "scenario.csv")
    output_json  = Path(output_json  or BASE_DIR / "results" / "viewer_data.json")
    meta_csv     = result_csv.with_suffix('.meta.csv')

    if not result_csv.exists():
        raise FileNotFoundError(
            f"분석 결과 없음: {result_csv}\n"
            "먼저 python analyze.py 를 실행해주세요.")

    result_df   = pd.read_csv(result_csv)
    meta        = pd.read_csv(meta_csv).iloc[0].to_dict() if meta_csv.exists() else {}
    scenario_df = pd.read_csv(scenario_csv) if scenario_csv.exists() else pd.DataFrame()

    fps          = float(meta.get('fps', 30.0))
    total_frames = int(meta.get('total_frames', result_df['frame'].max()))
    duration_sec = float(meta.get('duration_sec', total_frames / fps))

    # ── 프레임별 집계 ────────────────────────────────────────────
    frames_data: dict = {}
    has_bbox = 'bbox_area' in result_df.columns

    for _, row in result_df.iterrows():
        fid = int(row['frame'])
        if fid not in frames_data:
            frames_data[fid] = {'persons': [], 'time_sec': float(row['time_sec'])}
        frames_data[fid]['persons'].append({
            'id':        str(row['id']),
            'emotion':   str(row['emotion']),
            'valence':   float(row['valence']),
            'arousal':   float(row['arousal']),
            'bbox_area': float(row['bbox_area']) if has_bbox else 1.0,
        })

    for fd in frames_data.values():
        persons    = fd['persons']
        total_area = sum(p['bbox_area'] for p in persons)
        if total_area > 1e-6:
            fd['avg_valence'] = sum(p['valence'] * p['bbox_area'] for p in persons) / total_area
            fd['avg_arousal'] = sum(p['arousal'] * p['bbox_area'] for p in persons) / total_area
        else:
            fd['avg_valence'] = sum(p['valence'] for p in persons) / len(persons)
            fd['avg_arousal'] = sum(p['arousal'] for p in persons) / len(persons)

    # ── 시나리오 보간 ────────────────────────────────────────────
    scenario_frames = []
    if not scenario_df.empty:
        scn_frame_idxs = (scenario_df['time_sec'] * fps).astype(int).values
        all_frames     = list(range(total_frames))
        v_interp = np.interp(all_frames, scn_frame_idxs, scenario_df['valence'].values).tolist()
        a_interp = np.interp(all_frames, scn_frame_idxs, scenario_df['arousal'].values).tolist()
        scenario_frames = [
            {'frame': f, 'time_sec': round(f / fps, 2),
             'valence': round(v, 4), 'arousal': round(a, 4)}
            for f, v, a in zip(all_frames, v_interp, a_interp)
        ]

    # ── 시나리오 구간 ────────────────────────────────────────────
    sections = []
    if not scenario_df.empty and 'description' in scenario_df.columns:
        for i in range(len(scenario_df) - 1):
            sections.append({
                'description': str(scenario_df.iloc[i]['description']),
                'time_start':  float(scenario_df.iloc[i]['time_sec']),
                'time_end':    float(scenario_df.iloc[i + 1]['time_sec']),
            })

    # ── JSON 저장 ────────────────────────────────────────────────
    output_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'meta': {'fps': fps, 'total_frames': total_frames, 'duration_sec': duration_sec},
        'frames':   {str(k): v for k, v in frames_data.items()},
        'scenario': scenario_frames,
        'sections': sections,
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"JSON 저장 완료: {output_json}")
    print(f"프레임 수: {len(frames_data):,} | 시나리오 구간: {len(sections)}개")
    return str(output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result",   type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output",   type=str, default=None)
    args = parser.parse_args()
    export(args.result, args.scenario, args.output)