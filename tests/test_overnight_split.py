"""tools/overnight_split.py(2026-11-16)의 날짜 맞춤 규칙 시험: 한국 날짜 D 에는 달력 날짜가 D 보다 앞선 마지막 미국 거래일의 값,
전 한국 거래일과 같은 미국 날이면(미국 휴장) 새 정보가 없으므로 NaN. 조각 정의도 확인."""
import numpy as np
import pandas as pd

from tools.overnight_split import segments, us_before_open


def test_us_before_open_strictly_earlier_and_holiday():
    us = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))  # 1/4(목) 미국 휴장 가정
    kr = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    x = us_before_open(kr, us)
    # 1/3 한국 시가 전 끝난 미국 장 = 1/2 · 1/4 = 1/3 · 1/5 = 1/3 과 같음(1/4 휴장) → NaN · 1/8 = 1/5
    assert x.iloc[0] == 0.01 and x.iloc[1] == 0.02 and np.isnan(x.iloc[2]) and x.iloc[3] == 0.03


def test_segments_definition():
    px = pd.DataFrame({"open": [100.0, 102.0, 99.0], "close": [101.0, 100.0, 98.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    s = segments(px)
    assert np.isclose(s["gap"].iloc[1], 102 / 101 - 1) and np.isclose(s["intra"].iloc[1], 100 / 102 - 1)
    assert np.isclose(s["next"].iloc[1], 99 / 100 - 1) and np.isclose(s["engine"].iloc[1], 99 / 102 - 1)
    assert np.isclose((1 + s["intra"].iloc[1]) * (1 + s["next"].iloc[1]) - 1, s["engine"].iloc[1])   # 엔진 정답 = 장중 × 다음 밤
