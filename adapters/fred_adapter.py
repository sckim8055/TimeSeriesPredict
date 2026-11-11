"""FRED (미국 세인트루이스 연준). 키: FRED_API_KEY. 예) DGS10(미 10년물, 일), VIXCLS(VIX, 일), FEDFUNDS(월).
주의: 작업 환경에서 미검증."""
from __future__ import annotations

import os

import pandas as pd
import requests

from adapters.base import RawCache

NAME = "FRED"
URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, start: str, end: str | None = None, *, api_key: str | None = None, cache: RawCache | None = None) -> pd.Series:
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY 가 없습니다 (.env)")
    params = {"series_id": series_id, "api_key": key, "file_type": "json", "observation_start": start}
    if end:
        params["observation_end"] = end
    js = requests.get(URL, params=params, timeout=60).json()
    if cache:
        cache.save(series_id, js)
    obs = js.get("observations")
    if not obs:
        raise RuntimeError(f"FRED 응답 없음: {js}")
    df = pd.DataFrame(obs)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"])).dropna()
    return s
