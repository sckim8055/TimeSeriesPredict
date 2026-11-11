"""한국은행 ECOS Open API. 통계코드/항목코드로 시계열 1개를 받는다. 키: ECOS_API_KEY.
예) 기준금리 722Y001/0101000 (월), 원달러 731Y001/0000001 (일). 주기: D/M/Q/A.
주의: 작업 환경에서 미검증."""
from __future__ import annotations

import os

import pandas as pd
import requests

from adapters.base import RawCache

NAME = "ECOS"
URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/{stat}/{freq}/{start}/{end}/{item}"


def fetch_series(stat_code: str, item_code: str, freq: str, start: str, end: str, *, api_key: str | None = None, cache: RawCache | None = None) -> pd.Series:
    key = api_key or os.environ.get("ECOS_API_KEY")
    if not key:
        raise RuntimeError("ECOS_API_KEY 가 없습니다 (.env)")
    fmt = {"D": "%Y%m%d", "M": "%Y%m", "Q": "%YQ%q", "A": "%Y"}
    url = URL.format(key=key, stat=stat_code, freq=freq, start=start, end=end, item=item_code)
    js = requests.get(url, timeout=60).json()
    if cache:
        cache.save(f"{stat_code}_{item_code}_{freq}", js)
    rows = js.get("StatisticSearch", {}).get("row")
    if not rows:
        raise RuntimeError(f"ECOS 응답 없음: {js.get('RESULT', js)}")
    df = pd.DataFrame(rows)
    t = df["TIME"].astype(str)
    if freq == "Q":
        date = pd.PeriodIndex(t.str.replace("Q", "Q"), freq="Q").to_timestamp()
    elif freq == "M":
        date = pd.to_datetime(t, format="%Y%m")
    elif freq == "A":
        date = pd.to_datetime(t, format="%Y")
    else:
        date = pd.to_datetime(t, format="%Y%m%d")
    s = pd.Series(pd.to_numeric(df["DATA_VALUE"], errors="coerce").values, index=date).dropna().sort_index()
    return s
