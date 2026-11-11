"""KRX 정보데이터시스템 — 투자자별 거래실적(수급). pykrx 경유, KRX 계정 필요(.env: KRX_ID, KRX_PW → pykrx 가 자동 로그인).

마스터 source=KRX, source_code 형식:
  flow:<티커>:<투자자>[:<값>]   투자자 = 외국인합계 | 기관합계 | 개인 | 기타법인 ,  값 = 순매수(기본) | 매수 | 매도
  예) flow:091160:외국인합계   → 091160 의 외국인 일별 순매수 거래대금(원)
공개 시점: KRX 는 당일 장 마감 후(18시 전후) 집계를 올린다 → 결정 시점 't+1 개장 직전' 기준 lag_days=0.
주의: 수급은 가격과 동시에 결정되는 값이다. t 의 수급으로 t+1 을 예측하는 것은 가능하나, 같은 날 가격을 설명하는 데 쓰면 동시성 문제가 있다.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from adapters.base import RawCache

NAME = "KRX"
INVESTORS = ("외국인합계", "기관합계", "개인", "기타법인")


def _check_login() -> None:
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        raise RuntimeError("KRX_ID / KRX_PW 가 .env 에 없습니다 (KRX 정보데이터시스템 계정)")


_YEAR_CACHE = None  # data/raw/krx/flow_<ticker>_<on>_<year>.parquet — 완결된 해(작년 이전)는 다시 요청하지 않는다


def _cache_dir(cache: RawCache | None) -> Path | None:
    return cache.dir if cache is not None else None


def _krx_call(fn, *args, pause: float = 1.0):
    """KRX 호출 + 차단 대응. HTML(JSON 아님) 응답이면 30s/60s/120s 쉬었다 재시도. 요청 사이 1초."""
    import json as _json

    last = None
    for i, wait in enumerate((0, 30, 60, 120)):
        if wait:
            print(f"      KRX 응답 오류({last}) → {wait}초 대기 후 재시도", flush=True)
            time.sleep(wait)
        try:
            df = fn(*args)
            time.sleep(pause)
            return df
        except (_json.JSONDecodeError, ValueError) as e:  # requests.JSONDecodeError 도 ValueError 계열
            last = e.__class__.__name__
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "Expecting value" in msg or "JSON" in msg:
                last = e.__class__.__name__
            else:
                raise
    raise RuntimeError(f"KRX 가 4회 연속 JSON 이 아닌 응답을 보냈습니다(요청 과다 차단 가능성). 10~30분 뒤 다시 시도하세요. 마지막 오류: {last}")


def fetch_flow(ticker: str, investor: str, start: str, end: str | None = None, *, on: str = "순매수", cache: RawCache | None = None) -> pd.Series:
    """일별 투자자별 거래대금(원). 반환 Series(index=날짜). 연 단위로 받고, 끝난 해는 파일 캐시를 재사용한다."""
    _check_login()
    from pykrx import stock

    if investor not in INVESTORS:
        raise ValueError(f"투자자 구분은 {INVESTORS} 중 하나")
    s0 = start.replace("-", "")
    e0 = (end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    this_year = pd.Timestamp.today().year
    cdir = _cache_dir(cache)
    # 주식과 ETF 는 KRX 화면(엔드포인트)이 다르다. 주식 종목표에 있으면 주식 경로, 없으면 ETF 경로.
    is_etf = False
    try:
        if not stock.get_market_ticker_name(ticker):
            is_etf = True
    except Exception:  # noqa: BLE001
        is_etf = True
    frames = []
    y0, y1 = int(s0[:4]), int(e0[:4])
    for y in range(y0, y1 + 1):
        a = s0 if y == y0 else f"{y}0101"
        b = e0 if y == y1 else f"{y}1231"
        cfile = (cdir / f"flow_{ticker}_{on}_{y}.parquet") if cdir is not None else None
        if cfile is not None and cfile.exists() and y < this_year:
            frames.append(pd.read_parquet(cfile))
            continue
        if is_etf:
            df = _krx_call(stock.get_etf_trading_volume_and_value, a, b, ticker, "거래대금", on)
            if df is not None and not df.empty:
                df = df.rename(columns={"외국인": "외국인합계", "기관": "기관합계"})
        else:
            df = _krx_call(lambda a_, b_, t_: stock.get_market_trading_value_by_date(a_, b_, t_, on=on), a, b, ticker)
        if df is not None and not df.empty:
            df.index = pd.DatetimeIndex(df.index)
            if cfile is not None:
                df.to_parquet(cfile)
            frames.append(df)
    if not frames:
        raise RuntimeError(f"KRX 수급 응답 없음: {ticker} {s0}~{e0} ({'ETF' if is_etf else '주식'} 경로)")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if investor not in df.columns:
        raise RuntimeError(f"KRX 응답에 '{investor}' 열이 없음. 받은 열: {list(df.columns)}")
    s = df[investor].astype(float)
    s.index = pd.DatetimeIndex(s.index).normalize()
    s.name = f"{ticker}:{investor}"
    return s


def fetch_series(source_code: str, start: str, end: str | None = None, *, cache: RawCache | None = None) -> pd.Series:
    """마스터 source_code 해석: flow:<티커>:<투자자>[:<값>]"""
    parts = source_code.split(":")
    if len(parts) < 3 or parts[0] != "flow":
        raise ValueError(f"KRX source_code 형식 오류: {source_code} (flow:티커:투자자[:순매수|매수|매도])")
    on = parts[3] if len(parts) > 3 else "순매수"
    return fetch_flow(parts[1], parts[2], start, end, on=on, cache=cache)
