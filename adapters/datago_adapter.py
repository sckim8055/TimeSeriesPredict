"""공공데이터포털 — 금융위원회 증권상품시세정보 (ETF 일별 시세). 공식·무료(키 발급 필요).
API(V2, 2026-11 확인): https://apis.data.go.kr/1160100/GetSecuritiesProductInfoService_V2/getETFPriceInfo_V2
구버전 경로(/service/GetSecuritiesProductInfoService)는 403 을 돌려준다.
응답 항목(기준일 basDt): mkp(시가) hipr(고가) lopr(저가) clpr(종가) trqu(거래량) nav vs(발행주식수) srtnCd(단축코드) itmsNm(종목명)
주의: 이 어댑터는 작업 환경에서 네트워크가 막혀 미검증. 노트북에서 collect.py 로 첫 실행 시 응답 항목 이름을 확인한다."""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from adapters.base import RawCache, to_price_frame

NAME = "DATAGO"
URL = os.environ.get("DATAGO_ETF_URL", "https://apis.data.go.kr/1160100/GetSecuritiesProductInfoService_V2/getETFPriceInfo_V2")


def _get(params: dict) -> dict:
    """1회 호출 + 재시도. 연결 오류·HTTP 5xx·포털 SERVICETIMEOUT(코드 05) 는 5s/10s/20s 간격으로 다시 시도."""
    last = ""
    for attempt_no in range(4):
        try:
            r = requests.get(URL, params=params, timeout=90)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = f"{e.__class__.__name__}"
            r = None
        if r is not None:
            if r.status_code == 200:
                try:
                    js = r.json()
                except ValueError:
                    raise RuntimeError(f"DATAGO 응답이 JSON 이 아님(키/엔드포인트 확인): {r.text[:300]}")
                hdr = js.get("response", {}).get("header", {})
                if str(hdr.get("resultCode", "00")) not in ("0", "00"):
                    raise RuntimeError(f"DATAGO 오류 {hdr.get('resultCode')}: {hdr.get('resultMsg')}")
                return js
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code < 500 and '"05"' not in r.text:
                raise RuntimeError(f"DATAGO {last} (URL={URL})")
        if attempt_no < 3:
            time.sleep(5 * 2**attempt_no)
    raise RuntimeError(f"DATAGO 4회 실패 (URL={URL}): {last}")


def fetch_prices(ticker: str, start: str, end: str | None = None, *, api_key: str | None = None, cache: RawCache | None = None) -> pd.DataFrame:
    key = api_key or os.environ.get("DATAGO_API_KEY")
    if not key:
        raise RuntimeError("DATAGO_API_KEY 가 없습니다 (.env)")
    # 포털이 긴 기간 조회에서 SERVICETIMEOUT(504) 을 내므로 연 단위로 쪼개 받는다.
    t0 = pd.Timestamp(start)
    t_end = pd.Timestamp(end) if end else pd.Timestamp.today()
    rows: list[dict] = []
    y = t0
    while y <= t_end:
        y_end = min(pd.Timestamp(year=y.year, month=12, day=31), t_end)
        begin, stop = y.strftime("%Y%m%d"), (y_end + pd.Timedelta(days=1)).strftime("%Y%m%d")  # endBasDt 는 끝 날짜 미포함
        page = 1
        while True:
            params = {"serviceKey": key, "resultType": "json", "numOfRows": 500, "pageNo": page,
                      "likeSrtnCd": ticker, "beginBasDt": begin, "endBasDt": stop}
            js = _get(params)
            if cache:
                cache.save(f"etf_{ticker}_{y.year}_p{page}", js)
            body = js.get("response", {}).get("body", {})
            items = body.get("items", {}).get("item", []) if isinstance(body.get("items"), dict) else []
            if not items:
                break
            rows.extend(items)
            if page * 500 >= int(body.get("totalCount", 0)):
                break
            page += 1
            time.sleep(0.2)
        y = pd.Timestamp(year=y.year + 1, month=1, day=1)
        time.sleep(0.2)
    if not rows:
        raise RuntimeError(f"DATAGO 응답 없음: {ticker} {start}~{t_end.date()}")
    df = pd.DataFrame(rows)
    df = df[df["srtnCd"].astype(str).str.zfill(6) == ticker.zfill(6)]
    df["date"] = pd.to_datetime(df["basDt"], format="%Y%m%d")
    missing = [c for c in ("mkp", "hipr", "lopr", "clpr", "trqu") if c not in df.columns]
    if missing:
        raise RuntimeError(f"DATAGO 응답에 열이 없음 {missing}. 받은 열: {sorted(df.columns)[:30]}")
    for c in ("mkp", "hipr", "lopr", "clpr", "trqu"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return to_price_frame(df, {"mkp": "open", "hipr": "high", "lopr": "low", "clpr": "close", "trqu": "volume"}, "date")
