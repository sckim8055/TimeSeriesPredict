"""pykrx. 주의: adjusted=True(기본)는 네이버 차트 API, adjusted=False 는 KRX(계정 KRX_ID/KRX_PW 필요).
본인용 연구 단계에서만 사용. 공개 서비스 단계에서는 datago/KRX 계정 경로로."""
from __future__ import annotations

import pandas as pd

from adapters.base import RawCache, to_price_frame

NAME = "KRX"


def fetch_prices(ticker: str, start: str, end: str | None = None, *, adjusted: bool = True, cache: RawCache | None = None) -> pd.DataFrame:
    from pykrx import stock

    e = (end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    df = stock.get_market_ohlcv_by_date(start.replace("-", ""), e, ticker, adjusted=adjusted)
    if df is None or df.empty:
        raise RuntimeError(f"pykrx 응답 없음: {ticker}")
    if cache:
        cache.save(f"prices_{ticker}_adj{int(adjusted)}", df.reset_index())
    df = df.reset_index().rename(columns={"날짜": "date", "시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
    return to_price_frame(df, {}, "date")
