"""Yahoo Finance (yfinance). 비공식·무료. 국내 ETF는 '091160.KS'. 교차검증·미국 지표용.
수정종가(Adj Close)가 아닌 원 OHLC를 쓴다 — 수익률 계산은 엔진이 open 기준으로 하므로 배당 보정은 별도 변수로 처리."""
from __future__ import annotations

import pandas as pd

from adapters.base import RawCache, to_price_frame

NAME = "YF"


def fetch_prices(ticker: str, start: str, end: str | None = None, *, cache: RawCache | None = None) -> pd.DataFrame:
    import yfinance as yf

    sym = ticker if "." in ticker or not ticker.isdigit() else f"{ticker}.KS"
    df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False)
    if df is None or df.empty:
        raise RuntimeError(f"YF 응답 없음: {sym}")
    if cache:
        cache.save(f"prices_{sym}", df.reset_index())
    df = df.reset_index()
    return to_price_frame(df, {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, "date")


def fetch_series(symbol: str, start: str, end: str | None = None, *, field: str = "Close", cache: RawCache | None = None) -> pd.Series:
    """지수·종목 종가 1개를 변수로 (예: ^SOX, NVDA, KRW=X)."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False)
    if df is None or df.empty:
        raise RuntimeError(f"YF 응답 없음: {symbol}")
    if cache:
        cache.save(f"series_{symbol}", df.reset_index())
    s = df[field].astype(float)
    s.index = pd.to_datetime(s.index)
    return s
