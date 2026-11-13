"""합성 데이터 생성기 — 정답을 아는 가상 시세. 엔진·모델 검증용 (실데이터 아님)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def random_walk_ohlcv(n: int = 1500, seed: int = 0, mu: float = 0.0003, sigma: float = 0.015, start_price: float = 10000.0) -> pd.DataFrame:
    """로그 랜덤워크. 시가 = 전일 종가 × (1 + 작은 갭). 예측 불가능해야 정상인 데이터."""
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    close = start_price * np.exp(np.cumsum(r))
    gap = rng.normal(0, sigma / 4, n)
    open_ = np.r_[start_price, close[:-1]] * (1 + gap)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma / 3, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma / 3, n)))
    vol = rng.lognormal(12, 0.3, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=trading_days(n))


def predictable_ohlcv(n: int = 1500, seed: int = 0, sigma: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
    """'다음 수익률'이 숨은 신호 s_t 로 결정되는 시장. s_t 를 변수로 주면 모델이 맞혀야 한다.
    r_{t+1} = 0.02 * s_t + noise,  s_t ~ AR(1)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.7 * s[t - 1] + rng.normal(0, 1)
    # 결정일 t 의 타겟 = open[t+2]/open[t+1]-1 (t+1 시가 진입, t+2 시가 청산). 그 수익률을 s[t]가 결정하도록 심는다.
    r_open = 0.02 * np.r_[0, 0, s[:-2]] + rng.normal(0, sigma, n)  # open[t+2]/open[t+1]-1 = 0.02*s[t] + noise
    open_ = 10000 * np.exp(np.cumsum(r_open))
    close = open_ * (1 + rng.normal(0, sigma / 2, n))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.lognormal(12, 0.3, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=trading_days(n))
    return df, pd.Series(s, index=df.index, name="S_signal")


def sine_ohlcv(n: int = 1500, period: int = 40, amp: float = 0.05) -> pd.DataFrame:
    """완전 주기 시세. 과거 창을 보는 모델(window)이 쉽게 맞혀야 한다."""
    t = np.arange(n)
    open_ = 10000 * (1 + amp * np.sin(2 * np.pi * t / period))
    close = 10000 * (1 + amp * np.sin(2 * np.pi * (t + 0.5) / period))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999, "close": close, "volume": 1e6}, index=trading_days(n))


def volatility_ohlcv(n: int = 1500, seed: int = 0, planted: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """'앞으로의 변동성'이 숨은 신호 s_t(0~1 사이를 천천히 오가는 값)로 정해지는 시장. 수익률의 방향은 예측할 수 없다(평균 0).
    r_{t+1} ~ N(0, σ_t²),  σ_t = 0.004 + 0.02·s_t  (planted=False 면 σ 가 늘 같다 — 변동성에 예측할 것이 없는 시장)."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.97 * z[t - 1] + rng.normal(0, 0.25)
    s = 1 / (1 + np.exp(-z))
    sig = (0.004 + 0.02 * s) if planted else np.full(n, 0.012)
    r = np.r_[0, rng.normal(0, 1, n - 1) * sig[:-1]]  # r[t+1] 의 크기를 s[t] 가 정한다
    close = 10000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999, "close": close,
                       "volume": rng.lognormal(12, 0.3, n)}, index=trading_days(n))
    return df, pd.Series(s, index=df.index, name="S_signal")
