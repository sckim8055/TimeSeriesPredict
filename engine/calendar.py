"""날짜·정렬 연산의 유일한 지점 (CLAUDE.md 절대규칙 1).

다른 파일에서는 shift / resample / fillna / 날짜 비교를 직접 하지 않는다. 전부 여기 함수를 쓴다.
모든 함수는 pandas DatetimeIndex(거래일) 기준으로 동작하고, "미래 정보가 과거로 새는" 연산은 이름에 그 사실을 드러낸다.

용어
- 거래일 캘린더(trading calendar): 매매 대상 ETF가 실제 거래된 날짜들. 모든 변수는 이 캘린더에 맞춰 정렬된다.
- 발표시차(lag_days): 지표의 기준일(= 기간 시작일, 예: 1월 CPI = 1-01)과 공개일(2-05) 사이의 **달력일** 수(정수, 주말·공휴일 포함). 공개일 전에는 그 값을 쓸 수 없다.
- fill: 저주기(월간 등) 값을 일간으로 펼칠 때 빈 날을 채우는 방식. ffill만 미래 정보가 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Freq = Literal["D", "W", "M", "Q", "H", "Y"]
Fill = Literal["ffill", "bfill", "interp"]

FUTURE_LEAKING_FILLS: frozenset[str] = frozenset({"bfill", "interp"})


# ---------------------------------------------------------------------------
# 캘린더
# ---------------------------------------------------------------------------
def trading_calendar(prices: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """가격 데이터의 인덱스에서 거래일 캘린더를 만든다 (정렬·중복 제거)."""
    idx = pd.DatetimeIndex(prices.index).normalize()
    idx = idx[~idx.duplicated()].sort_values()
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("거래일 캘린더가 정렬·유일하지 않습니다")
    return idx


def resample_prices(ohlcv: pd.DataFrame, freq: Freq) -> pd.DataFrame:
    """일간 OHLCV를 매매 주기(D/W/M)로 리샘플. 주기 마지막 *거래일*을 대표일로 쓴다.

    - 시가 = 주기 첫 거래일 시가, 고가 = max, 저가 = min, 종가 = 주기 마지막 거래일 종가, 거래량 = 합.
    - 인덱스는 캘린더상 마지막 날이 아니라 실제 마지막 거래일(그날 체결 가능)이다.
    """
    if freq == "D":
        return ohlcv.copy()
    rule = {"W": "W-FRI", "M": "ME", "Q": "QE", "Y": "YE"}[freq]
    g = ohlcv.groupby(pd.Grouper(freq=rule))
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(min_count=1),
        }
    ).dropna(subset=["close"])
    last_day = g.apply(lambda d: d.index.max() if len(d) else pd.NaT)
    out.index = pd.DatetimeIndex(last_day.loc[out.index].values, name="date")
    return out


# ---------------------------------------------------------------------------
# 저주기 변수 → 거래일 캘린더로 정렬
# ---------------------------------------------------------------------------
def align_to_calendar(
    s: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    lag_days: int = 0,
    fill: Fill = "ffill",
    apply_lag: bool = True,
) -> pd.Series:
    """변수 s(기준일 인덱스)를 거래일 캘린더에 맞춘다.

    1) 발표시차: 기준일 + lag_days = 공개일. 공개일 이후부터 그 값을 쓸 수 있다 (apply_lag=False면 기준일부터).
    2) 채움: 공개일 사이의 빈 거래일을 fill 방식으로 채운다.
       - ffill: 마지막 공개값 유지 (미래 정보 없음)
       - bfill: 다음 공개값을 앞으로 끌어옴 (미래 정보 포함!)
       - interp: 앞뒤 공개값 사이 선형보간 (다음 값을 알아야 하므로 미래 정보 포함!)
    반환: calendar 인덱스의 Series. 첫 공개 전 구간은 NaN.
    """
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    if apply_lag and lag_days:
        s.index = s.index + pd.to_timedelta(int(lag_days), unit='D')
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # 공개일이 비거래일이면 그 다음 거래일부터 사용 가능 (ffill 의미와 일치)
    union = calendar.union(s.index)
    on_union = s.reindex(union)
    if fill == "ffill":
        filled = on_union.ffill()
    elif fill == "bfill":
        filled = on_union.bfill()
    elif fill == "interp":
        filled = on_union.interpolate(method="time", limit_area="inside")
    else:
        raise ValueError(f"알 수 없는 fill: {fill}")
    return filled.reindex(calendar).rename(s.name)


def fill_leaks_future(fill: str) -> bool:
    """이 채움 방식이 미래 정보를 포함하는지 (UI 경고 배지용)."""
    return fill in FUTURE_LEAKING_FILLS


# ---------------------------------------------------------------------------
# 타겟 정렬 — 모델이 맞히는 값은 항상 입력 구간 바깥의 미래
# ---------------------------------------------------------------------------
TargetKind = Literal["return", "logreturn", "direction", "event", "volatility"]


def _log_close_returns(ohlcv: pd.DataFrame) -> pd.Series:
    c = ohlcv["close"].astype(float)
    return np.log(c / c.shift(1))


def realized_volatility_target(ohlcv: pd.DataFrame, k: int) -> pd.Series:
    """변동성 타겟(2026-11-12): 결정일 t(마감 후)의 값 = 그 뒤 k 기간의 실현 변동성
        RV_t = sqrt( (1/k) · Σ_{j=1..k} ( ln(C[t+j] / C[t+j−1]) )² )      (C = 종가, 1기간 단위 — 연율화하지 않는다)
    t+1 … t+k 의 종가만 쓴다(t 의 종가는 첫 수익률의 출발점으로만). 마지막 k 행은 미래가 없어 NaN.
    k=1 이면 |다음 기간 수익률| 하나라 잡음이 크다 — 5(일간이면 1주) 이상을 권한다."""
    r2 = _log_close_returns(ohlcv) ** 2
    fut = sum(r2.shift(-j) for j in range(1, k + 1)) / k   # 하나라도 없으면 NaN
    return np.sqrt(fut)


def naive_volatility(ohlcv: pd.DataFrame, k: int) -> pd.Series:
    """변동성의 단순 기준 예측: 결정일 t 까지의 최근 k 기간 실현 변동성(t−k+1 … t 의 수익률). '변동성은 이어진다'만 아는 예측 — 미래 값을 쓰지 않는다."""
    r2 = _log_close_returns(ohlcv) ** 2
    past = sum(r2.shift(j) for j in range(0, k)) / k
    return np.sqrt(past)


def make_target(
    ohlcv: pd.DataFrame,
    k: int = 1,
    kind: TargetKind = "return",
    *,
    threshold: float = 0.0,
    exec_price: Literal["open", "close"] = "open",
) -> pd.Series:
    """결정일 t(마감 후)에 대한 타겟.

    체결 가정: t일 마감 후 판단 → t+1일 exec_price(기본 시가)에 진입 → t+1+k일 exec_price에 청산.
      수익률 r_t = P[t+1+k] / P[t+1] - 1  (exec_price 기준)
    - return: r_t
    - logreturn: log(1 + r_t)
    - direction: 1 if r_t > 0 else 0
    - event: 1 if r_t > threshold else 0  (threshold 음수면 하락 이벤트로 해석하지 않음 — 별도 타겟으로)
    - volatility: 수익률이 아니라 그 뒤 k 기간의 실현 변동성(realized_volatility_target — 종가 기준, 체결 가정과 무관)
    마지막 1+k 행은 미래가 없어 NaN. 인덱스는 결정일 t.
    """
    if k < 1:
        raise ValueError("k는 1 이상")
    if kind == "volatility":
        y = realized_volatility_target(ohlcv, k)
        y.name = f"target_volatility_k{k}"
        return y
    p = ohlcv[exec_price].astype(float)
    entry = p.shift(-1)
    exit_ = p.shift(-(1 + k))
    r = exit_ / entry - 1.0
    if kind == "return":
        y = r
    elif kind == "logreturn":
        y = np.log1p(r)
    elif kind == "direction":
        y = (r > 0).astype(float).where(r.notna())
    elif kind == "event":
        y = (r > threshold).astype(float).where(r.notna())
    else:
        raise ValueError(kind)
    y.name = f"target_{kind}_k{k}"
    return y


def make_forecast_target(level: pd.Series, k: int = 1, kind: TargetKind = "return", *, threshold: float = 0.0) -> pd.Series:
    """예측 연구 모드의 타겟 — 체결 없이 '지금 값 대비 k기간 뒤 값'.

      r_t = v[t+k] / v[t] − 1   (v = 대상 시계열의 기준기간 값, 인덱스 = 기준기간 시작일)
    매매 모드의 make_target(t+1 시가 진입)과 달리 t 의 값 자체를 기준으로 한다. 마지막 k 행은 NaN.
    purging 은 k 행이면 충분하다(라벨이 t+k 까지만 본다).
    """
    if k < 1:
        raise ValueError("k는 1 이상")
    v = level.astype(float)
    r = v.shift(-k) / v - 1.0
    if kind == "return":
        y = r
    elif kind == "logreturn":
        y = np.log1p(r)
    elif kind == "direction":
        y = (r > 0).astype(float).where(r.notna())
    elif kind == "event":
        y = (r > threshold).astype(float).where(r.notna())
    else:
        raise ValueError(kind)
    y.name = f"forecast_{kind}_k{k}"
    return y


def naive_forecast(y: pd.Series, k: int) -> pd.Series:
    """지속(persistence) 기준선: t 시점에 알 수 있는 가장 최근의 완결된 k기간 변화율 = y[t−k] (v[t]/v[t−k]−1).
    예측 연구 모드에서 0예측과 함께 반드시 이겨야 하는 기준. 평활된 지수(부동산 등)에서는 0예측보다 훨씬 강하다."""
    return y.shift(k).rename("naive")


def assert_no_target_in_inputs(X: pd.DataFrame, y: pd.Series, k: int) -> None:
    """누출 방어: 입력 X의 행 t 는 t 시점까지의 정보만, 타겟 y[t]는 t+1 이후 정보만 담아야 한다.

    이 함수는 '타겟을 만든 원천 가격의 미래 값이 X 열로 들어왔는지'를 상관으로 점검한다.
    완벽한 검사는 아니다(엔진 테스트의 타겟 섞기 검사가 보완). 여기서는 명백한 사고 — y 자체 또는 y의 미래 시프트가
    X 열과 거의 같은 경우 — 만 잡는다.
    """
    yy = y.dropna()
    for c in X.columns:
        x = X[c].reindex(yy.index)
        if x.notna().sum() < 30:
            continue
        corr = np.corrcoef(x.dropna().values, yy.loc[x.dropna().index].values)[0, 1]
        if np.isfinite(corr) and abs(corr) > 0.999:
            raise AssertionError(f"입력 열 '{c}'가 타겟과 사실상 동일합니다 (corr={corr:.4f}) — 누출 의심")


# ---------------------------------------------------------------------------
# 분할·워크포워드·purging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    val_from: pd.Timestamp | None = None  # 1회 분할에서 검증 시작일을 날짜로 정한 경우. None 이면 학습 창 뒤쪽 val_frac


def walkforward_folds(
    calendar: pd.DatetimeIndex,
    *,
    train_len: int,
    test_len: int,
    k: int,
    embargo: int | None = None,
    start: int | None = None,
    sealed_from: pd.Timestamp | None = None,
) -> list[Fold]:
    """워크포워드 분할.

    - 학습 창 train_len(행 수) → 바로 뒤 test_len 행을 예측 → test_len 만큼 창 이동.
    - purging/embargo: 결정일 t의 타겟은 t+1+k 까지의 가격을 쓴다. 학습 창 마지막 결정일의 타겟이 테스트 첫날 이후
      가격을 포함하면 누출이므로, 학습 창 끝에서 embargo(기본 1+k) 행을 제거한다.
    - sealed_from: 이 날짜 이후는 봉인 테스트 구간 — 폴드를 만들지 않는다 (봉인 평가는 별도 함수).
    - 마지막 폴드는 남은 날만큼만 테스트한다(빠지는 날 없음). 개발 평가는 봉인 전날까지, 봉인 평가는 봉인 첫날부터.
    """
    emb = (1 + k) if embargo is None else embargo
    cal = calendar if sealed_from is None else calendar[calendar < pd.Timestamp(sealed_from)]
    folds: list[Fold] = []
    i = train_len if start is None else max(start, train_len)  # start: 첫 테스트 행 번호(평가 시작일). 학습 창이 모자라면 train_len 부터
    while i < len(cal):
        tr = cal[i - train_len : i - emb] if emb else cal[i - train_len : i]
        te = cal[i : i + test_len]  # 마지막 폴드는 test_len 보다 짧을 수 있다(남은 날만큼) — 2026-11-06 부터. 그 전에는 버려서 봉인 직전·데이터 끝의 최대 test_len−1 일이 평가되지 않았다
        if len(tr) > 0:
            folds.append(Fold(tr, te))
        i += test_len
    return folds


# ---------------------------------------------------------------------------
# 입력 형태 변환
# ---------------------------------------------------------------------------
def make_windows(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """table [T, k] → window [N, seq_len, k]. 윈도우 마지막 행 = 결정일 t, 타겟 = y[t] (이미 미래로 정렬된 값)."""
    if len(X) != len(y):
        raise ValueError("X, y 길이 불일치")
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    return X[idx], y[seq_len - 1 :]
