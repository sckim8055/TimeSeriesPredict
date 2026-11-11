"""A그룹(가격·기술지표) 파생변수. 입력은 OHLCV(거래일 인덱스), 출력은 같은 인덱스의 DataFrame.

규칙: 모든 지표는 t일 마감 시점까지의 정보만 쓴다 (rolling·shift는 과거 방향만). 미래를 보는 연산 금지.
각 지표의 공식은 docstring에 적는다 — 논문·교재에서 그대로 인용할 수 있게.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """과거 수익률 r_{t-l+1..t} = close_t / close_{t-l} - 1 (l일 누적)."""
    return pd.DataFrame({f"ret_{l}": close / close.shift(l) - 1.0 for l in lags})


def moving_average_ratio(close: pd.Series, windows: tuple[int, ...] = (5, 20, 60, 120)) -> pd.DataFrame:
    """종가 / 단순이동평균(w) - 1. 수준이 아닌 비율로 두어 비정상성을 피한다."""
    return pd.DataFrame({f"ma_ratio_{w}": close / close.rolling(w).mean() - 1.0 for w in windows})


def volatility(close: pd.Series, windows: tuple[int, ...] = (20, 60), periods_per_year: int = 252) -> pd.DataFrame:
    """로그수익률의 롤링 표준편차 × sqrt(periods_per_year) (연율화). 주기가 주간이면 52, 월간이면 12."""
    lr = np.log(close).diff()
    return pd.DataFrame({f"vol_{w}": lr.rolling(w).std() * np.sqrt(periods_per_year) for w in windows})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI. RS = EMA(상승폭)/EMA(하락폭), RSI = 100 − 100/(1+RS). 범위 0~100."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD = EMA_fast − EMA_slow, Signal = EMA_signal(MACD), Hist = MACD − Signal. 종가로 나눠 비율화."""
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    m = ef - es
    s = m.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": m / close, "macd_signal": s / close, "macd_hist": (m - s) / close})


def volume_features(volume: pd.Series, windows: tuple[int, ...] = (5, 20)) -> pd.DataFrame:
    """거래량 / 롤링 평균 거래량 − 1."""
    return pd.DataFrame({f"vol_ratio_{w}": volume / volume.rolling(w).mean() - 1.0 for w in windows})


PRESETS = {
    # 주기별 창 (기간 수). 일간 기준을 주간·월간의 같은 달력 길이로 환산
    "D": dict(lags=(1, 2, 3, 5, 10, 20), ma=(5, 20, 60, 120), vol=(20, 60), rsi=14, macd=(12, 26, 9), volume=(5, 20)),
    "W": dict(lags=(1, 2, 4, 8, 13, 26), ma=(4, 13, 26, 52), vol=(13, 26), rsi=8, macd=(6, 13, 5), volume=(4, 13)),
    "M": dict(lags=(1, 2, 3, 6, 12), ma=(3, 6, 12, 24), vol=(6, 12), rsi=6, macd=(3, 6, 3), volume=(3, 12)),
}


def technical_block(ohlcv: pd.DataFrame, freq: str = "D", *, with_volume: bool = True) -> pd.DataFrame:
    """A그룹 기본 세트 한 번에. 열 이름 접두어 'A_'. 창 길이는 매매 주기(freq)에 맞춘 PRESETS 사용.
    with_volume=False 면 거래량 지표를 뺀다(예측 연구 모드: 지수에는 거래량이 없음)."""
    P = PRESETS[freq]
    c = ohlcv["close"].astype(float)
    ppy = {"D": 252, "W": 52, "M": 12}[freq]
    parts = [returns(c, P["lags"]), moving_average_ratio(c, P["ma"]), volatility(c, P["vol"], ppy), rsi(c, P["rsi"]).to_frame(),
             macd(c, *P["macd"])]
    if with_volume:
        parts.append(volume_features(ohlcv["volume"].astype(float), P["volume"]))
    out = pd.concat(parts, axis=1)
    out.columns = [f"A_{x}" for x in out.columns]
    return out


# ---------------------------------------------------------------------------
# 기술지표 열 이름 → 한글 설명 (화면·보고서용). 숫자는 주기 수(일간이면 거래일, 월간이면 개월).
# ---------------------------------------------------------------------------
import re as _re

_DESC_RULES = [  # 화면·계산 따라가기에 쓰는 설명. 위 함수의 식과 글자 그대로 맞출 것(2026-11-06 점검: 변동성 연율화·MACD 비율화·거래량 −1 누락 정정)
    (r"^A_ret_(\d+)$", lambda m: f"{m.group(1)}주기 수익률 = 종가_t ÷ 종가_(t−{m.group(1)}) − 1"),
    (r"^A_ma_ratio_(\d+)$", lambda m: f"종가 ÷ {m.group(1)}주기 단순이동평균 − 1 (추세 위치)"),
    (r"^A_vol_(\d+)$", lambda m: f"{m.group(1)}주기 로그수익률 표본표준편차(n−1) × √(연간 주기 수) (연율 변동성)"),
    (r"^A_rsi_(\d+)$", lambda m: f"RSI({m.group(1)}) = 100 − 100 ÷ (1 + 상승 평균 ÷ 하락 평균), Wilder 평활, 0~100"),
    (r"^A_macd$", lambda m: "MACD = (단기 EMA − 장기 EMA) ÷ 종가"),
    (r"^A_macd_signal$", lambda m: "MACD 시그널 = (MACD 의 EMA) ÷ 종가"),
    (r"^A_macd_hist$", lambda m: "MACD 히스토그램 = (MACD − 시그널) ÷ 종가"),
    (r"^A_vol_ratio_(\d+)$", lambda m: f"거래량 ÷ {m.group(1)}주기 평균거래량 − 1 (거래 활기)"),
]


def feature_description(name: str) -> str:
    for pat, fn in _DESC_RULES:
        m = _re.match(pat, name)
        if m:
            return fn(m)
    return ""


def describe_feature(name: str, registry=None) -> str:
    """기술지표면 규칙으로, 레지스트리 변수면 '이름 — 설명' 으로, '_pct' 등 변환 접미사도 풀어 쓴다."""
    d = feature_description(name)
    if d:
        return d
    base, _, tr = name.partition("_") if name.count("_") == 1 and name.rsplit("_", 1)[-1] in ("pct", "diff", "log", "logdiff") else (name, "", "")
    if not tr:
        m = _re.match(r"^(.*)_(pct\d*|diff\d*|log|logdiff)$", name)
        if m:
            base, tr = m.group(1), m.group(2)
    tr_desc = {"pct": "변화율", "diff": "차분(전 관측 대비 변화)", "log": "로그", "logdiff": "로그수익률"}
    t = ""
    if tr:
        mm = _re.match(r"^(pct|diff)(\d+)$", tr)
        t = f" · {mm.group(2)}주기 {tr_desc[mm.group(1)]}" if mm else f" · {tr_desc.get(tr, tr)}"
    if registry is not None and base in registry:
        v = registry.get(base)
        return f"{v.name}{(' — ' + v.description) if v.description else ''}{t}"
    return (base + t) if t else ""


# 기술지표 묶음(화면의 묶음 단위 켜고 끄기·설명용). 이름 접두어로 나눈다 — 'A_vol_ratio' 는 거래량, 'A_vol_' 은 변동성이라 순서가 중요.
TECH_GROUPS = [("A_ret_", "수익률"), ("A_ma_ratio_", "이동평균 비율"), ("A_vol_ratio_", "거래량"), ("A_vol_", "변동성"),
               ("A_rsi_", "RSI"), ("A_macd", "MACD")]


def technical_group(name: str) -> str:
    for pre, label in TECH_GROUPS:
        if name.startswith(pre):
            return label
    return "기타"


def technical_columns(freq: str, *, with_volume: bool = True) -> list[str]:
    """technical_block 이 만드는 열 이름 전체(데이터 없이). 이름은 주기별 창 길이(PRESETS)로만 정해진다."""
    n = 400
    idx = pd.bdate_range("2000-01-03", periods=n)
    c = pd.Series(100 + np.cumsum(np.ones(n)), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    return list(technical_block(df, freq, with_volume=with_volume).columns)


def select_technical(block: pd.DataFrame, wanted: list[str] | None) -> pd.DataFrame:
    """설정의 technical_features 로 기술지표 열을 고른다. None = 전부(기본). 없는 이름·빈 목록은 오류(조용히 넘어가지 않음)."""
    if wanted is None:
        return block
    if not wanted:
        raise ValueError("technical_features 가 빈 목록입니다 — 기술지표를 모두 빼려면 use_technical 을 끄세요")
    unknown = [w for w in wanted if w not in block.columns]
    if unknown:
        raise ValueError(f"기술지표 {unknown} 는 이 주기·모드에 없습니다. 있는 것: {list(block.columns)}")
    return block[[c for c in block.columns if c in set(wanted)]]
