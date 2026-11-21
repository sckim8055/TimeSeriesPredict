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
    # 주기별 창 (기간 수). 출처(2026-11-14 기록, DECISIONS 같은 날): 일간(D)은 창안자·차트 프로그램의 관례값(RSI 14 Wilder,
    # MACD 12·26·9 Appel, 이동평균 5·20·60·120 한국 HTS 기본)과 Claude 가 고른 기본 기간. 주간(W)·월간(M)은 관례가 아니라
    # Claude 가 2026-11-03 에 정한 값(월간 워밍업을 줄이려고) — 같은 달력 길이 환산이 아니다(preset_calendar_weeks 표 참고).
    # 어느 값도 이 데이터의 결과를 보고 고르지 않았다.
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


# ---------------------------------------------------------------------------
# 기술지표 설명서 — 한 곳에서(2026-11-14). 변수 사전 화면(/api/technical-info)과 도움말 20장 5절이 모두 이것을 읽는다.
# 사용자 요청: "왜 있는지, 어떤 역할인지, 예제와 함께 — 초등학생부터 교수님까지 한눈에".
# why = 왜 넣었나(모델에게 주려는 정보) · analogy = 쉬운 비유 · read = 값 읽는 법 · formula = 계산(위 함수와 같은 식) · caution = 주의
# origin = 누가 만들었나·기간은 어디서 왔나(2026-11-14 추가, 근거 DECISIONS 2026-11-14 「기술지표 18개·창 길이의 출처 기록」)
# ---------------------------------------------------------------------------
def _w(key: str) -> str:
    """일간 창 길이를 '5·20·60' 꼴 글로(설명서의 숫자를 PRESETS 에서 직접 — 손으로 쓰지 않음)."""
    return "·".join(map(str, PRESETS["D"][key]))


TECH_GROUP_DOCS: list[dict] = [
    {"group": "수익률", "prefix": "A_ret_",
     "why": "최근 며칠 동안 얼마나 올랐고 내렸는지를 모델에게 알려 줍니다. 시장에는 '오르던 것이 조금 더 오르는' 경향(모멘텀)과 '많이 오른 것이 되돌아오는' 경향(평균회귀)이 있다는 가설이 있어, 모델이 어느 쪽인지 자료에서 확인해 보게 하려는 것입니다.",
     "analogy": "달리기 선수가 최근 1초·5초·20초 동안 얼마나 빨리 달렸나. 짧은 기간과 긴 기간을 같이 보면 막 출발했는지, 계속 달리고 있는지 알 수 있습니다.",
     "read": "+0.05 = 그 기간에 5% 올랐다, −0.02 = 2% 내렸다. 숫자(n)가 다른 여러 개를 함께 넣어 짧은 흐름과 긴 흐름을 함께 봅니다.",
     "formula": "종가_t ÷ 종가_(t−n) − 1",
     "caution": "긴 기간끼리는 같은 날들이 겹쳐 값이 서로 닮습니다(예: 10일과 20일 수익률). 과거 수익률이 미래를 맞힌다는 보장은 없고, 그것을 확인하는 것이 실험입니다.",
     "origin": "특정 창안자 없음 — 가격 변화율이라는 기본 계산입니다. 기간 " + _w("lags") + "일은 Claude 가 정했습니다(1·5·20 = 하루·약 1주·약 1달, 나머지는 그 사이를 채움)."},
    {"group": "이동평균 비율", "prefix": "A_ma_ratio_",
     "why": "지금 가격이 최근 평균보다 위에 있는지 아래에 있는지(추세 속 위치)를 알려 줍니다. 가격 자체(예: 3만 원)는 시기마다 크기가 달라 비교가 안 되므로, 평균 대비 몇 %인지 비율로 바꿔 넣습니다.",
     "analogy": "내 키가 우리 반 평균 키보다 얼마나 큰가. 반 평균(5일)과 학년 평균(120일)을 함께 보면 '요즘만 큰지, 원래 큰지'를 알 수 있습니다.",
     "read": "+0.03 = 평균보다 3% 위(오름세 쪽), −0.05 = 평균보다 5% 아래. 5는 약 1주, 20은 약 1달, 60은 약 3달, 120은 약 반년(일간 기준).",
     "formula": "종가_t ÷ (최근 n일 종가의 단순평균) − 1",
     "caution": "평균은 과거로만 계산합니다(오늘까지). 처음 n−1일은 값이 없습니다.",
     "origin": "특정 창안자 없음 — 이동평균은 오래된 기본 기법입니다. 기간 " + _w("ma") + "일은 한국 증권사 차트(HTS·MTS)의 기본 이동평균선(약 1주·1달·분기·반년)을 따랐고, 평균 대비 비율로 바꾼 것은 이 솔루션이 정한 형태입니다."},
    {"group": "거래량", "prefix": "A_vol_ratio_",
     "why": "오늘 거래가 평소보다 많았는지를 알려 줍니다. 거래량이 크게 늘면 새 정보나 관심이 몰렸다는 뜻일 수 있고, 같은 가격 변화라도 거래가 많은 날의 변화가 더 의미 있다는 관점이 있습니다.",
     "analogy": "가게에 오늘 손님이 평소보다 얼마나 많이 왔나.",
     "read": "+1.0 = 평소(최근 n일 평균)의 2배, 0 = 평소와 같음, −0.5 = 평소의 절반.",
     "formula": "거래량_t ÷ (최근 n일 거래량의 평균) − 1",
     "caution": "예측 연구 모드(지수·경제지표 예측)는 거래량이 없어 이 묶음이 빠집니다. 분배금·정기 변경일처럼 일정 때문에 거래가 늘어난 날도 있습니다.",
     "origin": "특정 창안자 없음. 기간 " + _w("volume") + "일은 이동평균선의 관례 기간을 따라 Claude 가 정했습니다."},
    {"group": "변동성", "prefix": "A_vol_",
     "why": "요즘 가격이 얼마나 출렁이는지(위험의 크기)를 알려 줍니다. 크게 출렁인 뒤에는 크게 출렁이는 날이 이어지는 경향(변동성 군집)이 잘 알려져 있어, 위험이 커진 시기를 모델이 구분하게 하려는 것입니다.",
     "analogy": "바다의 파도 높이. 최근 20일과 60일의 파도를 비교하면 바다가 거칠어지는 중인지 잔잔해지는 중인지 알 수 있습니다.",
     "read": "0.30 = 1년으로 환산하면 약 30% 출렁이는 정도. 숫자가 클수록 위험이 큽니다. 일간은 √252, 주간은 √52, 월간은 √12 를 곱해 1년 기준으로 맞춥니다.",
     "formula": "최근 n일 로그수익률(ln 종가_t − ln 종가_(t−1))의 표준편차 × √(1년의 거래 주기 수)",
     "caution": "방향(오를지 내릴지)은 알려 주지 않고 크기만 알려 줍니다. 이름이 비슷한 'A_vol_ratio'(거래량)와 다릅니다.",
     "origin": "특정 창안자 없음 — 표준편차라는 기본 통계입니다. 기간 " + _w("vol") + "일(약 1달·분기)은 흔히 쓰는 길이로 Claude 가 정했습니다."},
    {"group": "RSI", "prefix": "A_rsi_",
     "why": "최근 오른 날의 힘과 내린 날의 힘을 비교해, 너무 많이 오른(과열) 상태인지 너무 많이 내린(침체) 상태인지를 0~100 점수로 알려 줍니다.",
     "analogy": "줄다리기. 사려는 쪽(오른 날들)과 팔려는 쪽(내린 날들) 중 누가 얼마나 세게 당기고 있나.",
     "read": "50 = 오른 힘과 내린 힘이 같음. 관례상 70 이상은 과열(과매수), 30 이하는 침체(과매도)로 봅니다 — 관례일 뿐 규칙은 아닙니다.",
     "formula": "RSI = 100 − 100 ÷ (1 + 상승폭 평균 ÷ 하락폭 평균). 평균은 Wilder 방식(최근 값에 1/n 무게를 주는 지수평활)",
     "caution": "내린 날이 한 번도 없으면 나눗셈이 안 되어 값이 비고, 오른 날만 있으면 100입니다. 과열 뒤에도 더 오를 수 있습니다.",
     "origin": "J. Welles Wilder 가 1978년 책 『New Concepts in Technical Trading Systems』에서 만들었습니다. " + str(PRESETS["D"]["rsi"]) + "일과 70·30 기준은 Wilder 가 제안한 기본값이고, 지금도 대부분의 차트 프로그램 기본값입니다."},
    {"group": "MACD", "prefix": "A_macd",
     "why": "짧은 기간 평균과 긴 기간 평균 사이의 간격으로 추세의 방향과, 그 추세가 강해지는지 약해지는지(전환점)를 알려 줍니다. 세 값을 함께 넣습니다: MACD(간격), 시그널(간격의 평균), 히스토그램(둘의 차이).",
     "analogy": "두 대의 자동차(빠르게 반응하는 차 = 12일 평균, 느긋한 차 = 26일 평균) 사이의 거리. 거리가 벌어지면 추세가 세지는 중, 좁혀지면 약해지는 중입니다.",
     "read": "MACD > 0 = 짧은 평균이 긴 평균보다 위(오름세). 히스토그램 > 0 = 오름세가 강해지는 중, < 0 = 약해지는 중. 종가로 나눠 비율로 만들어 가격 크기와 상관없이 비교됩니다.",
     "formula": "MACD = (12일 지수평균 − 26일 지수평균) ÷ 종가, 시그널 = (MACD의 9일 지수평균) ÷ 종가, 히스토그램 = MACD − 시그널 (일간 기준 창)",
     "caution": "지수평균은 첫날부터 계산이 시작되어, 자료 앞부분 수십 일은 값이 덜 안정적입니다. 이동평균 비율과 비슷한 정보를 담아 서로 닮습니다.",
     "origin": "Gerald Appel 이 1970년대 후반에 만들었습니다. " + "·".join(map(str, PRESETS["D"]["macd"])) + "는 Appel 의 기본값으로 차트 프로그램의 기본값이 되었습니다(예전 주 6일 거래 기준 약 2주·1달·1주 반). Appel 자신도 시장에 맞게 바꾸라고 했습니다."},
]


# 묶음 전체에 대한 출처 설명(2026-11-14). 화면·도움말이 같은 글을 쓴다.
TECH_ORIGIN = {
    "summary": "이 18개 묶음은 Claude 가 엔진을 처음 만들 때(2026-11-02~03) 정했습니다. 원래 연구 코드에는 기술지표 계산이 없었습니다. "
               "RSI·MACD 는 만든 사람의 기본값, 이동평균은 한국 증권사 차트의 기본값을 따랐고, 나머지 기간은 Claude 가 흔히 쓰는 길이로 골랐습니다. "
               "어느 값도 이 데이터의 결과를 보고 고르지 않았습니다(결과를 보고 고르면 우연히 맞은 값을 고르게 됩니다).",
    "not_standard": "'이 18개'가 학계 표준 묶음인 것은 아닙니다. 연구마다 쓰는 지표와 개수가 다릅니다(예: 14개, 27개를 쓴 논문도 있음). "
                    "관례값은 '많이 쓰는 값'이지 '가장 잘 맞는 값'이라는 뜻이 아닙니다.",
    "weekly_monthly": "주간·월간 창은 관례가 아니라 Claude 가 정한 값입니다(2026-11-03, 월간 자료에서 지표가 처음 계산되기까지 너무 오래 기다리지 않도록). "
                      "예전 설명의 '같은 달력 길이로 환산'은 사실과 달랐습니다 — 아래 표처럼 긴 창은 주간·월간이 훨씬 긴 기간을 봅니다.",
}

_WEEKS_PER = {"D": 1 / 5, "W": 1.0, "M": 52 / 12}  # 한 관측이 대략 몇 주인가(일간 = 거래일 5일이 1주)


def preset_calendar_weeks() -> list[dict]:
    """PRESETS 의 창 길이를 대략의 주(週) 수로 바꾼 표(2026-11-14) — 주기 사이 창이 같은 달력 길이가 아님을 보이려고. 대략값(일 ÷ 5, 월 × 52/12)."""
    rows = []
    for key, label in (("lags", "수익률"), ("ma", "이동평균"), ("vol", "변동성"), ("rsi", "RSI"), ("macd", "MACD"), ("volume", "거래량")):
        row = {"key": key, "label": label}
        for f, P in PRESETS.items():
            v = P[key] if isinstance(P[key], tuple) else (P[key],)
            row[f] = {"periods": list(v), "weeks": [round(x * _WEEKS_PER[f], 1) for x in v]}
        rows.append(row)
    return rows


def technical_example() -> dict:
    """도움말·화면의 예시(2026-11-14): 10거래일 종가·거래량에 엔진 함수를 그대로 적용한 마지막 날 값.
    예시라 창을 줄였다(실제는 PRESETS): 수익률 1·5일, 이동평균·변동성·거래량 5일, RSI 5일, MACD (3, 6, 3)."""
    close = [100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 110.0, 108.0, 112.0, 115.0]
    volume = [1000.0, 1100.0, 900.0, 1500.0, 1200.0, 1000.0, 1800.0, 1300.0, 1600.0, 2500.0]
    idx = pd.bdate_range("2026-11-02", periods=len(close))
    c, v = pd.Series(close, index=idx), pd.Series(volume, index=idx)
    parts = [returns(c, (1, 5)), moving_average_ratio(c, (5,)), volatility(c, (5,), 252), rsi(c, 5).to_frame(), macd(c, 3, 6, 3), volume_features(v, (5,))]
    df = pd.concat(parts, axis=1)
    df.columns = [f"A_{x}" for x in df.columns]
    last = df.iloc[-1]
    return {"dates": [str(d.date()) for d in idx], "close": close, "volume": volume,
            "windows": "수익률 1·5일, 이동평균·변동성·거래량 5일, RSI 5일, MACD (3, 6, 3) — 예시라 실제(일간: 수익률 1~20일, 이동평균 5~120일 등)보다 줄임",
            "values": {k: float(last[k]) for k in df.columns}}
