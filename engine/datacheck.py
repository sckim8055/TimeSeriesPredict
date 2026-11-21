"""자료 점검표 (2026-11-14) — 값을 바꾸지 않는 검사. 실험 계산에는 쓰지 않는다(알림·기록용).

변수 하나마다 다섯 가지를 본다.
  ① 정상성(추세)  모델이 받는 꼴(변수의 변환을 원래 주기에서 적용한 값)에 대해 두 검정을 함께 본다.
                 ADF(귀무가설 = 단위근, 즉 추세가 남아 있다) · KPSS(귀무가설 = 정상).
                   ADF p < 0.05 이고 KPSS p > 0.05        → '정상'
                   ADF p ≥ 0.05 이고 KPSS p ≤ 0.05        → '추세 남음'
                   그 밖(둘 다 기각, 둘 다 기각 못 함)     → '판단 어려움' (구조 변화·과차분·자료가 짧음 등)
                 값 30개 미만 → '자료 부족', 값이 모두 같으면 → '판단 불가'.
                 KPSS 의 p 는 statsmodels 표(0.01~0.10) 안에서만 보간된다 — 범위 밖이면 경계값이므로 '<0.01'·'>0.10' 으로 적는다.
                 ADF 차수는 AIC 로 고르고(autolag='AIC'), 두 검정 모두 상수항만(regression='c', 수준 정상성).
  ② 계절성       월(M)·분기(Q) 자료만. 모델이 받는 꼴을 달(분기) 더미에 회귀한 F 검정의 p.
                   p < 0.01 → '계절성 남음' · p < 0.05 → '약한 계절성' · 그 밖 → '없음'.
                 3년치(월 36·분기 12) 미만 → '자료 부족'. 일·주 자료 → '해당 없음'.
                 근사 검정이다: 이웃한 값끼리 닮아 있으면(자기상관) p 가 작게 나올 수 있다 — 그래서 '알림'으로만 쓴다.
  ③ 계절조정 표시 변수 사전의 seasonal_adj 칸(SA=조정됨, NSA=조정 안 됨)이 있으면 그것.
                 없으면 이름·설명의 글자: '계절조정'·낱말 'SA'·'seasonally adjusted' → 조정됨, '원계열'·'계절조정 전'·낱말 'NSA' → 조정 안 됨,
                 둘 다 있거나 없으면 '확인 필요'. 비고(note)는 읽지 않는다(다른 이야기가 섞여 있어 틀림 — 언급이 있다는 사실만 알림).
                 일·주 자료(금리·가격·수급)는 '해당 없음'.
  ④ 빈 기간      주기별 기준(_GAP_DAYS)보다 긴 날짜 간격.
  ⑤ 확인해 볼 날 하루 크게 움직였다가 다음 날 되돌아온 날(gap_and_reversal 의 규칙 — 도움말 10장과 같은 말).

신호: '노랑' = 추세 남음·계절성 남음·약한 계절성·빈 기간·확인해 볼 날 중 하나 이상(reasons 에 이유) · '회색' = 정상성을 판단할 수 없음
(자료 부족·판단 불가) · '초록' = 걸린 것 없음. '판단 어려움'은 노랑으로 알린다(사람이 그래프를 봐야 함).
권고(advice)는 적기만 하고 값·설정을 바꾸지 않는다.

upto: 그 날짜 '앞'의 값만 쓴다. 실험 결과에 남길 때는 봉인 시작일을 넣어, 봉인 구간 값이 점검에 섞이지 않게 한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

GAP_DAYS = {"D": 12, "W": 21, "M": 45, "Q": 120, "H": 220, "Y": 400}   # 이보다 긴 빈 기간을 알린다(일간 12일: 가장 긴 휴장 = 2017 추석 연휴 11일은 정상)
OUTLIER_Z = 8.0                                                        # 하루 변화의 강건 z(중앙값·MAD) 기준
MIN_N_STATIONARITY = 30
RULE = ("정상성: ADF(귀무 = 추세 있음)와 KPSS(귀무 = 정상)를 함께 봅니다 — ADF p<0.05 이고 KPSS p>0.05 면 정상, 반대면 추세 남음, 그 밖은 판단 어려움. "
        "계절성(월·분기만): 달(분기) 더미 회귀 F 검정 p<0.01 계절성 남음, p<0.05 약한 계절성. 모두 '모델이 받는 꼴'(변환 후, 원래 주기)로 검사합니다. "
        "값은 바꾸지 않습니다.")


def gap_and_reversal(s: pd.Series, freq: str, skip_reversal_note: str | None = None) -> dict:
    """빈 기간·하루 튀었다 되돌아온 날·가장 큰 움직임(변수·종목 가격 공통, 2026-11-13; 2026-11-14 api/main.py 에서 옮김). 값을 바꾸지 않는다."""
    gaps = []
    lim = GAP_DAYS.get(freq, 45)
    d = s.index.to_series().diff().dt.days
    for t, days in d[d > lim].items():
        prev = s.index[s.index.get_loc(t) - 1]
        gaps.append({"from": str(prev.date()), "to": str(t.date()), "days": int(days)})
    reversals, moves, rev_note = [], [], skip_reversal_note
    if rev_note is None:
        ser = s.dropna()
        pos = bool((ser > 0).all())
        r = (np.log(ser).diff() if pos else ser.diff()).dropna()
        # 규칙(2026-11-13 검토 7 — 도움말 10장과 같은 말): 하루 변화 r 의 강건 z = |r − 중앙값| ÷ (1.4826·MAD).
        # 이틀 연속 z > 8 이고, 두 날의 움직임(중앙값을 뺀 값)이 방향이 반대이며, 둘째 날이 첫째 날 움직임의 2/3 이상을 되돌리면 '확인해 볼 날'.
        # (|a + b| < 0.5·min(|a|,|b|) 은 b = −x·a 일 때 x > 2/3 과 같다.) 값을 바꾸지 않고 알리기만 한다.
        med = float(r.median()) if len(r) else 0.0
        mad = float((r - med).abs().median()) * 1.4826 if len(r) >= 10 else 0.0
        if len(r) < 10:
            rev_note = "값이 10개 미만이라 튀었다 되돌아온 날을 점검하지 않았습니다"
        elif mad == 0:
            rev_note = "하루 변화의 절반 이상이 같은 값(대부분 변화 없음)이라 흩어짐(MAD)이 0 — 튀었다 되돌아온 날을 이 방법으로 점검할 수 없습니다. 그래프를 직접 보세요"
        if mad > 0:
            a_, ix = (r - med).to_numpy(), r.index
            for i in range(len(a_) - 1):
                if (abs(a_[i]) > OUTLIER_Z * mad and abs(a_[i + 1]) > OUTLIER_Z * mad and np.sign(a_[i]) != np.sign(a_[i + 1])
                        and abs(a_[i] + a_[i + 1]) < 0.5 * min(abs(a_[i]), abs(a_[i + 1]))):
                    t = ix[i]; k = ser.index.get_loc(t)
                    reversals.append({"date": str(t.date()), "before": float(ser.iloc[k - 1]), "value": float(ser.iloc[k]), "after": float(ser.iloc[k + 1])})
            for t in r.abs().sort_values(ascending=False).index[:5]:
                moves.append({"date": str(t.date()), "change": float(r[t]), "kind": "로그 변화" if pos else "차이", "value": float(ser[t])})
    return {"gaps": gaps, "gap_days_limit": lim, "reversals": reversals, "reversal_z": OUTLIER_Z, "reversal_note": rev_note, "largest_moves": moves}


def _p_text(p: float, lo: float | None = None, hi: float | None = None) -> str:
    if lo is None and p < 0.001:
        return "<0.001"
    if hi is not None and p >= hi - 1e-12:
        return f">{hi:.2f}"
    if lo is not None and p <= lo + 1e-12:
        return f"<{lo:.2f}"
    return f"{p:.3f}"


def stationarity(x: pd.Series) -> dict:
    """정상성 판정(머리말 ①). x = 모델이 받는 꼴의 값(빈 칸은 뺀다)."""
    v = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(v))
    out = {"n": n, "adf_p": None, "kpss_p": None, "adf_p_text": None, "kpss_p_text": None}
    if n < MIN_N_STATIONARITY:
        return {**out, "verdict": "자료 부족", "why": f"값이 {n}개 — {MIN_N_STATIONARITY}개 이상이어야 검정합니다"}
    if float(v.std()) == 0.0:
        return {**out, "verdict": "판단 불가", "why": "값이 모두 같습니다"}
    from statsmodels.tsa.stattools import adfuller, kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # KPSS 의 '표 범위 밖' 경고 — 대신 p 를 경계값 글자로 적는다
        try:
            ar = adfuller(v.to_numpy(), regression="c", autolag="AIC")
            adf_p = float(ar.pvalue if hasattr(ar, "pvalue") else ar[1])   # statsmodels 0.16 부터 결과 객체(.pvalue), 그 전은 튜플([1])
            kp = float(kpss(v.to_numpy(), regression="c", nlags="auto")[1])
        except Exception as e:  # noqa: BLE001
            return {**out, "verdict": "판단 불가", "why": f"검정을 계산하지 못했습니다: {type(e).__name__}"}
    out.update(adf_p=adf_p, kpss_p=kp, adf_p_text=_p_text(adf_p), kpss_p_text=_p_text(kp, 0.01, 0.10))
    if adf_p < 0.05 and kp > 0.05:
        return {**out, "verdict": "정상", "why": "ADF 가 추세(단위근)를 기각하고 KPSS 는 정상을 기각하지 못했습니다"}
    if adf_p >= 0.05 and kp <= 0.05:
        return {**out, "verdict": "추세 남음", "why": "ADF 가 추세(단위근)를 기각하지 못하고 KPSS 는 정상을 기각했습니다 — 차이·변화율로 바꾸는 것을 검토하세요"}
    why = ("두 검정이 모두 기각해 엇갈립니다(구조 변화·변동성 변화·과차분에서 흔함) — 그래프를 보세요" if adf_p < 0.05
           else "두 검정 모두 기각하지 못했습니다(자료가 짧거나 신호가 약함) — 그래프를 보세요")
    return {**out, "verdict": "판단 어려움", "why": why}


def seasonality(x: pd.Series, freq: str) -> dict:
    """계절성 판정(머리말 ②). 월·분기만."""
    if freq not in ("M", "Q"):
        return {"verdict": "해당 없음", "p": None, "n": None, "why": "일·주·반기·연 자료는 달(분기) 계절성 검사를 하지 않습니다"}
    v = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    need = 36 if freq == "M" else 12
    if len(v) < need:
        return {"verdict": "자료 부족", "p": None, "n": int(len(v)), "why": f"3년치({need}개) 미만"}
    if float(v.std()) == 0.0:
        return {"verdict": "없음", "p": None, "n": int(len(v)), "why": "값이 모두 같습니다"}
    per = v.index.month if freq == "M" else v.index.quarter
    import statsmodels.api as sm
    dum = pd.get_dummies(pd.Categorical(per), drop_first=True, dtype=float)
    X = sm.add_constant(dum.to_numpy())
    fit = sm.OLS(v.to_numpy(), X).fit()
    p = float(fit.f_pvalue) if np.isfinite(fit.f_pvalue) else 1.0
    verdict = "계절성 남음" if p < 0.01 else "약한 계절성" if p < 0.05 else "없음"
    return {"verdict": verdict, "p": p, "p_text": _p_text(p), "n": int(len(v)),
            "why": f"{'달' if freq == 'M' else '분기'}별 평균 차이의 F 검정 p = {_p_text(p)}"}


import re as _re
# 영문 SA·NSA 는 낱말로만 찾는다('Transaction' 안의 'nsa' 에 걸리지 않게 — 실제 사전의 H3·O3 설명). 한글은 글자 그대로.
_SA_NO = _re.compile(r"원계열|계절\s?조정\s?전|원자료|\bnsa\b|not seasonally adjusted")
_SA_YES = _re.compile(r"계절\s?조정(?!\s?전)|\bsa\b|\bsaar\b|seasonally adjusted")


def sa_label(v) -> tuple[str, str]:
    """계절조정 표시(머리말 ③). (표시, 근거)."""
    if v.freq not in ("M", "Q", "H", "Y"):
        return "해당 없음", "일·주 자료(금리·가격·수급)는 계절조정 개념을 쓰지 않습니다"
    col = (getattr(v, "seasonal_adj", "") or "").upper()
    if col == "SA":
        return "조정됨", "변수 사전 seasonal_adj = SA"
    if col == "NSA":
        return "조정 안 됨", "변수 사전 seasonal_adj = NSA"
    # 자동 판정은 이름·설명 칸에서만(2026-11-14). 비고(note)에는 발표시차 근거 등 다른 이야기가 섞여 있어(예: '원계열'·'계절조정계열로 확인'이
    # 한 비고에 함께) 글자로 뜻을 짐작하면 틀린다 — 실제 사전에서 33개가 '조정 안 됨'으로 잘못 나왔다. 비고에 언급이 있으면 그 사실만 알린다.
    text = " ".join([v.name or "", v.description or ""]).lower()
    no, yes = bool(_SA_NO.search(text)), bool(_SA_YES.search(text))
    if no and not yes:
        return "조정 안 됨", "이름·설명의 글자"
    if yes and not no:
        return "조정됨", "이름·설명의 글자"
    note = (v.note or "").lower()
    hint = " 비고에 계절조정 관련 언급이 있습니다 — 비고를 읽고 seasonal_adj 칸에 SA/NSA 를 적으면 그것을 씁니다." if (_SA_NO.search(note) or _SA_YES.search(note)) else ""
    if no and yes:
        return "확인 필요", "이름·설명에 '조정'과 '원계열' 표시가 함께 있습니다." + hint
    return "확인 필요", "이름·설명에 계절조정 여부가 적혀 있지 않습니다(seasonal_adj 칸에 SA/NSA 를 적으면 그것을 씁니다)." + hint


def series_of(long: pd.DataFrame) -> tuple[pd.Series, int]:
    """롱 → 날짜순 시계열. 같은 날짜가 두 줄이면 나중 줄(엔진 wide() 와 같은 규칙). (시계열, 겹친 날 수)."""
    if long.empty:
        return pd.Series(dtype=float), 0
    long = long.sort_values("date", kind="stable")
    dates = pd.to_datetime(long["date"])
    dup = int(dates.duplicated(keep="first").sum())
    s = pd.Series(long["value"].astype(float).to_numpy(), index=dates)
    return s[~s.index.duplicated(keep="last")], dup


def flow_like(s: pd.Series, transform: str) -> bool:
    """값이 0 근처를 오가는 흐름 자료(변환 none 이고 음수 있음, 예: 순매수대금) — 매일 크게 오가는 게 정상이라 되돌림 검사를 하지 않는다."""
    return (transform or "none").lower() == "none" and not bool((s.dropna() > 0).all())


def check_variable(v, long: pd.DataFrame, transform: str | None = None, upto: str | None = None) -> dict:
    """변수 하나의 점검 줄. transform 을 주면(실험의 변수별 변환 덮어쓰기) 그것으로 검사한다."""
    from engine.experiment import apply_transform
    tr_name = transform if transform is not None else v.transform
    s, _ = series_of(long)
    if upto is not None and len(s):
        s = s[s.index < pd.Timestamp(upto)]
    sa, sa_basis = sa_label(v)
    row = {"id": v.id, "name": v.name, "freq": v.freq, "transform": tr_name, "n": int(len(s)),
           "first": str(s.index.min().date()) if len(s) else None, "last": str(s.index.max().date()) if len(s) else None,
           "seasonal_adj": sa, "seasonal_adj_basis": sa_basis}
    if not len(s):
        return {**row, "stationarity": {"verdict": "자료 부족", "n": 0, "why": "값이 없습니다"}, "seasonality": {"verdict": "자료 부족", "p": None},
                "gaps": 0, "reversals": 0, "signal": "회색", "reasons": ["값이 없습니다"], "advice": []}
    try:
        x = apply_transform(s, tr_name)
    except Exception as e:  # noqa: BLE001
        x = pd.Series(np.nan, index=s.index)
        row["transform_error"] = str(e)
    st = stationarity(x)
    se = seasonality(x, v.freq)
    gr = gap_and_reversal(s, v.freq, "0 근처를 오가는 자료라 되돌림 검사를 하지 않습니다" if flow_like(s, tr_name) else None)
    reasons, advice = [], []
    if st["verdict"] == "추세 남음":
        reasons.append("추세 남음(정상성 검정)")
        advice.append("변환을 차이(diff)·변화율(pct)로 바꾸는 것을 검토")
    elif st["verdict"] == "판단 어려움":
        reasons.append("정상성 판단 어려움 — " + st["why"])
    if se["verdict"] in ("계절성 남음", "약한 계절성"):
        reasons.append(f"{se['verdict']}(F 검정 p = {se.get('p_text')})")
        advice.append("월 자료면 pct12(전년 같은 달 대비), 분기 자료면 pct4 로 바꾸는 것을 검토" if sa != "조정됨"
                      else "계절조정 자료인데도 계절 패턴이 남아 있음 — 출처·단위를 확인")
    if gr["gaps"]:
        reasons.append(f"빈 기간 {len(gr['gaps'])}곳(가장 긴 것 {max(g['days'] for g in gr['gaps'])}일)")
    if gr["reversals"]:
        reasons.append(f"확인해 볼 날 {len(gr['reversals'])}일")
    if "transform_error" in row:
        reasons.append("변환을 계산하지 못함")
    signal = "회색" if st["verdict"] in ("자료 부족", "판단 불가") else ("노랑" if reasons else "초록")
    return {**row, "stationarity": st, "seasonality": se, "gaps": len(gr["gaps"]), "gap_list": gr["gaps"][:20],
            "reversals": len(gr["reversals"]), "reversal_dates": [r["date"] for r in gr["reversals"]][:20],
            "signal": signal, "reasons": reasons, "advice": advice}


def check_table(dv, reg, ids: list[str] | None = None, transforms: dict | None = None, upto: str | None = None) -> list[dict]:
    """점검표. ids 를 주지 않으면 사용 가능한 변수 전부(사전 순서). transforms = 변수별 변환 덮어쓰기(실험 설정)."""
    ids = list(ids) if ids is not None else reg.ids()
    long = dv.read_variables(ids) if ids else pd.DataFrame(columns=["variable_id", "date", "value"])
    out = []
    for vid in ids:
        part = long[long["variable_id"] == vid] if len(long) else long
        out.append(check_variable(reg.get(vid), part, (transforms or {}).get(vid), upto))
    return out


# ---------------------------------------------------------------------------
# 다른 출처와 대조(2026-11-14, 사용자 결정: 표시만 — 값을 바꾸지 않는다).
# 변수 사전의 duplicate_of 짝(같은 자료를 다른 출처에서 받은 것, 예: 엑셀 L1 ↔ 야후 USDKRW)을 같은 기간끼리 맞춰 본다.
#   ① 기간 맞추기  두 주기 중 느린 쪽(월이 있으면 월, 아니면 일)으로 묶는다. 느린 쪽 자료는 그 기간의 마지막 값.
#                  빠른 쪽(일간)은 월말·월평균·월초 세 정의로 각각 묶어 보고 차이의 흩어짐(MAD)이 가장 작은 정의를 쓴다
#                  (출처마다 '월 값'의 정의가 다르다 — 예: 엑셀 국고채는 월평균으로 보임). 쓴 정의를 기록한다.
#   ② 차이        두 값이 모두 양수면 로그 차 d = ln a − ln b(≈ 상대 차이), 아니면 그냥 차 d = a − b.
#   ③ 확인해 볼 기간  |d − 중앙값(d)| > max(8 × 1.4826·MAD(d), 바닥) — 바닥: 로그 차 0.01(≈1%), 그냥 차는 b 크기 중앙값의 1%.
#                  (8 은 되돌림 검사와 같은 강건 z. 바닥은 두 출처가 거의 같을 때(MAD = 0) 아주 작은 차이까지 걸리지 않게.)
#   ④ 변화 상관   두 출처의 기간별 변화(차분)의 상관 — 0.9 미만이면 정의·단위가 다를 수 있다고 알린다.
# 신호: 회색 = 한쪽에 값이 없음·겹친 기간 12개 미만 · 노랑 = 확인해 볼 기간이 있음 또는 변화 상관 < 0.9 · 초록 = 걸린 것 없음.
# 두 출처만으로는 어느 쪽이 맞는지 알 수 없다 — 각 기간의 두 값을 보여 주고, 각 출처 자신의 '확인해 볼 날'과 겹치면 그 사실을 적는다.
# ---------------------------------------------------------------------------
CROSS_Z = OUTLIER_Z
CROSS_FLOOR_LOG = 0.01
CROSS_FLOOR_REL = 0.01
CROSS_MIN_OVERLAP = 12
CROSS_MIN_CORR = 0.9
CROSS_RULE = ("다른 출처와 대조: 변수 사전에서 같은 자료로 묶인 짝(duplicate_of)을 같은 기간끼리 맞춰 차이를 봅니다. "
              "일간 자료를 월 자료와 맞출 때는 월말·월평균·월초 중 가장 잘 맞는 정의를 씁니다. "
              "차이가 평소 차이에서 크게 벗어난 기간(강건 z > 8, 최소 약 1%)을 '확인해 볼 기간'으로, 두 출처의 변화 상관이 0.9 미만이면 '정의·단위가 다를 수 있음'으로 알립니다. "
              "두 출처만으로는 어느 쪽이 맞는지 알 수 없습니다. 값은 바꾸지 않습니다.")
_AGG = {"말일": "last", "평균": "mean", "첫날": "first"}


def _period_values(s: pd.Series, per: str, how: str) -> pd.Series:
    g = s.groupby(s.index.to_period(per))
    return getattr(g, how)()


def cross_pair(va, sa: pd.Series, vb, sb: pd.Series, upto: str | None = None) -> dict:
    """한 짝(a = duplicate_of 를 가진 변수, b = 대표)의 대조 줄. 값을 바꾸지 않는다."""
    if upto is not None:
        sa, sb = sa[sa.index < pd.Timestamp(upto)], sb[sb.index < pd.Timestamp(upto)]
    sa, sb = sa.dropna(), sb.dropna()
    row = {"id": va.id, "name": va.name, "source": va.source, "freq": va.freq,
           "ref_id": vb.id, "ref_name": vb.name, "ref_source": vb.source, "ref_freq": vb.freq,
           "n_a": int(len(sa)), "n_b": int(len(sb))}
    if not len(sa) or not len(sb):
        who = va.id if not len(sa) else vb.id
        return {**row, "signal": "회색", "n": 0, "reasons": [f"{who} 의 값이 이 데이터 버전에 없어 대조할 수 없습니다"], "flags": []}
    per = "M" if "M" in (va.freq, vb.freq) or "Q" in (va.freq, vb.freq) else "D"
    if "Q" in (va.freq, vb.freq):
        per = "Q"
    # 느린 쪽은 기간의 마지막 값, 빠른 쪽은 세 정의 중 가장 잘 맞는 것
    fast_a = per != "D" and va.freq == "D"
    fast_b = per != "D" and vb.freq == "D"
    cands = list(_AGG) if (fast_a or fast_b) else ["말일"]
    best = None
    for name in cands:
        pa = _period_values(sa, per, _AGG[name] if fast_a else "last")
        pb = _period_values(sb, per, _AGG[name] if fast_b else "last")
        j = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
        if len(j) < 2:
            continue
        pos = bool((j > 0).all().all())
        d = (np.log(j["a"]) - np.log(j["b"])) if pos else (j["a"] - j["b"])
        med = float(d.median()); mad = float((d - med).abs().median()) * 1.4826
        if best is None or mad < best["mad"]:
            best = {"def": name, "j": j, "d": d, "pos": pos, "med": med, "mad": mad}
    if best is None or len(best["j"]) < CROSS_MIN_OVERLAP:
        n = 0 if best is None else len(best["j"])
        return {**row, "signal": "회색", "n": int(n), "reasons": [f"겹친 기간이 {n}개 — {CROSS_MIN_OVERLAP}개 이상이어야 대조합니다"], "flags": []}
    j, d, pos, med, mad = best["j"], best["d"], best["pos"], best["med"], best["mad"]
    floor = CROSS_FLOOR_LOG if pos else CROSS_FLOOR_REL * float(j["b"].abs().median())
    tol = max(CROSS_Z * mad, floor)
    hit = (d - med).abs() > tol
    ch = j.diff().dropna()
    corr = float(ch["a"].corr(ch["b"])) if len(ch) >= 3 and ch["a"].std() > 0 and ch["b"].std() > 0 else None
    # 각 출처 자신의 '확인해 볼 날'(되돌림)과 겹치는지 — 어느 쪽이 튄 것인지 짐작할 단서(단정하지 않음)
    rev_a = {r["date"] for r in gap_and_reversal(sa, va.freq)["reversals"]}
    rev_b = {r["date"] for r in gap_and_reversal(sb, vb.freq)["reversals"]}
    flags = []
    for t in d[hit].abs().sort_values(ascending=False).index:
        p = t if isinstance(t, pd.Period) else pd.Period(t, per)
        days = {str(x.date()) for x in pd.date_range(p.start_time, p.end_time, freq="D")} if per != "D" else {str(p.start_time.date())}
        clue = []
        if days & rev_a:
            clue.append(f"{va.id} 쪽에 하루 튀었다 되돌아온 날이 있음")
        if days & rev_b:
            clue.append(f"{vb.id} 쪽에 하루 튀었다 되돌아온 날이 있음")
        flags.append({"period": str(p), "a": float(j.loc[t, "a"]), "b": float(j.loc[t, "b"]),
                      "diff": float(d[t]), "diff_kind": "로그 차(≈상대 차이)" if pos else "차", "clue": clue})
    reasons = []
    if flags:
        reasons.append(f"확인해 볼 기간 {len(flags)}개(차이가 평소보다 크게 벗어남)")
    if corr is not None and corr < CROSS_MIN_CORR:
        reasons.append(f"두 출처의 변화 상관 {corr:.2f} — 정의·단위가 다를 수 있습니다")
    return {**row, "signal": "노랑" if reasons else "초록", "period": per, "definition": best["def"] if (fast_a or fast_b) else None,
            "n": int(len(j)), "first": str(j.index.min()), "last": str(j.index.max()),
            "diff_kind": "로그 차(≈상대 차이)" if pos else "차", "median_diff": med, "mad": mad, "tolerance": tol,
            "max_abs_diff": float(d.abs().max()), "change_corr": corr, "reasons": reasons, "flags": flags[:30], "n_flags": len(flags)}


def cross_table(dv, reg, ids: list[str] | None = None, upto: str | None = None) -> list[dict]:
    """다른 출처 대조표 — duplicate_of 짝 전부(ids 를 주면 그 변수 또는 그 대표가 들어간 짝만). 값을 바꾸지 않는다."""
    pairs = []
    for vid in reg.ids(usable_only=False):
        v = reg.get(vid)
        if v.duplicate_of and v.duplicate_of in reg:
            if ids is None or vid in ids or v.duplicate_of in ids:
                pairs.append((v, reg.get(v.duplicate_of)))
    if not pairs:
        return []
    need = sorted({x.id for p in pairs for x in p})
    long = dv.read_variables(need)
    out = []
    for va, vb in pairs:
        sa, _ = series_of(long[long["variable_id"] == va.id])
        sb, _ = series_of(long[long["variable_id"] == vb.id])
        out.append(cross_pair(va, sa, vb, sb, upto))
    return out
