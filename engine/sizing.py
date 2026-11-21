"""투자 비율 정하기(2026-11-15): 켈리 기준 + 자신감 확인표.

켈리 기준(Kelly 1956, 소프가 블랙잭·시장에 적용, 르네상스 초기 벌캠프도 썼다고 전함): 이길 확률·크기를 알면 오래 보아 재산을 가장 빨리
불리는 베팅 비율이 있다. 수익률이 평균 μ·분산 σ² 이면 그 비율은 f* = μ / σ² (현금 수익 0, 작은 수익률 근사).
  · 실제 μ·σ² 는 모른다 — 모델 예측을 그대로 μ 로 쓰면 자신감이 지나치다(예측은 보통 크기가 맞지 않음).
  · 그래서 '과거에 이 정도로 자신 있게 예측했을 때 실제로 얼마였나'로 바꾼다(보정):
      결정일 t 에 결과가 이미 나온 과거 예측(위치 s ≤ t − 1 − k: 다음 날 시가에 사서 k 기간 뒤 시가에 판 수익이 t 이전에 확정)을
      예측 크기 순으로 n_bins(기본 5) 묶음으로 나누고, t 의 예측이 든 묶음의 과거 실현 수익 평균 μ_b·분산 σ²_b 로
      비율 w_t = 분수 × μ_b / σ²_b 를 0~1 로 자른다(빌리지 않음, 공매도 없음).
  · 분수(fraction, 기본 0.5 = '절반 켈리'): 전부 켈리는 μ·σ² 추정이 조금만 틀려도 크게 잃을 수 있어 보통 절반 이하를 쓴다(소프도 그보다 적게 걸었다고 적음).
  · 과거가 min_history 개가 될 때까지는 현금(0) — 준비 기간. 그만큼 평가 구간 앞부분이 현금으로 계산된다(숨기지 않고 결과에 개수를 적는다).
자신감 확인표: 평가 구간의 예측을 크기 순 5묶음으로 나눠 묶음마다 개수·평균 예측·적중률(실현 수익 > 0)·평균 실현 수익.
  자신감이 클수록 실제 수익이 좋아지면(묶음 번호와 평균 실현 수익의 순위상관 ≥ 0.9 이고 맨 위 − 맨 아래 > 0) '맞음'.
  이 표는 평가 구간 전체를 한꺼번에 본 진단이라 매매 결정에는 쓰지 않는다(켈리는 위처럼 과거만).
  k > 1 이면 이웃한 관측의 수익 기간이 겹쳐 서로 독립이 아니다 — t 값은 참고용으로만 적는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KELLY_MIN_HISTORY = {"D": 250, "W": 52, "M": 24}


def realized_hold_return(prices: pd.DataFrame, k: int) -> pd.Series:
    """결정일 t 의 실현 보유 수익 = t+1 시가에 사서 t+1+k 시가에 판 수익(엔진의 매매 회계와 같은 정의)."""
    o = prices["open"]
    return o.shift(-(1 + k)) / o.shift(-1) - 1


def kelly_positions(pred: pd.Series, realized: pd.Series, *, k: int, fraction: float = 0.5, min_history: int = 250,
                    n_bins: int = 5, cap: float = 1.0) -> tuple[pd.Series, dict]:
    """반환 (비율 w — 예측이 없는 날은 NaN, 정보 dict). 결정일 t 에는 위치 ≤ t−1−k 인 과거만 쓴다."""
    p = pred.to_numpy(dtype=float)
    r = realized.reindex(pred.index).to_numpy(dtype=float)
    n = len(p)
    w = np.full(n, np.nan)
    ok = np.isfinite(p)
    usable = ok & np.isfinite(r)
    n_warm, rows = 0, 0
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for t in np.flatnonzero(ok):
        rows += 1
        last = t - 1 - k
        if last < 0:
            w[t] = 0.0; n_warm += 1; continue
        m = usable[: last + 1]
        if m.sum() < min_history:
            w[t] = 0.0; n_warm += 1; continue
        pp, rr = p[: last + 1][m], r[: last + 1][m]
        edges = np.quantile(pp, qs)
        b = np.searchsorted(edges, p[t], side="right")
        sel = rr[np.searchsorted(edges, pp, side="right") == b]
        if len(sel) < 2:
            w[t] = 0.0; continue
        mu, var = float(sel.mean()), float(sel.var(ddof=1))
        w[t] = 0.0 if var <= 0 else float(np.clip(fraction * mu / var, 0.0, cap))
    ws = pd.Series(w, index=pred.index)
    return ws, {"n_decisions": int(rows), "n_warmup": int(n_warm), "fraction": float(fraction), "min_history": int(min_history), "n_bins": int(n_bins),
                "mean_weight": float(np.nanmean(w)) if rows else None}


def confidence_table(pred: pd.Series, realized: pd.Series, n_bins: int = 5) -> dict:
    """자신감 확인표(진단용, 평가 구간 전체). 반환 {rows, rank_corr, top_minus_bottom, t_top_bottom, verdict, rule}."""
    d = pd.concat([pred.rename("p"), realized.reindex(pred.index).rename("r")], axis=1).dropna()
    rule = ("예측을 크기 순으로 5묶음(같은 개수)으로 나눠 묶음마다 실제 결과를 셉니다. 자신감(예측)이 클수록 실제 평균 수익이 커지면 — "
            "묶음 번호와 평균 실현 수익의 순위상관 0.9 이상, 맨 위 묶음 − 맨 아래 묶음 > 0 — '맞음'. 평가 구간 전체를 본 진단이라 매매 결정에는 쓰지 않습니다.")
    if len(d) < n_bins * 10:
        return {"rows": [], "rank_corr": None, "top_minus_bottom": None, "t_top_bottom": None, "verdict": "자료 부족", "rule": rule, "n": int(len(d))}
    b = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    rows = []
    for i in range(n_bins):
        g = d[b == i]
        rows.append({"bin": i + 1, "n": int(len(g)), "pred_mean": float(g["p"].mean()), "pred_min": float(g["p"].min()), "pred_max": float(g["p"].max()),
                     "hit_rate": float((g["r"] > 0).mean()), "mean_realized": float(g["r"].mean()), "sd_realized": float(g["r"].std(ddof=1))})
    m = np.array([x["mean_realized"] for x in rows])
    rc = float(pd.Series(m).rank().corr(pd.Series(np.arange(n_bins, dtype=float)), method="pearson"))
    top, bot = d[b == n_bins - 1]["r"], d[b == 0]["r"]
    diff = float(top.mean() - bot.mean())
    se = float(np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot)))
    tstat = diff / se if se > 0 else None
    verdict = "맞음" if (rc >= 0.9 and diff > 0) else ("반대" if (rc <= -0.9 and diff < 0) else "맞지 않음")
    return {"rows": rows, "rank_corr": rc, "top_minus_bottom": diff, "t_top_bottom": tstat, "verdict": verdict, "rule": rule, "n": int(len(d))}
