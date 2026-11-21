"""섀넌 재조정 기준선(2026-11-15). 시험을 먼저 쓴다.
핵심: 비율은 가격을 따라 흘러가고(오르면 늘고 내리면 줄고), 정한 때에만 다시 맞추며 그때만 비용을 낸다.
기존 모델(rebalance 없음)의 계산은 한 글자도 바뀌지 않아야 한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest import CostModel, period_end_mask, run_backtest


def _prices(opens, start="2024-01-01", freq="B"):
    idx = pd.date_range(start, periods=len(opens), freq=freq)
    o = pd.Series(opens, index=idx, dtype=float)
    return pd.DataFrame({"open": o, "high": o, "low": o, "close": o, "volume": 1.0})


def test_drift_by_hand():
    """손으로 계산: 시가 100→110→99→99, 비율 0.5, 첫날만 맞춤(그 뒤 재조정 없음), 비용 0.001.
    t0 결정 → t1 시가에 0.5 로 맞춤(비용 0.5×0.001). 보유 수익 t0: 99/110−1 = −10% → 수익 −5%, 비용 0.0005.
    흘러간 비율 = 0.5×0.9 / (1 − 0.05) = 0.47368… → t1 의 보유 비율(맞추지 않음, 비용 0)."""
    p = _prices([100, 110, 99, 99, 120, 120])
    sig = pd.Series(0.5, index=p.index)
    reb = pd.Series(False, index=p.index)
    daily, _ = run_backtest(p, sig, cost=CostModel(fee_rate=0.001, slippage_bp=0), rebalance=reb)
    r0 = 99 / 110 - 1
    assert abs(daily["position"].iloc[0] - 0.5) < 1e-12 and abs(daily["turnover"].iloc[0] - 0.5) < 1e-12
    assert abs(daily["net_return"].iloc[0] - (0.5 * r0 - 0.0005)) < 1e-12
    w1 = 0.5 * (1 + r0) / (1 + 0.5 * r0)
    assert abs(daily["position"].iloc[1] - w1) < 1e-12 and daily["turnover"].iloc[1] == 0 and daily["cost"].iloc[1] == 0
    r2 = 120 / 99 - 1                                                     # t2 보유: 99 → 120
    w2 = w1 * (1 + 0) / (1 + w1 * 0)                                      # t1 보유 수익 99/99−1 = 0 → 비율 그대로
    assert abs(daily["position"].iloc[2] - w2) < 1e-12 and abs(daily["net_return"].iloc[2] - w2 * r2) < 1e-12


def test_rebalance_day_trades_back_to_target_and_pays_cost():
    p = _prices([100, 110, 99, 99, 120, 120])
    sig = pd.Series(0.5, index=p.index)
    reb = pd.Series([False, True, False, False, False, False], index=p.index)
    daily, s = run_backtest(p, sig, cost=CostModel(fee_rate=0.001, slippage_bp=0), rebalance=reb)
    w_drift = 0.5 * (99 / 110) / (1 + 0.5 * (99 / 110 - 1))
    assert abs(daily["position"].iloc[1] - 0.5) < 1e-12
    assert abs(daily["turnover"].iloc[1] - abs(0.5 - w_drift)) < 1e-12
    assert abs(daily["cost"].iloc[1] - 0.001 * abs(0.5 - w_drift)) < 1e-12
    assert s["n_trades"] == 2 and abs(s["turnover"] - (0.5 + abs(0.5 - w_drift))) < 1e-12


def test_no_rebalance_argument_is_unchanged():
    """rebalance 를 주지 않으면 예전과 같은 계산(0/1 신호, 0.5 같은 중간값 모두)."""
    r = np.random.default_rng(0)
    p = _prices(100 * np.exp(np.cumsum(r.normal(0, 0.02, 300))))
    for sig in (pd.Series((r.random(300) > 0.5).astype(float), index=p.index), pd.Series(0.5, index=p.index)):
        d, s = run_backtest(p, sig)
        pos = sig.clip(0, 1)
        hold = p["open"].shift(-2) / p["open"].shift(-1) - 1
        tov = pos.diff().abs().fillna(pos.abs())
        np.testing.assert_allclose(d["net_return"].values, (pos * hold - tov * CostModel().one_way).values)
        np.testing.assert_allclose(d["turnover"].values, tov.values)


def test_01_signals_same_with_or_without_drift():
    """0/1 신호는 흘러가도 0·1 그대로라, 바뀔 때만 사고파는 drift 계산과 예전 계산이 같다(정합성 확인)."""
    r = np.random.default_rng(1)
    p = _prices(100 * np.exp(np.cumsum(r.normal(0, 0.02, 200))))
    sig = pd.Series((r.random(200) > 0.5).astype(float), index=p.index)
    a, _ = run_backtest(p, sig)
    b, _ = run_backtest(p, sig, rebalance=pd.Series(False, index=p.index))
    np.testing.assert_allclose(a["net_return"].values, b["net_return"].values, atol=1e-12)


def test_period_end_mask():
    idx = pd.bdate_range("2024-01-25", "2024-04-05")
    m = period_end_mask(idx, "M")
    assert [str(d.date()) for d in idx[m]] == ["2024-01-31", "2024-02-29", "2024-03-29"]   # 데이터 끝의 기간(4월)은 끝났는지 알 수 없어 뺀다
    dm = period_end_mask(idx, "D")
    assert dm[:-1].all() and not dm[-1]
    q = period_end_mask(idx, "Q")
    assert [str(d.date()) for d in idx[q]] == ["2024-03-29"]


def test_shannon_model_and_benchmark_rule():
    """섀넌 모델: 비율은 날마다 같은 목표, 다시 맞추는 날 = 주기의 마지막 결정일(다음 주기 첫 시가에 체결).
    마지막 미완성 기간(데이터 끝)은 재조정 날이 아니다."""
    from models.shannon import Model, PARAMS
    p = _prices(np.linspace(100, 130, 70))
    m = Model({"weight": 0.3, "every": "M"})
    w = m.positions(p)
    assert np.allclose(w, 0.3)
    reb = m.rebalance_days(p)
    assert reb.dtype == bool and reb.sum() == len(set(p.index.to_period("M"))) - 1
    assert PARAMS["weight"]["default"] == 0.5 and PARAMS["every"]["default"] == "M"


def test_shannon_benchmark_helper():
    from engine.backtest import shannon_benchmark
    r = np.random.default_rng(2)
    p = _prices(100 * np.exp(np.cumsum(r.normal(0, 0.02, 400))))
    lo, hi = p.index[50], p.index[350]
    ret, met = shannon_benchmark(p, lo, hi, cost=CostModel(), periods_per_year=252)
    assert ret.index.min() == lo and ret.index.max() == hi
    assert met["weight"] == 0.5 and met["every"] == "M" and met["n_periods"] > 0
    assert set(["total_return", "cagr", "sharpe", "mdd", "total_cost"]) <= set(met)
    # 직접 계산과 같다
    sig = pd.Series(np.nan, index=p.index); sig.loc[lo:hi] = 0.5
    d, _ = run_backtest(p, sig, cost=CostModel(), rebalance=pd.Series(period_end_mask(p.index, "M"), index=p.index))
    tr = float((1 + d.loc[lo:hi, "net_return"].fillna(0)).prod() - 1)
    assert abs(met["total_return"] - tr) < 1e-12
