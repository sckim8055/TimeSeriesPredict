"""켈리 투자 비율 + 자신감 확인표(2026-11-15). 시험을 먼저 쓴다.
켈리: 결정일 t 의 비율은 't 에 이미 결과가 나온' 과거 예측·실현 수익만으로 정한다(누출 없음).
  과거 예측을 자신감 순 5묶음으로 나눔 → t 의 예측이 든 묶음의 과거 실현 수익 평균 μ·분산 σ² → 비율 = 분수 × μ/σ², 0~1 로 자름.
자신감 확인표: 평가 구간의 예측을 5묶음으로 나눠 묶음별 적중률·평균 실현 수익 — 자신감이 클수록 실제로 좋아지는지."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.sizing import confidence_table, kelly_positions


def _series(n=1200, seed=0, signal=0.004, noise=0.02, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    z = rng.normal(size=n)
    pred = pd.Series(z, index=idx)
    realized = pd.Series(drift + signal * z + noise * rng.normal(size=n), index=idx)
    return pred, realized


def test_warmup_is_cash_and_formula_by_hand():
    pred, real = _series(400, seed=1)
    w, info = kelly_positions(pred, real, k=1, fraction=0.5, min_history=100, n_bins=5)
    # 처음에는 결과가 나온 과거 예측이 100개가 될 때까지 현금(0)
    assert (w.iloc[:101] == 0).all() and info["n_warmup"] >= 101
    # 손 계산: 한 결정일에서 과거(위치 ≤ t−1−k)만으로
    t = 300
    past = np.arange(0, t - 1 - 1 + 1)
    pp, rr = pred.values[past], real.values[past]
    edges = np.quantile(pp, [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(edges, pred.values[t], side="right")
    pb = np.searchsorted(edges, pp, side="right")
    mu, var = rr[pb == b].mean(), rr[pb == b].var(ddof=1)
    assert np.isclose(w.iloc[t], float(np.clip(0.5 * mu / var, 0, 1)))


def test_no_future_used():
    """j 이후 실현 수익을 바꿔도, 그 값이 아직 나오지 않은 결정일(t − 1 − k < j)의 비율은 그대로."""
    pred, real = _series(600, seed=2)
    for k in (1, 3):
        w1, _ = kelly_positions(pred, real, k=k, fraction=0.5, min_history=100)
        j = 350
        r2 = real.copy(); r2.iloc[j:] = -r2.iloc[j:] * 5
        w2, _ = kelly_positions(pred, r2, k=k, fraction=0.5, min_history=100)
        safe = np.arange(len(pred)) - 1 - k < j
        assert np.allclose(w1.values[safe], w2.values[safe]), k
        assert not np.allclose(w1.values[~safe], w2.values[~safe])     # 바꾼 뒤는 실제로 달라짐(시험이 헛돌지 않음)


def test_more_confident_gets_bigger_weight_and_caps():
    pred, real = _series(1500, seed=3, signal=0.006)
    w, info = kelly_positions(pred, real, k=1, fraction=0.5, min_history=200)
    late = w.iloc[600:]
    hi, lo = late[pred.iloc[600:] > pred.quantile(0.8)], late[pred.iloc[600:] < pred.quantile(0.2)]
    assert hi.mean() > 0.5 and lo.mean() < 0.05
    assert (w >= 0).all() and (w <= 1).all()
    w_full, _ = kelly_positions(pred, real, k=1, fraction=1.0, min_history=200)
    assert (w_full.iloc[600:] >= w.iloc[600:] - 1e-12).all()          # 분수가 크면 비율도 크다(같은 자료)


def test_missing_predictions_stay_missing():
    pred, real = _series(300, seed=4)
    pred.iloc[250:260] = np.nan
    w, _ = kelly_positions(pred, real, k=1, fraction=0.5, min_history=50)
    assert w.iloc[250:260].isna().all() and w.iloc[100:250].notna().all()


def test_confidence_table_detects_informative_and_uninformative():
    pred, real = _series(2000, seed=5, signal=0.006)
    t = confidence_table(pred, real)
    assert len(t["rows"]) == 5 and sum(r["n"] for r in t["rows"]) == 2000
    m = [r["mean_realized"] for r in t["rows"]]
    assert m == sorted(m) and t["rank_corr"] > 0.9 and t["verdict"] == "맞음"
    assert t["rows"][-1]["hit_rate"] > t["rows"][0]["hit_rate"]
    pred0, real0 = _series(2000, seed=6, signal=0.0)
    t0 = confidence_table(pred0, real0)
    assert t0["verdict"] != "맞음"


def test_engine_kelly_end_to_end(tmp_path):
    from engine.experiment import ExperimentConfig, Validation, run_experiment
    from tests.synthetic import predictable_ohlcv
    from tests.test_engine import _dv_with
    prices, s = predictable_ohlcv(1000, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    base = dict(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=250, test_len=40))
    a = ExperimentConfig(**base)
    b = ExperimentConfig(**base, trade={"sizing": "kelly"})
    # 기본값(문턱)이면 설정 지문은 켈리 칸이 생기기 전과 같다
    assert a.config_hash() == ExperimentConfig(**base, trade={"sizing": "threshold", "kelly_fraction": 0.5}).config_hash()
    assert a.config_hash() != b.config_hash()
    ra, rb = run_experiment(a, dv, reg), run_experiment(b, dv, reg)
    assert rb["trade_rule_used"]["kind"] == "kelly" and rb["trade_rule_used"]["fraction"] == 0.5
    pos = rb["_daily"]["position"].loc[rb["evaluation_period"][0]: rb["evaluation_period"][1]]
    assert (pos >= 0).all() and (pos <= 1).all() and ((pos > 0) & (pos < 1)).any()
    for r in (ra, rb):
        ct = r["confidence_table"]
        assert len(ct["rows"]) == 5 and ct["verdict"] == "맞음"      # 심은 신호 → 자신감이 클수록 실제로 좋음
    # 예측 자체는 같다(켈리는 예측 뒤 비율만 바꿈)
    pa, pb = ra["_daily"]["prediction"].dropna(), rb["_daily"]["prediction"].dropna()
    assert np.allclose(pa.values, pb.reindex(pa.index).values)


def test_engine_kelly_rejected_where_meaningless(tmp_path):
    from engine.experiment import ExperimentConfig, Validation, run_experiment
    from tests.synthetic import predictable_ohlcv
    from tests.test_engine import _dv_with
    prices, s = predictable_ohlcv(600, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    with pytest.raises(ValueError, match="켈리"):
        run_experiment(ExperimentConfig(ticker="TEST", model="ma_cross", trade={"sizing": "kelly"}), dv, reg)
    with pytest.raises(ValueError, match="켈리"):
        run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], target={"kind": "volatility", "k": 5},
                                        trade={"sizing": "kelly"}, validation=Validation(train_len=250, test_len=40)), dv, reg)
    with pytest.raises(Exception):
        ExperimentConfig(ticker="TEST", model="ridge", trade={"sizing": "kelly", "kelly_fraction": 1.5})


def test_engine_kelly_future_perturbation(tmp_path):
    """T 뒤 가격만 흔들어도 결정일이 T 앞(결과가 나온 날 기준)인 켈리 비율은 그대로."""
    from engine.experiment import ExperimentConfig, Validation, run_experiment
    from tests.synthetic import predictable_ohlcv
    from tests.test_engine import _shake_after, _write_dv
    prices, s = predictable_ohlcv(900, seed=8)
    T = prices.index[650]
    dv1, reg1 = _write_dv(tmp_path / "a", {"AAA": prices}, {"S_signal": s})
    dv2, reg2 = _write_dv(tmp_path / "b", {"AAA": _shake_after(prices, T, 4)}, {"S_signal": s})
    c = ExperimentConfig(ticker="AAA", model="ridge", variable_ids=["S_signal"], use_technical=False,
                         trade={"sizing": "kelly", "kelly_min_history": 100, "kelly_fraction": 0.02}, validation=Validation(train_len=250, test_len=40))
    w1 = run_experiment(c, dv1, reg1)["_daily"]["position"]
    w2 = run_experiment(c, dv2, reg2)["_daily"]["position"]
    early = w1.index[(w1.index < prices.index[640])]
    assert np.allclose(w1.loc[early].fillna(-1).values, w2.reindex(early).fillna(-1).values, atol=1e-10)
    assert not np.allclose(w1.loc[w1.index >= T].fillna(-1).values[:80], w2.reindex(w1.index[w1.index >= T][:80]).fillna(-1).values)


def test_default_trade_hash_equals_pre_kelly_hash():
    """설정 지문: 켈리 칸이 생기기 전 클래스로 만든 JSON(새 세 칸을 뺀 것)의 해시와 기본값(문턱) 설정의 해시가 같다 — 예전 결과와의 연결 유지.
    (2026-11-15 클라우드 results/ 49개 결과로도 확인: 새 지문 = 켈리 전 방식 지문 49/49.)"""
    import hashlib, json
    from engine.experiment import ExperimentConfig
    for c in (ExperimentConfig(ticker="TEST", model="ridge"), ExperimentConfig(ticker="TEST", model="lightgbm_model", trade={"threshold": 0.001, "fee_rate": 0.0002}, n_seeds=3)):
        ex = {"name"} | ({"technical_features"} if c.technical_features is None else set()) | ({"n_seeds"} if c.n_seeds == 1 else set()) | {"select_top", "pool_tickers"}
        o = json.loads(c.model_dump_json(exclude=ex))
        for k in ("sizing", "kelly_fraction", "kelly_min_history"):
            o["trade"].pop(k)
        assert c.config_hash() == hashlib.sha1(json.dumps(o, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:12]
