"""HMM 시장 상태(2026-11-15). 새 모델 등록 조건: 심은 신호를 찾고, 랜덤워크에서 무능하고, 미래를 쓰지 않는다.
여기에 더해 앞방향 계산을 모든 경로를 다 세는 손 계산과 맞추고, EM 이 로그우도를 줄이지 않음을 본다."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from engine.experiment import ExperimentConfig, Validation, run_experiment
from models.base import resolve_params
from models.hmm import PARAMS, Model, forward
from tests.synthetic import random_walk_ohlcv
from tests.test_engine import _dv_with, _shake_after, _write_dv


def _regimes(n=3000, seed=0, stay=0.98, mu=(-0.003, 0.002), sd=(0.03, 0.01)):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    return rng.normal(np.array(mu)[s], np.array(sd)[s]), s


def _model(**kw):
    return Model(resolve_params(PARAMS, kw, "D"), n_features=1, target_type="regression")


def test_forward_matches_brute_force_enumeration():
    """앞방향 필터 P(s_T | r_1..r_T)와 로그우도를, 상태 경로 2^6 개를 모두 세는 손 계산과 맞춘다."""
    r = np.array([0.5, -1.2, 0.3, 2.0, -0.4, 0.1])
    pi = np.array([0.6, 0.4]); A = np.array([[0.9, 0.1], [0.2, 0.8]]); mu = np.array([-0.5, 0.7]); var = np.array([1.5, 0.4])
    dens = lambda x, j: np.exp(-0.5 * (np.log(2 * np.pi * var[j]) + (x - mu[j]) ** 2 / var[j]))
    for T in range(1, len(r) + 1):
        tot = np.zeros(2)
        for path in itertools.product([0, 1], repeat=T):
            p = pi[path[0]] * dens(r[0], path[0])
            for t in range(1, T):
                p *= A[path[t - 1], path[t]] * dens(r[t], path[t])
            tot[path[-1]] += p
        a, c = forward(r[:T], pi, A, mu, var)
        assert np.allclose(a[-1], tot / tot.sum(), atol=1e-12)
        assert np.isclose(np.sum(np.log(c)), np.log(tot.sum()), atol=1e-10)


def test_recovers_planted_regimes_and_tracks_state():
    r, s = _regimes()
    m = _model(); m.fit_series(r[:2000])
    d = m.diag
    assert d["converged"] and abs(d["mean_per_period"][0] + 0.003) < 0.0015 and abs(d["mean_per_period"][1] - 0.002) < 0.001
    assert abs(d["vol_per_period"][0] - 0.03) < 0.003 and abs(d["vol_per_period"][1] - 0.01) < 0.0015
    assert min(d["stay_prob"]) > 0.93
    acc = np.mean([(m.filtered(r[:t + 1]).argmax() == 0) == (s[t] == 0) for t in range(2000, 3000)])
    assert acc > 0.85, acc
    # 예측 = (p · A^h) · μ — 상태가 '하락기'일 확률이 높을 때 예측도 낮다
    f = np.array([m.forecast(r[:t + 1], 1)[0] for t in range(2000, 3000)])
    assert f[s[2000:3000] == 0].mean() < f[s[2000:3000] == 1].mean()


def test_em_never_lowers_likelihood_and_is_deterministic():
    r, _ = _regimes(1500, seed=4)
    m = _model(n_states=3); m.fit_series(r)
    h = np.array(m.loglik_history)
    assert np.all(np.diff(h) >= -1e-6), np.diff(h).min()
    m2 = _model(n_states=3); m2.fit_series(r)
    assert np.array_equal(m.mu, m2.mu) and np.array_equal(m.A, m2.A)
    assert list(m.mu) == sorted(m.mu) and len(m.diag["tried"]) == 2 and m.diag["loglik"] >= max(t["loglik"] for t in m.diag["tried"]) - 1e-6


def test_incremental_filter_equals_fresh_and_forecast_formula():
    """같은 폴드 안에서 앞 계산을 이어 쓰는 것(속도용)과 처음부터 다시 계산한 것이 같다. 앞이 다른 history 면 처음부터."""
    r, _ = _regimes(800, seed=2)
    m = _model(); m.fit_series(r[:500])
    for t in (500, 501, 530, 799):
        fresh = _model(); fresh.pi, fresh.A, fresh.mu, fresh.var, fresh.scale, fresh._cache = m.pi, m.A, m.mu, m.var, m.scale, None
        assert np.allclose(m.filtered(r[:t]), fresh.filtered(r[:t]), atol=1e-13)
    other = r[:600].copy(); other[10] += 0.05                       # 앞부분이 다르다 → 이어 쓰면 안 됨
    fresh = _model(); fresh.pi, fresh.A, fresh.mu, fresh.var, fresh.scale, fresh._cache = m.pi, m.A, m.mu, m.var, m.scale, None
    assert np.allclose(m.filtered(other), fresh.filtered(other), atol=1e-13)
    p = m.filtered(r[:700]); fc = m.forecast(r[:700], 3)
    for h in range(3):
        assert np.isclose(fc[h], (p @ np.linalg.matrix_power(m.A, h + 1)) @ m.mu / m.scale)


def test_hmm_has_no_edge_on_random_walk(tmp_path):
    prices = random_walk_ohlcv(1500, seed=2)
    dv, reg = _dv_with(prices, None, tmp_path)
    res = run_experiment(ExperimentConfig(ticker="TEST", model="hmm", use_technical=False, validation=Validation(train_len=500, test_len=40)), dv, reg)
    acc = res["prediction_metrics"]["dir_acc"]
    assert 0.40 < acc < 0.60, f"랜덤워크에서 방향 정확도 {acc:.3f} — 누출 의심"
    fd = res["periods"]["first_fold"]["model_diag"]
    assert "tried" not in fd and len(fd["stay_prob"]) == 2


def test_hmm_future_perturbation_leaves_past_predictions_unchanged(tmp_path):
    """T 이후 가격만 흔들어도 결정일이 T 앞인 예측은 그대로(소수 10자리) — 학습은 학습 창, 상태 확률은 앞방향만 쓴다는 주장의 직접 시험."""
    prices = random_walk_ohlcv(900, seed=7)
    T = prices.index[600]
    dv1, reg1 = _write_dv(tmp_path / "a", {"AAA": prices}, {})
    dv2, reg2 = _write_dv(tmp_path / "b", {"AAA": _shake_after(prices, T, 3)}, {})
    c = ExperimentConfig(ticker="AAA", model="hmm", use_technical=False, validation=Validation(train_len=300, test_len=50))
    p1 = run_experiment(c, dv1, reg1)["_daily"]["prediction"].dropna()
    p2 = run_experiment(c, dv2, reg2)["_daily"]["prediction"].dropna()
    early = p1.index[p1.index < T - pd.Timedelta(days=3)]
    assert len(early) >= 200
    assert np.allclose(p1.loc[early].values, p2.reindex(early).values, atol=1e-10, rtol=0)
    late = p1.index[p1.index >= T][:50]
    assert not np.allclose(p1.loc[late].values, p2.reindex(late).values, atol=1e-8)


def test_hmm_runs_in_forecast_mode(tmp_path):
    """예측 연구(모델 비교) 모드에서도 돈다 — ARIMA 와 같은 자리."""
    import tests.test_engine as te
    idx = pd.date_range("2005-01-31", periods=200, freq="ME")
    r, _ = _regimes(200, seed=9, mu=(-0.004, 0.004), sd=(0.01, 0.004))
    lvl = pd.Series(100 * np.cumprod(1 + r), index=idx)
    prices = random_walk_ohlcv(1500, seed=1)
    dv, reg = _dv_with(prices, lvl.rename("X"), tmp_path)
    vid = [v for v in reg.ids() if v not in ("TEST",)][0]
    res = run_experiment(ExperimentConfig(model="hmm", mode="forecast", target_variable=vid, use_technical=False,
                                          validation=Validation(train_len=100, test_len=12)), dv, reg)
    assert res["n_folds"] > 3 and res["prediction_metrics"]["dir_acc"] is not None, res["prediction_metrics"]
