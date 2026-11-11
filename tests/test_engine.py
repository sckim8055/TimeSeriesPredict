"""엔진 검증 — 합성 데이터로 '값이 맞는지'를 본다 (돌아가는지가 아니라)."""
import json
import sys

import numpy as np
from pathlib import Path

import pandas as pd
import pytest

from engine import calendar as cal
from engine.backtest import CostModel, diebold_mariano, run_backtest
from engine.experiment import ExperimentConfig, TargetSpec, Validation, run_experiment
from engine.registry import Registry, Variable
from engine.store import DataVersion
from tests.synthetic import predictable_ohlcv, random_walk_ohlcv


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def test_target_alignment_is_strictly_future():
    """타겟 y[t] 는 open[t+1], open[t+1+k] 만 쓴다. 손으로 계산한 값과 일치해야 한다."""
    df = random_walk_ohlcv(50)
    for k in (1, 3):
        y = cal.make_target(df, k=k, kind="return")
        t = 10
        expected = df["open"].iloc[t + 1 + k] / df["open"].iloc[t + 1] - 1
        assert abs(y.iloc[t] - expected) < 1e-12
        assert y.iloc[-(1 + k) :].isna().all()  # 미래가 없는 마지막 행은 NaN


def test_align_ffill_has_no_future_info():
    """월간 값(기준일 월초, 발표시차 35일)을 일간으로 펼치면 공개일 전에는 이전 값이어야 한다."""
    calendar = pd.bdate_range("2024-01-01", "2024-06-30")
    monthly = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]))
    a = cal.align_to_calendar(monthly, calendar, lag_days=35, fill="ffill")
    # 2월 값(2.0)의 공개일 = 2024-03-07. 3월 6일에는 아직 1월 값(1.0)
    assert a.loc["2024-03-06"] == 1.0
    assert a.loc["2024-03-07"] == 2.0
    # 첫 공개 전 구간 NaN
    assert np.isnan(a.loc["2024-01-15"])


def test_align_bfill_leaks_future():
    calendar = pd.bdate_range("2024-01-01", "2024-03-31")
    monthly = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    b = cal.align_to_calendar(monthly, calendar, lag_days=0, fill="bfill")
    assert b.loc["2024-01-15"] == 2.0  # 아직 오지 않은 2월 값이 1월에 보임 = 누출
    assert cal.fill_leaks_future("bfill") and not cal.fill_leaks_future("ffill")


def test_walkforward_embargo_removes_overlap():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    k = 5
    folds = cal.walkforward_folds(calendar, train_len=200, test_len=20, k=k)
    for f in folds:
        gap = calendar.get_loc(f.test[0]) - calendar.get_loc(f.train[-1])
        assert gap >= 1 + k + 1 - 1  # 학습 마지막 결정일의 타겟(t+1+k)이 테스트 첫날 이전에 끝남
        assert f.train[-1] < f.test[0]


def test_resample_prices_uses_last_trading_day():
    df = random_walk_ohlcv(300)
    m = cal.resample_prices(df, "M")
    # 각 월의 대표일은 그 월의 실제 마지막 거래일
    for d in m.index[:5]:
        same_month = df.index[(df.index.year == d.year) & (df.index.month == d.month)]
        assert d == same_month.max()
        assert m.loc[d, "close"] == df.loc[d, "close"]
        assert m.loc[d, "open"] == df.loc[same_month.min(), "open"]


# ---------------------------------------------------------------------------
# backtest 회계
# ---------------------------------------------------------------------------
def test_backtest_matches_hand_calculation():
    df = random_walk_ohlcv(30)
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[5:10] = 1.0  # 5일 마감 결정 → 6일 시가 진입 … 10일 결정 0 → 11일 시가 청산
    cost = CostModel(fee_rate=0.001, slippage_bp=0)
    daily, perf = run_backtest(df, sig, cost=cost)
    o = df["open"].values
    expected_gross = np.prod([o[t + 2] / o[t + 1] for t in range(5, 10)]) - 1  # 6일 시가→11일 시가
    got_gross = np.prod(1 + daily["gross_return"].fillna(0)) - 1
    assert abs(got_gross - expected_gross) < 1e-12
    # 비용: 진입 1회 + 청산 1회 = 2 × 0.1%
    assert abs(daily["cost"].sum() - 0.002) < 1e-12
    assert perf["n_trades"] == 2


def test_buy_and_hold_equals_benchmark():
    df = random_walk_ohlcv(200)
    sig = pd.Series(1.0, index=df.index)
    daily, _ = run_backtest(df, sig, cost=CostModel(0, 0))
    v = daily.dropna()
    assert np.allclose(v["net_return"], v["hold_return"])


def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1, 500)
    e_bad = rng.normal(0, 2, 500)
    dm, p = diebold_mariano(e_good, e_bad)
    assert dm < 0 and p < 0.01


# ---------------------------------------------------------------------------
# 엔드투엔드: 모델 플러그인이 지켜야 할 두 테스트 (CLAUDE.md 절대규칙 6)
# ---------------------------------------------------------------------------
def _dv_with(prices: pd.DataFrame, extra: pd.Series | None, tmp_path) -> tuple[DataVersion, Registry]:
    dv = DataVersion(tmp_path, "t")
    dv.write_prices("TEST", prices)
    reg = Registry()
    if extra is not None:
        reg.add(Variable(id=extra.name, freq="D", source="DERIVED"))
        long = extra.rename("value").rename_axis("date").reset_index().assign(variable_id=extra.name)
        dv.write_variables(long, source="syn")
    return dv, reg


@pytest.mark.parametrize("model", ["ridge", "lightgbm_model"])
def test_model_finds_planted_signal(model, tmp_path):
    """숨은 신호 S 가 다음 수익률을 결정하는 시장에서, S 를 변수로 주면 방향 정확도가 우연(50%)을 크게 넘어야 한다."""
    prices, s = predictable_ohlcv(1200, seed=1)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=400, test_len=40))
    res = run_experiment(cfg, dv, reg)
    assert res["prediction_metrics"]["dir_acc"] > 0.75, res["prediction_metrics"]
    assert res["performance"]["sharpe"] > 1.0


@pytest.mark.parametrize("model", ["ridge", "lightgbm_model"])
def test_model_has_no_edge_on_random_walk(model, tmp_path):
    """랜덤워크에서는 어떤 모델도 방향 정확도가 우연 수준이어야 한다. 크게 넘으면 엔진 어딘가에 누출이 있다."""
    prices = random_walk_ohlcv(1500, seed=2)
    dv, reg = _dv_with(prices, None, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, validation=Validation(train_len=500, test_len=40))
    res = run_experiment(cfg, dv, reg)
    acc = res["prediction_metrics"]["dir_acc"]
    assert 0.40 < acc < 0.60, f"랜덤워크에서 방향 정확도 {acc:.3f} — 누출 의심"


def test_shuffled_target_destroys_signal(tmp_path):
    """누출 검사: 타겟을 섞으면 심어둔 신호도 사라져야 한다 (엔진이 타겟 외 경로로 정답을 흘리지 않는다는 뜻)."""
    prices, s = predictable_ohlcv(1200, seed=3)
    rng = np.random.default_rng(0)
    s_shuffled = pd.Series(rng.permutation(s.values), index=s.index, name="S_signal")
    dv, reg = _dv_with(prices, s_shuffled, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=400, test_len=40))
    res = run_experiment(cfg, dv, reg)
    assert 0.40 < res["prediction_metrics"]["dir_acc"] < 0.60


def test_rule_models_run(tmp_path):
    prices = random_walk_ohlcv(600, seed=4)
    dv, reg = _dv_with(prices, None, tmp_path)
    for m in ("buy_and_hold", "ma_cross", "momentum", "rsi_rule"):
        res = run_experiment(ExperimentConfig(ticker="TEST", model=m), dv, reg)
        assert "sharpe" in res["performance"], m
    bh = run_experiment(ExperimentConfig(ticker="TEST", model="buy_and_hold", trade={"fee_rate": 0, "slippage_bp": 0}), dv, reg)
    assert abs(bh["performance"]["total_return"] - bh["performance"]["benchmark_total_return"]) < 1e-9


# ---------------------------------------------------------------------------
# window 모델 (LSTM, BIVA)
# ---------------------------------------------------------------------------
BIVA_FAST = {"train_epochs": 60, "patience": 60, "batch_size": 32, "rnn_hid_size": 32}  # 원본 기본(배치 3·500 epoch)은 시험용으로 너무 느림


@pytest.mark.parametrize("model,params", [("lstm", {"epochs": 30}), ("biva", BIVA_FAST)])
def test_window_model_finds_planted_signal(model, params, tmp_path):
    prices, s = predictable_ohlcv(1000, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, model_params=params, variable_ids=["S_signal"], use_technical=False,
                           seq_len=5, validation=Validation(train_len=400, test_len=100))
    res = run_experiment(cfg, dv, reg)
    assert res["prediction_metrics"]["dir_acc"] > 0.70, res["prediction_metrics"]


@pytest.mark.parametrize("model,params", [("lstm", {"epochs": 20}), ("biva", {**BIVA_FAST, "train_epochs": 20})])
def test_window_model_no_edge_on_random_walk(model, params, tmp_path):
    prices = random_walk_ohlcv(1200, seed=6)
    dv, reg = _dv_with(prices, None, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, model_params=params, seq_len=10, validation=Validation(train_len=500, test_len=100))
    res = run_experiment(cfg, dv, reg)
    acc = res["prediction_metrics"]["dir_acc"]
    assert 0.38 < acc < 0.62, f"랜덤워크에서 방향 정확도 {acc:.3f} — 누출 의심"


def test_report_device():
    """실패하지 않는다 — 어떤 장치에서 테스트가 돌았는지 기록만 한다 (pytest -s 로 보임)."""
    import torch

    from engine.experiment import default_device

    print(f"\n[장치] torch {torch.__version__}, cuda build={torch.version.cuda}, available={torch.cuda.is_available()}, engine default={default_device()}")
    if torch.cuda.is_available():
        print("[장치]", torch.cuda.get_device_name(0))


def test_registry_duplicate_and_active_are_independent():
    """duplicate_of 는 active 와 별개 축: 값이 있으면 active=Y 여도 제외되고 대표로 치환된다."""
    reg = Registry([
        Variable(id="USDKRW", group="g", active=True),
        Variable(id="L1", group="g", active=True, duplicate_of="USDKRW"),
        Variable(id="OFF", group="g", active=False),
    ])
    assert reg.ids() == ["USDKRW"]  # L1(중복), OFF(비활성) 모두 제외
    ids, notes = reg.resolve_many(["L1", "USDKRW"])
    assert ids == ["USDKRW"] and len(notes) == 2
    with pytest.raises(ValueError):
        reg.resolve_many(["OFF"])  # 고른 변수가 모두 빠지면 오류
    bad = Registry([Variable(id="A", duplicate_of="B"), Variable(id="B", duplicate_of="C"), Variable(id="C")])
    assert len(bad.validate()) == 1  # 체인 금지


# ---------------------------------------------------------------------------
# 2026-11-04 추가: 변환·1회 분할·봉인·평가구간·규칙 분위수·분류 DM·Deflated
# ---------------------------------------------------------------------------
from engine.experiment import apply_transform  # noqa: E402
from engine.backtest import deflated_sharpe_pvalue, diebold_mariano_brier  # noqa: E402


def test_transform_uses_only_past():
    """pct/diff/logdiff 는 t 와 그 이전 값만 쓴다: 미래 한 점을 바꿔도 그 이전 변환값은 그대로."""
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    s = pd.Series(np.linspace(100, 150, 50), index=idx)
    s2 = s.copy(); s2.iloc[30:] += 999.0
    for tr in ("pct", "pct5", "diff", "diff5", "logdiff", "log"):
        a, b = apply_transform(s, tr), apply_transform(s2, tr)
        pd.testing.assert_series_equal(a.iloc[:30], b.iloc[:30])


def test_split_mode_has_embargo_and_fit(tmp_path):
    """1회 분할: 학습은 split_date 이전에서 embargo(1+k) 만큼 더 뗀 구간, 테스트는 그 이후. 적합값(fit)은 학습 구간에만."""
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    split = str(prices.index[600].date())
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(mode="split", split_date=split))
    res = run_experiment(cfg, dv, reg)
    d = res["_daily"]
    assert res["n_folds"] == 1
    assert d["prediction"].first_valid_index() >= pd.Timestamp(split)
    assert d["fit"].last_valid_index() < pd.Timestamp(split) - pd.Timedelta(days=1)  # embargo
    assert res["prediction_metrics"]["dir_acc"] > 0.75


def test_evaluation_window_excludes_sealed(tmp_path):
    """성과는 신호가 있는 개발 구간에서만. 봉인 구간(sealed_from 이후)은 포지션·성과에 들어가지 않는다."""
    prices = random_walk_ohlcv(1200, seed=6)
    dv, reg = _dv_with(prices, None, tmp_path)
    sealed = str(prices.index[1000].date())
    for m in ("ma_cross", "ridge"):
        res = run_experiment(ExperimentConfig(ticker="TEST", model=m, validation=Validation(train_len=400, test_len=40, sealed_from=sealed)), dv, reg)
        assert pd.Timestamp(res["evaluation_period"][1]) < pd.Timestamp(sealed), m
        assert res["_daily"]["signal"].loc[sealed:].isna().all(), m


def test_sealed_mode_predicts_only_sealed(tmp_path):
    """봉인 평가: 예측·성과가 봉인 구간에만 있고, 학습은 워크포워드를 그대로 이어 간다."""
    prices, s = predictable_ohlcv(1300, seed=7)
    dv, reg = _dv_with(prices, s, tmp_path)
    sealed = str(prices.index[1000].date())
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=400, test_len=40, sealed_from=sealed))
    res = run_experiment(cfg, dv, reg, sealed=True)
    assert res["sealed_evaluation"] is True
    assert pd.Timestamp(res["evaluation_period"][0]) >= pd.Timestamp(sealed)
    assert res["prediction_metrics"]["dir_acc"] > 0.75


def test_rule_rolling_quantile_excludes_today(tmp_path):
    """sox_gap_fade: 임계치는 t−1 까지의 분위수. 전체 표본 분위수(미래 포함)를 쓰면 안 된다."""
    import importlib
    m = importlib.import_module("models.sox_gap_fade")
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    x.iloc[300] = 0.5  # 극단값 하나
    X = pd.DataFrame({"SOX_pct": x})
    prices = pd.DataFrame({"open": 1.0, "high": 1, "low": 1, "close": 1.0, "volume": 1}, index=idx)
    pos = m.Model({"q": 0.9, "window": 100, "mode": "avoid_top", "column": "SOX_pct"}).positions(prices, X)
    assert pos[300] == 0.0  # 극단값 당일은 회피
    assert np.nanmean(pos[150:300]) > 0.85  # 그 이전 결정은 극단값(미래)의 영향을 받지 않음


def test_classification_dm_and_deflated():
    """분류용 DM: 정보 있는 확률은 0.5 보다 유의하게 낫다. Deflated: 시도 수가 늘면 p 가 커진다(보정)."""
    rng = np.random.default_rng(1)
    y = (rng.random(2000) > 0.5).astype(float)
    prob = np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.1, 2000), 0, 1)
    dm, p = diebold_mariano_brier(prob, y)
    assert dm < 0 and p < 0.01
    p1 = deflated_sharpe_pvalue(0.8, 1000, 1)
    p50 = deflated_sharpe_pvalue(0.8, 1000, 50)
    p50v = deflated_sharpe_pvalue(0.8, 1000, 50, trial_sr_var=0.05**2)
    assert p1 < p50 < p50v


def test_ridge_uses_full_training_window(tmp_path):
    """USES_VALIDATION=False 인 모델은 학습 창 전체를 쓴다 (검증 15% 를 버리지 않는다)."""
    import models.ridge as r
    assert r.Model.USES_VALIDATION is False


def test_forecast_mode_naive_and_targets(tmp_path):
    """예측 연구 모드: 타겟 = v[t+k]/v[t]−1, naive = y[t−k]; 지속성이 강한 시계열에서 Ridge 가 0예측은 이기고 naive 와 비슷해야 한다."""
    rng = np.random.default_rng(11)
    n = 400
    r = np.zeros(n); r[0] = 0.002
    for t in range(1, n):
        r[t] = 0.85 * r[t - 1] + rng.normal(0, 0.002)  # AR(1) 변화율(평활 지수 흉내)
    level = pd.Series(100 * np.cumprod(1 + r), index=pd.date_range("2000-01-01", periods=n, freq="MS"), name="IDX")
    dv = DataVersion(tmp_path, "t")
    dv.write_prices("TEST", random_walk_ohlcv(10))  # 사용 안 함(구조상 필요 없음)
    reg = Registry([Variable(id="IDX", freq="M", source="XLSX")])
    dv.write_variables(level.rename("value").rename_axis("date").reset_index().assign(variable_id="IDX"), source="syn")
    from engine import calendar as cal
    y = cal.make_forecast_target(level, k=2)
    assert abs(y.iloc[0] - (level.iloc[2] / level.iloc[0] - 1)) < 1e-12 and y.iloc[-2:].isna().all()
    assert abs(cal.naive_forecast(y, 2).iloc[5] - y.iloc[3]) < 1e-12
    cfg = ExperimentConfig(mode="forecast", target_variable="IDX", model="ridge", model_params={"alpha": 10.0}, use_technical=True,
                           target={"k": 1}, validation=Validation(train_len=60, test_len=1))
    res = run_experiment(cfg, dv, reg)
    pm = res["prediction_metrics"]
    assert res["performance"] == {} and "rmse_naive" in pm
    assert pm["dm_vs_zero"] < 0 and pm["dm_vs_zero_p"] < 0.05  # 0예측은 쉽게 이김
    assert pm["rmse"] < 1.3 * pm["rmse_naive"]  # naive 와 같은 급
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(mode="forecast", target_variable="IDX", model="ma_cross"), dv, reg)


def test_series_models_find_ar_signal_and_fail_on_noise():
    """series 플러그인(ARIMA): 심은 AR(1) 신호는 찾고(방향 >58%), 백색잡음에서는 무능(|상관| < 0.25)."""
    import numpy as np
    from models.arima import Model as Arima

    rng = np.random.default_rng(0)
    r = np.zeros(600)
    for t in range(1, 600):
        r[t] = 0.5 * r[t - 1] + rng.normal(0, 0.01)
    from models.arima import PARAMS as AP
    from models.base import resolve_params
    noise = rng.normal(0, 0.01, 600)
    for enforce in (True, False):  # 추정 제약 켬(기본)·끔 둘 다
        m = Arima(resolve_params(AP, {"enforce_stationarity": enforce, "enforce_invertibility": enforce}), n_features=1, target_type="regression")
        m.fit(r[:400], np.array([]))
        pred = np.array([m.forecast(r[: t + 1], 1)[0] for t in range(400, 599)])
        assert np.mean(np.sign(pred) == np.sign(r[401:600])) > 0.58, enforce
        m.fit(noise[:400], np.array([]))
        pred = np.array([m.forecast(noise[: t + 1], 1)[0] for t in range(400, 599)])
        assert abs(np.corrcoef(pred, noise[401:600])[0, 1]) < 0.25, enforce


def test_series_model_runs_in_engine_with_no_variables(tmp_path):
    """변수 0개 + series 모델이 엔진을 끝까지 통과하는지 (.empty 가 열 0개에서도 True 였던 회귀 방지)."""
    import pandas as pd
    from engine.experiment import _walkforward_predict, load_plugin, Dataset
    import engine.calendar as cal
    from models.base import resolve_params

    idx = pd.bdate_range("2015-01-01", periods=400)
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 400)), index=idx)
    prices = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    y = close.shift(-1) / close - 1
    ds = Dataset(prices, pd.DataFrame(index=idx), y, [], False, [], {}, {})
    folds = cal.walkforward_folds(idx, train_len=200, test_len=20, k=1, embargo=1, sealed_from=None)
    plugin = load_plugin("arima")
    from engine.experiment import ExperimentConfig
    cfg = ExperimentConfig(model="arima", mode="forecast", target_variable="X")
    pred, n, _ = _walkforward_predict(cfg, plugin, resolve_params(plugin.PARAMS, {}, "D"), ds, "cpu", folds[:3], 1, "regression")
    assert n == 3 and pred.notna().sum() == 60
    # logreturn 타겟이면 k기간 로그합: k=1 에서 log1p(return 예측) 과 같아야 한다
    cfg2 = ExperimentConfig(model="arima", mode="forecast", target_variable="X", target={"kind": "logreturn", "k": 1})
    pred2, _, _ = _walkforward_predict(cfg2, plugin, resolve_params(plugin.PARAMS, {}, "D"), ds, "cpu", folds[:1], 1, "regression")
    pr = pred.dropna().iloc[:20]
    assert np.allclose(pred2.dropna().values, np.log1p(pr.values))



def test_alignment_on_real_data_respects_lag():
    """실데이터 검산(전 변수, 전 기간): 거래일 d 에 모델이 보는 값 = '기준일 + lag ≤ d' 를 만족하는 가장 최근 관측값.
    이것이 발표시차의 정의 그 자체이며, 하루라도 어긋나면 실패한다(공개 전 값을 보거나, 공개된 값을 놓치거나)."""
    import pytest
    from pathlib import Path
    import pandas as pd
    import engine.calendar as cal
    from engine.registry import Registry
    from engine.store import DataVersion

    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "versions" / "20261103T141116Z").exists():
        pytest.skip("실데이터 버전 없음")
    reg, _ = Registry.load_master(root / "data" / "variable_master.xlsx")
    dv = DataVersion(root / "data", "20261103T141116Z")
    calendar = cal.trading_calendar(dv.read_prices("091160"))
    wide = dv.wide(reg.ids(usable_only=False))
    checked = 0
    for vid in wide.columns:
        lag = reg.get(vid).lag_days
        s = wide[vid].dropna()
        got = cal.align_to_calendar(s, calendar, lag_days=lag, fill="ffill")
        # 독립 구현(merge_asof): 각 거래일 d 에 대해 공개일(기준일+lag) ≤ d 인 마지막 관측
        rel = pd.DataFrame({"pub": s.index + pd.Timedelta(days=lag), "val": s.values}).sort_values("pub")
        exp = pd.merge_asof(pd.DataFrame({"d": calendar}), rel, left_on="d", right_on="pub", direction="backward").set_index("d")["val"]
        pd.testing.assert_series_equal(got.astype(float), exp.astype(float), check_names=False, check_freq=False, obj=f"{vid}(lag {lag})")
        checked += 1
    assert checked >= 40


def test_vecm_finds_error_correction_and_matches_statsmodels():
    """VECM 플러그인: (1) 수동 재귀 예측 = statsmodels predict, (2) 대상이 다른 변수와의 균형으로 되돌아가는 시스템에서
    '변화 없음'보다 1기간 예측 오차가 작다(심은 신호 찾기)."""
    import warnings
    from statsmodels.tsa.vector_ar.vecm import VECM
    from models.vecm import Model as V

    rng = np.random.default_rng(3)
    T = 400
    x = np.cumsum(rng.normal(0, 1, T)) + 100
    y = np.zeros(T); y[0] = 100
    for t in range(1, T):  # y 가 x 쪽으로 30% 씩 되돌아감
        y[t] = y[t - 1] + 0.3 * (x[t - 1] - y[t - 1]) + rng.normal(0, 0.3)
    Y = np.column_stack([y, x])
    from models.base import resolve_params
    import models.vecm as VM
    m = V(resolve_params(VM.PARAMS, {"k_ar_diff": 1, "coint_rank": 1, "deterministic": "ci", "log_levels": False}, "M"), n_features=2, target_type="regression")
    m.fit(Y[:300], np.array([]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = VECM(Y[:300], k_ar_diff=1, coint_rank=1, deterministic="ci").fit().predict(steps=3)[:, 0]
    assert np.allclose(m.forecast_levels(Y[:300], 3), ref, atol=1e-8)
    e_m, e_rw = [], []
    for t in range(300, T - 1):
        e_m.append(m.forecast_levels(Y[: t + 1], 1)[0] - y[t + 1])
        e_rw.append(y[t] - y[t + 1])
    assert np.sqrt(np.mean(np.square(e_m))) < 0.8 * np.sqrt(np.mean(np.square(e_rw)))


# ---------------------------------------------------------------------------
# 기간 지정 (2026-11-05): 데이터 시작·끝, 검증 시작, 테스트 시작·끝, 구간 사이 purge
# ---------------------------------------------------------------------------
def test_period_order_is_checked():
    from engine.experiment import _check_periods
    bad = [
        dict(validation=Validation(mode="split", val_start="2021-01-01", test_start="2020-01-01")),
        dict(validation=Validation(test_end="2025-02-01", sealed_from="2025-01-01")),
        dict(start="2022-01-01", end="2021-01-01"),
        dict(validation=Validation(mode="walkforward", val_start="2020-01-01")),
    ]
    for kw in bad:
        with pytest.raises(ValueError, match="기간 설정 오류"):
            _check_periods(ExperimentConfig(ticker="TEST", model="ridge", **kw))


def test_split_with_explicit_validation_dates_and_purge(tmp_path):
    prices, s = predictable_ohlcv(900, seed=11)
    dv, reg = _dv_with(prices, s, tmp_path)
    cal_idx = prices.index
    vs, ts, te = str(cal_idx[400].date()), str(cal_idx[600].date()), str(cal_idx[800].date())
    cfg = ExperimentConfig(ticker="TEST", model="lightgbm_model", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(mode="split", val_start=vs, test_start=ts, test_end=te))
    res = run_experiment(cfg, dv, reg)
    f = res["periods"]["first_fold"]
    k, emb = 1, 2  # 매매 모드 기본 embargo = 1+k
    assert f["val"][0] == vs and f["test"][0] == ts and f["test"][1] <= te
    pos = {str(d.date()): i for i, d in enumerate(cal_idx)}
    assert pos[f["val"][0]] - pos[f["train"][1]] > emb, "학습 끝과 검증 시작 사이 purge"
    assert pos[f["test"][0]] - pos[f["val"][1]] > emb, "검증 끝과 테스트 시작 사이 purge"


def test_walkforward_test_start_and_end_align_evaluation(tmp_path):
    prices, s = predictable_ohlcv(1000, seed=12)
    dv, reg = _dv_with(prices, s, tmp_path)
    ts, te = str(prices.index[600].date()), str(prices.index[850].date())
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=300, test_len=50, test_start=ts, test_end=te))
    res = run_experiment(cfg, dv, reg)
    assert res["periods"]["test"][0] == ts and res["periods"]["test"][1] <= te
    assert res["periods"]["first_fold"]["val"] is None  # Ridge 는 검증을 쓰지 않음 → 학습 창 전체


def test_vecm_without_variables_explains(tmp_path):
    prices, s = predictable_ohlcv(400, seed=1)
    dv, reg = _dv_with(prices, s, tmp_path)
    with pytest.raises(ValueError, match="입력 변수를 1개 이상"):
        run_experiment(ExperimentConfig(ticker="TEST", model="vecm"), dv, reg)


def test_brits_fills_empty_cells_inside_window_only():
    """창 안에 변수가 아직 없던 칸(NaN): BRITS 가 채우고, 관측된 칸은 그대로 둔다. 창 전체가 빈 변수도 유한값이 나온다."""
    import torch
    from types import SimpleNamespace
    from models.biva_src.BIVA import Model as BIVANet
    torch.manual_seed(0)
    a = SimpleNamespace(seq_len=6, pred_len=1, label_len=1, channels=5, vae_latent_size=4, infer_hid_size=128, rnn_hid_size=16, conv_kernal=1,
                        moving_avg=3, batch_size=2, conv1d=True, RIN=False, combination=True, imputed_weight=0.05)
    net = BIVANet(a).eval()
    x = torch.rand(2, 6, 5)
    x[0, :, 2] = float("nan")
    x[1, :3, 4] = float("nan")
    with torch.no_grad():
        states, _, _, _, _, imp = net(x)
    assert torch.isfinite(states).all() and torch.isfinite(imp).all()
    assert torch.allclose(imp[1, 3:, 4], x[1, 3:, 4]) and torch.allclose(imp[0, :, :2], x[0, :, :2])


# ---------------------------------------------------------------------------
# VECM 점검 (2026-11-06)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("det", ["n", "ci", "co"])
@pytest.mark.parametrize("seasons", [0, 12])
def test_vecm_recursion_matches_statsmodels(det, seasons):
    """수동 재귀 예측 = statsmodels VECMResults.predict (학습 끝에서 5기간), 상수·계절 조합별."""
    import warnings
    from statsmodels.tsa.vector_ar.vecm import VECM
    from models.base import resolve_params
    import models.vecm as V
    rng = np.random.default_rng(3)
    T = 131
    x = np.cumsum(rng.normal(0, 1, T)) + 50
    Y = np.c_[x * 0.8 + rng.normal(0, 0.5, T) + 30 + 2 * np.sin(np.arange(T) * 2 * np.pi / 12), x, np.cumsum(rng.normal(0, 1, T)) + 40]
    m = V.Model(resolve_params(V.PARAMS, {"deterministic": det, "seasons": seasons}, "M"), n_features=3, target_type="regression")
    m.fit_levels(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = VECM(Y, k_ar_diff=2, coint_rank=1, deterministic=det, seasons=seasons).fit().predict(5)[:, 0]
    assert np.max(np.abs(m.forecast_levels(Y, 5, pos=T - 1) - ref)) < 1e-9


def test_vecm_guards():
    from models.base import resolve_params
    import models.vecm as V
    rng = np.random.default_rng(0)
    T = 120
    x = np.cumsum(rng.normal(size=T)) + 50
    Y = np.c_[x * 0.8 + rng.normal(size=T) + 30, x, np.full(T, 5.0)]
    m = V.Model(resolve_params(V.PARAMS, {}, "M"), n_features=3, target_type="regression")
    m.fit_levels(Y)  # 상수 열은 빼고 적합(전에는 Singular matrix 로 실험 전체 중단)
    assert m.diag["dropped_constant_cols"] == [2] and np.isfinite(m.forecast_levels(Y, 2)).all()
    big = np.c_[Y[:40, :2], np.cumsum(rng.normal(size=(40, 9)), axis=0) + 30]
    with pytest.raises(ValueError, match="자유도"):
        V.Model(resolve_params(V.PARAMS, {"k_ar_diff": 4}, "M"), n_features=11, target_type="regression").fit_levels(big)
    m2 = V.Model(resolve_params(V.PARAMS, {"k_ar_diff": 6, "lag_select": "bic"}, "M"), n_features=3, target_type="regression")
    m2.fit_levels(Y)
    assert 0 <= m2.p <= 6


# ---------------------------------------------------------------------------
# 모델 흐름도 (2026-11-06): 실제 모델로 잰 모양
# ---------------------------------------------------------------------------
def test_model_flow_shapes_come_from_real_models(tmp_path):
    from engine.flow import model_flow
    prices, s = predictable_ohlcv(400, seed=3)
    dv, reg = _dv_with(prices, s, tmp_path)
    base = dict(ticker="TEST", variable_ids=["S_signal"], use_technical=False, seq_len=7)
    f = model_flow({**base, "model": "lstm", "model_params": {"hidden": 16}}, dv, reg)
    assert f["input"]["n_features"] == 1 and f["trace"]["lstm"]["in"] == [2, 7, 1] and f["trace"]["lstm"]["out"][0] == [2, 7, 16]
    assert f["weights"]["weight_ih_l0"] == [64, 1]  # 게이트 4개 × 은닉 16
    b = model_flow({**base, "model": "biva", "model_params": {"rnn_hid_size": 8}}, dv, reg)
    assert b["trace"]["BRITS"]["out"]["imputations"] == [2, 7, 1] and b["trace"]["Linear_Trend"]["out"] == [2, 1, 1]
    sa = model_flow({**base, "model": "sarima", "model_params": {"p": 1, "q": 0, "P": 1, "Q": 0, "s": 5}}, dv, reg)
    assert sa["lags"]["ar"] == [1, 5, 6]
    for m in ["ridge", "lightgbm_model", "vecm", "ma_cross"]:
        assert not [n for n in model_flow({**base, "model": m}, dv, reg)["notes"] if "실패" in n]


def test_rule_model_is_scored_on_same_dates_as_trained_model(tmp_path):
    """규칙 모델(학습 없음)도 같은 검증 설정이면 학습 모델과 같은 날짜 구간으로 성적을 매긴다(공정 비교)."""
    prices, s = predictable_ohlcv(900, seed=4)
    dv, reg = _dv_with(prices, s, tmp_path)
    for val in (Validation(train_len=400, test_len=40), Validation(mode="split", test_start=str(prices.index[600].date()))):
        r = run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=val), dv, reg)
        q = run_experiment(ExperimentConfig(ticker="TEST", model="buy_and_hold", variable_ids=[], use_technical=False, validation=val), dv, reg)
        assert q["evaluation_period"] == r["evaluation_period"], (val.mode, q["evaluation_period"], r["evaluation_period"])


def test_flow_reports_real_fit_counts_and_no_validation_when_too_small(tmp_path):
    """흐름도의 학습·검증 개수는 엔진 규칙과 같다: 비움은 학습 창 안에서 빼고, 검증 10개 미만이면 검증 없음."""
    from engine.flow import model_flow
    prices, s = predictable_ohlcv(400, seed=3)
    dv, reg = _dv_with(prices, s, tmp_path)
    base = dict(ticker="TEST", variable_ids=["S_signal"], use_technical=False)
    big = model_flow({**base, "model": "lightgbm_model", "validation": {"train_len": 200, "test_len": 20}}, dv, reg)["input"]["fit_plan"]
    assert big == {"embargo": 2, "rows": 198, "uses_val": True, "n_val": 29, "n_fit": 198 - 29 - 2, "has_val": True}
    small = model_flow({**base, "model": "lstm", "seq_len": 5, "validation": {"train_len": 60, "test_len": 5}}, dv, reg)
    assert small["input"]["fit_plan"]["has_val"] is False and any("검증" in n for n in small["notes"])
    ridge = model_flow({**base, "model": "ridge", "validation": {"train_len": 200, "test_len": 20}}, dv, reg)["input"]["fit_plan"]
    assert ridge["uses_val"] is False and ridge["n_fit"] == 198


@pytest.mark.parametrize("kind", ["return", "direction"])
def test_explain_ridge_reproduces_model_output_by_hand(tmp_path, kind):
    """계산 따라가기(Ridge): 손으로 다시 계산한 표준화·가중합(·확률)·닫힌 해가 실제 모델 출력과 같아야 한다."""
    from engine.explain import explain
    prices, s = predictable_ohlcv(600, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = dict(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=True, target={"kind": kind, "k": 1},
               validation={"train_len": 250, "test_len": 25})
    o = explain(cfg, dv, reg)
    assert o["ok"] and len(o["checks"]) >= 2, o["checks"]
    ids = [st["id"] for st in o["steps"]]
    assert ids[:2] == ["fold", "target"] and "weighted_sum" in ids and "decide" in ids
    tab = next(st for st in o["steps"] if st["id"] == "train_table")["table"]
    assert len(tab["rows"]) > 200 and len(tab["columns"]) == len(o["steps"][3]["table"]["rows"]) + 2  # 생략 없음: 학습 표 전 행, 모든 열
    d2 = o["available_dates"][0]
    assert explain(cfg, dv, reg, date=d2)["decision_date"] == d2


@pytest.mark.parametrize("model,extra", [
    ("lightgbm_model", {}), ("arima", {}), ("sarima", {"model_params": {"s": 5}}), ("vecm", {"use_technical": False}),
    ("lstm", {"model_params": {"epochs": 3, "hidden": 8}, "seq_len": 5}), ("gru", {"model_params": {"epochs": 3, "hidden": 8}, "seq_len": 5}),
    ("buy_and_hold", {}), ("ma_cross", {}), ("momentum", {}), ("rsi_rule", {}),
])
def test_explain_every_model_reproduces_its_output(tmp_path, model, extra):
    """계산 따라가기: 모델마다 손 계산(독립 구현)이 실제 출력과 같아야 한다(정보용 검사 제외)."""
    from engine.explain import explain
    prices, s = predictable_ohlcv(600, seed=6)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = {**dict(ticker="TEST", model=model, variable_ids=["S_signal"], use_technical=True, target={"kind": "return", "k": 1},
                  validation={"train_len": 250, "test_len": 25}), **extra}
    o = explain(cfg, dv, reg)
    strict = [c for c in o["checks"] if not c.get("info")]
    assert strict and all(c["ok"] for c in strict), o["checks"]
    assert o["steps"][-1]["id"] in ("decide", "compound", "head") or any(st["id"] == "decide" for st in o["steps"])


def test_explain_biva_hand_computation_matches_original_modules(tmp_path):
    """BIVA 계산 따라가기: BRITS·분해·추세·인코더·μ·logσ²·VAE 손실·주기·합치기를 손 계산해 원본 모듈 출력과 맞춘다. epoch 은 계산 따라가기용으로 줄임."""
    from engine.explain import explain
    prices, s = predictable_ohlcv(500, seed=7)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = dict(ticker="TEST", model="biva", variable_ids=["S_signal"], use_technical=True, target={"kind": "return", "k": 1}, seq_len=6,
               validation={"train_len": 200, "test_len": 20}, model_params={"rnn_hid_size": 8, "batch_size": 8})
    o = explain(cfg, dv, reg, epochs=2)
    assert o["explain_epochs"] == 2 and o["notes"]
    assert len(o["checks"]) >= 8 and all(c["ok"] for c in o["checks"]), o["checks"]
    ids = [st["id"] for st in o["steps"]]
    for need in ("brits", "decomp", "trend", "vae_enc", "vae_z", "vae_dec", "seasonal", "combine", "noise", "decide"):
        assert need in ids, need


def test_dev_and_sealed_folds_cover_every_day_without_gap_or_overlap():
    """개발 폴드는 봉인 전날까지 빠짐없이, 봉인 폴드는 봉인 첫날부터(마지막 폴드는 남은 날만큼). 겹치거나 빠지는 날이 없어야 한다."""
    import engine.calendar as cal
    days = pd.bdate_range("2020-01-01", periods=400)
    seal = days[333]
    dev = cal.walkforward_folds(days, train_len=100, test_len=21, k=1, sealed_from=seal)
    dev_days = pd.DatetimeIndex(np.concatenate([f.test.values for f in dev]))
    assert dev_days[0] == days[100] and dev_days[-1] == days[332] and dev_days.is_unique and len(dev_days) == 233
    assert len(dev[-1].test) == 233 % 21 or len(dev[-1].test) == 21
    sealed = cal.walkforward_folds(days, train_len=100, test_len=21, k=1, start=int(days.searchsorted(seal)))
    s_days = pd.DatetimeIndex(np.concatenate([f.test.values for f in sealed]))
    assert s_days[0] == seal and s_days[-1] == days[-1] and len(s_days) == 400 - 333
    for f in sealed:  # 봉인 폴드의 학습은 그 테스트 시작 전(비움 포함)
        assert f.train[-1] < f.test[0]


@pytest.mark.parametrize("model", ["arima", "sarima"])
@pytest.mark.parametrize("enforce", [False, True])
def test_arima_constraints_switch_and_roots_step(tmp_path, model, enforce):
    """ARIMA·SARIMA 추정 제약(2026-11-06): 켜고 끌 수 있고, 계산 따라가기의 근 점검 단계가 설정을 그대로 보이며, 켜면 근 > 1."""
    from engine.explain import explain
    prices, s = predictable_ohlcv(600, seed=6)
    dv, reg = _dv_with(prices, s, tmp_path)
    mp = {"enforce_stationarity": enforce, "enforce_invertibility": enforce, **({"s": 5} if model == "sarima" else {})}
    cfg = dict(ticker="TEST", model=model, target={"kind": "return", "k": 1}, validation={"train_len": 250, "test_len": 25}, model_params=mp)
    o = explain(cfg, dv, reg)
    strict = [c for c in o["checks"] if not c.get("info")]
    assert strict and all(c["ok"] for c in strict), o["checks"]
    roots = next(st for st in o["steps"] if st["id"] == "roots")
    assert roots["values"]["가역성 제약(설정)"] is enforce and roots["values"]["정상성 제약(설정)"] is enforce
    if enforce:
        assert roots["values"]["MA 근 모두 > 1"] and roots["values"]["AR 근 모두 > 1"]


def test_pinned_params_reproduce_old_arima_results():
    """옛 결과(model.params 에 제약 키 없음)는 그때 값(False)으로, 새 결과는 기록 그대로 재현한다."""
    from engine.experiment import pinned_model_params
    mp, notes = pinned_model_params("arima", {"p": 2, "d": 0, "q": 2, "trend": "c"}, {})
    assert mp == {"enforce_stationarity": False, "enforce_invertibility": False, "scale": 1.0} and len(notes) == 3
    # 2026-11-06 결과(제약 키 있음, 배율 키 없음) → 배율만 그때 값(1)으로
    assert pinned_model_params("sarima", {"p": 1, "enforce_stationarity": True, "enforce_invertibility": True}, {})[0] == {"scale": 1.0}
    assert pinned_model_params("sarima", {"p": 1, "enforce_stationarity": True, "enforce_invertibility": True, "scale": 100.0}, {})[0] == {}
    assert pinned_model_params("ridge", {"alpha": 1.0}, {})[0] == {}


@pytest.mark.parametrize("model", ["arima", "sarima"])
def test_arima_scale_is_same_model_better_optimum(model):
    """수치 배율(2026-11-07): 변화율 × scale 로 추정해도 같은 모형 — 잘 수렴하는 자료에서는 φ·θ·예측이 같고(c·σ² 는 배율만큼),
    분산이 1e−5 수준인 작은 변화율에서는 배율을 쓴 쪽의 로그우도(원래 단위로 환산)가 같거나 높다."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from models.base import resolve_params
    import importlib
    M = importlib.import_module(f"models.{model}")
    rng = np.random.default_rng(3)
    n = 240
    e = rng.normal(0, 0.003, n + 2)
    r = np.zeros(n)
    for t in range(2, n):
        r[t] = 0.001 + 0.5 * r[t - 1] + e[t] + 0.3 * e[t - 1]
    extra = {"s": 12} if model == "sarima" else {}
    ms = {}
    for sc in (1.0, 100.0):
        m = M.Model(resolve_params(M.PARAMS, {"scale": sc, **extra}, "M"), n_features=1, target_type="regression")
        m.fit(r, np.array([]))
        ms[sc] = m
    res1, res100 = ms[1.0].res, ms[100.0].res
    n_eff = res100.nobs - res100.loglikelihood_burn
    assert res100.llf + n_eff * np.log(100.0) >= res1.llf - 1e-6  # 원래 단위로 환산한 로그우도: 배율 쪽이 같거나 높다
    f1, f100 = ms[1.0].forecast(r, 3), ms[100.0].forecast(r, 3)
    assert f100.shape == (3,) and np.all(np.isfinite(f100))
    if res1.mle_retvals.get("converged") and abs(res100.llf + n_eff * np.log(100.0) - res1.llf) < 1e-3:  # 같은 최대점이면 예측도 같다
        assert np.allclose(f1, f100, atol=1e-6)
    # 같은 모수면 배율과 무관하게 예측이 같다(모형 동일성): 배율 100 모수를 원래 단위로 바꿔 배율 1 모형에 넣는다
    names = list(res100.model.param_names)
    unit = np.array([1 / 100 if n_ == "intercept" else 1e-4 if n_ == "sigma2" else 1.0 for n_ in names])
    p0 = np.asarray(res100.params) * unit
    mod1 = SARIMAX(r, **({"order": (2, 0, 2), "trend": "c"} if model == "arima" else {"order": (1, 0, 1), "seasonal_order": (1, 1, 1, 12)}))
    fc_same = mod1.filter(p0).forecast(3)
    assert np.allclose(fc_same, f100, atol=1e-9)


@pytest.mark.parametrize("mode", ["walkforward", "split"])
def test_sealed_eval_covers_seal_to_end_with_dev_rule(tmp_path, mode):
    """봉인 평가(2026-11-06): 봉인 첫 거래일 ~ 데이터 끝을 빠짐없이 예측. 워크포워드는 봉인 첫날부터 test_len 씩,
    1회 분할은 '봉인 전 전부 − 비움'으로 한 번 학습. 개발 구간은 다시 계산하지 않는다(폴드 기록이 모두 봉인 이후)."""
    from engine.experiment import sealed_folds, prepare
    prices, s = predictable_ohlcv(1300, seed=7)
    dv, reg = _dv_with(prices, s, tmp_path)
    seal = prices.index[1000]
    v = dict(train_len=400, test_len=40, sealed_from=str(seal.date()), mode=mode)
    if mode == "split":
        v["test_start"] = str(prices.index[800].date())
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(**v))
    res = run_experiment(cfg, dv, reg, sealed=True)
    d = res["_daily"]
    first, last = d["prediction"].first_valid_index(), d["prediction"].last_valid_index()
    assert first == seal and last == prices.index[-1]
    assert d["prediction"].loc[seal:].notna().all()
    _, _, _, _, ds = prepare(cfg, dv, reg)
    folds = sealed_folds(cfg, ds, 1)
    assert res["n_folds"] == len(folds)
    if mode == "split":
        assert len(folds) == 1 and folds[0].train[-1] < seal and len(folds[0].train) == 1000 - 2
    else:
        assert len(folds) == int(np.ceil(300 / 40)) and folds[0].test[0] == seal and len(folds[0].train) == 400 - 2
    assert pd.Timestamp(res["periods"]["first_fold"]["test"][0]) >= seal  # 개발 폴드 기록 없음


def test_sealed_eval_rule_model_only_sealed_days(tmp_path):
    prices = random_walk_ohlcv(1200, seed=6)
    dv, reg = _dv_with(prices, None, tmp_path)
    seal = prices.index[1000]
    res = run_experiment(ExperimentConfig(ticker="TEST", model="ma_cross", validation=Validation(train_len=400, test_len=40, sealed_from=str(seal.date()))), dv, reg, sealed=True)
    assert pd.Timestamp(res["evaluation_period"][0]) >= seal and res["sealed_evaluation"] is True


def test_sealed_result_does_not_overwrite_dev_result(tmp_path):
    """봉인 결과는 같은 설정의 개발 결과와 다른 파일(_sealed, 재개봉 _sealed2)로 저장된다(2026-11-06 전에는 덮어씀)."""
    from engine.experiment import save_result
    prices, s = predictable_ohlcv(1300, seed=7)
    dv, reg = _dv_with(prices, s, tmp_path / "d")
    cfg = ExperimentConfig(name="cand", ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=400, test_len=40, sealed_from=str(prices.index[1000].date())))
    out = tmp_path / "res"
    a = save_result(run_experiment(cfg, dv, reg), out)
    b = save_result(run_experiment(cfg, dv, reg, sealed=True), out)
    c = save_result(run_experiment(cfg, dv, reg, sealed=True), out)
    h = json.loads(a.read_text(encoding="utf-8"))["config_hash"]
    assert a.stem == f"cand_{h}" and b.stem == a.stem + "_sealed" and c.stem == a.stem + "_sealed2"
    assert json.loads(a.read_text(encoding="utf-8"))["sealed_evaluation"] is False


def test_trial_variance_reads_mode_folder_and_skips_sealed(tmp_path):
    from engine.experiment import trial_sharpe_variance, ENGINE_REV
    d = tmp_path / "trading"; d.mkdir()
    # 봉인 평가 결과와, 계산 방식이 지금과 다른 옛 결과(engine_rev 불일치 — 2026-11-10)는 V 에 넣지 않는다
    for i, (sh, sealed, rev) in enumerate([(0.5, False, ENGINE_REV), (1.5, False, ENGINE_REV), (9.0, True, ENGINE_REV), (7.0, False, None)]):
        (d / f"r{i}.json").write_text(json.dumps({"config": {"frequency": "D"}, "performance": {"sharpe": sh}, "sealed_evaluation": sealed, "engine_rev": rev}), encoding="utf-8")
    v = trial_sharpe_variance(d, 252)
    assert v == pytest.approx(np.var([0.5 / np.sqrt(252), 1.5 / np.sqrt(252)], ddof=1))
    assert trial_sharpe_variance(tmp_path, 252) is None  # 상위 폴더에는 결과가 없다 — 2026-11-06 전 cli 가 이걸 넘겼다


def test_sealed_preview_blocks_without_seal_and_on_sealed_result(tmp_path):
    from engine.sealed import preview
    prices, s = predictable_ohlcv(900, seed=7)
    dv, reg = _dv_with(prices, s, tmp_path)
    base = dict(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False)
    r = run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40)), dv, reg); r.pop("_daily")
    assert any("봉인 시작일이 없습니다" in b for b in preview(r, "x", dv, reg, [], None)["blockers"])
    cfg = ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40, sealed_from=str(prices.index[700].date())))
    r = run_experiment(cfg, dv, reg); r.pop("_daily")
    assert ExperimentConfig(**r["config"]).config_hash() == r["config_hash"]  # 미리보기가 다시 계산한 해시 = 결과·기록의 해시
    pv = preview(r, "x", dv, reg, [{"hash": r["config_hash"], "sealed_from": cfg.validation.sealed_from}], None)
    assert not pv["blockers"] and pv["plan"]["first_test"] == str(prices.index[700].date()) and pv["plan"]["last_test"] == str(prices.index[-1].date())
    assert pv["plan"]["n_days"] == 200 and any("이미 1회" in w for w in pv["warnings"])
    rs = run_experiment(cfg, dv, reg, sealed=True); rs.pop("_daily")
    assert any("봉인 평가 결과" in b for b in preview(rs, "y", dv, reg, [], None)["blockers"])


def test_overrides_written_for_duplicate_id_apply_to_representative(tmp_path):
    """2026-11-07: 중복으로 치환되는 변수(L1→USDKRW 같은)에 원래 ID 로 적은 변환·lag 재정의가 대표 ID 에 적용된다(전에는 버려짐)."""
    from engine.experiment import build_dataset
    prices, s = predictable_ohlcv(400, seed=3)
    dv, reg = _dv_with(prices, (s.abs() + 1).rename("REP"), tmp_path)
    reg.get("REP").transform = "pct"
    reg.add(Variable(id="DUP", freq="D", source="DERIVED", duplicate_of="REP"))
    out, notes = reg.remap_overrides(["DUP"], {"DUP": "none"})
    assert out == {"REP": "none"} and notes
    assert reg.remap_overrides(["DUP"], {"DUP": "none", "REP": "diff"})[0] == {"REP": "diff"}  # 대표 ID 에 직접 적은 것이 우선
    ds = build_dataset(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["DUP"], use_technical=False,
                                        variable_transforms={"DUP": "none"}, lag_override={"DUP": 3}), dv, reg)
    assert list(ds.X.columns) == ["REP"] and ds.transforms_used == {"REP": "none"} and ds.lag_days_used == {"REP": 3}


def test_window_scaler_uses_all_train_rows_not_only_complete_rows(tmp_path):
    """2026-11-07: 창 모델의 눈금(MinMax)은 학습 폴드의 모든 행으로 맞춘다 — 늦게 시작하는 변수가 있어도 학습 창 값이 0~1 안."""
    from engine.experiment import _walkforward_predict, load_plugin, make_folds, prepare
    prices, s = predictable_ohlcv(500, seed=4)
    late = s.copy(); late.iloc[:250] = np.nan; late = late.rename("LATE")
    dv, reg = _dv_with(prices, s, tmp_path)
    reg.add(Variable(id="LATE", freq="D", source="DERIVED"))
    dv.write_variables(late.rename("value").rename_axis("date").reset_index().assign(variable_id="LATE"), source="syn2")
    cfg = ExperimentConfig(ticker="TEST", model="lstm", variable_ids=["S_signal", "LATE"], use_technical=False, seq_len=5,
                           model_params={"epochs": 1, "hidden": 4, "input_scaler": "minmax"} if "input_scaler" in load_plugin("lstm").PARAMS else {"epochs": 1, "hidden": 4},
                           validation=Validation(train_len=400, test_len=50))
    cfg, plugin, params, tt, ds = prepare(cfg, dv, reg)
    folds, _ = make_folds(cfg, ds, 1)
    cap: dict = {}
    _walkforward_predict(cfg, plugin, params, ds, "cpu", [folds[0]], 1, tt, capture=cap)
    full = ds.X.loc[folds[0].train]
    sc = cap["scaler"]
    # 눈금이 학습 폴드 전체(빈 칸 무시)의 통계와 같아야 한다
    if hasattr(sc, "data_min_"):
        assert np.allclose(sc.data_min_, full.min().values) and np.allclose(sc.data_max_, full.max().values)
    else:
        assert np.allclose(sc.mean_, full.mean().values)


# ---------------------------------------------------------------------------
# 2026-11-09: 엑셀 입력은 variable_master.xlsx 한 파일, API 원본은 JSON + 받은 이력
# ---------------------------------------------------------------------------
def test_master_data_sheets_equal_stored_version():
    """변수 사전의 데이터 시트(df_M·df_Q)는 연구실 엑셀에서 옮긴 값 — 지금 정식 데이터 버전의 엑셀 변수와 한 칸도 다르면 안 된다."""
    root = Path(__file__).resolve().parents[1]
    master = root / "data" / "variable_master.xlsx"
    stored = root / "data" / "versions" / "20261103T141116Z" / "variables_xlsx_macro_seoul34.parquet"
    if not (master.exists() and stored.exists()):
        pytest.skip("데이터 없음")
    from engine.store import excel_to_long
    reg, _ = Registry.load_master(master)
    new = excel_to_long(master, reg).sort_values(["variable_id", "date"]).reset_index(drop=True)
    old = pd.read_parquet(stored).sort_values(["variable_id", "date"]).reset_index(drop=True)
    assert new.equals(old)
    # XLSX 변수는 모두 데이터 시트에 열이 있다
    xl = pd.ExcelFile(master)
    cols = {str(c) for sh in xl.sheet_names if sh.startswith("df_") for c in xl.parse(sh, nrows=0).columns}
    xlsx_vars = [v for v in reg.ids(usable_only=False) if reg.get(v).source == "XLSX"]
    have = [v for v in xlsx_vars if reg.get(v).source_code.split("!")[-1] in cols]
    assert len(have) >= 34


def test_rawcache_appends_fetch_log(tmp_path):
    """API 원본은 JSON 으로 남고, 같은 키를 다시 받아도 덮어쓰지 않으며 fetch_log.jsonl 에 한 줄씩 쌓인다."""
    import json
    from adapters.base import RawCache
    c = RawCache(tmp_path, "fred")
    p1 = c.save("DGS10", {"observations": [{"date": "2024-01-02", "value": "3.9"}]})
    p2 = c.save("DGS10", {"observations": [{"date": "2024-01-02", "value": "4.0"}]})
    assert p1 != p2 and p1.exists() and p2.exists()
    lines = [json.loads(l) for l in (tmp_path / "raw" / "fetch_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["key"] for l in lines] == ["DGS10", "DGS10"] and lines[0]["sha256"] != lines[1]["sha256"]
    assert json.loads(p1.read_text(encoding="utf-8"))["observations"][0]["value"] == "3.9"


def test_manifest_keeps_created_time(tmp_path):
    """같은 데이터 버전에 이어 받아도 처음 만든 시각은 바뀌지 않는다(updated_utc 만 바뀜)."""
    from engine.store import DataVersion
    dv = DataVersion(tmp_path, "20990101T000000Z")
    dv.write_manifest({"a": 1})
    c0 = dv.manifest()["created_utc"]
    dv.write_manifest({"a": 2})
    m = dv.manifest()
    assert m["created_utc"] == c0 and m["a"] == 2 and "updated_utc" in m


# ---------------------------------------------------------------------------
# 2026-11-09: active·duplicate_of 규칙(사용자 확정)과 실행 기록
# ---------------------------------------------------------------------------
def test_resolve_rule_table():
    """duplicate_of 있음 → 대표로(자기 active 무관) / 없음+Y → 포함 / 없음+N → 빼고 기록 / 대표가 N → 빼고 기록."""
    reg = Registry([
        Variable(id="REP", group="g"),
        Variable(id="DUP_Y", group="g", duplicate_of="REP", active=True),
        Variable(id="DUP_N", group="g", duplicate_of="REP", active=False),
        Variable(id="OK", group="g"),
        Variable(id="OFF", group="g", active=False),
        Variable(id="REP_OFF", group="g", active=False),
        Variable(id="TO_OFF", group="g", duplicate_of="REP_OFF"),
    ])
    d = reg.resolve_detail(["DUP_Y", "DUP_N", "OK", "OFF", "TO_OFF"])
    assert d["used"] == ["REP", "OK"]
    assert d["substituted"] == [{"from": "DUP_Y", "to": "REP"}, {"from": "DUP_N", "to": "REP"}, {"from": "TO_OFF", "to": "REP_OFF"}]
    assert d["merged"] == [{"from": "DUP_N", "to": "REP"}]
    assert [(x["id"], x["requested_as"]) for x in d["excluded"]] == [("OFF", "OFF"), ("REP_OFF", "TO_OFF")]
    ids, notes = reg.resolve_many(["OK", "OFF"])
    assert ids == ["OK"] and any("제외" in n for n in notes)


def test_run_record_lists_excluded_variable(tmp_path):
    """active=N 변수는 실험에서 빠지고, 결과의 run_record 에 무엇이 왜 빠졌는지 남는다."""
    prices, s = predictable_ohlcv(600, seed=3)
    dv, reg = _dv_with(prices, s, tmp_path)
    reg.add(Variable(id="OFFV", freq="D", source="DERIVED", active=False))
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal", "OFFV"], use_technical=False,
                           validation=Validation(train_len=300, test_len=40))
    res = run_experiment(cfg, dv, reg)
    rr = res["run_record"]
    assert res["feature_names"] == ["S_signal"]
    assert rr["variables"]["used"] == ["S_signal"]
    assert rr["variables"]["excluded"] == [{"id": "OFFV", "requested_as": "OFFV", "reason": "active=N"}]
    for k in ("run_id", "started_utc", "finished_utc", "libraries", "host", "data_version", "master"):
        assert k in rr
    assert rr["libraries"]["pandas"] == pd.__version__


def test_save_result_keeps_previous_and_logs(tmp_path):
    """같은 이름의 결과를 다시 저장하면 옛 결과는 history/ 로 가고, run_log.jsonl 에 저장 줄이 쌓인다."""
    from engine.experiment import save_result
    prices, s = predictable_ohlcv(600, seed=4)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(name="rr", ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=300, test_len=40))
    out = tmp_path / "res"
    p1 = save_result(run_experiment(cfg, dv, reg), out)
    p2 = save_result(run_experiment(cfg, dv, reg), out)
    assert p1 == p2 and p2.exists()
    hist = sorted((out / "history").iterdir())
    assert len(hist) == 2 and {h.suffix for h in hist} == {".json", ".parquet"}
    lines = [json.loads(l) for l in (out / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["event"] for l in lines] == ["saved", "saved"] and lines[1]["replaced_previous"]
    assert json.loads(p2.read_text(encoding="utf-8"))["run_record"]["replaced_previous"]


def test_snapshot_master_copies_once(tmp_path):
    from engine import runlog
    m = tmp_path / "variable_master.xlsx"
    m.write_bytes(b"abc")
    a = runlog.snapshot_master(m, tmp_path)
    b = runlog.snapshot_master(m, tmp_path)
    assert a == b and len(list((tmp_path / "master_history").iterdir())) == 1
    m.write_bytes(b"abcd")
    assert runlog.snapshot_master(m, tmp_path)["sha256"] != a["sha256"]
    assert len(list((tmp_path / "master_history").iterdir())) == 2


def test_technical_feature_selection(tmp_path):
    """기술지표 개별 선택: None = 전부(해시 그대로), 고른 것만 입력에, 없는 이름·빈 목록은 오류."""
    prices, _ = predictable_ohlcv(600, seed=5)
    dv, reg = _dv_with(prices, None, tmp_path)
    base = dict(ticker="TEST", model="ridge", validation=Validation(train_len=300, test_len=40))
    a = ExperimentConfig(**base)
    assert a.config_hash() == ExperimentConfig(**base, technical_features=None).config_hash()
    assert "technical_features" not in a.model_dump_json(exclude={"name", "technical_features"})
    sel = ["A_ret_1", "A_rsi_14"]
    b = ExperimentConfig(**base, technical_features=sel)
    assert b.config_hash() != a.config_hash()
    res = run_experiment(b, dv, reg)
    assert res["feature_names"] == sel
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(**base, technical_features=["A_nope"]), dv, reg)
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(**base, technical_features=[]), dv, reg)
    from engine.features import technical_columns, technical_group
    assert len(technical_columns("D")) == 18 and len(technical_columns("D", with_volume=False)) == 16
    assert technical_group("A_vol_ratio_5") == "거래량" and technical_group("A_vol_20") == "변동성"


# ---------------------------------------------------------------------------
# 2026-11-09: 한 변수 = 한 출처 (engine/master_check) — 규칙대로 알아서 처리, 짐작할 수 없는 것만 멈춤
# ---------------------------------------------------------------------------
def _mini_master(path, var_rows, sheets):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "variables"
    cols = ["id", "name", "freq", "source", "source_code", "lag_days", "duplicate_of", "active"]
    ws.append(cols)
    for r in var_rows:
        ws.append([r.get(c, "") for c in cols])
    for sh, header, data in sheets:
        s = wb.create_sheet(sh); s.append(header)
        for row in data:
            s.append(row)
    wb.save(path)


def test_current_master_passes_checks():
    from engine.master_check import check_master
    master = Path(__file__).resolve().parent.parent / "data" / "variable_master.xlsx"
    reg, _ = Registry.load_master(master)
    assert check_master(reg, master).errors == []


def test_master_check_auto_rules(tmp_path):
    """API 변수 열은 읽지 않고(알림), 새 열은 자동 등록, 머리글 공백은 지움."""
    import datetime as dt
    from engine.master_check import check_master, register_new_columns
    from engine.store import excel_to_long
    m = tmp_path / "variable_master.xlsx"
    d = [[dt.datetime(2020, i, 1), 1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 4)]
    _mini_master(m, [{"id": "X1", "freq": "M", "source": "XLSX", "source_code": "df_M!X1", "lag_days": 30},
                     {"id": "KOSPI", "freq": "D", "source": "YF", "source_code": "^KS11", "lag_days": 1}],
                 [("df_M", ["date", " X1 ", "KOSPI", "NEW1"], d)])
    reg, _ = Registry.load_master(m)
    msgs = register_new_columns(reg, m)
    assert len(msgs) == 1 and "NEW1" in msgs[0]
    reg, _ = Registry.load_master(m)
    assert reg.get("NEW1").source == "XLSX" and reg.get("NEW1").freq == "M" and not reg.get("NEW1").lag_confirmed
    r = check_master(reg, m)
    assert r.errors == [] and any("KOSPI" in n for n in r.notices) and any("공백" in n for n in r.notices)
    long = excel_to_long(m, reg)
    assert set(long["variable_id"]) == {"X1", "NEW1"}  # KOSPI 열(API 변수)은 읽지 않음


@pytest.mark.parametrize("case,expect", [
    ("two_sheets", "같은 이름의 열"), ("freq", "옮겨"), ("same_api", "두 번 받습니다"), ("bad_name", "띄어쓰기"), ("empty_header", "비어"),
])
def test_master_check_stops_when_ambiguous(tmp_path, case, expect):
    import datetime as dt
    from engine.master_check import check_master
    m = tmp_path / "variable_master.xlsx"
    d = [[dt.datetime(2020, 1, 1), 1.0, 2.0]]
    vars_ = [{"id": "X1", "freq": "M", "source": "XLSX", "source_code": "df_M!X1", "lag_days": 30}]
    sheets = [("df_M", ["date", "X1"], [[dt.datetime(2020, 1, 1), 1.0]])]
    if case == "two_sheets":
        sheets.append(("df_Q", ["date", "X1"], [[dt.datetime(2020, 3, 1), 1.0]]))
    elif case == "freq":
        vars_[0]["freq"] = "Q"
    elif case == "same_api":
        vars_ += [{"id": "V1", "freq": "D", "source": "FRED", "source_code": "DGS10", "lag_days": 1},
                  {"id": "V2", "freq": "D", "source": "FRED", "source_code": "DGS10", "lag_days": 1}]
    elif case == "bad_name":
        sheets = [("df_M", ["date", "X1", "가격 지수"], d)]
    elif case == "empty_header":
        sheets = [("df_M", ["date", "X1", None, "X2"], [[dt.datetime(2020, 1, 1), 1.0, 5.0, 2.0]])]
        vars_.append({"id": "X2", "freq": "M", "source": "XLSX", "source_code": "df_M!X2", "lag_days": 30})
    _mini_master(m, vars_, sheets)
    reg, _ = Registry.load_master(m)
    errs = check_master(reg, m).errors
    assert errs and any(expect in e for e in errs), errs


def test_same_api_code_allowed_with_duplicate_of(tmp_path):
    from engine.master_check import check_master
    m = tmp_path / "variable_master.xlsx"
    _mini_master(m, [{"id": "V1", "freq": "D", "source": "FRED", "source_code": "DGS10", "lag_days": 1},
                     {"id": "V2", "freq": "D", "source": "FRED", "source_code": "DGS10", "lag_days": 1, "duplicate_of": "V1"}], [])
    reg, _ = Registry.load_master(m)
    assert check_master(reg, m).errors == []


def test_master_check_stops_without_touching_file(tmp_path):
    """멈추는 경우에는 새 열도 등록하지 않는다(엑셀을 고치지 않음). 값 없는 빈 제목 열은 무시."""
    import datetime as dt
    from engine.master_check import check_master, register_new_columns
    m = tmp_path / "variable_master.xlsx"
    _mini_master(m, [{"id": "X1", "freq": "M", "source": "XLSX", "source_code": "df_M!X1", "lag_days": 30}],
                 [("df_M", ["date", "X1", "NEW1", None, "X1"], [[dt.datetime(2020, 1, 1), 1.0, 2.0, None, 3.0]])])
    before = m.read_bytes()
    reg, _ = Registry.load_master(m)
    assert check_master(reg, m).errors and register_new_columns(reg, m) == [] and m.read_bytes() == before
    _mini_master(m, [{"id": "X1", "freq": "M", "source": "XLSX", "source_code": "df_M!X1", "lag_days": 30}],
                 [("df_M", ["date", "X1", None], [[dt.datetime(2020, 1, 1), 1.0, None]])])
    reg, _ = Registry.load_master(m)
    assert check_master(reg, m).errors == []


def test_lag_confirmed_blank_means_unconfirmed(tmp_path):
    """lag_confirmed 는 Y 라고 적은 것만 확인됨. 빈칸·열 없음 = 확인 안 됨(2026-11-09)."""
    import openpyxl
    m = tmp_path / "variable_master.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "variables"
    ws.append(["id", "freq", "source", "source_code", "lag_days", "lag_confirmed"])
    ws.append(["V1", "M", "FRED", "A", 30, "Y"]); ws.append(["V2", "M", "FRED", "B", 30, ""]); ws.append(["V3", "M", "FRED", "C", 30, "N"])
    wb.save(m)
    reg, _ = Registry.load_master(m)
    assert [reg.get(i).lag_confirmed for i in ("V1", "V2", "V3")] == [True, False, False]


# ---------------------------------------------------------------------------
# 2026-11-10: 시도 원장 · 저장 한 묶음 · 사전 지문 시점
# ---------------------------------------------------------------------------
def test_trial_ledger_counts_and_detects_tampering(tmp_path):
    """원장: 옛 카운터에서 시작(genesis), 개발 시도만 세고(sealed·deleted 제외), 줄을 고치거나 빼면 verify 가 잡는다. 사본 카운터를 지워도 N 은 그대로."""
    from engine import ledger
    d = tmp_path / "trading"; d.mkdir()
    (d / "experiment_counter.json").write_text('{"count": 98}')
    assert ledger.count(d) == 98 and not (d / ledger.LEDGER).exists()  # 읽기만으로 원장을 만들지 않는다
    assert ledger.record(d, "trial", config_hash="aaa", name="x") == 99
    assert ledger.record(d, "sealed", config_hash="aaa") == 99
    assert ledger.record(d, "ablation", config_hash="bbb") == 100
    assert ledger.record(d, "trial", config_hash="aaa") == 101  # 같은 설정의 재실행도 한 번의 시도
    assert ledger.record(d, "deleted", result="x_aaa") == 101
    assert ledger.distinct_configs(d) == 2
    assert json.loads((d / "experiment_counter.json").read_text())["count"] == 101
    (d / "experiment_counter.json").unlink()
    assert ledger.count(d) == 101 and ledger.verify(d)["ok"]
    lines = (d / ledger.LEDGER).read_text(encoding="utf-8").splitlines()
    (d / ledger.LEDGER).write_text("\n".join(lines[:2] + lines[3:]) + "\n", encoding="utf-8")  # 한 줄 빼기
    assert not ledger.verify(d)["ok"]
    (d / ledger.LEDGER).write_text("\n".join([lines[0].replace('"count": 98', '"count": 3')] + lines[1:]) + "\n", encoding="utf-8")  # 시작 횟수 고치기
    v = ledger.verify(d)
    assert not v["ok"] and v["broken_at"] == 1


def test_save_result_failure_keeps_old_result(tmp_path, monkeypatch):
    """저장 도중 실패해도 옛 결과(json·일별)가 제자리에 온전히 남고 임시 파일이 남지 않는다."""
    from engine.experiment import save_result
    from engine import runlog
    prices, s = predictable_ohlcv(600, seed=6)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(name="at", ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40))
    out = tmp_path / "res"
    p1 = save_result(run_experiment(cfg, dv, reg), out)
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".parquet")}
    real = Path.rename
    def boom(self, target):  # 일별 자료는 옮겨지고 json 을 옮길 때 실패
        if self.suffix == ".json" and "history" in str(target):
            raise OSError("simulated")
        return real(self, target)
    monkeypatch.setattr(Path, "rename", boom)
    with pytest.raises(OSError):
        save_result(run_experiment(cfg, dv, reg), out)
    monkeypatch.undo()
    after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".parquet")}
    assert after == before and not list(out.glob(".*tmp")) and not list((out / "history").glob("*"))
    (out / p1.name).unlink()  # 짝 잃은 일별 자료도 덮지 않고 보관한다
    save_result(run_experiment(cfg, dv, reg), out)
    assert len(list((out / "history").glob("*.parquet"))) == 1


def test_run_record_points_to_loaded_master_not_disk(tmp_path):
    """실험 기록의 사전 지문·사본은 '읽은 그 바이트'다 — 읽은 뒤 파일이 바뀌어도."""
    import openpyxl, hashlib
    from engine import runlog
    m = tmp_path / "data" / "variable_master.xlsx"; m.parent.mkdir()
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "variables"
    ws.append(["id", "freq", "source", "source_code", "lag_days", "active"]); ws.append(["S_signal", "D", "DERIVED", "", 0, ""])
    wb.save(m)
    loaded = m.read_bytes()
    reg, _ = Registry.load_master(m)
    assert reg.get("S_signal").active is True  # active 빈칸 = Y
    ws.append(["LATER", "D", "DERIVED", "", 0, "Y"]); wb.save(m)  # 읽은 뒤 사전이 바뀜
    r = runlog.snapshot_master(reg.source_path, tmp_path / "data", raw=reg.source_bytes, sha=reg.source_sha256)
    assert r["sha256"] == hashlib.sha256(loaded).hexdigest() != hashlib.sha256(m.read_bytes()).hexdigest()
    assert (tmp_path / "data" / r["snapshot"]).read_bytes() == loaded


# ---------------------------------------------------------------------------
# 2026-11-10: Clark–West · 통계 재계산 · 결과 삭제는 보관
# ---------------------------------------------------------------------------
def test_clark_west_formula_size_and_power():
    """CW: 0예측 기준이면 통계량 = mean(2·y·f)/se. 신호 없는 추정 모델에서 기각률이 명목(5%) 근처 이하, DM '유의하게 나쁨'은 과다. 신호가 있으면 잡는다."""
    from engine.backtest import clark_west, diebold_mariano
    rng = np.random.default_rng(1)
    y, f = rng.normal(size=500), rng.normal(size=500)
    t, p = clark_west(y, f, 0.0)
    a = 2 * y * f
    assert np.isclose(t, a.mean() / (a.std(ddof=0) / np.sqrt(len(a))))
    assert np.isclose(clark_west(y + 0.5, f + 0.5, 0.5)[0], t)  # 기준을 옮겨도 같은 식(분류: 0.5)
    assert np.isnan(clark_west(y[:5], f[:5])[0])

    def sim(b, n=900, win=150, k=5):
        X = rng.normal(size=(n, k)); yy = b * X[:, 0] + rng.normal(size=n); ff = np.full(n, np.nan)
        for i in range(win, n):
            ff[i] = X[i] @ np.linalg.lstsq(X[i - win:i], yy[i - win:i], rcond=None)[0]
        m = ~np.isnan(ff)
        return yy[m], ff[m]
    R = 120
    null = [sim(0.0) for _ in range(R)]
    cw_rej = np.mean([clark_west(a, b)[1] < 0.05 for a, b in null])
    dm_bad = np.mean([(lambda d, pv: d > 0 and pv < 0.05)(*diebold_mariano(b - a, 0 - a)) for a, b in null])
    assert cw_rej <= 0.10 and dm_bad >= 0.5
    assert np.mean([clark_west(*sim(0.15))[1] < 0.05 for _ in range(40)]) >= 0.8


def test_restat_matches_engine_and_separates_stale(tmp_path):
    """재계산 도구: 저장된 일별 자료로 다시 구한 DM 이 엔진 값과 같고(같은 표본), CW·통일 DSR 을 내며, 옛 엔진 결과는 V·Holm 에서 뺀다. 원 결과는 고치지 않는다."""
    import importlib.util
    from engine.experiment import save_result, ENGINE_REV
    from engine import ledger
    spec = importlib.util.spec_from_file_location("restat", Path(__file__).resolve().parents[1] / "tools" / "restat.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    prices, s = predictable_ohlcv(700, seed=8)
    dv, reg = _dv_with(prices, s, tmp_path)
    out = tmp_path / "results" / "trading"
    for nm, vt in (("a", ["S_signal"]), ("b", [])):
        cfg = ExperimentConfig(name=nm, ticker="TEST", model="ridge", variable_ids=vt, use_technical=not vt, validation=Validation(train_len=300, test_len=40))
        res = run_experiment(cfg, dv, reg)
        assert "cw_vs_zero" in res["prediction_metrics"]
        p = save_result(res, out); ledger.record(out, "trial", config_hash=res["config_hash"])
    old = json.loads(p.read_text(encoding="utf-8")); old["engine_rev"] = None; old["config"]["name"] = "old"
    (out / "old_x.json").write_text(json.dumps(old), encoding="utf-8")
    (out / "old_x_daily.parquet").write_bytes(p.with_name(p.stem + "_daily.parquet").read_bytes())
    before = {q.name: q.read_bytes() for q in out.iterdir() if q.is_file()}
    r = rs.restat("trading", tmp_path / "results")
    assert r["info"]["n_results"] == 3 and r["info"]["n_stale"] == 1 and r["info"]["n_trials_ledger"] == 2 and not r["info"]["dm_recalc_mismatch"]
    rows = {x["name"]: x for x in r["rows"]}
    assert rows["a"]["dm_matches_stored"] and rows["a"]["cw_vs_zero_p"] < 0.01  # 심은 신호를 CW 가 잡는다
    assert rows["a"]["holm_cw_zero_p_m"] == 2 and "holm_cw_zero_p" not in rows["old"]
    assert rows["a"]["deflated_basis"]["n_trials"] == 2
    assert {q.name: q.read_bytes() for q in out.iterdir() if q.is_file()} == before
    assert ENGINE_REV


def test_api_delete_archives_and_ignores_non_results(tmp_path, monkeypatch):
    """화면 서버: 카운터·봉인 기록은 결과가 아니다(조회·삭제 불가). 결과 삭제는 history/ 보관 + 이력. 봉인은 일반 실행 길로 못 연다."""
    import api.main as m
    from fastapi import HTTPException
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    d = tmp_path / "trading"; d.mkdir()
    (d / "experiment_counter.json").write_text('{"count": 7}'); (d / "sealed_log.json").write_text("[]")
    (d / "x_abc.json").write_text('{"config": {}}'); (d / "x_abc_daily.parquet").write_bytes(b"p")
    for bad in ("experiment_counter", "sealed_log", "ablation_1", "..\\x", "a/b", "C:x", ""):
        assert m._find_result(bad) is None
    assert m.delete_result("experiment_counter") == {"deleted": 0} and (d / "experiment_counter.json").exists()
    r = m.delete_result("x_abc")
    assert r["deleted"] == 2 and not (d / "x_abc.json").exists() and len(list((d / "history").iterdir())) == 2
    assert '"deleted"' in (d / "run_log.jsonl").read_text(encoding="utf-8")
    from engine import ledger
    assert ledger.count(d) == 7 and ledger.entries(d)[-1]["kind"] == "deleted"
    with pytest.raises(HTTPException):
        m.submit(m.SubmitBody(config={"ticker": "T", "model": "ridge"}, sealed=True))


# ---------------------------------------------------------------------------
# 2026-11-10: 엑셀 칸 단위 읽기 · 일부 출처 수집의 이어받기
# ---------------------------------------------------------------------------
def test_excel_cells_dates_and_values(tmp_path):
    """숫자로 적은 연·연월·날짜, 여러 날짜 표기를 그대로 읽고(전에는 1970-01-01 로 뭉치거나 줄이 빠짐), 못 읽는 칸은 버리지 않고 문제로 알린다."""
    import datetime as dt
    from engine.store import excel_to_long, parse_date_cell, parse_value_cell, read_data_sheets
    from engine.master_check import check_master
    assert [str(parse_date_cell(x).date()) for x in (2019, 202003, 20200131, "2020/02/29", "2020.03.31", "2020-04", "2020년 5월", "2020Q3", dt.date(2021, 5, 5))] == \
        ["2019-01-01", "2020-03-01", "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-01", "2020-05-01", "2020-09-01", "2021-05-05"]  # 분기는 마지막 달 1일(도움말 2-4절의 약속)
    assert all(parse_date_cell(x) is None for x in (43831, "abc", "2020-13", 1850, True, None))
    assert parse_value_cell("1,234.5") == (1234.5, False) and parse_value_cell(" - ") == (None, False) and parse_value_cell("3.5%") == (None, True)
    m = tmp_path / "variable_master.xlsx"
    rows = [{"id": "Y1", "freq": "Y", "source": "XLSX", "source_code": "df_Y!Y1", "lag_days": 400}, {"id": "M1", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": 30}]
    _mini_master(m, rows, [("df_Y", ["date", "Y1"], [[2019, 1.0], [2020, 2.0], [2021, 3.0]]),
                           ("df_M", ["date", "M1", "비고"], [["2020-01", "1,000", "메모"], ["2020/02/01", 2.0, None], ["2020년 3월", "-", None]])])
    reg, _ = Registry.load_master(m)
    chk = check_master(reg, m)
    assert chk.errors == [] and chk.new_vars == [] and any("메모 열" in n for n in chk.notices)  # 글자뿐인 새 열은 변수로 등록하지 않는다
    long = excel_to_long(m, reg)
    y = long[long.variable_id == "Y1"]
    assert list(y["date"].dt.year) == [2019, 2020, 2021] and list(y["value"]) == [1.0, 2.0, 3.0]
    assert list(long[long.variable_id == "M1"]["value"]) == [1000.0, 2.0]
    _mini_master(m, rows, [("df_M", ["date", "M1"], [[43831, 1.0], ["2020-02", "3.5%"], ["2020-03", 1.0], ["2020-03", 2.0]])])
    reg, _ = Registry.load_master(m)
    errs = check_master(reg, m).errors
    assert any("A2" in e and "날짜" in e for e in errs) and any("B3" in e and "숫자가 아닌" in e for e in errs) and any("B5" in e and "두 번" in e for e in errs)
    with pytest.raises(ValueError):
        excel_to_long(m, reg)
    assert len(read_data_sheets(m)[0]) == 1  # 문제 칸은 넣지 않는다


def test_master_check_formula_sheetname_case(tmp_path):
    """수식이 있는데 새 열을 등록해야 하면 멈춘다(다시 저장하면 수식 값이 지워지므로). 시트 이름 오타·대소문자만 다른 변수 이름도 멈춘다."""
    import openpyxl
    from engine.master_check import check_master
    m = tmp_path / "variable_master.xlsx"
    rows = [{"id": "X1", "freq": "M", "source": "XLSX", "source_code": "df_M!X1", "lag_days": 30}]
    _mini_master(m, rows, [("df_M", ["date", "X1", "NEW1"], [["2020-01", 1.0, 5.0], ["2020-02", 2.0, 6.0]])])
    wb = openpyxl.load_workbook(m); wb["df_M"]["B3"] = "=B2*2"; wb.save(m)
    reg, _ = Registry.load_master(m)
    errs = check_master(reg, m).errors
    assert any("수식" in e for e in errs)
    _mini_master(m, rows, [("df_m", ["date", "X1"], [["2020-01", 1.0]])])
    reg, _ = Registry.load_master(m)
    assert any("df_M" in e and "고쳐" in e for e in check_master(reg, m).errors)
    _mini_master(m, rows, [("df_M", ["date", "X1", "x1"], [["2020-01", 1.0, 2.0]])])
    reg, _ = Registry.load_master(m)
    assert any("대소문자" in e for e in check_master(reg, m).errors)


def test_collect_partial_sources_carry_forward(tmp_path, monkeypatch):
    """--sources xlsx 로 새 버전을 만들면 받지 않은 출처(가격·API 변수)는 직전 버전에서 이어받는다 — 전에는 가격 없는 버전이 최신이 됐다."""
    import collect
    data = tmp_path / "data"; (data / "versions" / "20200101T000000Z").mkdir(parents=True)
    rows = [{"id": "M1", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": 30},
            {"id": "KOSPI", "freq": "D", "source": "YF", "source_code": "^KS11", "lag_days": 0},
            {"id": "GONE", "freq": "D", "source": "XLSX", "source_code": "df_D!GONE", "lag_days": 0}]  # 전에는 YF 였다가 출처가 바뀐 변수
    _mini_master(data / "variable_master.xlsx", rows, [("df_M", ["date", "M1"], [["2020-01", 1.0], ["2020-02", 2.0]])])
    old = data / "versions" / "20200101T000000Z"
    prices, _ = predictable_ohlcv(60, seed=1)
    for t in collect.TICKERS:
        prices.to_parquet(old / f"prices_{t}_yf.parquet"); prices.to_parquet(old / f"prices_{t}.parquet")
    idx = prices.index
    pd.DataFrame({"variable_id": ["KOSPI"] * 3 + ["GONE"] * 3, "date": list(idx[:3]) * 2, "value": [1.0, 2, 3, 9, 9, 9]}).to_parquet(old / "variables_yf.parquet", index=False)
    pd.DataFrame({"variable_id": ["M1"], "date": [pd.Timestamp("2019-12-01")], "value": [99.0]}).to_parquet(old / "variables_xlsx_oldfile.parquet", index=False)
    (old / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "xlsx", "--data-dir", str(data)])
    assert collect.main() == 0
    new = sorted(p.name for p in (data / "versions").iterdir())[-1]
    assert new != "20200101T000000Z"
    from engine.store import DataVersion
    dv = DataVersion(data, new)
    v = dv.read_variables()
    assert sorted(v.variable_id.unique()) == ["KOSPI", "M1"]  # 출처가 바뀐 GONE 은 이어받지 않고, 옛 엑셀 파일의 M1(99)도 들어오지 않는다
    assert list(v[v.variable_id == "M1"]["value"]) == [1.0, 2.0]
    assert len(dv.read_prices(collect.TICKERS[0])) == 60
    run = json.loads((data / "versions" / new / "manifest.json").read_text(encoding="utf-8"))["collect_runs"][-1]
    assert run["carried_from"]["version"] == "20200101T000000Z" and run["multi_source_ids"] == {} and run["master_snapshot"]
