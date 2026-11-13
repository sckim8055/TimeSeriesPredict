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
    # 첫 봉인 결정일 = 수익에 봉인 시작일의 시가가 처음 쓰이는 결정일(매매 k=1 이면 봉인 시작 2 거래일 전) — 2026-11-10
    assert pd.Timestamp(res["evaluation_period"][0]) == prices.index[1000 - 2]
    assert res["seal_boundary"]["first_sealed_decision"] == str(prices.index[998].date())
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
    assert mp == {"enforce_stationarity": False, "enforce_invertibility": False, "scale": 1.0, "starts": "default"} and len(notes) == 4
    # 2026-11-06 결과(제약 키 있음, 배율 키 없음) → 배율만 그때 값(1)으로
    assert pinned_model_params("sarima", {"p": 1, "D": 1, "enforce_stationarity": True, "enforce_invertibility": True}, {})[0] == {"scale": 1.0, "starts": "default", "trend": "n", "D": 1}
    # 2026-11-07~09 결과(배율 있음, 상수항 칸 없음) → 상수항은 그때 값(n), 계절 차분 D 는 결과에 기록된 값(1 — 2026-11-10 에 기본값이 0 으로 바뀜)
    assert pinned_model_params("sarima", {"p": 1, "D": 1, "enforce_stationarity": True, "enforce_invertibility": True, "scale": 100.0}, {})[0] == {"starts": "default", "trend": "n", "D": 1}
    assert pinned_model_params("sarima", {"p": 1, "D": 1, "scale": 100.0, "trend": "n", "enforce_stationarity": True, "enforce_invertibility": True, "starts": "multi"}, {"D": 0})[0] == {"D": 0}  # 설정에 직접 적은 값은 그대로
    assert pinned_model_params("sarima", {"p": 1, "D": 0, "trend": "auto", "starts": "multi", "enforce_stationarity": True, "enforce_invertibility": True, "scale": 100.0}, {})[0] == {}
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
        # 출발점 하나(starts=default)로 본다 — 이 시험은 '배율이 모형을 바꾸지 않는다'를 보는 것이고, 여러 출발점 추정은 따로 시험한다
        m = M.Model(resolve_params(M.PARAMS, {"scale": sc, "starts": "default", **extra}, "M"), n_features=1, target_type="regression")
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
    # SARIMA 기본은 2026-11-10 부터 계절 차분 없음(D=0) + 상수항(trend=auto → c)
    mod1 = SARIMAX(r, **({"order": (2, 0, 2), "trend": "c"} if model == "arima" else {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12), "trend": "c"}))
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
    cut = prices.index[1000 - 2]  # 첫 봉인 결정일: 수익에 봉인 시작일 시가가 처음 쓰이는 결정(매매 k=1 → 2 거래일 앞) — 2026-11-10
    assert first == cut and last == prices.index[-1]
    assert d["prediction"].loc[cut:].notna().all()
    _, _, _, _, ds = prepare(cfg, dv, reg)
    folds = sealed_folds(cfg, ds, 1)
    assert res["n_folds"] == len(folds)
    if mode == "split":
        assert len(folds) == 1 and folds[0].train[-1] < cut and len(folds[0].train) == 998 - 2
    else:
        assert len(folds) == int(np.ceil(302 / 40)) and folds[0].test[0] == cut and len(folds[0].train) == 400 - 2
    assert pd.Timestamp(res["periods"]["first_fold"]["test"][0]) >= cut  # 개발 폴드 기록 없음


def test_sealed_eval_rule_model_only_sealed_days(tmp_path):
    prices = random_walk_ohlcv(1200, seed=6)
    dv, reg = _dv_with(prices, None, tmp_path)
    seal = prices.index[1000]
    res = run_experiment(ExperimentConfig(ticker="TEST", model="ma_cross", validation=Validation(train_len=400, test_len=40, sealed_from=str(seal.date()))), dv, reg, sealed=True)
    assert pd.Timestamp(res["evaluation_period"][0]) == prices.index[1000 - 2] and res["sealed_evaluation"] is True  # 첫 봉인 결정일


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
    assert not pv["blockers"] and pv["plan"]["first_test"] == str(prices.index[700 - 2].date()) and pv["plan"]["last_test"] == str(prices.index[-1].date())
    assert pv["plan"]["n_days"] == 202 and any("이미 1회" in w for w in pv["warnings"])  # 봉인 결정일 = 봉인 시작 2 거래일 전부터(2026-11-10)
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


# ---------------------------------------------------------------------------
# 2026-11-10: 봉인 경계 — 개발 구간의 어떤 수치도 봉인 시작일 이후의 가격을 쓰지 않는다
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model,freq,k", [("ridge", "D", 1), ("ma_cross", "D", 1), ("ridge", "W", 1), ("ridge", "D", 3)])
def test_dev_results_do_not_depend_on_sealed_prices(tmp_path, model, freq, k):
    """봉인 시작일 이후의 가격을 통째로 바꿔도 개발 결과(성과·검정·일별 수익)가 한 자리도 달라지지 않아야 한다.
    전에는 개발의 마지막 결정이 봉인 첫 시가로 채점돼 달라졌다(점검 T5). 그리고 개발 결정 + 봉인 결정 = 빠짐·겹침 없는 전체."""
    prices, s = predictable_ohlcv(1300, seed=11)
    sealed = str(prices.index[1000].date())
    tampered = prices.copy()
    tampered.loc[sealed:, ["open", "high", "low", "close"]] *= 3.0
    kw = dict(ticker="TEST", model=model, frequency=freq, target={"kind": "return", "k": k}, variable_ids=["S_signal"] if model == "ridge" else [],
              use_technical=False, validation=Validation(train_len=400 if freq == "D" else 100, test_len=40 if freq == "D" else 8, sealed_from=sealed))
    out = []
    for i, pr in enumerate((prices, tampered)):
        dv, reg = _dv_with(pr, s, tmp_path / f"v{i}")
        out.append(run_experiment(ExperimentConfig(**kw), dv, reg))
    a, b = out
    lo, hi = a["evaluation_period"]
    assert a["evaluation_period"] == b["evaluation_period"] and pd.Timestamp(hi) < pd.Timestamp(sealed)
    da, db = a["_daily"].loc[lo:hi], b["_daily"].loc[lo:hi]
    for col in ("signal", "net_return", "hold_return", "prediction", "target"):
        assert np.allclose(da[col].fillna(-9).values, db[col].fillna(-9).values), col
    assert a["performance"]["sharpe"] == pytest.approx(b["performance"]["sharpe"]) and a["performance"]["total_return"] == pytest.approx(b["performance"]["total_return"])
    assert a["prediction_metrics"] == pytest.approx(b["prediction_metrics"])
    dv, reg = _dv_with(prices, s, tmp_path / "s")
    sr = run_experiment(ExperimentConfig(**kw), dv, reg, sealed=True)
    dev_days = a["_daily"]["signal"].dropna().index
    seal_days = sr["_daily"]["signal"].dropna().index
    calendar = a["_daily"].index
    assert len(dev_days.intersection(seal_days)) == 0
    assert calendar.get_loc(seal_days[0]) == calendar.get_loc(dev_days[-1]) + 1  # 빠지는 결정일 없음
    assert sr["seal_boundary"]["horizon_rows"] == 1 + k


def test_forecast_dev_results_do_not_depend_on_sealed_values(tmp_path):
    """예측 모드도 같다: 봉인 시작일 이후의 대상 값을 바꿔도 개발 결과가 달라지지 않는다(정답이 k 주기 뒤 값이므로 경계는 k 행 앞)."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2005-01-01", periods=240, freq="MS")
    lvl = pd.Series(100 * np.exp(np.cumsum(0.002 + 0.004 * rng.normal(size=240))), index=idx)
    sealed = str(idx[200].date())
    res = []
    for i, mult in enumerate((1.0, 2.0)):
        v = lvl.copy(); v.loc[sealed:] *= mult
        dv = DataVersion(tmp_path / f"f{i}", "t")
        dv.write_variables(v.rename("value").rename_axis("date").reset_index().assign(variable_id="TGT"), source="syn")
        reg = Registry(); reg.add(Variable(id="TGT", freq="M", source="DERIVED"))
        cfg = ExperimentConfig(mode="forecast", target_variable="TGT", ticker="TGT", frequency="M", model="ridge", use_technical=True,
                               validation=Validation(train_len=60, test_len=6, sealed_from=sealed))
        res.append(run_experiment(cfg, dv, reg))
    a, b = res
    assert a["evaluation_period"] == b["evaluation_period"]
    assert a["prediction_metrics"] == pytest.approx(b["prediction_metrics"])
    assert a["seal_boundary"]["horizon_rows"] == 1


# ---------------------------------------------------------------------------
# 2026-11-10: 촘촘한 주기의 변수를 성긴 주기 모델에 넣을 때 변화율의 '한 칸' = 모델 주기
# ---------------------------------------------------------------------------
def test_finer_variable_change_is_over_model_period(tmp_path):
    """일간 변수의 pct 가 월간·주간 모델에서는 '결정일 사이(모델 주기 한 칸)의 변화'여야 한다(전에는 그 달 마지막 하루의 변화).
    일간 모델에서는 그대로 하루 변화, 월간 변수는 일간 모델에서 원래 주기의 변화, pct5 처럼 간격을 적은 변환은 원래 주기 그대로."""
    from engine.experiment import build_dataset
    prices, _ = predictable_ohlcv(900, seed=12)
    rng = np.random.default_rng(0)
    lvl = pd.Series(100 * np.exp(np.cumsum(0.01 * rng.normal(size=900))), index=prices.index, name="value")
    mon = lvl.resample("MS").first()
    dv = DataVersion(tmp_path, "t"); dv.write_prices("TEST", prices)
    for vid, sr in (("DV", lvl), ("DV5", lvl), ("MV", mon)):
        dv.write_variables(sr.rename("value").rename_axis("date").reset_index().assign(variable_id=vid), source=f"syn_{vid}")
    reg = Registry()
    reg.add(Variable(id="DV", freq="D", source="DERIVED", transform="pct")); reg.add(Variable(id="DV5", freq="D", source="DERIVED", transform="pct5"))
    reg.add(Variable(id="MV", freq="M", source="DERIVED", transform="pct"))
    base = dict(ticker="TEST", model="ridge", variable_ids=["DV", "DV5", "MV"], use_technical=False)
    for fq in ("M", "W"):
        ds = build_dataset(ExperimentConfig(**base, frequency=fq, validation=Validation(train_len=10, test_len=2)), dv, reg)
        dec = ds.X.index
        asof = lvl.reindex(dec, method="ffill")  # 결정일에 알 수 있는 수준값(lag 0)
        want = asof.pct_change()
        got = ds.X["DV_pct"]
        ok = want.notna() & got.notna()
        assert ok.sum() > 20 and np.allclose(got[ok], want[ok]), fq
        assert not np.allclose(got[ok], lvl.pct_change().reindex(dec)[ok])  # 하루 변화가 아니다
        assert ds.transform_basis["DV"] == "model_period" and ds.transform_basis["DV5"] == "native"
        assert np.allclose(ds.X["DV5_pct5"].dropna(), lvl.pct_change(5).reindex(dec).dropna())
        assert ds.transform_basis["MV"] == "native"  # 월간 변수는 월간·주간 모델보다 촘촘하지 않다 → 원래 주기의 변화
    ds = build_dataset(ExperimentConfig(**base, frequency="D", validation=Validation(train_len=300, test_len=40)), dv, reg)
    assert ds.transform_basis == {"DV": "native", "DV5": "native", "MV": "native"}
    g = ds.X["DV_pct"].dropna()
    assert np.allclose(g, lvl.pct_change().reindex(g.index))
    mv = ds.X["MV_pct"].dropna()
    assert mv.nunique() <= len(mon) and np.isclose(mv.iloc[-1], mon.pct_change().iloc[-1])  # 월간 변화가 그 달 내내 이어진다


# ---------------------------------------------------------------------------
# 폴드 시드 · 시드 반복 · 학습 진단 (2026-11-10)
# ---------------------------------------------------------------------------
def _wf_setup(tmp_path, model, n=700, seed=21, **cfg_kw):
    from engine.experiment import make_folds, prepare
    prices, s = predictable_ohlcv(n, seed=seed)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=300, test_len=40), **cfg_kw)
    cfg, plugin, params, tt, ds = prepare(cfg, dv, reg, device="cpu")
    folds, _ = make_folds(cfg, ds, cfg.target.k)
    return cfg, plugin, params, tt, ds, [f for f in folds if len(f.test)]


@pytest.mark.parametrize("model,extra", [("lightgbm_model", {}), ("lstm", {"model_params": {"epochs": 3, "hidden": 4}, "seq_len": 5})])
def test_fold_results_do_not_depend_on_fold_order(model, extra, tmp_path):
    """폴드의 난수는 그 폴드의 날짜로 정해진다 → 폴드를 한꺼번에 돌리든, 거꾸로 하나씩 돌리든 같은 예측.
    (전에는 실험 시작에 한 번만 시드를 고정해, 폴드 하나만 다시 계산하면 실험 때와 다른 값이 나왔다.)"""
    from engine.experiment import _walkforward_predict
    cfg, plugin, params, tt, ds, folds = _wf_setup(tmp_path, model, **extra)
    folds = folds[:4]
    p_all, _, _ = _walkforward_predict(cfg, plugin, params, ds, "cpu", folds, 1, tt)
    p_one = None
    for f in reversed(folds):
        q, _, _ = _walkforward_predict(cfg, plugin, params, ds, "cpu", [f], 1, tt)
        p_one = q if p_one is None else p_one.where(q.isna(), q)
    assert p_all.notna().sum() == p_one.notna().sum() > 100
    assert np.allclose(p_all.dropna().values, p_one.dropna().values, rtol=0, atol=1e-9)
    # 실험 시드를 바꾸면 달라진다(LightGBM 도 — 전에는 random_state=0 고정이라 같았다)
    p_s, _, _ = _walkforward_predict(cfg.model_copy(update={"seed": 7}), plugin, params, ds, "cpu", folds, 1, tt)
    assert not np.allclose(p_s.dropna().values, p_all.dropna().values)


def test_n_seeds_averages_replicates_and_records_spread(tmp_path, monkeypatch):
    """n_seeds=2 의 예측 = 반복 0 과 반복 1 을 따로 돌린 예측의 평균. 시드 사이 흩어짐이 진단에 남는다."""
    import engine.experiment as E
    cfg, plugin, params, tt, ds, folds = _wf_setup(tmp_path, "lightgbm_model")
    folds = folds[:3]
    dg: list = []
    p2, _, imp2 = E._walkforward_predict(cfg.model_copy(update={"n_seeds": 2}), plugin, params, ds, "cpu", folds, 1, tt, diags=dg)
    p_r0, _, imp0 = E._walkforward_predict(cfg, plugin, params, ds, "cpu", folds, 1, tt)
    orig = E.fold_seed
    monkeypatch.setattr(E, "fold_seed", lambda seed, d, rep=0: orig(seed, d, rep + 1))
    p_r1, _, imp1 = E._walkforward_predict(cfg, plugin, params, ds, "cpu", folds, 1, tt)
    monkeypatch.setattr(E, "fold_seed", orig)
    assert not np.allclose(p_r0.dropna(), p_r1.dropna())
    assert np.allclose(p2.dropna().values, ((p_r0 + p_r1) / 2).dropna().values, atol=1e-12)
    assert np.allclose(np.mean(imp2, axis=0), (np.mean(imp0, axis=0) + np.mean(imp1, axis=0)) / 2)
    sm = E.diag_summary(dg)
    assert sm["seeds"]["n_seeds"] == 2 and sm["seeds"]["pred_sd_across_seeds_mean"] > 0
    assert sm["trees_used"]["n"] == 2 * len(folds) and sm["early_stopping_share"] == 1.0
    assert [r["seed"] for r in dg[0]["reps"]] == [orig(cfg.seed, folds[0].test[0], 0), orig(cfg.seed, folds[0].test[0], 1)]


def test_n_seeds_ignored_by_deterministic_models_and_hash_stable(tmp_path):
    """릿지처럼 난수를 쓰지 않는 모델은 n_seeds 를 줘도 한 번만 학습(같은 예측). n_seeds 기본값(1)은 설정 해시를 바꾸지 않는다."""
    import hashlib
    import engine.experiment as E
    cfg, plugin, params, tt, ds, folds = _wf_setup(tmp_path, "ridge")
    dg: list = []
    p1, _, _ = E._walkforward_predict(cfg, plugin, params, ds, "cpu", folds[:3], 1, tt)
    p5, _, _ = E._walkforward_predict(cfg.model_copy(update={"n_seeds": 5}), plugin, params, ds, "cpu", folds[:3], 1, tt, diags=dg)
    assert np.array_equal(p1.dropna().values, p5.dropna().values) and all(d["n_seeds"] == 1 for d in dg)
    old = hashlib.sha1(cfg.model_dump_json(exclude={"name", "n_seeds", "technical_features", "select_top", "pool_tickers"}).encode()).hexdigest()[:12]  # 이 칸들이 생기기 전의 해시
    assert cfg.config_hash() == old and cfg.model_copy(update={"n_seeds": 3}).config_hash() != old
    with pytest.raises(Exception):
        ExperimentConfig(ticker="TEST", model="ridge", n_seeds=0)


def test_result_records_model_diag_and_constant_prediction_is_flagged(tmp_path):
    """결과에 학습 진단이 남는다(쓰인 트리 수·시드 규칙). 분할이 하나도 안 되는 설정(잎 최소 표본 > 학습 표본)은 상수 예측 → 퇴화로 표시.
    트리가 생겨도 예측이 전혀 변하지 않으면 퇴화(전에는 중요도 합이 0 일 때만 잡았다)."""
    prices, s = predictable_ohlcv(700, seed=22)
    dv, reg = _dv_with(prices, s, tmp_path)
    base = dict(ticker="TEST", model="lightgbm_model", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40))
    res = run_experiment(ExperimentConfig(**base), dv, reg)
    md = res["model_diag"]
    assert md["trees_used"]["median"] >= 1 and md["limit"] == 300 and md["seeds"]["n_seeds"] == 1 and "sha256" in md["seeds"]["rule"]
    assert res["degenerate_model"] is False and res["prediction_metrics"]["pred_sd_ratio"] > 0.05
    assert res["prediction_metrics"]["rmse_zero"] > res["prediction_metrics"]["rmse"]  # 심은 신호: 0 예측보다 정확
    assert res["periods"]["first_fold"]["model_diag"]["trees_used"] >= 1
    res0 = run_experiment(ExperimentConfig(**base, model_params={"min_child_samples": 1000}), dv, reg)
    assert res0["degenerate_model"] is True and res0["model_diag"]["constant_folds"] == res0["model_diag"]["folds_checked_for_constant"] > 0
    rr = run_experiment(ExperimentConfig(**{**base, "model": "ridge"}), dv, reg)
    assert rr["model_diag"]["seeds"]["n_seeds"] == 1 and "trees_used" not in rr["model_diag"] and rr["degenerate_model"] is False


def test_collect_failed_source_keeps_previous_and_is_not_latest(tmp_path, monkeypatch):
    """요청한 출처가 실패하면(접속 불가) 직전 버전 값을 그대로 두고, 그 버전은 '일부 받지 못함'으로 표시되어 자동 최신에서 빠지며 종료 코드는 0 이 아니다.
    받은 것이 직전보다 짧아도 같은 처리(짧은 자료로 긴 자료를 덮지 않는다). 전에는 실패한 출처의 자료가 통째로 빠진 버전이 최신이 됐다(독립 검토 F1)."""
    import collect
    from adapters import yf_adapter
    from engine.store import DataVersion, latest_version_id, list_versions
    data = tmp_path / "data"; old = data / "versions" / "20200101T000000Z"; old.mkdir(parents=True)
    rows = [{"id": "KOSPI", "freq": "D", "source": "YF", "source_code": "^KS11", "lag_days": 0},
            {"id": "SOX", "freq": "D", "source": "YF", "source_code": "^SOX", "lag_days": 0}]
    _mini_master(data / "variable_master.xlsx", rows, [])
    prices, _ = predictable_ohlcv(60, seed=1)
    for t in collect.TICKERS:
        prices.to_parquet(old / f"prices_{t}_yf.parquet"); prices.to_parquet(old / f"prices_{t}.parquet")
    idx = prices.index
    pd.DataFrame({"variable_id": ["KOSPI"] * 5 + ["SOX"] * 5, "date": list(idx[:5]) * 2, "value": [1.0, 2, 3, 4, 5] * 2}).to_parquet(old / "variables_yf.parquet", index=False)
    (old / "manifest.json").write_text(json.dumps({"start": "2006-01-01"}))

    def boom(*a, **k):
        raise ConnectionError("접속 불가")
    monkeypatch.setattr(yf_adapter, "fetch_prices", boom)
    # KOSPI: 접속 실패. SOX: 받았지만 직전(5줄)보다 짧다(2줄)
    monkeypatch.setattr(yf_adapter, "fetch_series", lambda code, *a, **k: boom() if code == "^KS11" else pd.Series([7.0, 8.0], index=idx[3:5]))
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "yf", "--data-dir", str(data)])
    assert collect.main() == 2
    new = sorted(p.name for p in (data / "versions").iterdir())[-1]
    assert new != old.name
    dv = DataVersion(data, new)
    v = dv.read_variables()
    assert len(v[v.variable_id == "KOSPI"]) == 5 and len(v[v.variable_id == "SOX"]) == 5  # 둘 다 직전 값 그대로
    assert len(dv.read_prices(collect.TICKERS[0])) == 60
    man = dv.manifest()
    assert man["incomplete"] is True and set(man["kept_previous"]) >= {"KOSPI", "SOX"} and man["failed_items"]
    assert latest_version_id(data) == old.name and [x["incomplete"] for x in list_versions(data)] == [False, True]
    # 없는 버전 id 로 이어 받기는 거절(폴더를 만들지 않는다)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "yf", "--data-dir", str(data), "--version", "20200101T00000"])
    assert collect.main() == 3 and not (data / "versions" / "20200101T00000").exists()   # 검토 5: 시작 전에 멈춤 = 3(받지 못한 항목 2 와 구분)
    with pytest.raises(ValueError):
        DataVersion(data, "")
    # 같은 버전에 이어 받아 성공하면 온전한 버전이 된다
    monkeypatch.setattr(yf_adapter, "fetch_prices", lambda *a, **k: prices)
    monkeypatch.setattr(yf_adapter, "fetch_series", lambda code, *a, **k: pd.Series(np.arange(6.0), index=idx[:6]))
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "yf", "--data-dir", str(data), "--version", new, "--refresh"])
    assert collect.main() == 0
    assert DataVersion(data, new).manifest()["incomplete"] is False and latest_version_id(data) == new
    assert len(DataVersion(data, new).read_variables()) == 12


# ---------------------------------------------------------------------------
# 2026-11-10 독립 검토(F1~F22) 반영 시험
# ---------------------------------------------------------------------------
def _rolling_ols_forecasts(rng, n_eval, W, h, rho, mu, k=3):
    """정보 없는 설명변수 k 개 + 절편의 굴리는 창 OLS 예측(타겟 = h 기간 합, 평균 mu). 반환 (y, 예측, 학습 창 평균)."""
    T = W + h + n_eval + h + 5
    e = rng.normal(size=(T, k))
    X = e.copy()
    if rho:
        X[0] = e[0] / np.sqrt(1 - rho**2)
        for t in range(1, T):
            X[t] = rho * X[t - 1] + e[t]
    r = mu + rng.normal(size=T + h + 1)
    cs = np.cumsum(r); y = np.array([cs[t + h] - cs[t] for t in range(T)])
    Z = np.column_stack([np.ones(T), X])
    P = np.einsum("ti,tj->tij", Z, Z).cumsum(0); Q = (Z * y[:, None]).cumsum(0); C = np.cumsum(y)
    ts = np.arange(W + h, W + h + n_eval); hi = ts - h; lo = hi - W
    beta = np.linalg.solve(P[hi] - P[lo], (Q[hi] - Q[lo])[:, :, None])[:, :, 0]
    return y[ts], (Z[ts] * beta).sum(1), (C[hi] - C[lo]) / W


def test_cw_against_training_mean_does_not_reward_knowing_the_mean():
    """타겟 평균이 0 이 아니면 '0 기준' CW 는 평균만 맞히는 예측도 기각한다 — 변수의 정보를 말하려면 '학습 창 평균 기준'을 써야 한다(F2).
    정보 없는 변수 + 절편 모델: 0 기준은 대부분 기각, 평균 기준은 명목 수준 근처."""
    from engine.backtest import clark_west
    rng = np.random.default_rng(7)
    rej0 = rejm = 0; R = 300
    for _ in range(R):
        y, f, b = _rolling_ols_forecasts(rng, n_eval=120, W=60, h=1, rho=0, mu=0.6)
        rej0 += clark_west(y, f, 0.0)[1] < 0.05
        rejm += clark_west(y, f, b)[1] < 0.05
    assert rej0 / R > 0.9, rej0 / R
    assert rejm / R < 0.10, rejm / R


def test_cw_and_dm_size_with_overlapping_targets():
    """겹치는 타겟(h=5)과 지속적인 설명변수에서 CW 의 크기: 절단 핵(h−1)이면 명목 5% 근처, 옛 Bartlett 가중치는 그보다 많이 기각했다(F12)."""
    from engine.backtest import clark_west, diebold_mariano
    rng = np.random.default_rng(11)
    R = 400; rej = rej_old = 0
    for _ in range(R):
        y, f, _ = _rolling_ols_forecasts(rng, n_eval=600, W=250, h=5, rho=0.95, mu=0.0)
        rej += clark_west(y, f, 0.0, h=5)[1] < 0.05
        adj = 2 * y * f; dc = adj - adj.mean(); n = len(adj); var = dc @ dc / n
        for lag in range(1, 5):
            var += 2 * (1 - lag / 5) * (dc[lag:] @ dc[:-lag]) / n
        rej_old += (adj.mean() / np.sqrt(var / n)) > 1.645
    assert rej / R < 0.075, rej / R
    assert rej_old > rej  # 옛 가중치가 더 많이 기각
    # DM: h=1 은 전과 같은 값(표준정규), h>1 은 보정으로 통계량 크기가 줄어든다
    e1, e2 = rng.normal(size=300), rng.normal(size=300) * 1.1
    d = e1**2 - e2**2; dc = d - d.mean()
    assert np.isclose(diebold_mariano(e1, e2, h=1)[0], d.mean() / np.sqrt(dc @ dc / 300 / 300))
    s5, p5 = diebold_mariano(e1, e2, h=5)
    assert np.isfinite(s5) and 0 <= p5 <= 1


def test_result_has_mean_benchmark_metrics_and_bench_column(tmp_path):
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    res = run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40)), dv, reg)
    pm, d = res["prediction_metrics"], res["_daily"]
    assert pm["cw_vs_mean_p"] < 0.01 and pm["r2_oos_vs_mean"] > 0.3 and pm["dm_vs_mean"] < 0
    ev = d.dropna(subset=["prediction", "target"])
    assert ev["bench_mean"].notna().all() and d["bench_mean"].notna().sum() == d["prediction"].notna().sum()
    # 기준 = 그 결정일 앞 학습 창의 타겟 평균(결정일 이후 값이 들어가지 않는다): 첫 폴드에서 손으로
    f0 = res["periods"]["first_fold"]
    y = d["target"]
    tr = y.loc[f0["train"][0]:f0["train"][1]]
    assert np.isclose(ev["bench_mean"].iloc[0], tr.mean())
    assert pd.Timestamp(f0["train"][1]) < ev.index[0]
    rn = run_experiment(ExperimentConfig(ticker="TEST", model="ma_cross"), dv, reg)
    assert "cw_vs_mean" not in (rn["prediction_metrics"] or {})  # 규칙 모델은 예측값이 없다


def test_holm_family_and_one_sided_direction():
    """가족 = 대상·주기·k·타겟 종류(평가 구간이 달라도 한 가족). naive 보다 '나쁜' 쪽으로 유의한 것은 보정 p 가 작아지지 않는다(F3·F13)."""
    from engine.multitest import better_one_sided, holm
    rows = [{"mode": "trading", "ticker": "T", "frequency": "D", "k": 1, "target_kind": "return", "evaluation_period": ["2012-01-02", "2024-12-30"], "p": 0.006},
            {"mode": "trading", "ticker": "T", "frequency": "D", "k": 1, "target_kind": "return", "evaluation_period": ["2012-01-13", "2024-12-30"], "p": 0.30},
            {"mode": "trading", "ticker": "T", "frequency": "W", "k": 1, "target_kind": "return", "p": 0.04},
            {"mode": "trading", "ticker": "T", "frequency": "D", "k": 1, "target_kind": "return", "p": 0.001, "stale": True}]
    holm(rows, "p", "h", skip=lambda r: r.get("stale"))
    assert rows[0]["h_m"] == 2 and np.isclose(rows[0]["h"], 0.012) and rows[0]["h_all_m"] == 3 and np.isclose(rows[0]["h_all"], 0.018)
    assert rows[2]["h_m"] == 1 and "h" not in rows[3]
    assert np.isclose(better_one_sided(-2.0, 0.04), 0.02) and np.isclose(better_one_sided(3.58, 0.0004), 0.9998)
    assert better_one_sided(None, 0.1) is None


def test_ledger_detects_tail_truncation_and_survives_partial_line(tmp_path):
    """원장: 끝 줄을 지우면 사본의 머리·결과 파일의 기록과 맞지 않아 드러난다. 쓰다 끊긴 줄이 있어도 N 을 셀 수 있고(줄이지 않는 쪽으로) 다음 줄은 새 줄에 쓴다(F9·F15)."""
    from engine import ledger
    d = tmp_path / "r"
    for i in range(3):
        before = ledger.head(d)
        ledger.record(d, "trial", config_hash=f"h{i}", name=f"n\u2028{i}", run_id=f"r{i}")
        (d / f"res{i}.json").write_text(json.dumps({"config": {}, "run_record": {"run_id": f"r{i}", "ledger_before": before}}), encoding="utf-8")
    assert ledger.verify(d)["ok"] and ledger.count(d) == 3 and len(ledger.entries(d)) == 4  # 이름의 줄 나눔 문자가 줄을 가르지 않는다
    p = d / "trial_ledger.jsonl"
    full = p.read_text(encoding="utf-8")
    p.write_text("\n".join(full.split("\n")[:-2]) + "\n", encoding="utf-8")       # 끝 줄 삭제
    v = ledger.verify(d)
    assert v["ok"] is False and "끝 줄" in v["why"]
    (d / "experiment_counter.json").unlink()                                          # 사본까지 지워도 결과 파일의 기록으로 드러난다
    v = ledger.verify(d)
    assert v["ok"] is False and v["n_mismatch"] >= 1
    p.write_text(full + '{"seq": 4, "ut', encoding="utf-8")                           # 쓰다 끊긴 줄
    assert ledger.count(d) == 4 and ledger.verify(d)["ok"] is False                   # 읽을 수 없는 줄은 시도 1회로 센다
    ledger.record(d, "trial", config_hash="x", run_id="r9")
    assert ledger.count(d) == 5 and len(p.read_text(encoding="utf-8").strip().split("\n")) == 6


def test_ledger_concurrent_appends_keep_chain(tmp_path):
    import threading
    from engine import ledger
    d = tmp_path / "r"
    def work(tag):
        for i in range(40):
            ledger.record(d, "trial", config_hash=f"{tag}{i}", run_id=f"{tag}{i}")
    ts = [threading.Thread(target=work, args=(t,)) for t in "abcd"]
    [t.start() for t in ts]; [t.join() for t in ts]
    v = ledger.verify(d)
    assert v["ok"], v
    assert ledger.count(d) == 160 and sum(e["kind"] == "genesis" for e in ledger.entries(d)) == 1


def test_seal_policy_fills_blank_blocks_later_date_and_records_override(tmp_path, monkeypatch):
    """봉인 정책: 대상의 봉인 시작일이 정해져 있으면 빈 설정에는 채우고, 더 늦은 날짜는 막고, 일부러 넘기면 이유가 결과에 남는다(F8)."""
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    seal = str(prices.index[700].date())
    pol = tmp_path / "seal_policy.json"
    pol.write_text(json.dumps({"_comment": "x", "trading": {"TEST": seal}}), encoding="utf-8")
    monkeypatch.setenv("TSP_SEAL_POLICY", str(pol))
    base = dict(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False)
    r = run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40)), dv, reg)
    assert r["seal_policy"]["status"] == "applied" and r["config"]["validation"]["sealed_from"] == seal
    assert pd.Timestamp(r["evaluation_period"][1]) < pd.Timestamp(seal)
    later = str(prices.index[800].date())
    with pytest.raises(ValueError, match="봉인 구간"):
        run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40, sealed_from=later)), dv, reg)
    monkeypatch.setenv("TSP_SEAL_OVERRIDE", "원 논문 분할 재현")
    r2 = run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40)), dv, reg)
    assert r2["seal_policy"]["status"] == "override" and r2["seal_policy"]["reason"] == "원 논문 분할 재현"
    assert pd.Timestamp(r2["evaluation_period"][1]) > pd.Timestamp(seal)
    monkeypatch.delenv("TSP_SEAL_OVERRIDE")
    with pytest.raises(ValueError, match="봉인 평가를 할 수 없습니다"):  # 봉인 평가: 정책보다 앞선 봉인 시작일은 개발 구간을 봉인 평가로 채점한다
        run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40, sealed_from=str(prices.index[500].date()))), dv, reg, sealed=True)
    rs = run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40)), dv, reg, sealed=True)
    assert rs["config"]["validation"]["sealed_from"] == seal and rs["sealed_evaluation"]
    # 정책에 없는 대상은 전과 같다
    pol.write_text(json.dumps({"trading": {"OTHER": seal}}), encoding="utf-8")
    assert run_experiment(ExperimentConfig(**base, validation=Validation(train_len=300, test_len=40)), dv, reg)["seal_policy"] is None
    # 봉인 구간을 이미 채점한 개발 결과 찾기
    from engine.experiment import save_result
    from engine.sealed import seal_exposures
    out = tmp_path / "res"
    save_result(r2, out); save_result(r, out)
    ex = seal_exposures(out, {**r["config"]})
    assert len(ex) == 1 and ex[0]["override_reason"] == "원 논문 분할 재현"


def test_excel_cells_strict_dates_values_and_stale_dimension(tmp_path):
    """뜻이 둘인 날짜 글자·소수점 쉼표·무한대는 짐작하지 않고 문제로 알린다. 시트의 크기 표시가 낡아도 끝 줄까지 읽는다(F5·F6·F19)."""
    import datetime as dt
    import zipfile
    from engine.store import parse_date_cell as D, parse_value_cell as V, read_data_sheets
    for bad in ["20-01-31", dt.time(12, 0), "12:00", "now", "today", "1Q20", "1/2/2020", "31/01/2020"]:
        assert D(bad) is None, bad
    assert D("2020-01-31 00:00:00") == pd.Timestamp("2020-01-31") and D("2020Q3") == pd.Timestamp("2020-09-01") and D(202001) == pd.Timestamp("2020-01-01")
    for bad in ["1.234,5", "1,5", "inf", "1,23", float("inf"), "3.5%"]:
        assert V(bad) == (None, True), bad
    assert V("1,234.5") == (1234.5, False) and V("-1,234,567") == (-1234567.0, False) and V("nan") == (None, False)
    import openpyxl
    x = tmp_path / "m.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "df_D"; ws.append(["date", "V1"])
    for i in range(120):
        ws.append([dt.date(2020, 1, 1) + dt.timedelta(days=i), float(i)])
    wb.save(x)
    y = tmp_path / "m2.xlsx"                                           # 크기 표시(dimension)를 A1:B50 으로 바꾼 사본
    with zipfile.ZipFile(x) as zi, zipfile.ZipFile(y, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            b = zi.read(it.filename)
            if it.filename == "xl/worksheets/sheet1.xml":
                import re as _re
                b2 = _re.sub(rb'<dimension ref="[^"]*"', b'<dimension ref="A1:B50"', b)
                assert b2 != b
                b = b2
            zo.writestr(it, b)
    long, _ = read_data_sheets(y)
    assert len(long) == 120


def test_save_result_failure_after_archive_restores_old(tmp_path, monkeypatch):
    """옛 결과를 history 로 옮긴 뒤 새 파일을 제 이름으로 바꾸다 실패하면 옛 결과를 제자리로 되돌린다(F17). json 으로 못 바꾸는 값은 옮기기 전에 드러난다."""
    import os as _os
    from engine.experiment import save_result
    prices, s = predictable_ohlcv(600, seed=3)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", name="x", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40))
    out = tmp_path / "res"
    p1 = save_result(run_experiment(cfg, dv, reg), out)
    old = p1.read_text(encoding="utf-8")
    r2 = run_experiment(cfg, dv, reg)
    real = _os.replace
    def bad_replace(a, b):
        if str(b).endswith(".json") and ".tmp" in str(a):
            raise OSError("디스크 오류(시험)")
        return real(a, b)
    monkeypatch.setattr("engine.experiment.os.replace", bad_replace)
    with pytest.raises(OSError):
        save_result(r2, out)
    monkeypatch.setattr("engine.experiment.os.replace", real)
    assert p1.read_text(encoding="utf-8") == old and p1.with_name(p1.stem + "_daily.parquet").exists()
    assert not list(out.glob(".*.tmp")) and not list((out / "history").glob("*.json"))
    r3 = run_experiment(cfg, dv, reg); r3["oops"] = {1, 2}.__iter__()       # json 으로 못 바꾸는 값
    class _X: pass
    r3["oops"] = _X()
    monkeypatch.setattr("engine.experiment.json.dumps", lambda *a, **k: (_ for _ in ()).throw(TypeError("직렬화 불가(시험)")))
    with pytest.raises(TypeError):
        save_result(r3, out)
    monkeypatch.undo()
    assert p1.read_text(encoding="utf-8") == old and not (out / "history").exists() or not list((out / "history").glob("*.json"))


def test_fill_defaults_gives_result_hash_and_real_frequency_normalisation(tmp_path):
    """실행 전에 '결과와 같은 해시'를 알 수 있다(봉인 예약 줄이 쓴다, F7). 예측 연구 모드의 기술지표 선택은 대상 변수의 실제 주기로 정규화된다(F18)."""
    from engine.experiment import fill_defaults
    from engine.features import technical_columns
    prices, s = predictable_ohlcv(700, seed=3)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40))
    res = run_experiment(cfg, dv, reg)
    assert cfg.config_hash() != res["config_hash"] and fill_defaults(cfg, reg).config_hash() == res["config_hash"]
    reg.add(Variable(id="MV", freq="M", source="DERIVED"))
    allm = technical_columns("M", with_volume=False)
    a = fill_defaults(ExperimentConfig(mode="forecast", target_variable="MV", model="ridge", frequency="D", technical_features=list(allm)), reg)
    b = fill_defaults(ExperimentConfig(mode="forecast", target_variable="MV", model="ridge", frequency="M", technical_features=list(reversed(allm))), reg)
    c = fill_defaults(ExperimentConfig(mode="forecast", target_variable="MV", model="ridge", frequency="M"), reg)
    d = fill_defaults(ExperimentConfig(mode="forecast", target_variable="MV", model="ridge", frequency="D"), reg)
    assert a.technical_features is None and b.technical_features is None
    assert b.config_hash() == c.config_hash() and a.config_hash() == d.config_hash()  # '전부 고름'은 고르지 않은 것과 같은 설정(frequency 칸을 무엇으로 적었든)


# ---------------------------------------------------------------------------
# 여러 출발점 추정 · 심은 신호(GRU·SARIMA·엔진 경유 VECM) (2026-11-10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["arima", "sarima"])
def test_multi_start_never_lowers_likelihood_and_is_deterministic(model):
    """여러 출발점 추정: 고른 해의 로그우도는 기본 출발점의 것보다 낮지 않다. 난수를 쓰지 않아 다시 해도 같은 해. starts=default 면 전과 같은 한 번 추정."""
    import importlib
    from models.base import resolve_params
    M = importlib.import_module(f"models.{model}")
    rng = np.random.default_rng(5)
    n = 90
    r = np.zeros(n); e = rng.normal(0, 0.004, n + 2)
    for t in range(2, n):
        r[t] = 0.003 + 0.4 * r[t - 1] - 0.3 * r[t - 2] + e[t] + 0.5 * e[t - 1]
    extra = {"s": 12} if model == "sarima" else {}
    m = M.Model(resolve_params(M.PARAMS, extra, "M"), n_features=1, target_type="regression"); m.fit(r, np.array([]))
    d = M.Model(resolve_params(M.PARAMS, {**extra, "starts": "default"}, "M"), n_features=1, target_type="regression"); d.fit(r, np.array([]))
    assert m.diag["n_starts"] == 7 and d.diag["n_starts"] == 1 and d.diag["start_used"] == "default"
    assert np.isclose(m.diag["llf_default"], d.res.llf) and m.res.llf >= d.res.llf - 1e-9 and m.diag["llf_gain"] >= -1e-9
    assert len(m.diag["tried"]) == 7 and max(t["llf"] for t in m.diag["tried"] if t["llf"] is not None) <= m.diag["llf"] + 1e-4  # 고른 해 = 최대(1e−4 안)
    m2 = M.Model(resolve_params(M.PARAMS, extra, "M"), n_features=1, target_type="regression"); m2.fit(r, np.array([]))
    assert m2.diag["start_used"] == m.diag["start_used"] and np.allclose(m2.forecast(r, 3), m.forecast(r, 3))
    assert M.LEGACY_DEFAULTS["starts"] == "default"


def test_series_result_records_multi_start_summary(tmp_path):
    prices = random_walk_ohlcv(500, seed=4)
    dv, reg = _dv_with(prices, None, tmp_path)
    res = run_experiment(ExperimentConfig(ticker="TEST", model="arima", use_technical=False, validation=Validation(train_len=120, test_len=60)), dv, reg)
    ms = res["model_diag"]["multi_start"]
    assert ms["folds"] == res["n_folds"] and sum(ms["start_used"].values()) == ms["folds"] and ms["max_llf_gain"] >= 0
    assert res["periods"]["first_fold"]["model_diag"]["n_starts"] == 7 and "tried" not in res["periods"]["first_fold"]["model_diag"]


def test_sarima_finds_seasonal_signal_and_fails_on_noise():
    """SARIMA: 심은 계절 신호(12기간 전 값의 0.6배)는 찾고(방향 > 58%), 백색잡음에서는 무능(|상관| < 0.25). 기본 설정(계절 차분 없음 + 상수항)."""
    import models.sarima as S
    from models.base import resolve_params
    rng = np.random.default_rng(1)
    n = 420
    r = np.zeros(n)
    for t in range(12, n):
        r[t] = 0.6 * r[t - 12] + rng.normal(0, 0.01)
    noise = rng.normal(0, 0.01, n)
    m = S.Model(resolve_params(S.PARAMS, {"s": 12}, "M"), n_features=1, target_type="regression")
    m.fit(r[:300], np.array([]))
    assert m.trend_used == "c"
    pred = np.array([m.forecast(r[: t + 1], 1)[0] for t in range(300, n - 1)])
    assert np.mean(np.sign(pred) == np.sign(r[301:n])) > 0.58
    assert np.corrcoef(pred, r[301:n])[0, 1] > 0.4
    m.fit(noise[:300], np.array([]))
    pred = np.array([m.forecast(noise[: t + 1], 1)[0] for t in range(300, n - 1)])
    assert abs(np.corrcoef(pred, noise[301:n])[0, 1]) < 0.25


def test_gru_finds_planted_signal_and_has_no_edge_on_random_walk(tmp_path):
    """GRU 도 LSTM·BIVA 와 같은 시험: 심은 신호는 찾고(방향 > 70%), 랜덤워크에서는 우연 수준."""
    prices, s = predictable_ohlcv(1000, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path / "a")
    res = run_experiment(ExperimentConfig(ticker="TEST", model="gru", model_params={"epochs": 30}, variable_ids=["S_signal"], use_technical=False,
                                          seq_len=5, validation=Validation(train_len=400, test_len=100)), dv, reg)
    assert res["prediction_metrics"]["dir_acc"] > 0.70, res["prediction_metrics"]
    assert res["model_diag"]["epoch_used"]["median"] >= 1 and res["prediction_metrics"]["cw_vs_mean_p"] < 0.01
    dv2, reg2 = _dv_with(random_walk_ohlcv(1200, seed=6), None, tmp_path / "b")
    res2 = run_experiment(ExperimentConfig(ticker="TEST", model="gru", model_params={"epochs": 20}, seq_len=10, validation=Validation(train_len=500, test_len=100)), dv2, reg2)
    assert 0.38 < res2["prediction_metrics"]["dir_acc"] < 0.62


def test_vecm_finds_error_correction_through_engine(tmp_path):
    """VECM 을 엔진 경유(예측 연구 모드, 워크포워드)로: 대상이 다른 변수와의 균형으로 되돌아가는 자료에서 '변화 없음'(0예측)보다 정확하고,
    평균 기준 Clark–West 가 유의하다. 서로 무관한 랜덤워크 둘에서는 그렇지 않다."""
    def make(seed, coint):
        rng = np.random.default_rng(seed)
        T = 260
        idx = pd.date_range("2000-01-01", periods=T, freq="MS")
        x = np.cumsum(rng.normal(0, 1, T)) + 100
        y = np.zeros(T); y[0] = 100
        for t in range(1, T):
            y[t] = y[t - 1] + (0.3 * (x[t - 1] - y[t - 1]) if coint else 0.0) + rng.normal(0, 0.3 if coint else 1.0)
        return pd.Series(y, index=idx), pd.Series(x, index=idx)

    def run(seed, coint, d):
        y, x = make(seed, coint)
        dv = DataVersion(d, "t")
        for vid, sr in (("YV", y), ("XV", x)):
            dv.write_variables(sr.rename("value").rename_axis("date").reset_index().assign(variable_id=vid), source=f"syn_{vid}")
        reg = Registry()
        reg.add(Variable(id="YV", freq="M", source="DERIVED", lag_days=0)); reg.add(Variable(id="XV", freq="M", source="DERIVED", lag_days=0))
        cfg = ExperimentConfig(mode="forecast", target_variable="YV", model="vecm", variable_ids=["XV"], use_technical=False,
                               model_params={"k_ar_diff": 1, "coint_rank": 1, "deterministic": "ci", "log_levels": False},
                               validation=Validation(train_len=120, test_len=12))
        return run_experiment(cfg, dv, reg)["prediction_metrics"]
    pm = run(3, True, tmp_path / "c")
    assert pm["rmse"] < 0.8 * pm["rmse_zero"] and pm["cw_vs_mean_p"] < 0.01 and pm["dm_vs_zero"] < 0, pm
    pm0 = run(4, False, tmp_path / "n")
    assert pm0["rmse"] > 0.95 * pm0["rmse_zero"], pm0


# ---------------------------------------------------------------------------
# 변동성 타겟 (2026-11-12) — 시험을 먼저 쓰고 구현
# ---------------------------------------------------------------------------
def test_volatility_target_is_future_only_and_matches_hand_calc():
    """변동성 타겟: 결정일 t 의 값 = 그 뒤 k 기간(t+1 … t+k) 종가 로그수익률의 제곱평균의 제곱근. t 까지의 가격을 바꿔도 변하지 않고, 마지막 k 행은 없다."""
    from engine import calendar as cal
    from tests.synthetic import volatility_ohlcv
    prices, _ = volatility_ohlcv(120, seed=1)
    k = 5
    y = cal.make_target(prices, k=k, kind="volatility")
    c = prices["close"].values
    t = 40
    hand = np.sqrt(np.mean([np.log(c[t + j] / c[t + j - 1]) ** 2 for j in range(1, k + 1)]))
    assert np.isclose(y.iloc[t], hand) and y.iloc[-k:].isna().all() and y.iloc[:-k].notna().all() and (y.dropna() >= 0).all()
    p2 = prices.copy(); p2.iloc[: t, :] *= 1.3          # t 앞의 가격을 바꿔도(t 행 포함 안 함) t 의 타겟은 같다
    assert np.isclose(cal.make_target(p2, k=k, kind="volatility").iloc[t], hand)
    nv = cal.naive_volatility(prices, k=k)                # 기준 예측: 결정일까지의 최근 k 기간 변동성(미래 값을 쓰지 않는다)
    assert np.isclose(nv.iloc[t], np.sqrt(np.mean([np.log(c[t - j] / c[t - j - 1]) ** 2 for j in range(0, k)])))
    p3 = prices.copy(); p3.iloc[t + 1:, :] *= 1.3
    assert np.isclose(cal.naive_volatility(p3, k=k).iloc[t], nv.iloc[t])


def test_volatility_experiment_finds_planted_volatility_and_sizes_position(tmp_path):
    """심은 변동성 신호: 모델이 '학습 창 평균 변동성'만 아는 예측보다 낫고(CW 유의, 표본 밖 R² > 0), 포지션은 0~1 사이의 비중(예측 변동성이 높으면 줄임).
    변동성이 늘 같은 시장에서는 평균 이상의 정보가 없다. 방향 지표(dir_acc·0예측 비교)는 변동성에는 뜻이 없어 만들지 않는다."""
    from tests.synthetic import volatility_ohlcv
    prices, s = volatility_ohlcv(1500, seed=2)
    dv, reg = _dv_with(prices, s, tmp_path / "a")
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, target=TargetSpec(kind="volatility", k=5),
                           validation=Validation(train_len=500, test_len=60))
    res = run_experiment(cfg, dv, reg)
    pm, d = res["prediction_metrics"], res["_daily"]
    assert pm["cw_vs_mean_p"] < 0.01 and pm["r2_oos_vs_mean"] > 0.2, pm
    assert {"rmse", "rmse_naive", "dm_vs_naive", "dm_vs_naive_p", "corr"} <= set(pm) and not ({"dir_acc", "dm_vs_zero", "cw_vs_zero"} & set(pm))
    ev = d.dropna(subset=["prediction", "target"])
    w = d.loc[ev.index, "position"]
    assert w.between(0, 1).all() and 0.05 < w.mean() < 0.999 and w.nunique() > 50          # 0/1 이 아니라 비중
    hi, lo = ev["prediction"] > ev["prediction"].quantile(0.8), ev["prediction"] < ev["prediction"].quantile(0.2)
    assert w[hi].mean() < w[lo].mean()                                                       # 변동성이 높다고 본 날 덜 싣는다
    assert np.allclose(w.values, np.clip(ev["bench_mean"].values / ev["prediction"].clip(lower=1e-12).values, 0, 1))
    assert res["trade_rule_used"]["kind"] == "volatility_target" and res["performance"]["n_periods"] > 500
    flat, s0 = volatility_ohlcv(1500, seed=3, planted=False)
    dv0, reg0 = _dv_with(flat, s0, tmp_path / "b")
    pm0 = run_experiment(cfg, dv0, reg0)["prediction_metrics"]
    assert pm0["cw_vs_mean_p"] > 0.01 and pm0["r2_oos_vs_mean"] < 0.05, pm0


def test_volatility_target_refuses_models_and_modes_it_does_not_fit(tmp_path):
    from tests.synthetic import volatility_ohlcv
    prices, s = volatility_ohlcv(700, seed=2)
    dv, reg = _dv_with(prices, s, tmp_path)
    for model in ("arima", "ma_cross"):
        with pytest.raises(ValueError, match="변동성"):
            run_experiment(ExperimentConfig(ticker="TEST", model=model, use_technical=False, target=TargetSpec(kind="volatility", k=5),
                                            validation=Validation(train_len=300, test_len=60)), dv, reg)
    reg.add(Variable(id="MV", freq="M", source="DERIVED"))
    with pytest.raises(ValueError, match="변동성"):
        run_experiment(ExperimentConfig(mode="forecast", target_variable="MV", model="ridge", target=TargetSpec(kind="volatility", k=1)), dv, reg)
    # 변동성 타겟이 아니면 설정 해시는 그대로(옛 설정과 이어짐)
    a = ExperimentConfig(ticker="TEST", model="ridge")
    assert "volatility" not in a.model_dump_json()


# ---------------------------------------------------------------------------
# 입력 변수의 연관성 순위·선별 — 폴드마다 학습 창 안에서만 (2026-11-12, 시험을 먼저 쓰고 구현)
# ---------------------------------------------------------------------------
def _dv_with_noise(prices, s, tmp_path, n_noise=6, seed=9):
    """심은 신호 S_signal 과, 수익률과 무관한 잡음 변수 N1…Nn 을 함께 넣은 데이터."""
    rng = np.random.default_rng(seed)
    dv = DataVersion(tmp_path, "t")
    dv.write_prices("TEST", prices)
    reg = Registry()
    cols = {s.name: s}
    for i in range(1, n_noise + 1):
        z = np.zeros(len(prices))
        for t in range(1, len(z)):
            z[t] = 0.7 * z[t - 1] + rng.normal()
        cols[f"N{i}"] = pd.Series(z, index=prices.index, name=f"N{i}")
    for name, sr in cols.items():
        reg.add(Variable(id=name, freq="D", source="DERIVED"))
        dv.write_variables(sr.rename("value").rename_axis("date").reset_index().assign(variable_id=name), source=f"syn_{name}")
    return dv, reg, list(cols)


def test_feature_relevance_ranking_is_computed_inside_training_windows(tmp_path):
    """연관성 순위: 폴드마다 '학습 창 안'의 입력과 정답만으로 |순위상관|을 재서 모은다. 심은 신호가 1위, 모든 폴드에서 상위. 선별을 켜지 않아도 순위는 남는다.
    학습 창 밖(테스트 구간)의 값을 바꿔도 그 폴드의 순위 계산은 변하지 않는다(미래를 보고 고르지 않는다)."""
    from engine.experiment import fold_relevance
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg, ids = _dv_with_noise(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=ids, use_technical=False, validation=Validation(train_len=300, test_len=40))
    res = run_experiment(cfg, dv, reg)
    fr = res["feature_relevance"]
    assert fr["selected"] is False and fr["n_folds"] == res["n_folds"] and len(fr["features"]) == len(ids)
    top = fr["features"][0]
    assert top["name"] == "S_signal" and top["mean_rank"] == 1.0 and top["top_share"] == 1.0 and top["mean_abs_corr"] > 0.5
    assert all(f["mean_abs_corr"] < 0.2 for f in fr["features"][1:])
    # 함수 자체: 학습 행만 받는다 — 순위상관의 절댓값, 상수 열은 0
    rng = np.random.default_rng(0)
    y = rng.normal(size=200); X = np.column_stack([y + rng.normal(0, 0.1, 200), -np.exp(y), rng.normal(size=200), np.ones(200)])
    r = fold_relevance(X, y)
    assert r[1] > 0.999 and r[0] > 0.9 and r[2] < 0.2 and r[3] == 0.0


def test_select_top_uses_only_selected_inputs_each_fold(tmp_path):
    """select_top=1: 폴드마다 학습 창에서 연관성이 가장 높은 변수 1개만으로 학습 → 심은 신호만 넣은 실험과 예측이 같다.
    잡음 변수만 있으면 고르는 변수가 폴드마다 바뀐다(선택 안정성이 낮다 — 그것이 '믿을 순위가 아니다'라는 신호). 기본값(None)은 설정 해시를 바꾸지 않는다."""
    import hashlib
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg, ids = _dv_with_noise(prices, s, tmp_path / "a")
    base = dict(ticker="TEST", model="ridge", use_technical=False, validation=Validation(train_len=300, test_len=40))
    r_sel = run_experiment(ExperimentConfig(**base, variable_ids=ids, select_top=1), dv, reg)
    r_one = run_experiment(ExperimentConfig(**base, variable_ids=["S_signal"]), dv, reg)
    a, b = r_sel["_daily"]["prediction"].dropna(), r_one["_daily"]["prediction"].dropna()
    assert len(a) == len(b) > 400 and np.allclose(a.values, b.values, atol=1e-10)
    fs = r_sel["feature_relevance"]
    assert fs["selected"] is True and fs["top_n"] == 1 and fs["features"][0]["name"] == "S_signal" and fs["features"][0]["selected_share"] == 1.0
    assert all(f["selected_share"] == 0.0 for f in fs["features"][1:])
    imp = dict(zip(r_sel["feature_names"], r_sel["feature_importance"]))
    assert imp["S_signal"] > 0 and all(v == 0 for k_, v in imp.items() if k_ != "S_signal")   # 고르지 않은 변수의 중요도는 0
    # 잡음만: 선택이 폴드마다 흔들린다
    r_noise = run_experiment(ExperimentConfig(**base, variable_ids=[i for i in ids if i != "S_signal"], select_top=1), dv, reg)
    shares = [f["selected_share"] for f in r_noise["feature_relevance"]["features"]]
    assert max(shares) < 0.9 and abs(sum(shares) - 1.0) < 1e-9
    # LightGBM·창 입력 모델도 돈다
    for model, extra in (("lightgbm_model", {}), ("lstm", {"model_params": {"epochs": 2, "hidden": 4}, "seq_len": 5})):
        rr = run_experiment(ExperimentConfig(**{**base, "model": model, **extra}, variable_ids=ids, select_top=2), dv, reg)
        assert rr["feature_relevance"]["top_n"] == 2 and rr["n_predictions"] > 400
    c = ExperimentConfig(**base, variable_ids=ids)
    old = hashlib.sha1(c.model_dump_json(exclude={"name", "n_seeds", "technical_features", "select_top", "pool_tickers"}).encode()).hexdigest()[:12]
    assert c.config_hash() == old and ExperimentConfig(**base, variable_ids=ids, select_top=3).config_hash() != old
    with pytest.raises(ValueError, match="입력 변수 수"):
        run_experiment(ExperimentConfig(**base, variable_ids=ids, select_top=50), dv, reg)


# ---------------------------------------------------------------------------
# 화면에서 변수 등록 · 데이터 가져오기 (2026-11-12, 시험을 먼저 쓰고 구현)
# ---------------------------------------------------------------------------
def test_api_register_variable_checks_source_then_appends_master(tmp_path, monkeypatch):
    """화면의 변수 등록: 출처·코드를 실제로 받아 본 뒤에만 변수 사전에 행을 추가한다. 받아지지 않는 코드·이미 있는 id·규칙에 안 맞는 id·지원하지 않는 출처는 거절.
    사전을 고치기 전의 사본이 남는다. 접속 키는 받지 않는다(.env 에만)."""
    import api.main as m
    from fastapi import HTTPException
    monkeypatch.setattr(m, "DATA", tmp_path)
    master = tmp_path / "variable_master.xlsx"
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "variables"
    ws.append(["id", "name", "description", "group", "domain", "role", "freq", "lag_days", "lag_confirmed", "source", "source_code", "alt_sources", "duplicate_of", "default_fill", "transform", "unit", "active", "note"])
    ws.append(["KOSPI", "코스피", "", "B", "", "", "D", 0, "Y", "YF", "^KS11", "", "", "ffill", "pct", "", "Y", ""])
    wb.save(master)
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    def probe(source, code, freq):
        if code == "BAD":
            raise RuntimeError("404 Not Found")
        return pd.Series(np.arange(30.0) + 100, index=idx)
    monkeypatch.setattr(m, "_probe_series", probe)
    ok = m.check_variable(m.VariableCheckBody(source="YF", source_code="^SOX"))
    assert ok["ok"] is True and ok["rows"] == 30 and ok["last"] == str(idx[-1].date()) and len(ok["sample"]) == 5
    bad = m.check_variable(m.VariableCheckBody(source="YF", source_code="BAD"))
    assert bad["ok"] is False and "받지 못했습니다" in bad["error"]
    r = m.add_variable(m.VariableBody(id="SOX2", name="필라델피아 반도체", source="YF", source_code="^SOX", freq="D", group="C", transform="pct"))
    assert r["added"] == "SOX2" and "데이터 가져오기" in r["note"]
    reg, _ = Registry.load_master(master)
    v = reg.get("SOX2")
    assert v.source == "YF" and v.source_code == "^SOX" and v.freq == "D" and v.transform == "pct" and v.active and v.lag_confirmed is False
    assert list((tmp_path / "master_history").glob("*.xlsx"))            # 고치기 전 사본
    for body, why in ((dict(id="SOX2", source="YF", source_code="^SOX"), "이미"), (dict(id="A_x", source="YF", source_code="^SOX"), "A_"),
                      (dict(id="new var", source="YF", source_code="^SOX"), "띄어쓰기"), (dict(id="NEW1", source="YF", source_code="BAD"), "받지 못했습니다"),
                      (dict(id="NEW2", source="ECOS", source_code="722Y001"), "통계표코드/항목코드/주기")):
        with pytest.raises(HTTPException) as e:
            m.add_variable(m.VariableBody(**body))
        assert why in str(e.value.detail), (body, e.value.detail)
    with pytest.raises(Exception):
        m.VariableBody(id="X", source="NAVER", source_code="x")          # 지원하지 않는 출처
    src = m.variable_sources()
    assert {x["source"] for x in src} == {"YF", "FRED", "ECOS"} and all("key_present" in x and "example" in x for x in src)
    assert not any("api_key" in k.lower() for k in m.VariableBody.model_fields)   # 키는 화면에서 받지 않는다


def test_api_collect_job_runs_once_and_reports(tmp_path, monkeypatch):
    """화면의 '데이터 가져오기': collect.py 를 한 번에 하나만 돌리고, 끝난 뒤 종료 코드와 로그 끝을 보여 준다. 실험이 도는 중이면 시작하지 않는다."""
    import sys as _sys, time as _time
    import api.main as m
    from fastapi import HTTPException
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "_collect_cmd", lambda: [_sys.executable, "-c", "import time; print('받는 중'); time.sleep(0.5); print('완료'); raise SystemExit(2)"])
    monkeypatch.setattr(m.jobs, "list", lambda limit=100000: [])
    monkeypatch.setattr(m, "_COLLECT", {})
    assert m.collect_status()["state"] == "none"
    st = m.collect_start()
    assert st["state"] == "running"
    with pytest.raises(HTTPException):
        m.collect_start()                                                 # 이미 도는 중
    for _ in range(100):
        if m.collect_status()["state"] != "running":
            break
        _time.sleep(0.1)
    fin = m.collect_status()
    assert fin["state"] == "done" and fin["returncode"] == 2 and "받지 못한 항목" in fin["message"] and any("완료" in ln for ln in fin["log_tail"])
    monkeypatch.setattr(m.jobs, "list", lambda limit=100000: [{"status": "running"}])
    with pytest.raises(HTTPException):
        m.collect_start()                                                 # 실험이 도는 중


def test_secrets_never_reach_screen_logs_or_manifest(tmp_path, monkeypatch):
    """접속 키는 오류 문구에 섞여도 화면·로그·manifest 에 나가지 않는다. 통신 오류는 주소(키가 든)를 그대로 담고 있어, 걸러 내지 않으면 키가 새어 나간다."""
    import api.main as m
    from adapters.base import scrub_secrets
    monkeypatch.setenv("FRED_API_KEY", "abcd1234secretkey")
    monkeypatch.setenv("ECOS_API_KEY", "ECOSKEY99887766")
    t = scrub_secrets("ProxyError: /fred/series?series_id=DGS10&api_key=abcd1234secretkey&file_type=json · /api/StatisticSearch/ECOSKEY99887766/json · serviceKey=zzz111yyy222")
    assert "abcd1234secretkey" not in t and "ECOSKEY99887766" not in t and "zzz111yyy222" not in t and "DGS10" in t and "***" in t
    def boom(source, code, freq):
        raise RuntimeError("Max retries exceeded with url: /fred/series/observations?series_id=X&api_key=abcd1234secretkey")
    monkeypatch.setattr(m, "_probe_series", boom)
    r = m.check_variable(m.VariableCheckBody(source="FRED", source_code="X"))
    assert r["ok"] is False and "abcd1234secretkey" not in json.dumps(r, ensure_ascii=False)
    # 수집: 실패 문구가 manifest 와 화면 출력에 키 없이 남는다
    import collect
    from adapters import fred_adapter
    data = tmp_path / "data"; data.mkdir()
    _mini_master(data / "variable_master.xlsx", [{"id": "US10Y", "freq": "D", "source": "FRED", "source_code": "DGS10", "lag_days": 1}], [])
    monkeypatch.setattr(fred_adapter, "fetch_series", lambda *a, **k: boom(0, 0, 0))
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "fred", "--data-dir", str(data)])
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        collect.main()
    out = buf.getvalue()
    assert "abcd1234secretkey" not in out and "[fail] fred US10Y" in out and "api_key=***" in out
    for f in data.rglob("*"):                                    # 만들어진 어떤 파일에도 키가 없다
        if f.is_file() and f.suffix in (".json", ".jsonl", ".csv", ".txt", ".log"):
            assert "abcd1234secretkey" not in f.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 여러 종목 묶어 학습(표본 확대) — 학습에만 보태고 평가는 대상 종목만 (2026-11-12, 시험을 먼저 쓰고 구현)
# ---------------------------------------------------------------------------
def _shared_signal_pair(n=900, seed=3, sigma_a=0.03, sigma_b=0.005):
    """같은 신호 s 로 움직이는 두 종목: A 는 잡음이 크고(배우기 어렵다) B 는 잡음이 작다."""
    pa, s = predictable_ohlcv(n, seed=seed, sigma=sigma_a)
    rng = np.random.default_rng(seed + 100)
    r_open = 0.02 * np.r_[0, 0, s.values[:-2]] + rng.normal(0, sigma_b, n)
    open_ = 20000 * np.exp(np.cumsum(r_open))
    close = open_ * (1 + rng.normal(0, sigma_b / 2, n))
    pb = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
                       "close": close, "volume": rng.lognormal(12, 0.3, n)}, index=pa.index)
    return pa, pb, s


def _pair_dv(tmp_path, pa, pb, s):
    dv = DataVersion(tmp_path, "t")
    dv.write_prices("AAA", pa); dv.write_prices("BBB", pb)
    reg = Registry()
    reg.add(Variable(id="S_signal", freq="D", source="DERIVED"))
    dv.write_variables(s.rename("value").rename_axis("date").reset_index().assign(variable_id="S_signal"), source="syn")
    return dv, reg


def test_pool_tickers_adds_training_rows_but_evaluates_target_only(tmp_path):
    """pool_tickers: 다른 종목의 (입력, 정답) 행을 '학습에만' 보탠다. 예측·평가는 대상 종목의 날짜에서만 → 검정 표본은 그대로(종목끼리 같은 날 함께 움직이는
    문제를 검정에 끌어들이지 않는다). 기본값(빈 목록)은 해시·숫자를 바꾸지 않는다. 같은 신호를 따르는 덜 시끄러운 종목을 보태면 오차가 준다."""
    pa, pb, s = _shared_signal_pair()
    dv, reg = _pair_dv(tmp_path, pa, pb, s)
    base = dict(ticker="AAA", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=40, test_len=20))
    c0, c1 = ExperimentConfig(**base), ExperimentConfig(**base, pool_tickers=["BBB"])
    assert c0.config_hash() == ExperimentConfig(**base, pool_tickers=[]).config_hash() != c1.config_hash()
    r0, r1 = run_experiment(c0, dv, reg), run_experiment(c1, dv, reg)
    assert r0.get("pool") is None
    p0, p1 = r0["_daily"]["prediction"].dropna(), r1["_daily"]["prediction"].dropna()
    assert p0.index.equals(p1.index) and r0["n_predictions"] == r1["n_predictions"]          # 평가 표본은 그대로
    pool = r1["pool"]
    assert pool["tickers"] == ["BBB"] and pool["n_folds"] == r1["n_folds"] and 20 <= pool["rows_min"] <= pool["rows_mean"] <= 40
    assert r1["prediction_metrics"]["rmse"] < r0["prediction_metrics"]["rmse"]                                    # 보탠 행이 실제로 학습에 쓰였다


def test_pool_rows_never_use_prices_from_validation_or_test_period(tmp_path):
    """보태는 행의 정답은 그 종목의 미래 가격(결정일 뒤 1+k 행)으로 만든다 → 그 마지막 가격의 날짜가 경계(검증·테스트 시작) 앞인 행만 쓴다.
    거래일이 다른 종목(해외)도 날짜로 자르므로 안전하다."""
    from engine.experiment import build_dataset, pool_training_rows
    pa, pb, s = _shared_signal_pair(400)
    pb = pb.drop(pb.index[[150, 151, 152, 153]])        # B 만 쉬는 날(달력이 다르다)
    dv, reg = _pair_dv(tmp_path, pa, pb, s)
    cfg = ExperimentConfig(ticker="BBB", model="ridge", variable_ids=["S_signal"], use_technical=False, target=TargetSpec(k=3))
    pds = build_dataset(cfg, dv, reg)
    lo, cut, h = pa.index[100], pa.index[156], 4      # h = 1 + k
    rows = pool_training_rows(pds, lo, cut, h)
    cal_b = pds.prices.index
    assert len(rows) > 20 and rows.index.min() >= lo and list(rows.columns) == pds.feature_names + ["y"] and rows.notna().all().all()
    last_used = [cal_b[cal_b.get_loc(t) + h] for t in rows.index]
    assert max(last_used) < cut
    nxt = cal_b[cal_b.get_loc(rows.index.max()) + 1]     # 바로 다음 행은 경계를 넘는다(그래서 빠졌다)
    assert cal_b[cal_b.get_loc(nxt) + h] >= cut
    # 값을 경계 뒤에서 바꿔도 고른 행은 같다(미래를 보지 않는다)
    pb2 = pb.copy(); pb2.loc[pb2.index >= cut, ["open", "high", "low", "close"]] *= 3.0
    dv2, reg2 = _pair_dv(tmp_path / "b", pa, pb2, s)
    rows2 = pool_training_rows(build_dataset(cfg, dv2, reg2), lo, cut, h)
    assert rows.equals(rows2)


def test_pool_tickers_with_validation_models_and_rejections(tmp_path):
    """검증을 쓰는 모델(LightGBM): 검증은 대상 종목의 뒤쪽 행 그대로, 보탠 행은 검증 시작 앞에서 끝난다. 지원하지 않는 조합은 이유를 말하고 거절."""
    pa, pb, s = _shared_signal_pair(700)
    dv, reg = _pair_dv(tmp_path, pa, pb, s)
    base = dict(ticker="AAA", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=200, test_len=50))
    r = run_experiment(ExperimentConfig(**base, model="lightgbm_model", pool_tickers=["BBB"]), dv, reg)
    assert r["pool"]["rows_min"] > 50 and r["n_predictions"] > 300 and r["prediction_metrics"]["rmse"] < r["prediction_metrics"]["rmse_zero"]
    with pytest.raises(ValueError, match="묶"):
        run_experiment(ExperimentConfig(**base, model="lstm", seq_len=5, pool_tickers=["BBB"]), dv, reg)          # 창 입력 모델은 아직
    with pytest.raises(ValueError, match="묶"):
        run_experiment(ExperimentConfig(**base, model="ridge", pool_tickers=["AAA"]), dv, reg)                     # 대상 자신
    with pytest.raises(ValueError, match="묶"):
        run_experiment(ExperimentConfig(**base, model="ridge", pool_tickers=["ZZZ"]), dv, reg)                     # 가격 자료 없음
    with pytest.raises(ValueError, match="묶"):
        run_experiment(ExperimentConfig(**base, model="arima", pool_tickers=["BBB"]), dv, reg)


# ---------------------------------------------------------------------------
# 종목 목록(묶어 학습에 쓸 종목을 화면에서 추가) — 2026-11-12, 시험을 먼저 쓰고 구현
# ---------------------------------------------------------------------------
def test_ticker_list_file_default_add_and_rejections(tmp_path):
    """종목 목록은 data/tickers.json. 없으면 기본 둘(KODEX·TIGER 반도체). 추가는 6자리 숫자 코드만, 중복·잘못된 종류는 거절. 고치기 전 사본을 남긴다."""
    from engine import tickers as tk
    assert [t["id"] for t in tk.load(tmp_path)] == ["091160", "091230"] and all(t["kind"] == "etf" for t in tk.load(tmp_path))
    tk.add(tmp_path, "005930", "삼성전자", "stock")
    got = tk.load(tmp_path)
    assert [t["id"] for t in got] == ["091160", "091230", "005930"] and got[-1] == {"id": "005930", "name": "삼성전자", "kind": "stock"}
    tk.add(tmp_path, "000660", "SK하이닉스", "stock")
    assert len(list((tmp_path / "tickers_history").glob("*.json"))) >= 1            # 두 번째 고치기 전 사본
    for args, why in ((("005930", "x", "stock"), "이미"), (("5930", "x", "stock"), "6자리"), (("00593A", "x", "stock"), "6자리"),
                      (("123456", "x", "bond"), "종류")):
        with pytest.raises(ValueError, match=why):
            tk.add(tmp_path, *args)


def test_collect_uses_ticker_list_and_official_etf_source_only_for_etfs(tmp_path, monkeypatch):
    """수집기는 종목 목록 파일의 종목을 모두 받는다. 공식 출처(data.go.kr)는 ETF 시세 API 라 주식에는 묻지 않는다
    (물었다가 실패로 기록되면 그 버전 전체가 '일부 받지 못함'이 된다). 주식은 야후 가격으로 조립한다."""
    import collect
    from adapters import datago_adapter, yf_adapter
    from engine import tickers as tk
    from engine.store import DataVersion
    data = tmp_path / "data"; data.mkdir()
    _mini_master(data / "variable_master.xlsx", [{"id": "M1", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": 30}],
                 [("df_M", ["date", "M1"], [["2020-01", 1.0]])])
    tk.add(data, "005930", "삼성전자", "stock")
    prices, _ = predictable_ohlcv(60, seed=2)
    asked_dg, asked_yf = [], []
    monkeypatch.setenv("DATAGO_API_KEY", "dummy")
    monkeypatch.setattr(datago_adapter, "fetch_prices", lambda t, *a, **k: asked_dg.append(t) or prices)
    monkeypatch.setattr(yf_adapter, "fetch_prices", lambda t, *a, **k: asked_yf.append(t) or prices)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "datago,yf,xlsx", "--data-dir", str(data)])
    assert collect.main() == 0
    assert sorted(asked_dg) == ["091160", "091230"] and sorted(asked_yf) == ["005930", "091160", "091230"]
    new = sorted(p.name for p in (data / "versions").iterdir())[-1]
    dv = DataVersion(data, new)
    assert len(dv.read_prices("005930")) == 60 and "005930" in dv.manifest()["tickers"] and not dv.manifest().get("incomplete")


def test_api_tickers_check_add_and_meta(tmp_path, monkeypatch):
    """화면의 종목 추가: 야후에서 가격을 실제로 받아 본 뒤에만 목록에 넣는다. 화면 정보(meta)의 종목·데이터 버전별 보유 종목에 반영."""
    import api.main as m
    from fastapi import HTTPException
    monkeypatch.setattr(m, "DATA", tmp_path)
    prices, _ = predictable_ohlcv(40, seed=4)
    def probe(code):
        if code == "999999":
            raise RuntimeError("YF 응답 없음: 999999.KS")
        return prices
    monkeypatch.setattr(m, "_probe_prices", probe)
    ok = m.check_ticker(m.TickerCheckBody(id="005930"))
    assert ok["ok"] is True and ok["rows"] == 40 and len(ok["sample"]) == 5
    bad = m.check_ticker(m.TickerCheckBody(id="999999"))
    assert bad["ok"] is False and "받지 못했습니다" in bad["error"]
    r = m.add_ticker(m.TickerBody(id="005930", name="삼성전자", kind="stock"))
    assert r["added"] == "005930" and "데이터 가져오기" in r["note"]
    assert [t["id"] for t in m.list_tickers()] == ["091160", "091230", "005930"]
    for body, why in ((dict(id="005930", name="x", kind="stock"), "이미"), (dict(id="999999", name="x", kind="stock"), "받지 못했습니다"),
                      (dict(id="12", name="x", kind="stock"), "6자리")):
        with pytest.raises(HTTPException) as e:
            m.add_ticker(m.TickerBody(**body))
        assert why in str(e.value.detail), (body, e.value.detail)
    monkeypatch.setattr(m, "_registry", lambda: (Registry(), []))
    meta = m.meta()
    assert [t["id"] for t in meta["tickers"]] == ["091160", "091230", "005930"] and set(meta["price_api"]) == {"091160", "091230", "005930"}


# ---------------------------------------------------------------------------
# 외부 공개 주소(Tailscale Funnel)에서 '바꾸는 동작'에 암호 — 2026-11-12, 시험을 먼저 쓰고 구현
# ---------------------------------------------------------------------------
def _asgi_call(app, method, path, headers=None):
    """httpx 없이 ASGI 앱을 한 번 부른다 → (상태, 본문 dict)."""
    import asyncio
    sent = []
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "", "headers": hdrs, "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 8000)}
    msgs = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return msgs.pop(0) if msgs else {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    try:
        return status, json.loads(body or b"{}")
    except ValueError:
        return status, {}


def test_write_password_guards_changes_but_not_viewing(monkeypatch):
    """암호(.env 의 TSP_WRITE_PASSWORD)가 있으면 /api 의 바꾸는 요청(POST·DELETE 등)은 머리글 X-TSP-Password 가 맞을 때만 통과.
    보기(GET)는 그대로 열려 있다. 틀린 암호·빈 암호는 401(화면이 암호를 묻게 need_password 표시). 암호는 응답에 나오지 않는다."""
    import api.main as m
    monkeypatch.setenv("TSP_WRITE_PASSWORD", "long-test-pass-123")
    monkeypatch.setattr(m, "_WRONG_DELAY", 0.0)
    st, b = _asgi_call(m.app, "POST", "/api/auth/check")
    assert st == 401 and b["need_password"] is True and "암호" in b["detail"]
    st, b = _asgi_call(m.app, "POST", "/api/auth/check", {"X-TSP-Password": "wrong"})
    assert st == 401 and "틀" in b["detail"] and "long-test-pass-123" not in json.dumps(b, ensure_ascii=False)
    st, b = _asgi_call(m.app, "POST", "/api/auth/check", {"X-TSP-Password": "long-test-pass-123"})
    assert st == 200 and b["ok"] is True
    st, _ = _asgi_call(m.app, "DELETE", "/api/results/nothing")
    assert st == 401
    st, b = _asgi_call(m.app, "GET", "/api/auth")
    assert st == 200 and b["password_set"] is True and "long-test-pass-123" not in json.dumps(b)
    st, _ = _asgi_call(m.app, "GET", "/api/sealed/log")
    assert st == 200                                                    # 보기는 암호 없이


def test_without_password_external_changes_are_blocked(monkeypatch):
    """암호를 정하지 않았으면: 노트북에서 직접(127.0.0.1) 바꾸는 것은 지금처럼 되고, Funnel 로 들어온 요청(머리글 Tailscale-Funnel-Request)의
    바꾸는 동작은 막는다 — 암호를 정하라는 안내와 함께(403). 보기(GET)는 열려 있다."""
    import api.main as m
    monkeypatch.delenv("TSP_WRITE_PASSWORD", raising=False)
    st, b = _asgi_call(m.app, "POST", "/api/auth/check")
    assert st == 200 and b["ok"] is True
    st, b = _asgi_call(m.app, "POST", "/api/auth/check", {"Tailscale-Funnel-Request": "?1"})
    assert st == 403 and "TSP_WRITE_PASSWORD" in b["detail"]
    st, b = _asgi_call(m.app, "GET", "/api/auth", {"Tailscale-Funnel-Request": "?1"})
    assert st == 200 and b["password_set"] is False and b["external"] is True


def test_collect_naver_sliding_window_is_merged_not_failure(tmp_path, monkeypatch):
    """네이버 수정주가(교차검증용, pykrx)는 '최근 N일'만 주는 이동 창이라 받을 때마다 시작일이 뒤로 밀린다(2026-11-12 실제: 2014-08-08 → 2014-08-20).
    짧아졌다고 버전 전체를 '일부 받지 못함'으로 만들면 매번 실패한다 → 교차검증 출처만: 겹치는 날 값이 같으면 직전 앞부분 + 새 자료로 잇고,
    값이 같은 배수로 바뀌었으면 앞부분도 그 배수로 맞춰 잇고(→ test_review13_naver_…), 설명할 수 없게 바뀌었으면 직전 파일을 그대로 둔다. 어느 쪽이든 실패가 아니라 기록으로 남긴다. 주 출처(datago·yf)의 짧아짐은 여전히 실패."""
    import collect
    from adapters import pykrx_adapter, yf_adapter
    from engine.store import DataVersion
    data = tmp_path / "data"; old = data / "versions" / "20200101T000000Z"; old.mkdir(parents=True)
    _mini_master(data / "variable_master.xlsx", [{"id": "M1", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": 30}],
                 [("df_M", ["date", "M1"], [["2020-01", 1.0]])])
    full, _ = predictable_ohlcv(80, seed=5)
    for t in collect.TICKERS:
        full.iloc[:70].to_parquet(old / f"prices_{t}_naver.parquet"); full.iloc[:70].to_parquet(old / f"prices_{t}_yf.parquet"); full.iloc[:70].to_parquet(old / f"prices_{t}.parquet")
    (old / "manifest.json").write_text(json.dumps({"start": "2006-01-01"}))
    changed = {"on": False}

    def naver(t, *a, **k):
        d = full.iloc[10:80].copy()                       # 앞 10일이 빠지고 뒤 10일이 새로 생김(이동 창)
        if changed["on"]:
            d["close"] *= 1.5                             # 수정주가가 소급해 바뀐 경우
        return d
    monkeypatch.setattr(pykrx_adapter, "fetch_prices", naver)
    monkeypatch.setattr(yf_adapter, "fetch_prices", lambda t, *a, **k: full)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "yf,pykrx,xlsx", "--data-dir", str(data)])
    assert collect.main() == 0
    new = sorted(p.name for p in (data / "versions").iterdir())[-1]
    dv = DataVersion(data, new)
    nv = dv.read_prices(f"{collect.TICKERS[0]}_naver")
    assert nv.index.min() == full.index[0] and nv.index.max() == full.index[79] and len(nv) == 80     # 직전 앞부분 + 새 자료
    man = dv.manifest()
    assert not man.get("incomplete") and any("이었다" in str(v) for v in (man.get("notes") or {}).values())
    # 겹치는 날 값이 다르면 직전 파일 그대로, 그래도 실패는 아님
    changed["on"] = True
    import time as _time; _time.sleep(1.1)              # 버전 id 는 초 단위
    assert collect.main() == 0
    new2 = sorted(p.name for p in (data / "versions").iterdir())[-1]
    dv2 = DataVersion(data, new2)
    assert len(dv2.read_prices(f"{collect.TICKERS[0]}_naver")) == 80 and not dv2.manifest().get("incomplete")
    assert any("설명할 수 없" in str(v) for v in (dv2.manifest().get("notes") or {}).values())   # 종가만 1.5배 = 열마다 다른 비율(2026-11-13 규칙)
    # 주 출처(yf)가 짧아지면 여전히 실패
    monkeypatch.setattr(yf_adapter, "fetch_prices", lambda t, *a, **k: full.iloc[5:])
    _time.sleep(1.1)
    assert collect.main() == 2


def test_viewing_explanations_need_no_password(monkeypatch):
    """화면을 열 때 저절로 부르는 설명 계산(/api/flow, /api/explain)은 아무것도 저장하지 않는 '보기'라 암호를 묻지 않는다(2026-11-13 —
    실험 설정 화면을 열자마자 암호 창이 떠서 거부감). 실험 실행·데이터 가져오기 같은 바꾸는 동작은 여전히 암호."""
    import api.main as m
    monkeypatch.setenv("TSP_WRITE_PASSWORD", "long-test-pass-123")
    monkeypatch.setattr(m, "_WRONG_DELAY", 0.0)
    for path in ("/api/flow", "/api/explain"):
        st, b = _asgi_call(m.app, "POST", path)
        assert st != 401, (path, st, b)                     # 암호가 아니라 본문 검사(422) 등으로 간다
    for path in ("/api/experiments", "/api/collect", "/api/variables", "/api/tickers", "/api/sealed/run", "/api/ensemble"):
        st, b = _asgi_call(m.app, "POST", path)
        assert st == 401 and b.get("need_password"), (path, st)



# ---------------------------------------------------------------------------
# 2026-11-13 — 종목 추가 한 번에 · 변수 값 보기(+CSV) · 화면에서 발표시차·변환 고치기 (시험을 먼저 쓰고 구현)
# ---------------------------------------------------------------------------
def test_api_add_ticker_checks_and_reports_in_one_step(tmp_path, monkeypatch):
    """[종목 추가] 한 번으로 '받아 보기'까지 한다 — 응답에 받아 본 요약(줄 수·기간·최근 값)이 들어 있다."""
    import api.main as m
    monkeypatch.setattr(m, "DATA", tmp_path)
    prices, _ = predictable_ohlcv(40, seed=4)
    monkeypatch.setattr(m, "_probe_prices", lambda code: prices)
    r = m.add_ticker(m.TickerBody(id="005930", name="삼성전자", kind="stock"))
    assert r["added"] == "005930" and r["check"]["rows"] == 40 and len(r["check"]["sample"]) == 5 and r["check"]["first"] == str(prices.index.min().date())


def _values_fixture(tmp_path, monkeypatch):
    import api.main as m
    from engine.registry import Variable
    data = tmp_path / "data"
    for vid_, (n, bump) in {"20260101T000000Z": (30, False), "20260201T000000Z": (60, True)}.items():
        d = data / "versions" / vid_; d.mkdir(parents=True)
        dates = list(pd.bdate_range("2025-01-01", periods=n))
        vals = [100.0 + i for i in range(n)]
        if bump:
            vals[40] = 10_000.0                          # 튀는 값
            dates = dates[:20] + dates[30:]              # 10거래일(2주) 빈 기간
            vals = vals[:20] + vals[30:]
            dates.append(dates[-1]); vals.append(vals[-1])   # 같은 날짜 두 번
        pd.DataFrame({"variable_id": "X", "date": dates, "value": vals}).to_parquet(d / "variables_fred.parquet", index=False)
        (d / "manifest.json").write_text(json.dumps({"start": "2006-01-01"}))
    reg = Registry([Variable(id="X", name="시험 변수", freq="D", lag_days=1, source="FRED", source_code="XX", transform="pct"),
                    Variable(id="Y", name="자료 없음", freq="M", source="FRED", source_code="YY")])
    monkeypatch.setattr(m, "DATA", data)
    monkeypatch.setattr(m, "_registry", lambda: (reg, []))
    return m


def test_api_variable_values_and_checks(tmp_path, monkeypatch):
    """[값 보기]: 기본은 화면이 쓰는 데이터 버전(가장 새 온전한 버전). 값 전체·처음/끝·개수, 점검(빈 기간·같은 날짜 중복·
    하루 튀었다 되돌아온 날(자료 오류의 흔한 모양)·변화율인데 0 이하). 실데이터의 큰 움직임은 '가장 큰 움직임'으로 참고만."""
    m = _values_fixture(tmp_path, monkeypatch)
    r = m.variable_values("X")
    assert r["version"] == "20260201T000000Z" and r["transform"] == "pct" and r["n"] == 50          # 같은 날짜 두 번은 하나로
    assert r["first"] == "2025-01-01" and len(r["values"]) == 50 and {"date", "value", "transformed"} <= set(r["values"][0])
    c = r["checks"]
    assert c["duplicate_dates"] == 1
    assert any(g["days"] >= 14 for g in c["gaps"])                                   # 2주 빈 기간
    assert [o["value"] for o in c["reversals"]] == [10_000.0] and len(c["largest_moves"]) == 5
    assert c["nonpositive"] == 0
    old = m.variable_values("X", version="20260101T000000Z")
    assert old["version"] == "20260101T000000Z" and old["n"] == 30 and old["checks"]["gaps"] == [] and old["checks"]["reversals"] == []
    none = m.variable_values("Y")
    assert none["n"] == 0 and "없습니다" in none["message"]
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        m.variable_values("NOPE")
    assert e.value.status_code == 404


def test_api_variable_values_csv_opens_in_excel(tmp_path, monkeypatch):
    """CSV 내려받기: 엑셀에서 한글이 깨지지 않게 UTF-8 BOM, 머리글 date,value,변환값(…), 버전 이름이 파일 이름에."""
    m = _values_fixture(tmp_path, monkeypatch)
    resp = m.variable_values_csv("X", version="20260101T000000Z")
    body = resp.body
    assert body.startswith("﻿".encode("utf-8")) and resp.media_type.startswith("text/csv")
    lines = body.decode("utf-8-sig").splitlines()
    assert lines[0] == "date,value,변환값(pct)" and len(lines) == 31
    assert "X_20260101T000000Z" in resp.headers["content-disposition"]


def test_api_update_variable_lag_and_transform(tmp_path, monkeypatch):
    """화면에서 발표시차·확인 여부·변환을 고친다. 고치기 전 사전 사본을 남기고, 근거를 비고에 날짜와 함께 남긴다.
    잘못된 값(변환 이름·음수 시차)과 없는 변수는 거절. 바뀐 것이 없으면 거절."""
    import api.main as m
    from fastapi import HTTPException
    data = tmp_path / "data"; data.mkdir()
    _mini_master(data / "variable_master.xlsx", [{"id": "M1", "name": "시험", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": ""}],
                 [("df_M", ["date", "M1"], [["2020-01", 1.0]])])
    monkeypatch.setattr(m, "DATA", data)
    before, _ = Registry.load_master(data / "variable_master.xlsx")
    assert before.get("M1").lag_confirmed is False
    r = m.update_variable("M1", m.VariableEditBody(lag_days=20, lag_confirmed=True, transform="diff", basis="한국은행 공표일정(매월 20일 전후)"))
    assert set(r["changed"]) == {"lag_days", "lag_confirmed", "transform"}
    reg, _ = Registry.load_master(data / "variable_master.xlsx")
    v = reg.get("M1")
    assert v.lag_days == 20 and v.lag_confirmed is True and v.transform == "diff" and "한국은행 공표일정" in v.note
    assert len(list((data / "master_history").glob("*.xlsx"))) >= 1
    for body, code in ((m.VariableEditBody(transform="abc"), 400), (m.VariableEditBody(lag_days=20, lag_confirmed=True, transform="diff"), 400)):
        with pytest.raises(HTTPException) as e:
            m.update_variable("M1", body)
        assert e.value.status_code == code
    with pytest.raises(HTTPException) as e:
        m.update_variable("NOPE", m.VariableEditBody(lag_days=3))
    assert e.value.status_code == 404
    with pytest.raises(Exception):
        m.VariableEditBody(lag_days=-1)
    # pct5 같은 변환도 받는다
    assert m.update_variable("M1", m.VariableEditBody(transform="pct5"))["changed"] == ["transform"]


# ---------------------------------------------------------------------------
# 2026-11-13 — 종목 종류(ETF/주식) 자동 판별 · 종목 가격 [값 보기]
# ---------------------------------------------------------------------------
def test_api_add_ticker_detects_etf_or_stock(tmp_path, monkeypatch):
    """종류를 사람이 고르지 않는다: 공식 ETF 시세(data.go.kr)에 그 코드가 있으면 ETF, 없으면 주식.
    키가 없거나 연결이 안 되면 판별 못 함 → 주식으로 두고(공식 출처를 묻지 않음 = 수집 실패 없음) 그 사실을 알린다."""
    import api.main as m
    from adapters import datago_adapter
    monkeypatch.setattr(m, "DATA", tmp_path)
    prices, _ = predictable_ohlcv(40, seed=4)
    monkeypatch.setattr(m, "_probe_prices", lambda code: prices)
    monkeypatch.setenv("DATAGO_API_KEY", "dummy")

    def dg(t, *a, **k):
        if t == "091170":
            return prices.tail(10)
        raise RuntimeError(f"DATAGO 응답 없음: {t}")
    monkeypatch.setattr(datago_adapter, "fetch_prices", dg)
    r = m.add_ticker(m.TickerBody(id="091170", name="KODEX 은행"))
    assert r["ticker"]["kind"] == "etf" and "ETF" in r["kind_basis"]
    r = m.add_ticker(m.TickerBody(id="005930", name="삼성전자"))
    assert r["ticker"]["kind"] == "stock" and "주식" in r["kind_basis"]
    monkeypatch.setattr(datago_adapter, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("timeout serviceKey=dummy")))
    r = m.add_ticker(m.TickerBody(id="000660", name="SK하이닉스"))
    assert r["ticker"]["kind"] == "stock" and "판별하지 못" in r["kind_basis"] and "dummy" not in r["kind_basis"]


def test_api_ticker_values_and_checks(tmp_path, monkeypatch):
    """종목 가격 [값 보기]: 실험이 읽는 조립된 가격(prices_<코드>) 그대로. 점검 = 빈 기간·중복·하루 튀었다 되돌아온 날(종가)·
    0 이하·고가/저가 앞뒤 안 맞음(고가 < 시가·종가, 저가 > 시가·종가)·네이버로 메운 날 수(repair_log). CSV 는 시가~거래량."""
    import api.main as m
    data = tmp_path / "data"; d = data / "versions" / "20260201T000000Z"; d.mkdir(parents=True)
    p, _ = predictable_ohlcv(60, seed=7)
    p = p.drop(p.index[20:30])                                 # 2주 빈 기간
    p.iloc[30, p.columns.get_loc("close")] = p["close"].iloc[30] * 20; p.iloc[30, p.columns.get_loc("high")] = p["close"].iloc[30]   # 튀었다 되돌아옴
    p.iloc[40, p.columns.get_loc("high")] = p["low"].iloc[40] * 0.5          # 고가가 저가보다 낮음
    p.to_parquet(d / "prices_091160.parquet")
    pd.DataFrame({"date": ["2025-01-10", "2025-01-11"], "action": "fill"}).to_csv(d / "repair_log_091160.csv", index=False, encoding="utf-8-sig")
    (d / "manifest.json").write_text(json.dumps({"start": "2006-01-01", "tickers": ["091160"]}))
    monkeypatch.setattr(m, "DATA", data)
    r = m.ticker_values("091160")
    assert r["version"] == "20260201T000000Z" and r["n"] == 50 and set(r["values"][0]) >= {"date", "open", "high", "low", "close", "volume"}
    c = r["checks"]
    assert any(g["days"] >= 14 for g in c["gaps"]) and len(c["reversals"]) == 1 and c["ohlc_bad"] >= 1 and c["repaired"] == 2 and c["nonpositive"] == 0
    csv = m.ticker_values_csv("091160").body.decode("utf-8-sig").splitlines()
    assert csv[0] == "date,open,high,low,close,volume" and len(csv) == 51
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        m.ticker_values("999999")
    assert e.value.status_code == 404


def test_sealed_history_counts_reopenings_and_skips_failed():
    """봉인 기록에서 '같은 후보(설정 지문)'·'같은 봉인 시작일'로 연 횟수를 센다. 수치를 보기 전에 실패한 시도는 세지 않고,
    실행 중이거나 강제 종료된 시도(started)는 연 것으로 센다(2026-11-13: 다시 열기는 막지 않고 경고·기록 — 사용자 결정)."""
    from engine.sealed import history
    log = [{"hash": "aaa", "sealed_from": "2025-01-01", "status": "done"},
           {"hash": "aaa", "sealed_from": "2025-01-01", "status": "failed"},
           {"hash": "bbb", "sealed_from": "2025-01-01", "status": "started"},
           {"hash": "ccc", "sealed_from": "2024-01-01"}]                        # 옛 기록: status 없음 = done
    h = history(log, "aaa", "2025-01-01")
    assert len(h["same_config"]) == 1 and len(h["same_seal_date"]) == 2 and h["total"] == 3 and h["failed"] == 1
    assert len(history(log, "zzz", "2026-01-01")["same_config"]) == 0


# ---------------------------------------------------------------------------
# 독립 검토 반영 (2026-11-13) — 시험을 먼저 쓰고 고친다
# ---------------------------------------------------------------------------
def test_review1_add_variable_freq_comes_from_data_not_from_form(tmp_path, monkeypatch):
    """검토 1(높음): 화면 기본값 '일(D)' 그대로 월 자료를 등록하면 발표시차가 1일로 잡혀 미래 정보가 샌다.
    주기는 받아 본 자료의 날짜 간격(한국은행은 코드 끝의 주기)으로 정하고, 사람이 다른 주기를 넣으면 거절한다."""
    import api.main as m
    from fastapi import HTTPException
    monkeypatch.setattr(m, "DATA", tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "x"); monkeypatch.setenv("ECOS_API_KEY", "x")
    master = tmp_path / "variable_master.xlsx"
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "variables"
    ws.append(["id", "name", "description", "group", "domain", "role", "freq", "lag_days", "lag_confirmed", "source", "source_code", "alt_sources", "duplicate_of", "default_fill", "transform", "unit", "active", "note"])
    ws.append(["KOSPI", "코스피", "", "B", "", "", "D", 0, "Y", "YF", "^KS11", "", "", "ffill", "pct", "", "Y", ""])
    wb.save(master)
    series = {"MON": pd.date_range("2022-01-01", periods=36, freq="MS"), "WK": pd.date_range("2023-01-06", periods=100, freq="W-FRI"),
              "DAY": pd.date_range("2025-01-01", periods=300, freq="B"), "QTR": pd.date_range("2020-01-01", periods=12, freq="QS"),
              "YR": pd.date_range("2020-01-01", periods=5, freq="YS"), "TWO": pd.date_range("2025-01-01", periods=2, freq="MS")}
    monkeypatch.setattr(m, "_probe_series", lambda s, code, f: pd.Series(np.arange(len(series[code.split("/")[0]])) + 100.0, index=series[code.split("/")[0]]))
    for code, want in (("MON", "M"), ("WK", "W"), ("DAY", "D"), ("QTR", "Q"), ("YR", "Y")):
        c = m.check_variable(m.VariableCheckBody(source="FRED", source_code=code))
        assert c["ok"] and c["freq"] == want and c["freq_basis"], (code, c)
    # 주기를 비우면 자료에서 정한 주기로 등록되고, 발표시차 기본값도 그 주기 것이 된다
    r = m.add_variable(m.VariableBody(id="CPI2", source="FRED", source_code="MON"))
    assert r["freq"] == "M" and "월" in r["note"]
    reg, _ = Registry.load_master(master)
    assert reg.get("CPI2").freq == "M" and reg.get("CPI2").lag_days == 45
    # 사람이 넣은 주기가 자료와 다르면 거절(조용히 바꾸지 않는다)
    with pytest.raises(HTTPException) as e:
        m.add_variable(m.VariableBody(id="CPI3", source="FRED", source_code="QTR", freq="D"))
    assert "주기" in e.value.detail and "분기" in e.value.detail
    # 같으면 그대로 받는다
    assert m.add_variable(m.VariableBody(id="DAY2", source="FRED", source_code="DAY", freq="D"))["freq"] == "D"
    # 값이 너무 적어 주기를 판별 못 하면 등록하지 않는다
    with pytest.raises(HTTPException) as e:
        m.add_variable(m.VariableBody(id="TWO2", source="FRED", source_code="TWO"))
    assert "주기" in e.value.detail
    # 한국은행은 코드 끝 주기를 따른다(A = 연 → Y)
    c = m.check_variable(m.VariableCheckBody(source="ECOS", source_code="YR/0101000/A"))
    assert c["ok"] and c["freq"] == "Y" and "코드" in c["freq_basis"]


def test_review2_seal_exposure_includes_near_identical_other_tickers(tmp_path):
    """검토 2(높음): 091230 은 봉인 구간에서 091160 과 하루 수익률 상관 0.99 — 091230 개발 결과로 그 구간을 채점해 보면 091160 의 봉인 구간을 본 것과 같다.
    같은 종목만 찾던 seal_exposures 가 봉인 구간 하루 로그수익률 상관 |ρ| ≥ 0.9 인 다른 종목의 결과도 찾아, 상관값과 함께 돌려준다.
    상관이 낮은 종목·봉인 전까지만 채점한 결과·데이터가 없을 때(dv 없음)는 넣지 않는다. 막지는 않는다(경고·기록)."""
    from engine.sealed import seal_exposures
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=600)
    common = rng.normal(0, 0.02, len(idx))
    def px(r):
        c = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    dv = DataVersion(tmp_path, "t")
    dv.write_prices("AAA", px(common))
    dv.write_prices("BBB", px(common + rng.normal(0, 0.002, len(idx))))    # 거의 같은 종목
    dv.write_prices("CCC", px(rng.normal(0, 0.02, len(idx))))             # 상관 없는 종목
    sf = str(idx[400].date())
    res = tmp_path / "res"; res.mkdir()
    def put(stem, ticker, to):
        (res / f"{stem}.json").write_text(json.dumps({"config": {"mode": "trading", "ticker": ticker, "validation": {}},
                                                       "evaluation_period": [str(idx[100].date()), to]}), encoding="utf-8")
    put("a_same", "AAA", str(idx[-1].date())); put("b_near", "BBB", str(idx[-1].date())); put("b_before", "BBB", str(idx[399].date()))
    put("c_far", "CCC", str(idx[-1].date()))
    cfg = {"mode": "trading", "ticker": "AAA", "validation": {"sealed_from": sf}}
    ex = seal_exposures(res, cfg, dv)
    got = {x["result"]: x for x in ex}
    assert set(got) == {"a_same", "b_near"}, got
    assert got["a_same"].get("related_ticker") is None
    assert got["b_near"]["related_ticker"] == "BBB" and got["b_near"]["corr"] > 0.95 and got["b_near"]["corr_period"][0] == sf
    assert {x["result"] for x in seal_exposures(res, cfg)} == {"a_same"}          # 데이터 없이 부르면 전과 같다(같은 종목만)


def test_review2_sealed_opening_records_exposures(tmp_path, monkeypatch):
    """봉인을 열면 결과의 sealed_info.exposures 에 '열기 전에 이 봉인 구간을 이미 채점한 개발 결과'(같은 종목 + 거의 같은 종목)가 남는다.
    (전에는 미리보기 경고로만 보여 주고 결과 파일에는 남지 않았다 — 나중에 결과만 보는 사람은 알 수 없었다.)"""
    import cli
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2022-01-03", periods=700)
    common = rng.normal(0, 0.02, len(idx))
    def px(r):
        c = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000.0}, index=idx)
    data = tmp_path / "data"
    dv = DataVersion(data, "v1")
    dv.write_prices("AAA", px(common)); dv.write_prices("BBB", px(common + rng.normal(0, 0.002, len(idx))))
    mf = data / "versions" / "v1" / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps({"id": "v1", "complete": True}), encoding="utf-8")
    (data / "registry.csv").write_text("id,freq,source,lag_days\n", encoding="utf-8")
    sf = str(idx[550].date())
    pol = tmp_path / "pol.json"; pol.write_text(json.dumps({"trading": {"AAA": sf}}), encoding="utf-8")
    monkeypatch.setenv("TSP_SEAL_POLICY", str(pol))
    monkeypatch.setattr(cli, "DATA", data); monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    from engine.experiment import results_dir
    res = results_dir(tmp_path / "results", "trading")
    (res / "dev_bbb.json").write_text(json.dumps({"config": {"mode": "trading", "ticker": "BBB", "validation": {}},
                                                  "evaluation_period": [str(idx[300].date()), str(idx[-1].date())]}), encoding="utf-8")
    cfgp = tmp_path / "c.json"
    cfgp.write_text(json.dumps({"name": "s", "ticker": "AAA", "model": "ridge", "validation": {"train_len": 250, "test_len": 40}}), encoding="utf-8")
    cli.cmd_run(str(cfgp), "v1", sealed=True)
    out = [p for p in res.glob("*.json") if p.name not in ("dev_bbb.json", "sealed_log.json", "experiment_counter.json")]
    rs = [json.loads(p.read_text(encoding="utf-8")) for p in out]
    rs = [r for r in rs if r.get("sealed_evaluation")]
    assert len(rs) == 1
    ex = rs[0]["sealed_info"]["exposures"]
    assert [x["result"] for x in ex] == ["dev_bbb"] and ex[0]["related_ticker"] == "BBB" and ex[0]["corr"] > 0.95


def test_review3_explain_refuses_volatility_target(tmp_path):
    """검토 3(중간): 계산 따라가기의 '결정' 단계는 '예측 > 문턱 → 보유'로 설명하는데, 변동성 타겟의 실제 규칙은 '비중 = 학습 창 평균 변동성 ÷ 예측 변동성'.
    틀린 설명을 보여 주지 않도록, 지원할 때까지 거절한다(묶음 학습·입력 선별과 같은 방식)."""
    from engine.explain import explain
    prices, s = predictable_ohlcv(600, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = dict(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, target={"kind": "volatility", "k": 5},
               validation={"train_len": 250, "test_len": 40})
    with pytest.raises(ValueError, match="변동성"):
        explain(cfg, dv, reg)


def test_review4_restat_handles_volatility_results(tmp_path):
    """검토 4(중간): 재계산 도구가 변동성 결과를 '예측값 없음(규칙 모델)'으로 적었다('0 예측 대비 DM' 이 없다는 이유로).
    변동성 결과는 '변동성 타겟' 줄이 되어, naive 대비 DM(저장값, 나음 한쪽 p)과 학습 창 평균 기준 CW 를 내고, 0 예측 대비 값은 비워 둔다.
    Holm 가족은 타겟 종류로 나뉘므로 수익률 결과와 섞이지 않는다."""
    import importlib.util
    from engine.experiment import save_result
    spec = importlib.util.spec_from_file_location("restat", Path(__file__).resolve().parents[1] / "tools" / "restat.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    prices, s = predictable_ohlcv(700, seed=8)
    dv, reg = _dv_with(prices, s, tmp_path)
    out = tmp_path / "results" / "trading"
    cv = ExperimentConfig(name="vol", ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=True, target={"kind": "volatility", "k": 5},
                          validation=Validation(train_len=300, test_len=40))
    cr = ExperimentConfig(name="ret", ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=300, test_len=40))
    rv = run_experiment(cv, dv, reg); save_result(rv, out); save_result(run_experiment(cr, dv, reg), out)
    rows = {x["name"]: x for x in rs.restat("trading", tmp_path / "results")["rows"]}
    v = rows["vol"]
    assert "예측값 없음" not in (v.get("note") or "") and "변동성" in v["note"]
    assert v["dm_vs_naive"] == rv["prediction_metrics"]["dm_vs_naive"] and v["dm_vs_naive_p_better"] is not None
    assert v.get("cw_vs_mean_p") is not None and v.get("n_eval", 0) > 10
    assert v.get("dm_vs_zero") is None and v.get("cw_vs_zero") is None
    assert v["holm_cw_mean_p_m"] == 1 and rows["ret"]["holm_cw_mean_p_m"] == 1     # 가족이 나뉜다
    assert v["holm_dm_naive_p_m"] == 1
    md = rs.to_markdown(rs.restat("trading", tmp_path / "results"))
    assert "naive" in md


def test_review5_collect_exit_code_3_means_stopped_before_collecting(tmp_path, monkeypatch):
    """검토 5(중간): 시작하기 전에 멈춘 경우(변수 사전 없음·엑셀 고칠 곳·없는 버전 id)도 종료 코드 2 를 써서, 화면이
    '받지 못한 항목이 있어 직전 값을 두었다'고 잘못 알렸다. 조기 멈춤은 3, 화면 문구도 따로."""
    import collect
    import api.main as m
    monkeypatch.setattr(sys, "argv", ["collect.py", "--data-dir", str(tmp_path / "없음")])
    assert collect.main() == 3                                            # 변수 사전 없음
    class P:
        def __init__(self, rc): self.rc = rc
        def poll(self): return self.rc
    for rc, word in ((3, "시작하기 전에"), (2, "받지 못한 항목"), (1, "종목 가격을 새로 받지 못했"), (0, "모두 받았")):
        monkeypatch.setattr(m, "_COLLECT", {"proc": P(rc), "log": str(tmp_path / "x.log"), "started": "t"})
        st = m.collect_status()
        assert st["returncode"] == rc and word in st["message"], (rc, st["message"])


def _write_dv(path, prices_by_ticker: dict, variables: dict) -> tuple[DataVersion, Registry]:
    dv = DataVersion(path, "t")
    for t, p in prices_by_ticker.items():
        dv.write_prices(t, p)
    reg = Registry()
    for name, sr in variables.items():
        reg.add(Variable(id=name, freq="D", source="DERIVED"))
        dv.write_variables(sr.rename("value").rename_axis("date").reset_index().assign(variable_id=name), source=f"syn_{name}")
    return dv, reg


def _shake_after(p: pd.DataFrame, T, seed: int) -> pd.DataFrame:
    """T 이후 가격만 흔든다(하루하루 다른 배수 — 수익률·변동성이 모두 바뀐다). T 앞은 그대로."""
    q = p.copy()
    m = q.index >= T
    f = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.05, int(m.sum()))))
    for c in ("open", "high", "low", "close"):
        q.loc[m, c] = q.loc[m, c].to_numpy() * f
    q.loc[m, "volume"] = q.loc[m, "volume"].to_numpy() * np.random.default_rng(seed + 1).lognormal(0, 1, int(m.sum()))
    return q


@pytest.mark.parametrize("case", ["select_top", "pool_ridge", "pool_lgbm", "volatility"])
def test_review6_future_perturbation_leaves_past_predictions_unchanged(case, tmp_path):
    """검토 6(중간): '폴드 안에서만 고른다·경계 앞 행만 보탠다·변동성도 미래를 안 쓴다'는 주장을 직접 시험한다.
    날짜 T 이후의 자료만 바꾼다 — 가격은 흔들고, 신호 변수 S 와 잡음 변수 N1 의 값은 맞바꾼다(그러면 '전체 자료로' 순위를 매기는 잘못된 코드는
    T 앞 폴드에서도 N1 을 고르게 되어 예측이 바뀐다). 결정일이 T 앞인 예측은 모두 그대로여야 한다(소수 10자리)."""
    pa, pb, s = _shared_signal_pair(700)
    rng = np.random.default_rng(11)
    noise = {f"N{i}": pd.Series(np.convolve(rng.normal(size=700), np.ones(5) / 5, "same"), index=pa.index) for i in (1, 2, 3)}
    T = pa.index[260]
    s2, n2 = s.copy(), noise["N1"].copy()
    s2[s.index >= T], n2[s.index >= T] = noise["N1"][s.index >= T] * 3, s[s.index >= T]
    dv1, reg1 = _write_dv(tmp_path / "a", {"AAA": pa, "BBB": pb}, {"S_signal": s, **noise})
    dv2, reg2 = _write_dv(tmp_path / "b", {"AAA": _shake_after(pa, T, 1), "BBB": _shake_after(pb, T, 2)}, {"S_signal": s2, **noise, "N1": n2})
    base = dict(ticker="AAA", use_technical=False, validation=Validation(train_len=150, test_len=30))
    cfg = {"select_top": dict(model="ridge", variable_ids=["S_signal", "N1", "N2", "N3"], select_top=1),
           "pool_ridge": dict(model="ridge", variable_ids=["S_signal"], pool_tickers=["BBB"]),
           "pool_lgbm": dict(model="lightgbm_model", variable_ids=["S_signal"], pool_tickers=["BBB"]),
           "volatility": dict(model="ridge", variable_ids=["S_signal", "N1"], target={"kind": "volatility", "k": 5}, use_technical=True)}[case]
    c = ExperimentConfig(**{**base, **cfg})
    r1, r2 = run_experiment(c, dv1, reg1), run_experiment(c, dv2, reg2)
    p1, p2 = r1["_daily"]["prediction"].dropna(), r2["_daily"]["prediction"].dropna()
    early = p1.index[p1.index < T]
    assert len(early) >= 60, len(early)
    assert np.allclose(p1.loc[early].values, p2.reindex(early).values, atol=1e-10, rtol=0), np.abs(p1.loc[early] - p2.reindex(early)).max()
    if case == "volatility":                                       # 비중(포지션)도 T 앞은 같다
        w1, w2 = r1["_daily"]["position"], r2["_daily"]["position"]
        assert np.allclose(w1.loc[early].values, w2.reindex(early).values, atol=1e-10, equal_nan=True)
    assert not np.allclose(p1.loc[p1.index >= T].values[:50], p2.reindex(p1.index[p1.index >= T][:50]).values, atol=1e-6)   # 바꾼 뒤는 실제로 달라졌다(시험이 헛돌지 않는다)


@pytest.mark.parametrize("model", ["ridge", "lightgbm_model"])
def test_review6_pool_rows_respect_validation_and_test_boundary_end_to_end(model, tmp_path):
    """묶어 학습의 경계를 실험 전체로 시험한다: 첫 폴드의 경계(검증을 쓰는 LightGBM 은 검증 시작일, 릿지는 테스트 시작일)부터 뒤의
    '보탠 종목(BBB)' 가격만 바꾼다. 첫 폴드의 예측은 그대로여야 한다 — 보탠 행이 경계 뒤 가격으로 정답을 만들면(또는 경계를 테스트 시작일로 잘못 잡으면) 달라진다."""
    pa, pb, s = _shared_signal_pair(700)
    base = dict(ticker="AAA", model=model, variable_ids=["S_signal"], use_technical=False, pool_tickers=["BBB"], validation=Validation(train_len=200, test_len=50))
    dv1, reg1 = _pair_dv(tmp_path / "a", pa, pb, s)
    r1 = run_experiment(ExperimentConfig(**base), dv1, reg1)
    ff = r1["periods"]["first_fold"]
    cut = pd.Timestamp((ff.get("val") or ff["test"])[0])
    if model == "lightgbm_model":
        assert ff.get("val"), ff                                     # 이 설정에서 검증이 실제로 쓰였다
    dv2, reg2 = _pair_dv(tmp_path / "b", pa, _shake_after(pb, cut, 5), s)
    r2 = run_experiment(ExperimentConfig(**base), dv2, reg2)
    te = pd.date_range(ff["test"][0], ff["test"][1])
    p1, p2 = r1["_daily"]["prediction"].dropna(), r2["_daily"]["prediction"].dropna()
    d = p1.index[p1.index.isin(te)]
    assert len(d) >= 40 and np.allclose(p1.loc[d].values, p2.reindex(d).values, atol=1e-10, rtol=0)


def test_review6_volatility_sealed_refused_incomplete_version_skipped_duplicate_keeps_last(tmp_path, monkeypatch):
    """검토 6 나머지: ① 변동성 타겟은 봉인 평가를 거절한다(통과 기준이 계획서에 없다). ② [값 보기]의 기본 버전은 '가장 새 온전한 버전' —
    더 새 버전이 '일부 받지 못함'이면 건너뛴다. ③ 같은 날짜가 두 번이면 화면과 엔진이 모두 '나중 값'을 쓴다(값이 다를 때)."""
    from tests.synthetic import volatility_ohlcv
    prices, s = volatility_ohlcv(700, seed=2)
    dv, reg = _dv_with(prices, s, tmp_path / "v")
    with pytest.raises(ValueError, match="봉인"):
        run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=[s.name], use_technical=False, target=TargetSpec(kind="volatility", k=5),
                                        validation=Validation(train_len=300, test_len=60, sealed_from=str(prices.index[600].date()))), dv, reg, sealed=True)
    import api.main as m
    from engine.registry import Variable
    data = tmp_path / "data"
    for vid_, (val2, incomplete) in {"20260101T000000Z": (7.0, False), "20260201T000000Z": (9.0, True)}.items():
        d = data / "versions" / vid_; d.mkdir(parents=True)
        dates = list(pd.bdate_range("2025-01-01", periods=20)); vals = [100.0 + i for i in range(20)]
        dates.append(dates[5]); vals.append(val2)                                   # 같은 날짜(1월 8일)에 다른 값이 뒤에 한 번 더
        pd.DataFrame({"variable_id": "X", "date": dates, "value": vals}).to_parquet(d / "variables_fred.parquet", index=False)
        (d / "manifest.json").write_text(json.dumps({"start": "2006-01-01", **({"incomplete": True} if incomplete else {})}))
    reg2 = Registry([Variable(id="X", freq="D", lag_days=1, source="FRED", source_code="XX", transform="pct")])
    monkeypatch.setattr(m, "DATA", data); monkeypatch.setattr(m, "_registry", lambda: (reg2, []))
    r = m.variable_values("X")
    assert r["version"] == "20260101T000000Z"                                     # 더 새 버전은 '일부 받지 못함'이라 건너뜀
    day = str(pd.bdate_range("2025-01-01", periods=20)[5].date())
    shown = {x["date"]: x["value"] for x in r["values"]}
    assert shown[day] == 7.0 and r["checks"]["duplicate_dates"] == 1              # 화면: 나중 값
    w = DataVersion(data, "20260101T000000Z").wide(["X"])                       # 엔진이 실험에 쓰는 길(build_dataset → wide)
    assert float(w.loc[pd.Timestamp(day), "X"]) == 7.0 and len(w) == 20          # 엔진: 나중 값 하나


def test_review7_reversal_rule_is_robust_z_from_median_and_reports_flat_series():
    """검토 7(낮음): '튀었다 되돌아온 날'의 기준을 문서와 같게 — 강건 z = |변화 − 중앙값| ÷ (1.4826·MAD) > 8 이 이틀 연속, 방향 반대,
    둘째 날이 첫째 날 움직임의 2/3 이상을 되돌림. 변화의 대부분이 0 이라 MAD 가 0 이면 점검할 수 없음을 말없이 건너뛰지 않고 알린다."""
    import api.main as m
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    # 꾸준히 오르는 값(하루 변화 중앙값 0.02) — 중앙값을 빼지 않으면 작은 잡음도 '큰 움직임'처럼 재진다
    r = 0.02 + rng.normal(0, 0.001, 300)
    s = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    c = m._gap_and_reversal_checks(s, "D")
    assert c["reversals"] == [] and c["reversal_note"] is None
    s2 = s.copy(); s2.iloc[150] *= 1.5                                             # 하루 50% 튀었다 다음 날 제자리
    c2 = m._gap_and_reversal_checks(s2, "D")
    assert [x["date"] for x in c2["reversals"]] == [str(idx[150].date())]
    s3 = s.copy(); s3.iloc[150:] *= 1.5; s3.iloc[151:] /= 1.2                      # 50% 오른 뒤 일부(1/3 쯤)만 되돌림 — 자료 오류 모양이 아니다
    assert m._gap_and_reversal_checks(s3, "D")["reversals"] == []
    flat = pd.Series([5.0] * 250 + [5.0, 50.0, 5.0] + [5.0] * 47, index=idx)      # 변화가 거의 다 0 → MAD 0
    cf = m._gap_and_reversal_checks(flat, "D")
    assert cf["reversals"] == [] and "MAD" in (cf["reversal_note"] or "")


def test_review8_transform_rejects_zero_and_leading_zero_lags():
    """검토 8(낮음): pct0·diff0 은 '0 관측 전 대비' = 모든 값이 0 인 입력이 되고, pct00·pct05 는 뜻이 모호하다. 엔진과 화면(같은 정규식) 모두 거절."""
    from engine.experiment import apply_transform, _TR_RE
    s = pd.Series(np.arange(1.0, 30.0))
    for bad in ("pct0", "diff0", "pct00", "pct05", "diff007"):
        assert not _TR_RE.match(bad), bad
        with pytest.raises(ValueError):
            apply_transform(s, bad)
    for ok, n in (("pct", 1), ("pct5", 5), ("diff12", 12), ("pct10", 10)):
        assert _TR_RE.match(ok)
        out = apply_transform(s, ok)
        assert out.isna().sum() == n


def test_review12_select_top_needs_learned_inputs_and_pool_rejects_duplicates(tmp_path):
    """검토 12(낮음): 입력 선별(select_top)은 입력 변수로 학습하는 모델(표·창 입력)에서만 뜻이 있다 — 규칙·ARIMA 류에서는 무시되면서 설정 해시만 바꿨다 → 거절.
    pool_tickers 에 같은 종목을 두 번 넣으면 결과는 같은데 해시만 달라졌다(같은 후보가 둘로 셈) → 거절."""
    prices, s = predictable_ohlcv(600, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    for model in ("arima", "ma_cross"):
        with pytest.raises(ValueError, match="입력 선별"):
            run_experiment(ExperimentConfig(ticker="TEST", model=model, variable_ids=["S_signal"], select_top=1, validation=Validation(train_len=250, test_len=40)), dv, reg)
    with pytest.raises(ValueError, match="두 번"):
        ExperimentConfig(ticker="TEST", model="ridge", pool_tickers=["BBB", "BBB"])
    assert ExperimentConfig(ticker="TEST", model="ridge", pool_tickers=["BBB", "CCC"]).pool_tickers == ["BBB", "CCC"]


def test_review13_other_sites_cannot_drive_local_server_and_wrong_password_delay_is_global(monkeypatch):
    """검토 13·의심(CSRF): 암호를 정하지 않은 노트북에서, 다른 웹사이트가 사용자 브라우저를 통해 127.0.0.1:8000 에 바꾸는 요청을 보낼 수 있었다
    (CORS 가 모든 출처 허용). → CORS 허용을 없애고, 바꾸는 요청에 Origin 이 있으면 서버 주소(Host)와 같아야 통과. Origin 이 없는 요청(화면 밖의 도구)은 그대로.
    외부(Funnel) 요청은 지금처럼 암호가 지킨다. 틀린 암호 지연은 서버 전체에서 한 번에 하나씩(동시에 여러 개를 넣어도 빨라지지 않게)."""
    import asyncio, time
    import api.main as m
    monkeypatch.delenv("TSP_WRITE_PASSWORD", raising=False)
    host = {"host": "127.0.0.1:8000"}
    assert _asgi_call(m.app, "POST", "/api/auth/check", {**host, "origin": "http://127.0.0.1:8000"})[0] == 200      # 우리 화면
    assert _asgi_call(m.app, "POST", "/api/auth/check", {"host": "localhost:5173", "origin": "http://localhost:5173"})[0] == 200   # 개발 화면(vite 프록시)
    assert _asgi_call(m.app, "POST", "/api/auth/check", host)[0] == 200                                               # Origin 없음(도구)
    assert _asgi_call(m.app, "POST", "/api/auth/check", {**host, "origin": "https://a.ts.net", "x-forwarded-host": "a.ts.net"})[0] == 200   # 프록시 뒤
    for bad in ("https://evil.example", "null", "http://127.0.0.1:9999"):
        st, b = _asgi_call(m.app, "POST", "/api/auth/check", {**host, "origin": bad})
        assert st == 403 and "다른 사이트" in b["detail"], bad
    assert _asgi_call(m.app, "GET", "/api/auth", {**host, "origin": "https://evil.example"})[0] == 200                # 보기는 그대로
    assert not any("CORSMiddleware" in str(mw.cls) for mw in m.app.user_middleware)
    # 틀린 암호 지연: 동시에 3개 → 지연이 겹치지 않고 한 줄로(≥ 3 × 지연)
    monkeypatch.setenv("TSP_WRITE_PASSWORD", "long-test-pass-123")
    monkeypatch.setattr(m, "_WRONG_DELAY", 0.2)
    async def three():
        t0 = time.perf_counter()
        await asyncio.gather(*[asyncio.to_thread(_asgi_call, m.app, "POST", "/api/auth/check", {**host, "x-tsp-password": "wrong"}) for _ in range(3)])
        return time.perf_counter() - t0
    assert asyncio.run(three()) >= 0.55


def test_review13_naver_retroactive_factor_is_rescaled_and_no_overlap_is_named(tmp_path, monkeypatch):
    """검토 13(낮음): ① 겹치는 날이 없는데도 '수정주가 소급 변경'이라고 적었다. ② 소급 변경이 생기면 직전 파일을 그대로 둬, 다음 실행에서도 또 달라 보여
    영영 갱신되지 않았다. → 겹치는 날의 '새 값 ÷ 직전 값'(시가·고가·저가·종가)이 모두 같은 배수면(배당·분할 조정) 직전 앞부분에 그 배수를 곱해 새 기준으로 잇는다.
    배수가 날마다·열마다 다르면 설명할 수 없는 변경이라 직전 파일을 두고 기록. 겹치는 날이 없으면 그렇게 적는다. 어느 쪽도 실패로 세지 않는다."""
    import collect
    from adapters import pykrx_adapter, yf_adapter
    from engine.store import DataVersion
    data = tmp_path / "data"; old = data / "versions" / "20200101T000000Z"; old.mkdir(parents=True)
    _mini_master(data / "variable_master.xlsx", [{"id": "M1", "freq": "M", "source": "XLSX", "source_code": "df_M!M1", "lag_days": 30}],
                 [("df_M", ["date", "M1"], [["2020-01", 1.0]])])
    full, _ = predictable_ohlcv(120, seed=5)
    for t in collect.TICKERS:
        full.iloc[:70].to_parquet(old / f"prices_{t}_naver.parquet"); full.iloc[:70].to_parquet(old / f"prices_{t}_yf.parquet"); full.iloc[:70].to_parquet(old / f"prices_{t}.parquet")
    (old / "manifest.json").write_text(json.dumps({"start": "2006-01-01"}))
    mode = {"m": "factor"}
    def naver(t, *a, **k):
        if mode["m"] == "factor":
            d = full.iloc[10:80].copy(); d[["open", "high", "low", "close"]] *= 0.8      # 전 기간에 같은 배수(소급 조정)
        else:
            d = full.iloc[100:120].copy()                                                # 직전 파일과 겹치는 날 없음
        return d
    monkeypatch.setattr(pykrx_adapter, "fetch_prices", naver)
    monkeypatch.setattr(yf_adapter, "fetch_prices", lambda t, *a, **k: full)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", "yf,pykrx,xlsx", "--data-dir", str(data)])
    assert collect.main() == 0
    dv = DataVersion(data, sorted(p.name for p in (data / "versions").iterdir())[-1])
    nv = dv.read_prices(f"{collect.TICKERS[0]}_naver")
    assert len(nv) == 80 and np.allclose(nv["close"].to_numpy(), full["close"].iloc[:80].to_numpy() * 0.8, rtol=1e-12)   # 앞부분도 새 기준
    note = " ".join(str(v) for v in (dv.manifest().get("notes") or {}).values())
    assert "배수" in note and "0.8" in note and not dv.manifest().get("incomplete")
    mode["m"] = "gap"
    import time as _time; _time.sleep(1.1)
    assert collect.main() == 0
    dv2 = DataVersion(data, sorted(p.name for p in (data / "versions").iterdir())[-1])
    note2 = " ".join(str(v) for v in (dv2.manifest().get("notes") or {}).values())
    assert "겹치는 날이 없" in note2 and "소급" not in note2 and len(dv2.read_prices(f"{collect.TICKERS[0]}_naver")) == 80


def test_review13_sealed_preview_blocks_volatility_candidates(tmp_path):
    """검토 13(낮음): 변동성 결과는 실행 단계에서 봉인 평가를 거절하는데, 미리보기에는 막는 이유가 없어 '열 수 있다'처럼 보였다 → 미리보기에서 미리 막는다."""
    from engine.sealed import preview
    from tests.synthetic import volatility_ohlcv
    prices, s = volatility_ohlcv(700, seed=2)
    dv, reg = _dv_with(prices, s, tmp_path)
    r = run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=[s.name], use_technical=False, target=TargetSpec(kind="volatility", k=5),
                                       validation=Validation(train_len=300, test_len=60, sealed_from=str(prices.index[600].date()))), dv, reg)
    r = json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, default=str))
    pv = preview(r, "x", dv, reg, [], None, None)
    assert any("변동성" in b for b in pv["blockers"]), pv["blockers"]


@pytest.mark.parametrize("model,extra", [("lightgbm_model", {}), ("lstm", {"model_params": {"epochs": 2, "hidden": 4}, "seq_len": 5})])
def test_review13_selection_ranks_only_rows_before_validation(model, extra, tmp_path):
    """검토 13(낮음): 검증을 쓰는 모델(LightGBM·LSTM 등)에서 입력 선별의 순위를 학습 창 전체(검증 행 포함)로 쟀다 — 검증 행은 조기 종료 판단에 쓰므로
    선별도 그 행을 보면 검증이 '처음 보는 자료'가 아니게 된다. → 폴드 기록의 '학습' 구간(검증과 그 앞 비움을 뺀 행)으로만 잰다."""
    from engine.experiment import _run_experiment, prepare, fold_relevance, make_folds
    prices, s = predictable_ohlcv(900, seed=5)
    dv, reg, ids = _dv_with_noise(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model=model, variable_ids=ids, use_technical=False, select_top=2, validation=Validation(train_len=300, test_len=40), **extra)
    log, diags = [], []
    _run_experiment(cfg, dv, reg, fold_log=log, diags=diags)
    _, _, _, _, ds = prepare(cfg, dv, reg, device="cpu")
    assert log[0]["val"] is not None
    a = make_folds(cfg, ds, cfg.target.k)[0][0].train[0]          # 폴드의 학습 행 시작(창 모델은 첫 창보다 앞의 행도 학습 행)
    b = pd.Timestamp(log[0]["train"][1])                           # 기록된 학습 끝 = 검증과 그 앞 비움을 뺀 마지막 결정일
    tr = ds.X.join(ds.y.rename("y")).dropna()
    tr = tr[(tr.index >= a) & (tr.index <= b)]
    want = fold_relevance(tr[ds.feature_names].values, tr["y"].values)
    got = np.array([d["relevance"] for d in diags if d.get("relevance") is not None][0])
    assert np.allclose(got, want, atol=1e-12), (got, want)


def test_review13_etf_probe_looks_back_far_enough_and_says_so(tmp_path, monkeypatch):
    """의심(검토 13): ETF 판별이 '최근 30일 공식 시세'만 물어, 공식 시세가 잠시 늦거나(연휴·지연) 비면 ETF 를 주식으로 판별할 수 있었다.
    → 최근 120일을 묻고, 판별 근거 문장에 그 기간을 적는다."""
    import api.main as m
    from adapters import datago_adapter
    seen = {}
    def dg(t, start, end, *a, **k):
        seen["start"] = start
        raise RuntimeError(f"DATAGO 응답 없음: {t}")
    monkeypatch.setenv("DATAGO_API_KEY", "dummy")
    monkeypatch.setattr(datago_adapter, "fetch_prices", dg)
    kind, basis = m._probe_kind("005930")
    assert kind == "stock" and (pd.Timestamp.today() - pd.Timestamp(seen["start"])).days >= 119 and "120일" in basis
