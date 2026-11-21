"""1회 분할 계획(engine/splitplan.py, 2026-11-18): 비율 → 개수 규칙이 원본 BIVA 와 같고, 계획의 구간·개수가 실제 실행의 폴드 기록과 같다."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.experiment import ExperimentConfig, run_experiment
from engine.splitplan import ratio_counts, split_plan
from tests.synthetic import predictable_ohlcv
from tests.test_engine import _dv_with


def test_ratio_counts_follow_original_biva_rule():
    """원본 data_loader.py: num_train = int(len*0.60), num_test = int(len*0.30), num_vali = 나머지. 모든 N 에서 같아야 한다."""
    for n in range(1, 3000):
        assert ratio_counts(n, (60, 10, 30)) == (int(n * 0.60), n - int(n * 0.60) - int(n * 0.30), int(n * 0.30))
    assert sum(ratio_counts(137, (62.5, 7.5, 30))) == 137
    assert ratio_counts(101, (70, 0, 30)) == (71, 0, 30)          # 검증 0% 면 나머지는 학습에
    with pytest.raises(ValueError, match="합이 100"):
        ratio_counts(100, (60, 10, 20))
    with pytest.raises(ValueError):
        ratio_counts(100, (110, -10, 0))


def _data(tmp_path, late: int = 200):
    prices, s = predictable_ohlcv(900, seed=11)
    s = s.copy()
    s.iloc[:late] = np.nan          # 늦게 시작하는 변수(앞쪽 빈 칸)
    return _dv_with(prices, s, tmp_path)


def _cfg(model, **kw):
    return dict(ticker="TEST", model=model, variable_ids=["S_signal"], use_technical=False, validation=dict(mode="split", test_start="2017-01-01"), **kw)


@pytest.mark.parametrize("model,extra", [("ridge", {}), ("lightgbm_model", {}), ("lstm", {"model_params": {"epochs": 2}})])
def test_plan_matches_the_run_fold_log(tmp_path, model, extra):
    """비율로 채운 날짜로 실제로 돌리면, 폴드 기록의 학습·검증·테스트 구간과 개수가 계획과 같다(표·창 입력, 검증 있음·없음)."""
    dv, reg = _data(tmp_path)
    plan = split_plan(_cfg(model, **extra), dv, reg, ratios=[60, 10, 30])
    assert not plan["blocking"]
    n_tr, n_va, n_te = plan["ratio"]["counts"]
    assert n_tr + n_va + n_te == plan["base"]["n"]
    d = plan["dates"]
    cfg = ExperimentConfig(**{**_cfg(model, **extra), "validation": dict(mode="split", **{k: v for k, v in d.items() if v})})
    ff = run_experiment(cfg, dv, reg)["periods"]["first_fold"]
    rows = {r["part"].split("(")[0]: r for r in plan["rows"]}
    assert [rows["학습"]["first"], rows["학습"]["last"]] == ff["train"] and rows["학습"]["n"] == ff["n_train"]
    assert [rows["테스트"]["first"], rows["테스트"]["last"]] == ff["test"] and rows["테스트"]["n"] == ff["n_test"]
    if ff["val"]:
        assert [rows["검증"]["first"], rows["검증"]["last"]] == ff["val"] and rows["검증"]["n"] == ff["n_val"]
    else:
        assert "검증" not in rows and plan["dates"]["val_start"] is None     # Ridge 는 검증을 쓰지 않아 검증 몫도 학습에


def test_ratio_dates_cut_the_base_at_exact_positions(tmp_path):
    """비율로 채운 날짜 = 바탕 결정일의 n_tr·n_tr+n_va 번째, 테스트 종료 = 마지막 결정일."""
    dv, reg = _data(tmp_path)
    plan = split_plan(_cfg("lightgbm_model"), dv, reg, ratios=[60, 10, 30])
    from engine.experiment import ExperimentConfig as EC, prepare
    from engine.splitplan import base_index
    cfg, _p, _pa, _t, ds = prepare(EC(**_cfg("lightgbm_model")), dv, reg)
    base = base_index(cfg, ds, cfg.target.k)
    n_tr, n_va, _ = plan["ratio"]["counts"]
    assert plan["dates"] == {"val_start": str(base[n_tr].date()), "test_start": str(base[n_tr + n_va].date()), "test_end": str(base[-1].date())}


def test_dates_that_are_not_decision_days_are_snapped(tmp_path):
    """주말 날짜를 넣으면 시작일은 다음 결정일, 종료일은 이전 결정일로 맞추고 그 사실을 알려 준다 — 실행도 같은 날짜를 쓴다."""
    dv, reg = _data(tmp_path)
    cfg = _cfg("lightgbm_model")
    cfg["validation"] = dict(mode="split", val_start="2017-01-01", test_start="2017-04-01", test_end="2018-01-07")  # 일·토·일
    plan = split_plan(cfg, dv, reg)
    assert len(plan["snapped"]) == 3
    ff = run_experiment(ExperimentConfig(**cfg), dv, reg)["periods"]["first_fold"]
    rows = {r["part"].split("(")[0]: r for r in plan["rows"]}
    assert rows["검증"]["first"] == ff["val"][0] == "2017-01-02"
    assert rows["테스트"]["first"] == ff["test"][0] == "2017-04-03" and rows["테스트"]["last"] == ff["test"][1] == "2018-01-05"


def test_blanks_and_blocking_are_reported(tmp_path):
    """빈 칸이 있는 변수와 처리 방법을 알려 주고, 입력이 모두 있는 학습 행이 30개 미만이면(엔진이 건너뜀) 막는 이유를 말한다."""
    dv, reg = _data(tmp_path, late=610)
    plan = split_plan(_cfg("lstm"), dv, reg, ratios=[60, 10, 30])
    assert plan["blanks"]["vars"][0]["var"] == "S_signal" and plan["blanks"]["vars"][0]["blank_train"] > 0
    assert "직전 값" in plan["blanks"]["how"]
    assert any("30" in b for b in plan["blocking"])
    c = _cfg("lstm", model_params={"epochs": 1})
    c["validation"] = dict(mode="split", **{k: v for k, v in plan["dates"].items() if v})
    with pytest.raises(RuntimeError):            # 계획이 막는다고 한 설정은 실제로도 결과가 나오지 않는다
        run_experiment(ExperimentConfig(**c), dv, reg)
    dv2, reg2 = _data(tmp_path / "b", late=0)
    assert not split_plan(_cfg("lstm"), dv2, reg2, ratios=[60, 10, 30])["blanks"]["vars"]


@pytest.mark.parametrize("model,extra", [("ridge", {}), ("lightgbm_model", {}), ("lstm", {"model_params": {"epochs": 2}})])
def test_walkforward_plan_matches_first_and_last_fold_of_the_run(tmp_path, model, extra):
    """워크포워드 계획의 첫·마지막 폴드 구간·개수 = 실행의 periods.first_fold·last_fold. 건너뛴 폴드 수도 같다. 자르는 순서의 끝 = 학습 개수."""
    from engine.splitplan import validation_plan
    dv, reg = _data(tmp_path, late=300)
    cfg = dict(ticker="TEST", model=model, variable_ids=["S_signal"], use_technical=False,
               validation=dict(mode="walkforward", train_len=200, test_len=50), **extra)
    plan = validation_plan(cfg, dv, reg)
    assert plan["mode"] == "walkforward" and plan["skipped_folds"]       # 앞쪽 폴드는 늦게 시작하는 변수 때문에 건너뜀
    res = run_experiment(ExperimentConfig(**cfg), dv, reg)
    per = res["periods"]
    assert plan["test_span"]["n_folds_run"] == res["n_folds"]
    for fp, ff in zip(plan["folds"], (per["first_fold"], per["last_fold"])):
        rows = {r["part"].split("(")[0]: r for r in fp["rows"]}
        assert [rows["학습"]["first"], rows["학습"]["last"]] == ff["train"] and rows["학습"]["n"] == ff["n_train"]
        assert [rows["테스트"]["first"], rows["테스트"]["last"]] == ff["test"] and rows["테스트"]["n"] == ff["n_test"]
        if ff["val"]:
            assert [rows["검증"]["first"], rows["검증"]["last"]] == ff["val"] and rows["검증"]["n"] == ff["n_val"]
        st = fp["steps"]
        assert st[0]["n"] == 200 and st[-1]["final"] and st[-1]["n"] == ff["n_train"]
        for a, b in zip(st[:-1], st[1:-1]):          # 매 단계: 앞 단계 개수 − 뺀 개수 = 이 단계 개수
            assert a["n"] - b["minus"] == b["n"]
    assert plan["test_span"]["first"] == per["first_fold"]["test"][0] and plan["test_span"]["last"] == per["last_fold"]["test"][1]
