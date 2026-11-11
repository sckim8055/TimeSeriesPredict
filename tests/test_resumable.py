"""tools/resumable_run.py — 폴드 단위 저장·재개가 '한 번에 돌린 것'과 같은 결과를 내는지 본다(2026-11-07).

창 모델(LSTM)은 폴드마다 전역 난수로 가중치를 초기화하고 미니배치를 섞는다. 그래서 중간에 멈췄다 새 프로세스에서 이어 돌릴 때
난수 상태를 되돌리지 않으면 뒤 폴드의 예측이 달라진다 — 이 시험이 그것까지 확인한다(이어 돌리기 전에 일부러 난수를 흐트러뜨림).
"""
from __future__ import annotations

import numpy as np
import pytest

import engine.experiment as E
from engine.experiment import ExperimentConfig, Validation, make_folds, prepare
from engine.registry import Variable
from tests.synthetic import predictable_ohlcv
from tests.test_engine import _dv_with


def _setup(tmp_path):
    prices, s = predictable_ohlcv(700, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path)
    cfg = ExperimentConfig(ticker="TEST", model="lstm", variable_ids=["S_signal"], use_technical=False, seq_len=5,
                           model_params={"epochs": 2, "hidden": 4}, validation=Validation(train_len=300, test_len=60), seed=11)
    cfg, plugin, params, tt, ds = prepare(cfg, dv, reg)
    folds, _ = make_folds(cfg, ds, 1)
    assert len(folds) >= 4
    return cfg, plugin, params, tt, ds, folds[:4]


def test_stop_and_resume_equals_uninterrupted(tmp_path, monkeypatch):
    from tools import resumable_run as R
    monkeypatch.setattr(E, "_walkforward_predict", E._walkforward_predict)  # 시험이 끝나면 원래 함수로 되돌림
    cfg, plugin, params, tt, ds, folds = _setup(tmp_path)

    E._seed(cfg.seed)
    log0: list = []
    ref, n0, imp0 = R._orig(cfg, plugin, params, ds, "cpu", folds, 1, tt, log=log0)

    ck = tmp_path / "ck.pkl"
    E._seed(cfg.seed)
    wf = R.make_resumable(ck, stop_after=2)
    with pytest.raises(R._Stop):
        wf(cfg, plugin, params, ds, "cpu", folds, 1, tt, log=[])
    assert ck.exists()

    E._seed(12345); np.random.rand(7)  # 새 프로세스: 시작 때 난수 상태가 다르다 → 저장한 상태로 되돌아가야 한다
    wf = R.make_resumable(ck)
    log1: list = []
    out, n1, imp1 = wf(cfg, plugin, params, ds, "cpu", folds, 1, tt, log=log1)

    assert n1 == n0 == 4
    assert ref.notna().sum() > 0
    assert np.array_equal(ref.isna().values, out.isna().values)
    assert np.max(np.abs(ref.dropna().values - out.dropna().values)) == 0.0
    assert log1 == log0


def test_resume_ignores_checkpoint_of_other_config(tmp_path, monkeypatch):
    from tools import resumable_run as R
    monkeypatch.setattr(E, "_walkforward_predict", E._walkforward_predict)
    cfg, plugin, params, tt, ds, folds = _setup(tmp_path)
    ck = tmp_path / "ck.pkl"
    E._seed(cfg.seed)
    with pytest.raises(R._Stop):
        R.make_resumable(ck, stop_after=1)(cfg, plugin, params, ds, "cpu", folds, 1, tt, log=[])
    E._seed(cfg.seed)
    ref, _, _ = R._orig(cfg, plugin, params, ds, "cpu", folds[:3], 1, tt)
    E._seed(cfg.seed)
    out, n, _ = R.make_resumable(ck)(cfg, plugin, params, ds, "cpu", folds[:3], 1, tt)  # 폴드 구성이 다름 → 저장 파일을 쓰지 않고 처음부터
    assert n == 3 and np.max(np.abs(ref.dropna().values - out.dropna().values)) == 0.0
