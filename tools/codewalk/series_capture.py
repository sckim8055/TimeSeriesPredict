"""ARIMA·SARIMA 실제 실행 붙잡기 — 엔진의 실제 경로(_walkforward_predict → models/arima.py·sarima.py → statsmodels SARIMAX)로 기록한다.

  python -m tools.codewalk.series_capture --model arima [--config configs/f11_arima_k1.json]

기록: 마지막 폴드의 학습 변화율, 최우추정(MLE) 반복마다의 모수·로그우도(statsmodels fit 에 callback 만 붙임 — 계산 불변),
칼만 필터의 한 걸음 예측오차 v_t·분산 F_t, 시작값, 근, 결정일의 예측(apply), 그리고 모든 폴드의 추정 모수(안정성 확인용).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.experiment import ExperimentConfig, _walkforward_predict, make_folds, prepare  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402


def capture(model: str = "arima", config: str | None = None, data_root: Path | None = None) -> dict:
    from statsmodels.tsa.statespace import sarimax as SX
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = config or f"configs/f11_{model}_k1.json"
    data_root = data_root or ROOT / "data"
    raw_cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    reg, _ = Registry.load_master(data_root / "variable_master.xlsx")
    cfg = ExperimentConfig(**raw_cfg)
    dv = DataVersion(data_root, cfg.data_version)
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    k = cfg.target.k
    folds, _ = make_folds(cfg, ds, k)
    folds = [f for f in folds if len(f.test)]
    fold = folds[-1]

    rec = {"iters": [], "start": None}
    orig_fit = SX.SARIMAX.fit

    def fit(self, *a, **kw):  # 원래 fit 그대로 + 반복마다 모수 기록(callback)
        def cb(x):
            th = self.transform_params(np.asarray(x)) if kw.get("transformed", True) else np.asarray(x)
            rec["iters"].append({"params": th.copy(), "llf": float(self.loglike(th))})
        rec["start"] = np.asarray(self.start_params, dtype=float).copy()
        rec["param_names"] = list(self.param_names)
        return orig_fit(self, *a, callback=cb, **kw)

    SX.SARIMAX.fit = fit
    cap: dict = {}
    try:
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [fold], k, target_type, capture=cap)
    finally:
        SX.SARIMAX.fit = orig_fit
    m = cap["model"]; res = m.res
    fr = res.filter_results
    r_all = cap["r_all"]
    t_dec = cap["pred"].dropna().index[-1]
    hist = r_all.loc[:t_dec].dropna()
    sc = float(getattr(m, "scale", 1.0))
    app = res.apply(hist.values * sc, refit=False)
    # 모든 폴드(같은 설정)의 추정 모수 — 폴드마다 다시 추정하는 워크포워드의 안정성
    allp = []
    for f in folds:
        c2: dict = {}
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [f], k, target_type, capture=c2)
        if "model" in c2:
            r2 = c2["model"].res
            allp.append({"test": str(f.test[0].date()), "params": np.asarray(r2.params, dtype=float).copy(),
                         "ar_roots": np.abs(r2.arroots).tolist() if len(r2.arroots) else [], "ma_roots": np.abs(r2.maroots).tolist() if len(r2.maroots) else [],
                         "pred": float(c2["pred"].dropna().iloc[-1]) if len(c2["pred"].dropna()) else np.nan, "llf": float(r2.llf) + (r2.nobs - r2.loglikelihood_burn) * np.log(sc),
                         "converged": bool((r2.mle_retvals or {}).get("converged", True)), "warnflag": int((r2.mle_retvals or {}).get("warnflag", 0)),
                         "iterations": int((r2.mle_retvals or {}).get("iterations", 0))})
    # 같은 폴드들을 배율 1(2026-11-07 전 방식)로도 추정해 수렴·로그우도 비교(원래 단위로 환산)
    p1 = {**params, "scale": 1.0}
    cmp1 = []
    for f, ap in zip(folds, allp):
        c3: dict = {}
        _walkforward_predict(cfg, plugin, p1, ds, "cpu", [f], k, target_type, capture=c3)
        r3 = c3["model"].res
        cmp1.append({"converged": bool((r3.mle_retvals or {}).get("converged", True)), "llf": float(r3.llf), "pred": float(c3["pred"].dropna().iloc[-1])})
    ssm = app.model.ssm
    lastm = lambda a_: np.asarray(a_)[..., -1] if np.asarray(a_).ndim == 3 or (np.asarray(a_).ndim == 2 and a_ is not None and np.asarray(a_).shape[-1] == len(hist)) else np.asarray(a_)
    mats = {"Z": np.asarray(ssm["design"])[:, :, -1] if np.asarray(ssm["design"]).ndim == 3 else np.asarray(ssm["design"]),
            "T": np.asarray(ssm["transition"])[:, :, -1] if np.asarray(ssm["transition"]).ndim == 3 else np.asarray(ssm["transition"]),
            "c": np.asarray(ssm["state_intercept"])[:, -1] if np.asarray(ssm["state_intercept"]).ndim == 2 else np.asarray(ssm["state_intercept"]),
            "d": np.asarray(ssm["obs_intercept"])[:, -1] if np.asarray(ssm["obs_intercept"]).ndim == 2 else np.asarray(ssm["obs_intercept"]),
            "a_last": np.asarray(app.filter_results.filtered_state[:, -1], dtype=float)}
    tr_idx = r_all.loc[fold.train].dropna().index
    # 학습 적합의 상태공간 행렬·초기값 — 대본이 칼만 필터를 손으로 다시 돌려 v_t·F_t·로그우도를 맞춰 본다
    s2 = res.model.ssm
    g = lambda k_: (lambda a_: a_[..., 0] if a_.ndim == 3 or (a_.ndim == 2 and k_ in ("state_intercept", "obs_intercept")) else a_)(np.asarray(s2[k_], dtype=float))
    a0, P0inf, P0 = s2.initialization(model=s2)
    kf = {"Z": g("design"), "d": g("obs_intercept"), "H": g("obs_cov"), "T": g("transition"), "c": g("state_intercept"), "R": g("selection"), "Q": g("state_cov"),
          "a0": np.asarray(a0, dtype=float), "P0": np.asarray(P0, dtype=float), "P0inf": np.asarray(P0inf, dtype=float), "y": np.asarray(cap["r_tr"], dtype=float) * sc,
          "uncon": np.asarray(res.model.untransform_params(res.params), dtype=float), "k_states": int(s2.k_states)}
    return {
        "model": model, "config": json.loads(cfg.model_dump_json()), "config_file": config, "params": params, "scale": sc,
        "fold": {"train": [str(fold.train[0].date()), str(fold.train[-1].date())], "test": [str(fold.test[0].date()), str(fold.test[-1].date())], "n_folds": len(folds)},
        "level": ds.prices["close"], "r_all": r_all, "r_tr": cap["r_tr"], "tr_idx": [str(d.date()) for d in tr_idx], "y": ds.y,
        "param_names": rec.get("param_names", list(res.model.param_names)), "start": rec["start"], "iters": rec["iters"],
        "est": np.asarray(res.params, dtype=float), "bse": np.asarray(res.bse, dtype=float), "llf": float(res.llf), "aic": float(res.aic), "bic": float(res.bic),
        "nobs": int(res.nobs), "burn": int(res.loglikelihood_burn), "mle_retvals": {k_: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k_, v in (res.mle_retvals or {}).items()},
        "v": np.asarray(fr.forecasts_error[0], dtype=float), "F": np.asarray(fr.forecasts_error_cov[0, 0], dtype=float), "yhat_in": np.asarray(fr.forecasts[0], dtype=float),
        "ar_poly": np.asarray(res.polynomial_ar, dtype=float), "ma_poly": np.asarray(res.polynomial_ma, dtype=float),
        "ar_roots": np.asarray(res.arroots), "ma_roots": np.asarray(res.maroots),
        "apply": {"date": str(t_dec.date()), "n_hist": int(len(hist)), "hist_from": str(hist.index[0].date()), "v": np.asarray(app.filter_results.forecasts_error[0], dtype=float),
                  "yhat": np.asarray(app.filter_results.forecasts[0], dtype=float), "F": np.asarray(app.filter_results.forecasts_error_cov[0, 0], dtype=float), "forecast": np.asarray(app.forecast(k), dtype=float), "hist": hist,
                  "pred": float(cap["pred"].loc[t_dec]), "actual": float(ds.y.get(t_dec, np.nan)), "level": float(ds.prices["close"].loc[t_dec])},
        "all_folds": allp, "all_folds_scale1": cmp1, "ssm": mats, "kf": kf,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="arima", choices=["arima", "sarima"])
    ap.add_argument("--config")
    ap.add_argument("--out")
    a = ap.parse_args()
    cap = capture(a.model, a.config)
    out = Path(a.out or ROOT / f"docs/guide_build/walk/{a.model}_capture.pkl")
    pickle.dump(cap, open(out, "wb"))
    print("저장:", out, "· 반복", len(cap["iters"]), "· 모수", dict(zip(cap["param_names"], np.round(cap["est"], 5))), "· 폴드", len(cap["all_folds"]), "· 예측", cap["apply"]["date"], cap["apply"]["pred"])
