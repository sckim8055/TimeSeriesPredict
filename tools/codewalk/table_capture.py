"""표 입력 모델(Ridge·LightGBM) 실제 실행 붙잡기 — 엔진의 실제 경로(prepare → build_dataset → _walkforward_predict → models/*.py)로 기록한다.

  python -m tools.codewalk.table_capture --model ridge            (configs/f11_ridge_macro_k1.json)
  python -m tools.codewalk.table_capture --model lightgbm_model   (configs/12_lgbm_ABCD_M.json)

기록: 데이터셋을 만드는 재료(변수 원자료·변환·발표시차·달력 정렬, 매매 모드면 일간 가격과 월간 리샘플), 마지막 폴드의 학습 표·눈금·
적합된 모델 상태(Ridge 계수 / LightGBM 나무 전체), 결정일 예측, 그리고 모든 폴드의 예측·계수(또는 중요도). 계산은 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import calendar as cal  # noqa: E402
from engine.experiment import ExperimentConfig, _embargo, _walkforward_predict, make_folds, prepare  # noqa: E402
from engine.features import describe_feature  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402

# LightGBM: 결과 목록에 있는 12_lgbm_ABCD_M 을 2026-11-07 엔진으로 다시 돌린 결과의 설정(모델 파라미터 = 월간 기본값). configs/12_lgbm_ABCD_M.json 은 그 뒤
# n_estimators 200·min_child_samples 8 로 바뀌어 결과가 없다 — 영상은 결과 화면과 맞춰 볼 수 있는 쪽을 쓴다.
DEFAULT_CFG = {"ridge": "configs/f11_ridge_macro_k1.json", "lightgbm_model": "results/trading/12_lgbm_ABCD_M_v1107_0b61d18f4d03.json"}


def capture(model: str = "ridge", config: str | None = None, data_root: Path | None = None) -> dict:
    config = config or DEFAULT_CFG[model]
    data_root = data_root or ROOT / "data"
    raw_cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    if "config" in raw_cfg and "prediction_metrics" in raw_cfg:   # 결과 파일을 주면 그 결과를 낸 설정 그대로(결과 화면의 숫자와 맞춰 보기 위해)
        raw_cfg = raw_cfg["config"]
    reg, _ = Registry.load_master(data_root / "variable_master.xlsx")
    cfg = ExperimentConfig(**raw_cfg)
    dv = DataVersion(data_root, cfg.data_version)
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    k = cfg.target.k
    folds, _ = make_folds(cfg, ds, k)
    folds = [f for f in folds if len(f.test)]
    cap: dict = {}
    _walkforward_predict(cfg, plugin, params, ds, "cpu", [folds[-1]], k, target_type, capture=cap)
    m = cap["model"]; feats = list(ds.feature_names)
    var_ids, _ = reg.resolve_many(cfg.variable_ids) if cfg.variable_ids else ([], [])
    wide = dv.wide(var_ids) if var_ids else pd.DataFrame()
    variables = {vid: {"raw": wide[vid].dropna(), "lag": int(ds.lag_days_used[vid]), "transform": ds.transforms_used[vid], "freq": reg.get(vid).freq,
                       "name": reg.get(vid).name, "unit": reg.get(vid).unit, "desc": describe_feature(vid if ds.transforms_used[vid] in ("", "none") else f"{vid}_{ds.transforms_used[vid]}", reg)}
                 for vid in var_ids}
    out = {"model": model, "config_file": config, "config": json.loads(cfg.model_dump_json()), "params": params, "target_type": target_type,
           "mode": cfg.mode, "k": k, "embargo": _embargo(cfg, k), "val_frac": cfg.validation.val_frac, "uses_val": bool(getattr(plugin.Model, "USES_VALIDATION", True)),
           "prices": ds.prices, "X": ds.X, "y": ds.y, "feats": feats, "desc": {c: describe_feature(c, reg) for c in feats}, "variables": variables,
           "n_folds": len(folds), "fold": {"train": list(folds[-1].train), "test": list(folds[-1].test)},
           "tr": cap["tr"], "te_valid": cap["te_valid"], "Xtr": cap["Xtr"], "ytr": cap["ytr"], "Xte": cap["Xte"], "n_tr": cap["n_tr"], "n_va": cap["n_va"],
           "scaler_mean": cap["scaler"].mean_.copy(), "scaler_scale": cap["scaler"].scale_.copy(), "pred": cap["pred"]}
    if cfg.mode == "trading":
        out["daily"] = dv.read_prices(cfg.ticker).loc[cfg.start or None:]
    if model == "ridge":
        sc2, est = m.pipe[0], m.pipe[-1]
        out["ridge"] = {"inner_mean": sc2.mean_.copy(), "inner_scale": sc2.scale_.copy(), "coef": np.ravel(est.coef_).copy(), "intercept": float(np.ravel(est.intercept_)[0])}
    else:
        est = m.est; bst = est.booster_
        Xte = cap["Xte"]
        out["lgbm"] = {"dump": bst.dump_model(), "best_iteration": int(getattr(est, "best_iteration_", 0) or 0), "n_trees": bst.num_trees(),
                       "importance_gain": bst.feature_importance(importance_type="gain").astype(float), "importance_split": bst.feature_importance(importance_type="split").astype(float),
                       "raw_te": bst.predict(Xte, raw_score=True), "contrib_te": bst.predict(Xte, pred_contrib=True),
                       "staged_tr": np.array([bst.predict(cap["Xtr"][: cap["n_tr"]], num_iteration=i) for i in range(1, bst.num_trees() + 1)]),
                       "leaf_tr": bst.predict(cap["Xtr"][: cap["n_tr"]], pred_leaf=True), "evals": getattr(est, "evals_result_", None) or {}}
    # 모든 폴드 — 폴드마다 같은 엔진 경로로 다시 학습해 예측과 계수(또는 중요도)를 모은다
    allf = []
    for f in folds:
        c2: dict = {}
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [f], k, target_type, capture=c2)
        if "model" not in c2:
            continue
        m2 = c2["model"]
        row = {"test": [str(d.date()) for d in c2["te_valid"].index], "pred": c2["pred"].values.astype(float).tolist(), "n_tr": int(c2["n_tr"]), "n_va": int(c2["n_va"])}
        if model == "ridge":
            row["coef"] = np.ravel(m2.pipe[-1].coef_).copy(); row["intercept"] = float(np.ravel(m2.pipe[-1].intercept_)[0])
        else:
            row["gain"] = m2.est.booster_.feature_importance(importance_type="gain").astype(float); row["best"] = int(getattr(m2.est, "best_iteration_", 0) or 0)
        allf.append(row)
    out["all_folds"] = allf
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridge", choices=["ridge", "lightgbm_model"])
    ap.add_argument("--config"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = capture(a.model, a.config)
    out = Path(a.out or ROOT / f"docs/guide_build/walk/{'lgbm' if a.model == 'lightgbm_model' else a.model}_capture.pkl")
    pickle.dump(cap, open(out, "wb"))
    print("저장:", out, "· 변수", len(cap["feats"]), "· 학습행", len(cap["tr"]), "· 폴드", len(cap["all_folds"]), "· 예측", dict(cap["pred"].items()))
