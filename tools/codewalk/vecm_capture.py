"""VECM 실제 실행 붙잡기 — 엔진의 실제 경로(prepare → build_dataset → _walkforward_predict 'levels' 분기 → models/vecm.py → statsmodels VECM)로 기록한다.

  python -m tools.codewalk.vecm_capture [--config configs/f11_vecm_k1.json]

기록: 수준값 행렬(대상 + 변수, 발표시차 적용·변환 없음), 마지막 폴드의 학습 행렬·적합된 α·β·Γ·진단, 결정일 예측,
Johansen 검정표(정보용), 그리고 모든 폴드의 예측·α·β·차수. 계산은 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.experiment import ExperimentConfig, _walkforward_predict, make_folds, prepare  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402


def capture(config: str = "configs/f11_vecm_k1.json", data_root: Path | None = None) -> dict:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
    warnings.filterwarnings("ignore")
    data_root = data_root or ROOT / "data"
    raw_cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    if "config" in raw_cfg and "prediction_metrics" in raw_cfg:
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
    m = cap["model"]
    var_ids, _ = reg.resolve_many(cfg.variable_ids)
    wide = dv.wide(var_ids)
    Z = m._to_model(cap["Ltr"].values[:, m.keep])
    joh = coint_johansen(Z, det_order=-1 if m.det == "n" else 0, k_ar_diff=m.p)
    out = {"config_file": config, "config": json.loads(cfg.model_dump_json()), "params": params, "k": k, "n_folds": len(folds),
           "fold": {"train": list(folds[-1].train), "test": list(folds[-1].test)},
           "Lall": cap["Lall"], "Ltr": cap["Ltr"], "Ld": cap["Ld"], "r0": cap["r0"], "need": cap["need"], "skip": cap["skip"], "pred": cap["pred"], "y": ds.y,
           "variables": {vid: {"raw": wide[vid].dropna(), "lag": int(ds.lag_days_used[vid]), "name": reg.get(vid).name, "unit": reg.get(vid).unit, "freq": reg.get(vid).freq,
                               "transform_ignored": ds.transforms_used[vid]} for vid in var_ids},
           "target_name": reg.get(cfg.target_variable).name,
           "model": {"alpha": m.alpha, "beta": m.beta, "gamma": m.gamma, "rank": m.rank, "p": m.p, "keep": m.keep, "log": m._log, "det": m.det, "seasons": m.seasons,
                     "c_in": m.c_in, "det_out": m.det_out, "diag": m.diag},
           "johansen": {"eig": joh.eig, "trace": joh.lr1, "trace_cv": joh.cvt, "maxeig": joh.lr2, "maxeig_cv": joh.cvm}}
    allf = []
    for f in folds:
        c2: dict = {}
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [f], k, target_type, capture=c2)
        if "model" not in c2:
            continue
        m2 = c2["model"]
        Z2 = m2._to_model(c2["Ltr"].values[:, m2.keep])
        try:
            auto = int(select_coint_rank(Z2, det_order=-1 if m2.det == "n" else 0, k_ar_diff=m2.p, method="trace", signif=0.05).rank)
        except Exception:  # noqa: BLE001
            auto = None
        allf.append({"test": [str(d.date()) for d in c2["pred"].index], "pred": c2["pred"].values.astype(float).tolist(), "alpha": m2.alpha.copy(), "beta": m2.beta.copy(),
                     "rank": m2.rank, "p": m2.p, "keep": m2.keep.tolist(), "auto_rank": auto})
    out["all_folds"] = allf
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/f11_vecm_k1.json"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = capture(a.config)
    out = Path(a.out or ROOT / "docs/guide_build/walk/vecm_capture.pkl")
    pickle.dump(cap, open(out, "wb"))
    print("저장:", out, "· 학습행", len(cap["Ltr"]), "· 차수", cap["model"]["rank"], "· p", cap["model"]["p"], "· 폴드", len(cap["all_folds"]), "· 예측", dict(cap["pred"].items()))
