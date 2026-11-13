"""앙상블 — 이미 저장된 예측 연구 결과들의 **표본 외(out-of-sample) 예측**을 시점별로 결합해 같은 지표로 다시 채점한다.

원 노트북 4의 "ARIMA 예측과 LSTM 예측의 평균"을 플랫폼 규약으로 옮긴 것이다.
- 결합 방식: mean(단순 평균) / median(중앙값). 가중치를 데이터로 정하지 않는다 — 가중치를 평가 구간 성적으로 고르면
  그 자체가 평가 구간을 엿본 것(누출)이 되기 때문. 사전에 고정한 동일 가중치만 허용한다.
- 조건: 같은 대상·같은 k·같은 맞히는 방식·같은 정보 시점. 결합은 **모든 구성 모델이 예측을 낸 시점만**(교집합) 쓴다.
- 결과는 일반 결과와 같은 형식으로 저장되고, 실험 카운터도 1 올린다(새로운 비교 대상이므로).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.experiment import forecast_metrics


def make_ensemble(stems: list[str], results_dir: Path, method: str = "mean") -> dict:
    if len(stems) < 2:
        raise ValueError("앙상블에는 결과가 2개 이상 필요합니다")
    if method not in ("mean", "median"):
        raise ValueError("method 는 mean 또는 median")
    rs, ds = [], []
    for st in stems:
        p = results_dir / f"{st}.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["config"].get("mode", "trading") != "forecast":
            raise ValueError(f"{st}: 지금은 예측 연구 결과만 결합할 수 있습니다(매매 결과는 체결·비용 재계산이 필요)")
        d = pd.read_parquet(p.with_name(f"{st}_daily.parquet"))
        rs.append(r); ds.append(d)
    key = lambda r: (r["config"].get("target_variable"), r["config"]["target"]["k"], r["config"]["target"]["kind"], r["config"].get("forecast_info_time", "stamp"))
    k0 = key(rs[0])
    for st, r in zip(stems, rs):
        if key(r) != k0:
            raise ValueError(f"{st}: 대상·k·맞히는 방식·정보 시점이 다릅니다 {key(r)} ≠ {k0}")
    if any(r.get("sealed_evaluation") for r in rs):
        raise ValueError("봉인 평가 결과는 결합하지 않습니다")
    # 계산 방식(engine_rev)이 다른 결과는 섞지 않는다. 앙상블 결과에도 그 값을 적는다 — 2026-11-10 전에는 적지 않아 새 앙상블이 전부 '옛 방식'으로
    # 표시되고 다중비교 가족에서 빠졌다(독립 검토 F4).
    revs = sorted({str(r.get("engine_rev")) for r in rs})
    if len(revs) > 1:
        raise ValueError(f"구성 결과의 계산 방식(engine_rev)이 서로 다릅니다 {revs} — 같은 방식으로 다시 돌린 결과끼리 결합하세요")
    preds = pd.concat([d["prediction"].rename(st) for st, d in zip(stems, ds)], axis=1)
    both = preds.dropna()
    if len(both) < 10:
        raise ValueError(f"모든 모델이 예측한 공통 시점이 {len(both)}개뿐입니다(10개 이상 필요)")
    comb = both.mean(axis=1) if method == "mean" else both.median(axis=1)
    base = ds[0].copy()
    base["prediction"] = comb.reindex(base.index)
    kind, k = k0[2], int(k0[1])
    g = (lambda x: 1 + x) if kind == "return" else (np.exp if kind == "logreturn" else None)
    base["pred_future_level"] = (base["level"] * g(base["prediction"])).where(base["prediction"].notna()) if g else np.nan
    base = base.drop(columns=[c for c in ("fit",) if c in base.columns])
    # 기준 예측(학습 창 평균)은 구성 모델들의 폴드가 같을 때만 뜻이 있다 — 공통 시점에서 모두 같으면 그대로, 아니면 뺀다(평균 기준 CW 는 계산하지 않음)
    if "bench_mean" in base.columns:
        same = all("bench_mean" in d.columns and np.allclose(d["bench_mean"].reindex(both.index).values, ds[0]["bench_mean"].reindex(both.index).values, equal_nan=True) for d in ds)
        base = base.assign(bench_mean=base["bench_mean"].where(base.index.isin(both.index))) if same else base.drop(columns=["bench_mean"])
    target_type = "classification" if kind in ("direction", "event") else "regression"
    pm, ev = forecast_metrics(base, kind, k, target_type)
    # 구성 모델 각각도 같은 공통 시점에서 다시 채점(공정 비교)
    members = {}
    for st, d in zip(stems, ds):
        dd = d.copy()
        dd["prediction"] = d["prediction"].where(d.index.isin(both.index))
        if "pred_future_level" in dd and g is not None:
            dd["pred_future_level"] = (dd["level"] * g(dd["prediction"])).where(dd["prediction"].notna())
        mp, _ = forecast_metrics(dd, kind, k, target_type)
        members[st] = {x: mp.get(x) for x in ("level_mape", "rmse", "dm_vs_naive", "dm_vs_naive_p", "skill_mape_vs_naive")}
    cfg = json.loads(json.dumps(rs[0]["config"]))
    names = [r["model"]["name"] for r in rs]
    cfg["model"] = "ensemble"
    cfg["model_params"] = {"method": method, "members": stems}
    cfg["name"] = f"ens_{method}_" + "+".join(r["config"].get("name") or r["model"]["file"] for r in rs)[:80]
    h = hashlib.sha1(json.dumps([sorted(stems), method]).encode()).hexdigest()[:12]
    return {
        "config": cfg, "config_hash": h,
        "model": {"file": "ensemble", "name": f"앙상블({method}: {' + '.join(names)})", "input_shape": "ensemble", "params": cfg["model_params"]},
        "data_version": rs[0].get("data_version"), "code_version": rs[0].get("code_version"), "engine_rev": rs[0].get("engine_rev"), "n_folds": None,
        "n_features": None, "feature_names": [], "feature_importance": None, "degenerate_model": False, "sealed_evaluation": False,
        "evaluation_period": [str(ev.index[0].date()), str(ev.index[-1].date())], "leak_warning": any(r.get("leak_warning") for r in rs),
        "lag_days_used": {st: r.get("lag_days_used") for st, r in zip(stems, rs)},
        "lag_unconfirmed": sorted({v for r in rs for v in (r.get("lag_unconfirmed") or [])}),
        "transforms_used": None, "prediction_metrics": pm, "performance": {},
        "period": rs[0].get("period"), "prediction_period": [str(ev.index[0].date()), str(ev.index[-1].date())], "n_predictions": int(len(ev)),
        "ensemble_members": members, "device": "cpu", "seconds": 0.0, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "_daily": base,
    }


def save_ensemble(stems: list[str], results_root: Path, method: str = "mean") -> Path:
    """앙상블을 만들고 results/forecast 에 저장, 예측 실험 카운터 +1."""
    from engine.experiment import results_dir, save_result

    rd = results_dir(results_root, "forecast")
    res = make_ensemble(stems, rd, method)
    from engine import ledger, runlog

    # 실행 기록: 앙상블은 새로 학습하지 않으므로 구성원과 그 실행 번호를 남긴다(2026-11-10 — 전에는 기록이 없었다)
    members = {}
    for st in stems:
        try:
            m = json.loads((rd / f"{st}.json").read_text(encoding="utf-8"))
            members[st] = {"config_hash": m.get("config_hash"), "run_id": (m.get("run_record") or {}).get("run_id"),
                           "data_version": m.get("data_version"), "code_version": m.get("code_version"), "master_sha256": ((m.get("run_record") or {}).get("master") or {}).get("sha256")}
        except Exception:  # noqa: BLE001
            members[st] = None
    run_id = runlog.new_run_id()
    res["run_record"] = {"run_id": run_id, "started_utc": runlog.now_utc(), "finished_utc": runlog.now_utc(), "command": sys.argv, "kind": "ensemble",
                         "method": method, "members": members, "libraries": runlog.library_versions(),
                         "notes": (["구성원의 데이터 버전이 서로 다릅니다"] if len({(v or {}).get("data_version") for v in members.values()}) > 1 else [])}
    out = save_result(res, rd)
    ledger.record(rd, "ensemble", config_hash=res.get("config_hash"), name=(res.get("config") or {}).get("name"), model="ensemble", run_id=run_id, result=Path(out).stem, members=stems)
    return out
