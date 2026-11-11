"""규칙 모델 + 백테스트 실제 실행 붙잡기 — 엔진의 실제 실행(_run_experiment)을 그대로 돌리고, 그 안의 단계(신호·평가 구간·체결·성적표·검정)를
같은 엔진 함수로 다시 불러 중간값을 기록한다.

  python -m tools.codewalk.rule_capture [--result results/trading/02_ma_cross_v1107_0a5e0dc0fcd8.json]

결과 파일을 주면 그 결과를 낸 설정·데이터 버전·실험 수(n_trials_so_far)·시도 분산 그대로 돌린다 — 결과 화면의 숫자와 맞춰 보기 위해.
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

from engine.experiment import ExperimentConfig, _eval_window, _run_experiment, load_plugin, prepare  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402

OTHERS = {"buy_and_hold": "results/trading/01_buyhold_v1107_803772f51e88.json", "momentum": "results/trading/03_momentum_v1107_76dcb718edeb.json",
          "rsi_rule": None, "sox_gap_fade": "configs/r1_soxfade_q90.json"}


def _cfg(path: str) -> tuple[ExperimentConfig, dict | None]:
    raw = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if "config" in raw and "performance" in raw:
        return ExperimentConfig(**raw["config"]), raw
    return ExperimentConfig(**raw), None


def capture(result: str = "results/trading/02_ma_cross_v1107_0a5e0dc0fcd8.json", data_root: Path | None = None) -> dict:
    data_root = data_root or ROOT / "data"
    reg, _ = Registry.load_master(data_root / "variable_master.xlsx")
    cfg, res = _cfg(result)
    perf0 = res["performance"]
    dv = DataVersion(data_root, res["data_version"])
    out_run = _run_experiment(cfg, dv, reg, n_trials_so_far=int(perf0["n_trials_so_far"]), device="cpu", trial_sr_var=perf0.get("trial_sr_var"))
    cfg2, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    pos_raw = pd.Series(plugin.Model(params).positions(ds.prices), index=ds.prices.index)
    lo, hi = _eval_window(cfg2, ds, cfg2.target.k)
    out = {"result_file": result, "config": json.loads(cfg.model_dump_json()), "params": params, "data_version": dv.id, "prices": ds.prices, "pos_raw": pos_raw,
           "eval": [lo, hi], "sealed_from": cfg.validation.sealed_from, "run": {k_: v_ for k_, v_ in out_run.items() if k_ != "_daily"}, "daily_full": out_run["_daily"],
           "result": {"performance": perf0, "evaluation_period": res.get("evaluation_period")}}
    # 다른 규칙들 — 같은 가격으로 결정일 하나의 포지션을 손으로 확인하기 위해
    others = {}
    for name, src in OTHERS.items():
        try:
            if src:
                c, r_ = _cfg(src)
                dvo = DataVersion(data_root, r_["data_version"]) if r_ else DataVersion(data_root, c.data_version)
            else:
                c, dvo = cfg.model_copy(update={"model": name, "model_params": {}}), dv
            c2, pl, prm, _, ds2 = prepare(c, dvo, reg, device="cpu")
            m = pl.Model(prm)
            pos = pd.Series(m.positions(ds2.prices, ds2.X) if getattr(m, "USES_VARIABLES", False) else m.positions(ds2.prices), index=ds2.prices.index)
            others[name] = {"params": prm, "pos": pos, "prices": ds2.prices, "X": ds2.X if getattr(m, "USES_VARIABLES", False) else None, "source": src or "(같은 설정, 모델만 RSI)",
                            "data_version": dvo.id}
        except Exception as e:  # noqa: BLE001
            others[name] = {"error": f"{type(e).__name__}: {e}"}
    out["others"] = others
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="results/trading/02_ma_cross_v1107_0a5e0dc0fcd8.json"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = capture(a.result)
    out = Path(a.out or ROOT / "docs/guide_build/walk/rule_capture.pkl")
    pickle.dump(cap, open(out, "wb"))
    print("저장:", out, "· 평가", cap["run"]["evaluation_period"], "· 샤프", cap["run"]["performance"]["sharpe"], "· 결과 파일 샤프", cap["result"]["performance"]["sharpe"],
          "· 다른 규칙", {k: ("오류 " + v["error"]) if "error" in v else "ok" for k, v in cap["others"].items()})
