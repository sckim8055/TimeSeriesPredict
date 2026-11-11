"""lag 민감도(lag-shift) 진단 — 추정한 발표시차가 너무 작지(낙관적이지) 않은지 확인한다.

원리: 모든 입력 변수의 lag_days 를 한꺼번에 +Δ일 늘려 같은 실험을 다시 돌린다.
  - lag 가 맞거나 넉넉하면: Δ를 늘려도 성과가 서서히만 나빠진다(정보가 조금 늦어질 뿐).
  - 어떤 변수의 lag 가 너무 작으면(아직 발표 전 값을 쓰고 있었으면): 작은 Δ에서 성과가 **급락**한다 — 누출로 얻은 성과였다는 뜻.
  - 반대로 Δ<0(더 빨리 쓰기)은 실제로 불가능하므로 진단용 참고치일 뿐이다.

결과는 저장하지 않고 실험 카운터도 올리지 않는다(진단이지 가설 검정이 아님). 표만 출력한다.

사용: python tools/lag_shift.py configs/h1_ridge_split2021.json [--shifts -1,0,1,2,5,10,30] [--only SOX,NVDA]
  --only 를 주면 그 변수들만 이동시키고 나머지는 그대로 둔다(변수 하나씩 범인 찾기).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from engine.experiment import ExperimentConfig, run_experiment  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shifts", default="-1,0,1,2,5,10,30")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cfg0 = json.load(open(a.config, encoding="utf-8"))
    reg, _ = Registry.load_master(ROOT / "data" / "variable_master.xlsx")
    dv = DataVersion(ROOT / "data", cfg0["data_version"])
    only = [v.strip() for v in a.only.split(",") if v.strip()]
    rows = []
    for d in [int(x) for x in a.shifts.split(",")]:
        cfg = ExperimentConfig(**cfg0)
        ids, _ = reg.resolve_many(cfg.variable_ids)
        base = {v: cfg.lag_override.get(v, reg.get(v).lag_days) for v in ids}
        cfg.lag_override = {v: max(0, base[v] + (d if (not only or v in only) else 0)) for v in ids}
        cfg.name = f"lagshift_{d:+d}"
        import io, contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            res = run_experiment(cfg, dv, reg, n_trials_so_far=1)
        pm, perf = res["prediction_metrics"], res.get("performance") or {}
        rows.append((d, cfg.lag_override, pm.get("rmse"), pm.get("dir_acc"), pm.get("dm_vs_zero_p"), pm.get("dm_vs_naive_p"), perf.get("sharpe"), perf.get("bootstrap_p_excess")))
        print(f"  Δ={d:+d} 완료", file=sys.stderr, flush=True)
    print(f"\nlag-shift 진단: {a.config}  (변수: {', '.join(base)}; 기준 lag: {base})")
    print("| Δ(일) | RMSE | 방향정확도 | DM p(0예측) | DM p(naive) | 샤프 | 초과수익 p |")
    print("|---|---|---|---|---|---|---|")
    for d, lo, rmse, da, p0, pn, sh, pe in rows:
        f = lambda x, n=4: "–" if x is None else f"{x:.{n}f}"
        print(f"| {d:+d} | {f(rmse,5)} | {f(da,3)} | {f(p0,3)} | {f(pn,3)} | {f(sh,2)} | {f(pe,3)} |")
    print("\n읽는 법: Δ가 커질수록 완만하게 나빠지면 정상. Δ=+1~+5 에서 급락하면 어느 변수의 lag 가 너무 작다(--only 로 하나씩 확인).")


if __name__ == "__main__":
    main()
