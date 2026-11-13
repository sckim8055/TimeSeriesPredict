"""그룹 단위 ablation — "이 변수 그룹이 실제로 정보를 갖는가"를 같은 잣대로 비교한다.

사용:
  python ablate.py configs/12_lgbm_ABCD_M.json                 # 기준 설정에서 그룹 하나씩 빼기(leave-one-out)
  python ablate.py configs/12_lgbm_ABCD_M.json --mode add      # 기준에 없는 그룹 하나씩 더하기
  python ablate.py configs/12_lgbm_ABCD_M.json --mode both
  python ablate.py ... --groups "금리·유동성,반도체 글로벌"       # 특정 그룹만
  python ablate.py ... --technical                             # 기술지표(A) 블록도 빼기 대상에 포함

그룹은 variable_master.xlsx 의 group 열. 기준 설정의 variable_ids 가 속한 그룹이 '기준 그룹'.
각 변형은 run_experiment 로 똑같이 실행되고(카운터에 모두 기록), 기준 대비:
  - ΔRMSE, Δ방향정확도, Δ샤프
  - Diebold–Mariano(변형 vs 기준, 같은 예측 구간): 통계량 < 0 이면 변형이 낫다. p 는 양측.
결과: results/ablation_<기준이름>_<타임스탬프>.md / .csv
해석 주의: 빼서 나빠지면(DM>0, p 작음) 그 그룹에 정보가 있다는 뜻. 빼서 좋아지면 그 그룹이 잡음(과적합 유발)이다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.backtest import diebold_mariano  # noqa: E402
from engine.experiment import ExperimentConfig, default_device, run_experiment, save_result, trial_sharpe_variance  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results" / "trading"  # ablation 은 매매 모드 기준 (예측 모드는 추후)


def _variant(cfg: ExperimentConfig, **upd) -> ExperimentConfig:
    """설정의 변형을 '검증을 다시 거쳐' 만든다. model_copy(update=…) 는 검증·정규화를 건너뛰어, 기술지표를 끈 변형에 기술지표 선택 목록이 남아
    저장된 해시와 다시 계산한 해시가 달랐다(2026-11-10 독립 검토 F18)."""
    return ExperimentConfig(**{**cfg.model_dump(mode="json"), **upd})


def _counter() -> int:
    from engine import ledger
    return ledger.count(RESULTS)  # 시도 원장 기준(2026-11-10 — 전에는 카운터 파일을 직접 고쳐 썼다)


def run_one(cfg: ExperimentConfig, dv: DataVersion, reg: Registry) -> dict:
    n = _counter() + 1
    ppy = {"D": 252, "W": 52, "M": 12}[cfg.frequency]
    from engine import runlog
    import os
    run_id = runlog.new_run_id()
    os.environ["TSP_RUN_ID"] = run_id
    runlog.append_log(RESULTS, {"event": "start", "run_id": run_id, "via": "ablate", "config_hash_given": cfg.config_hash(),
                                "config": cfg.model_dump(mode="json"), "data_version": dv.id})
    from engine import ledger
    ledger.ensure(RESULTS)
    led_before = ledger.head(RESULTS)  # 시작할 때의 원장 머리 — 결과에 적어 두어 원장의 끝 줄이 지워지면 드러나게 한다
    try:
        res = run_experiment(cfg, dv, reg, n_trials_so_far=n, trial_sr_var=trial_sharpe_variance(RESULTS, ppy))
        res.setdefault("run_record", {})["ledger_before"] = led_before
        out = save_result(res, RESULTS)
    except BaseException as e:
        runlog.append_log(RESULTS, {"event": "failed", "run_id": run_id, "error": f"{type(e).__name__}: {e}"[:2000]})
        raise
    ledger.record(RESULTS, "ablation", config_hash=res["config_hash"], name=cfg.name, model=cfg.model, run_id=run_id, result=Path(out).stem)
    return res


def compare(base: dict, var: dict, k: int) -> dict:
    """같은 예측 구간에서 변형 vs 기준."""
    b = base["_daily"][["prediction", "target"]].dropna()
    v = var["_daily"][["prediction", "target"]].dropna()
    idx = b.index.intersection(v.index)
    out = {}
    if len(idx) > 10:
        eb = (b.loc[idx, "prediction"] - b.loc[idx, "target"]).values
        ev = (v.loc[idx, "prediction"] - v.loc[idx, "target"]).values
        dm, p = diebold_mariano(ev, eb, h=k)
        out.update(dm_vs_base=round(float(dm), 3), dm_p=round(float(p), 4), n_common=int(len(idx)))
    pm_b, pm_v = base["prediction_metrics"], var["prediction_metrics"]
    pf_b, pf_v = base["performance"], var["performance"]
    out.update(
        rmse=pm_v.get("rmse"), d_rmse=_delta(pm_v.get("rmse"), pm_b.get("rmse")),
        dir_acc=pm_v.get("dir_acc"), d_dir_acc=_delta(pm_v.get("dir_acc"), pm_b.get("dir_acc")),
        sharpe=pf_v.get("sharpe"), d_sharpe=_delta(pf_v.get("sharpe"), pf_b.get("sharpe")),
        cagr=pf_v.get("cagr"), mdd=pf_v.get("mdd"), n_features=var["n_features"], seconds=var["seconds"],
    )
    return out


def _delta(a, b):
    return None if a is None or b is None else round(float(a) - float(b), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", choices=["drop", "add", "both"], default="drop")
    ap.add_argument("--groups", default=None, help="쉼표 구분. 비우면 전부")
    ap.add_argument("--technical", action="store_true", help="기술지표 블록(A)도 drop 대상에 포함")
    ap.add_argument("--version", default=None)
    a = ap.parse_args()

    base_cfg = ExperimentConfig(**json.loads(Path(a.config).read_text(encoding="utf-8")))
    dv = DataVersion(DATA, a.version or base_cfg.data_version)
    reg, warns = Registry.load_master(DATA / "variable_master.xlsx")
    for w in warns:
        print("  [마스터]", w)
    print("device:", default_device())

    available = set(dv.read_variables()["variable_id"].unique())
    base_ids, _ = reg.resolve_many(base_cfg.variable_ids)
    # 변형은 대표 id 로 만든다. 그래서 중복 id 에 적은 변환·lag 재정의도 대표 id 로 옮겨 둔다 — 안 그러면 기준 실험은 재정의를 쓰고
    # 변형은 사전 기본값으로 돌아가 '변수 제거 효과'에 재정의 차이가 섞인다(2026-11-10 점검 N10)
    vbase = _variant(base_cfg, variable_transforms=reg.remap_overrides(base_cfg.variable_ids, base_cfg.variable_transforms)[0],
                     lag_override=reg.remap_overrides(base_cfg.variable_ids, base_cfg.lag_override)[0])
    group_of = {vid: reg.get(vid).group for vid in reg.ids()}
    base_groups = sorted({group_of[v] for v in base_ids})
    all_groups = sorted({g for g in reg.groups() if g})
    only = {g.strip() for g in a.groups.split(",")} if a.groups else None
    variants: list[tuple[str, ExperimentConfig]] = []
    if a.mode in ("drop", "both"):
        for g in base_groups:
            if only and g not in only:
                continue
            ids = [v for v in base_ids if group_of[v] != g]
            variants.append((f"-{g}", _variant(vbase, variable_ids=ids, name=f"{base_cfg.name}_minus_{g}")))
        if a.technical and base_cfg.use_technical:
            variants.append(("-기술지표(A)", _variant(vbase, use_technical=False, name=f"{base_cfg.name}_minus_A")))
    if a.mode in ("add", "both"):
        for g in all_groups:
            if g in base_groups or (only and g not in only):
                continue
            cand = [v for v in reg.ids(group=g) if v not in base_ids]
            extra = [v for v in cand if v in available]
            if len(extra) < len(cand):
                print(f"  [+{g}] 데이터 없는 변수 제외: {[v for v in cand if v not in available]}")
            if not extra:
                continue
            variants.append((f"+{g}", _variant(vbase, variable_ids=base_ids + extra, name=f"{base_cfg.name}_plus_{g}")))
    if not variants:
        print("변형이 없습니다 (기준 그룹:", base_groups, ")")
        return 1
    print(f"기준: {base_cfg.name} 그룹 {base_groups} 변수 {len(base_ids)}개 → 변형 {len(variants)}개 + 기준 1회")

    print("\n[기준]")
    base = run_one(base_cfg, dv, reg)
    rows = [{"변형": "기준", **compare(base, base, base_cfg.target.k), "그룹": ", ".join(base_groups)}]
    for label, cfg in variants:
        print(f"\n[{label}] 변수 {len(cfg.variable_ids)}개")
        res = run_one(cfg, dv, reg)
        rows.append({"변형": label, **compare(base, res, cfg.target.k)})

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    stem = RESULTS / f"ablation_{base_cfg.name}_{stamp}"
    df.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    lines = [f"# Ablation — 기준 {base_cfg.name} ({base_cfg.model}, {base_cfg.frequency}, k={base_cfg.target.k})", "",
             f"- 데이터 버전 {dv.id}, 기준 그룹 {base_groups}, 실험 카운터 {_counter()}",
             f"- lag 미확인 변수: {base.get('lag_unconfirmed') or '없음'}",
             "- DM<0 & p<0.05: 변형이 기준보다 유의하게 낫다. DM>0 & p<0.05: 뺀(또는 더한) 그룹이 정보를 가진다(뺀 경우) / 잡음이다(더한 경우).", "",
             "| 변형 | 변수수 | RMSE | ΔRMSE | 방향정확도 | Δ방향 | 샤프 | Δ샤프 | CAGR | MDD | DM vs 기준 | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x, fmt: ("" if x is None or pd.isna(x) else format(x, fmt))
        lines.append(f"| {r['변형']} | {r['n_features']} | {f(r['rmse'], '.4f')} | {f(r['d_rmse'], '+.4f')} | {f(r['dir_acc'], '.1%')} | {f(r['d_dir_acc'], '+.1%')} | {f(r['sharpe'], '.3f')} | {f(r['d_sharpe'], '+.3f')} | {f(r['cagr'], '.1%')} | {f(r['mdd'], '.1%')} | {f(r.get('dm_vs_base'), '.2f')} | {f(r.get('dm_p'), '.3f')} |")
    Path(f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n저장: {stem}.md / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
