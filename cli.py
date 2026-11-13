"""실험 CLI.  python cli.py models | run <config.json> [--data-version ID] [--master 과거사전.xlsx] [--sealed] | ensemble 결과1 결과2 …"""
from __future__ import annotations

import json
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from engine.experiment import ExperimentConfig, list_plugins, run_experiment, save_result  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402
from engine import logs  # noqa: E402
import os  # noqa: E402

log = logs.setup("cli", os.environ.get("TSP_JOB_ID") or f"cli-{pd.Timestamp.now():%Y%m%d_%H%M%S}")

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def latest_version() -> str:
    from engine.store import latest_version_id
    v = latest_version_id(DATA)  # 가장 새 '온전한' 버전(일부를 받지 못한 버전은 자동으로 고르지 않는다)
    if not v:
        raise SystemExit("데이터 버전이 없습니다. 먼저 collect.py 를 실행하세요.")
    return v


def cmd_models():
    for m in list_plugins():
        print(f"{m['file']:18s} {m['name']:12s} {m['input_shape']:7s} {m['target_types']}  params={list(m['params'])}")


def cmd_run(path: str, version: str | None, sealed: bool = False, master: str | None = None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    log.info("실험 시작: 설정 파일 %s · 봉인 평가 %s", path, sealed)
    log.debug("받은 설정(JSON 그대로): %s", logs.describe(raw))
    cfg = ExperimentConfig(**raw)
    log.debug("검증된 설정(기본값 채움, 해시 %s): %s", cfg.config_hash(), logs.describe(cfg.model_dump()))
    vid = version or cfg.data_version or latest_version()
    if not (DATA / "versions" / vid / "manifest.json").exists():  # 오타 id 로 빈 버전 폴더가 생기던 것 방지(2026-11-10)
        raise SystemExit(f"데이터 버전 {vid} 가 없습니다(data/versions 아래 manifest.json 이 있는 폴더만 쓸 수 있습니다)")
    dv = DataVersion(DATA, vid)
    log.info("데이터 버전 %s", vid)
    # 변수 사전이 유일한 기준. --master 로 과거 사본(data/master_history/<sha>.xlsx)을 주면 그때의 정의로 다시 돌릴 수 있다(재현용, 2026-11-10)
    mpath = Path(master) if master else DATA / "variable_master.xlsx"
    if master and not mpath.exists():
        raise SystemExit(f"변수 사전 {mpath} 가 없습니다")
    if mpath.exists():
        reg, warns = Registry.load_master(mpath)
        for w in warns:
            print("  [마스터]", w)
    else:
        reg = Registry.load_csv(DATA / "registry.csv")
    from engine.experiment import results_dir
    from engine import ledger, runlog
    from engine.sealed import read_log, seal_exposures, write_log
    RES = results_dir(RESULTS, cfg.mode)
    RES.mkdir(parents=True, exist_ok=True)
    ledger.ensure(RES)  # 시도 원장(없으면 옛 카운터 값으로 시작)
    n_dev = ledger.count(RES)
    led_before = ledger.head(RES)  # 시작할 때의 원장 머리 — 결과(run_record.ledger_before)에 적어 두어 원장의 끝 줄이 지워지면 드러나게 한다
    # 봉인 평가는 '개발 시도'가 아니다: 원장에 sealed 로 적되 N 에 넣지 않고, 과최적화 보정에는 지금까지의 개발 시도 수를 쓴다
    n = max(1, n_dev) if sealed else n_dev + 1
    from engine.experiment import default_device
    print("device:", default_device())
    from engine.experiment import trial_sharpe_variance
    ppy = {"D": 252, "W": 52, "M": 12}[cfg.frequency]
    run_id = runlog.new_run_id()
    os.environ["TSP_RUN_ID"] = run_id  # 결과의 run_record.run_id 와 run_log.jsonl 의 줄을 잇는 값
    job_id = os.environ.get("TSP_JOB_ID")
    runlog.append_log(RES, {"event": "start", "run_id": run_id, "job_id": job_id, "config_file": str(path),
                            "config_hash_given": cfg.config_hash(), "config": raw, "data_version": vid, "sealed": bool(sealed),
                            "master_sha256": reg.source_sha256 or None, "master_path": str(mpath) if master else None})
    sealed_log = RES / "sealed_log.json"
    if sealed:
        # 시작할 때 '예약' 줄을 먼저 쓴다 — 전에는 끝난 뒤에만 써서, 연달아 두 번 실행하면 둘 다 통과했다(2026-11-10 점검 S4)
        hist = read_log(sealed_log)
        from engine import sealpolicy
        from engine.experiment import fill_defaults
        cfg_eff = fill_defaults(sealpolicy.apply(cfg, sealed=True)[0], reg)  # 봉인 정책·기본값을 채운 '실제로 돌 설정'
        h_res, sf_eff = cfg_eff.config_hash(), cfg_eff.validation.sealed_from  # 결과 파일과 같은 해시(창 길이 기본값을 채운 설정) — 중간에 죽어도 같은 설정의 개봉으로 잡힌다
        hist.append({"config": cfg.name, "hash": h_res, "when": pd.Timestamp.utcnow().isoformat(), "status": "started", "run_id": run_id,
                     "job_id": job_id, "source_result": os.environ.get("TSP_SEALED_SOURCE") or None, "note": os.environ.get("TSP_SEALED_NOTE") or "",
                     "model": cfg.model, "mode": cfg.mode, "sealed_from": sf_eff})
        write_log(sealed_log, hist)
    try:
        res = run_experiment(cfg, dv, reg, n_trials_so_far=n, sealed=sealed, trial_sr_var=trial_sharpe_variance(RES, ppy))
    except BaseException as e:  # 실패·중단도 기록(사중 관리). 오류는 그대로 올린다
        runlog.append_log(RES, {"event": "failed", "run_id": run_id, "job_id": job_id, "error": f"{type(e).__name__}: {e}"[:2000]})
        if sealed:  # 예약 줄을 '실패'로 — 봉인 수치를 보기 전에 멈췄으므로 개봉으로 세지 않되 기록은 남긴다
            hist = read_log(sealed_log)
            for x in hist:
                if x.get("run_id") == run_id:
                    x["status"] = "failed"; x["error"] = f"{type(e).__name__}: {e}"[:500]
            write_log(sealed_log, hist)
        raise
    # 저장·기록 단계의 실패도 실행 기록에 남긴다(전에는 run_experiment 안의 실패만 남았다 — 2026-11-10 독립 검토 F17).
    # 봉인 평가는 이 단계가 끝나기 전까지 수치를 어디에도 보여 주지 않으므로, 여기서 실패하면 '실패'(개봉으로 세지 않음)가 맞다.
    try:
        res.setdefault("run_record", {})["ledger_before"] = led_before
        if sealed:
            hist = read_log(sealed_log)
            h = res["config_hash"]  # 실제로 돈 설정(창 길이 등 기본값을 채운 것)의 해시 — 결과 파일·미리보기와 같은 값
            done = [e for e in hist if e.get("run_id") != run_id and e.get("status", "done") != "failed"]  # 실패한 시도·자기 예약 줄은 개봉 횟수에서 뺀다
            prev_same = sum(1 for e in done if e.get("hash") == h)
            sf_eff = (res["config"].get("validation") or {}).get("sealed_from")  # 실제로 쓴 봉인 시작일(정책이 채웠을 수 있다)
            prev_seal = sum(1 for e in done if e.get("sealed_from") == sf_eff)
            res["sealed_info"] = {
                "opened_utc": pd.Timestamp.utcnow().isoformat(), "job_id": job_id,
                "source_result": os.environ.get("TSP_SEALED_SOURCE") or None, "note": os.environ.get("TSP_SEALED_NOTE") or "",
                "opening_no_same_config": prev_same + 1, "opening_no_same_seal_date": prev_seal + 1, "opening_no_mode": len(done) + 1,
                "dev_trials": n,
                # 열기 전에 이 봉인 구간을 이미 채점한 개발 결과(같은 종목 + 거의 같은 종목, 2026-11-13 독립 검토 2) — 미리보기 경고와 같은 함수
                "exposures": seal_exposures(RES, res["config"], dv),
            }
            out = save_result(res, RES)
            pm, pf = res.get("prediction_metrics") or {}, res.get("performance") or {}
            entry = {"config": cfg.name, "hash": h, "when": res["sealed_info"]["opened_utc"], "status": "done", "run_id": run_id, "result": Path(out).stem,
                     "source_result": res["sealed_info"]["source_result"], "note": res["sealed_info"]["note"], "job_id": res["sealed_info"]["job_id"],
                     "model": cfg.model, "mode": cfg.mode, "sealed_from": sf_eff,
                     "period": (res.get("periods") or {}).get("test") or res.get("evaluation_period"), "n_folds": res.get("n_folds"),
                     "data_version": res.get("data_version"), "code_version": res.get("code_version"),
                     "metrics": {k: pm.get(k) for k in ("rmse", "rmse_naive", "dm_vs_naive_p", "cw_vs_zero_p", "dir_acc", "auc") if pm.get(k) is not None}
                                | {k: pf.get(k) for k in ("sharpe", "cagr", "mdd", "benchmark_cagr", "bootstrap_p_excess", "deflated_sharpe_p") if pf.get(k) is not None}}
            hist = [entry if e.get("run_id") == run_id else e for e in hist] if any(e.get("run_id") == run_id for e in hist) else hist + [entry]
            write_log(sealed_log, hist)
            ledger.record(RES, "sealed", config_hash=h, name=cfg.name, model=cfg.model, run_id=run_id, result=Path(out).stem)
            print(f"  [봉인 평가] 이 설정(해시 {h})의 봉인 평가 {prev_same + 1}회째 · 같은 봉인 시작일 {prev_seal + 1}회째 — 같은 설정을 2회 이상 열면 봉인의 의미가 없습니다")
            print(f"  [봉인 평가] 개발 시도 수는 올리지 않음(개발 시도 {n}회 기준으로 보정)")
        else:
            out = save_result(res, RES)
            # 원장에 한 줄(끝난 뒤 — 평가를 본 시도만 센다). 저장 시점의 원장으로 세므로 두 실험이 겹쳐도 횟수가 줄지 않는다
            n = ledger.record(RES, "trial", config_hash=res["config_hash"], name=cfg.name, model=cfg.model, run_id=run_id, result=Path(out).stem)
            sp = res.get("seal_policy") or {}
            if sp.get("status") == "override":  # 봉인 구간을 일부러 본 실행 — 원장에 이유와 함께 따로 한 줄(개발 시도로는 위에서 이미 셌다)
                ledger.record(RES, "seal_override", config_hash=res["config_hash"], name=cfg.name, run_id=run_id, result=Path(out).stem,
                              target=sp.get("target"), policy_date=sp.get("policy_date"), reason=sp.get("reason"))
    except BaseException as e:
        runlog.append_log(RES, {"event": "failed", "run_id": run_id, "job_id": job_id, "stage": "save", "error": f"{type(e).__name__}: {e}"[:2000]})
        if sealed:
            hist = read_log(sealed_log)
            for x in hist:
                if x.get("run_id") == run_id and x.get("status") == "started":
                    x["status"] = "failed"; x["error"] = f"저장 단계: {type(e).__name__}: {e}"[:500]
            write_log(sealed_log, hist)
        raise
    log.info("결과 저장: %s (실험 카운터 %d%s)", out, n, ", 봉인 평가" if sealed else "")
    log.debug("결과 요약: %s", logs.describe({"prediction_metrics": res.get("prediction_metrics"), "periods": res.get("periods"), "n_folds": res.get("n_folds")}))
    p = res["performance"]
    print(json.dumps({"saved": str(out), "model": res["model"]["name"], "device": res["device"], "folds": res["n_folds"], "pred": res["prediction_metrics"],
                      "sharpe": round(p.get("sharpe", float("nan")), 3), "cagr": round(p.get("cagr", float("nan")), 4), "mdd": round(p.get("mdd", float("nan")), 4),
                      "bh_cagr": round(p.get("benchmark_cagr", float("nan")), 4), "bh_mdd": round(p.get("benchmark_mdd", float("nan")), 4),
                      "p_excess": p.get("bootstrap_p_excess"), "trials": n}, ensure_ascii=False, indent=1, default=float))


def _main_with_log():
    try:
        _main()
    except SystemExit:
        raise
    except Exception:
        log.exception("실행 실패 — 위의 ▶ 단계 중 마지막으로 시작하고 ◀ 로 끝나지 않은 단계가 오류 위치")
        raise


def _main():
    if len(sys.argv) >= 2 and sys.argv[1] == "ensemble":  # python cli.py ensemble 결과1 결과2 [...] [--median]
        from engine.ensemble import save_ensemble

        stems = [a.removesuffix(".json").split("/")[-1].split("\\")[-1] for a in sys.argv[2:] if not a.startswith("--")]
        out = save_ensemble(stems, RESULTS, "median" if "--median" in sys.argv else "mean")
        r = json.loads(Path(out).read_text(encoding="utf-8"))
        print(json.dumps({"saved": str(out), "model": r["model"]["name"], "pred": r["prediction_metrics"], "members": r["ensemble_members"]}, ensure_ascii=False, indent=1, default=float))
        sys.exit(0)
    if len(sys.argv) < 2 or sys.argv[1] not in ("models", "run"):
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "models":
        cmd_models()
    else:
        ver = sys.argv[sys.argv.index("--data-version") + 1] if "--data-version" in sys.argv else None
        mst = sys.argv[sys.argv.index("--master") + 1] if "--master" in sys.argv else None
        if "--past-seal" in sys.argv:  # 봉인 정책을 일부러 넘긴다(재현 실험 등). 이유가 결과와 시도 원장에 남는다 — engine/sealpolicy.py
            i_ = sys.argv.index("--past-seal")
            why = sys.argv[i_ + 1].strip() if i_ + 1 < len(sys.argv) and not sys.argv[i_ + 1].startswith("--") else ""
            if not why:
                raise SystemExit('--past-seal 뒤에 이유를 적어야 합니다. 예: --past-seal "원 논문의 분할 재현"')
            os.environ["TSP_SEAL_OVERRIDE"] = why
        cmd_run(sys.argv[2], ver, sealed="--sealed" in sys.argv, master=mst)


if __name__ == "__main__":
    _main_with_log()
