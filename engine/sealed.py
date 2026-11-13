"""봉인 평가 미리보기·실행 준비(화면의 '봉인 평가' 버튼이 쓴다. 실제 실행은 cli.py run --sealed 와 같은 경로).

봉인(sealed) = 개발하는 동안 한 번도 보지 않고 남겨 둔 마지막 기간(봉인 시작일 ~ 데이터 끝). 후보 하나를 정한 뒤 **한 번만** 열어
'개발 구간에서 본 성적이 처음 보는 기간에서도 나오는가'를 확인한다. 같은 후보를 고쳐 가며 여러 번 열면 봉인 구간도 개발에 쓴 셈이 되어 의미가 없다.

이 모듈은
- candidate_config(결과): 개발 결과 파일의 설정을 그대로 가져오되, 기본값이 바뀐 파라미터는 그 결과를 낼 때의 값으로 고정한다(pinned_model_params).
- plan(설정, 데이터): 실행과 같은 함수(engine.experiment.sealed_folds)로 봉인 폴드를 미리 계산해 날짜·폴드 수·학습 구간을 보여 준다.
- preview(...): 위 둘 + 봉인 기록(같은 설정·같은 봉인 시작일로 몇 번 열었나) + 경고·막는 이유.
계산 결과는 돌려주기만 하고 아무것도 저장하지 않는다(기록은 실행이 끝난 뒤 cli 가 sealed_log.json 에 남긴다).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import engine.calendar as cal
from engine.experiment import ExperimentConfig, _code_version, _embargo, pinned_model_params, prepare, seal_cut, sealed_folds

CRITERIA = ("사전 등록 성공 기준(docs/DECISIONS.md 2026-11-03): 워크포워드에서 DM(vs 0예측) p<0.05 그리고 봉인 구간에서 방향 일치 "
            "그리고 Deflated Sharpe p<0.05 — 셋 다 통과해야 '신호 있음'. '방향 일치'를 어떤 지표의 부호로 볼지는 기준 문장에 정해져 있지 않으므로, "
            "봉인을 열기 전에 사전 기록 칸에 적어 두세요.")


def candidate_config(r: dict) -> tuple[dict, list[str]]:
    """개발 결과 r 의 설정 → 봉인 평가에 넘길 설정. 바꾸는 것은 model_params(옛 기본값 고정) 하나뿐이다."""
    c = json.loads(json.dumps(r["config"]))
    mp, notes = pinned_model_params(r["model"]["file"], r["model"].get("params"), c.get("model_params"))
    c["model_params"] = mp
    return c, notes


def plan(cfg_dict: dict, dv, reg) -> dict:
    """봉인 폴드 미리 계산(학습 없음). 실행(engine.experiment._predict_sealed)과 같은 sealed_folds 를 쓴다."""
    cfg = ExperimentConfig(**cfg_dict)
    cfg, plugin, params, _, ds = prepare(cfg, dv, reg)
    k = cfg.target.k
    v = cfg.validation
    calendar_all = cal.trading_calendar(ds.prices)
    sf = seal_cut(cfg, calendar_all, k)  # 첫 봉인 결정일(실행과 같은 경계)
    out = {"sealed_from": v.sealed_from, "first_sealed_decision": str(sf.date()) if sf is not None else None, "data_end": str(calendar_all[-1].date()), "data_start": str(calendar_all[0].date()),
           "mode": v.mode, "train_len": v.train_len, "test_len": v.test_len, "embargo": _embargo(cfg, k), "k": k,
           "params_now": params, "input_shape": plugin.INPUT_SHAPE}
    if plugin.INPUT_SHAPE == "rule":
        days = calendar_all[calendar_all >= sf]
        out.update(n_folds=0, first_test=str(days[0].date()) if len(days) else None, last_test=str(days[-1].date()) if len(days) else None,
                   n_days=int(len(days)), folds=[], how="규칙 모델: 학습 없이 봉인 구간의 포지션만 채점")
        return out
    folds = sealed_folds(cfg, ds, k)
    rows = [{"no": i + 1, "train": [str(f.train[0].date()), str(f.train[-1].date())], "n_train": int(len(f.train)),
             "test": [str(f.test[0].date()), str(f.test[-1].date())], "n_test": int(len(f.test))} for i, f in enumerate(folds)]
    n_days = int(sum(len(f.test) for f in folds))
    how = (f"1회 분할 규칙: 봉인 시작 전 전부 − 비움 {out['embargo']}일로 한 번 학습 → 봉인 구간 전체 예측" if v.mode == "split" else
           f"워크포워드 규칙: 학습 창 {v.train_len} · 비움 {out['embargo']} · 재학습 {v.test_len} 마다 — 봉인 첫날부터 창을 밀며 데이터 끝까지")
    out.update(n_folds=len(folds), first_test=rows[0]["test"][0], last_test=rows[-1]["test"][1], n_days=n_days, folds=rows, how=how)
    return out


def history(log: list[dict], cfg_hash: str, sealed_from: str | None) -> dict:
    # status: done(끝남. 옛 기록은 status 가 없고 done 으로 본다) · started(실행 중이거나 강제 종료됨 — 개봉으로 센다) · failed(봉인 수치를 보기 전에 오류 — 세지 않음)
    live = [e for e in log if e.get("status", "done") != "failed"]
    same = [e for e in live if e.get("hash") == cfg_hash]
    seal = [e for e in live if e.get("sealed_from") == sealed_from]
    return {"same_config": same, "same_seal_date": seal, "total": len(live), "failed": len(log) - len(live)}


def seal_exposures(res_dir: str | Path | None, cfg_dict: dict) -> list[dict]:
    """같은 대상의 '개발' 결과 가운데 봉인 구간(봉인 시작일 이후의 결정일)을 이미 채점한 것들. 지금 폴더와 history/ 를 모두 본다.
    봉인 평가 기록(sealed_log)에는 봉인 모드로 연 것만 남는다 — 봉인 시작일을 비우고 돌린 보통 실험, 봉인 정책을 일부러 넘긴 실험은 여기서 찾는다(독립 검토 F8)."""
    sf = (cfg_dict.get("validation") or {}).get("sealed_from")
    if not res_dir or not sf:
        return []
    d = Path(res_dir)
    mode = cfg_dict.get("mode", "trading")
    from engine.sealpolicy import base_ticker
    tgt = cfg_dict.get("target_variable") if mode == "forecast" else base_ticker(cfg_dict.get("ticker"))
    out = []
    files = list(d.glob("*.json")) + (list((d / "history").glob("*.json")) if (d / "history").exists() else [])
    for p in sorted(files):
        if p.name in ("experiment_counter.json", "sealed_log.json") or p.name.startswith("ablation_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            c = r["config"]
        except Exception:  # noqa: BLE001
            continue
        if r.get("sealed_evaluation") or c.get("mode", "trading") != mode or (c.get("target_variable") if mode == "forecast" else base_ticker(c.get("ticker"))) != tgt:
            continue
        # 평가 구간 기록이 없는 옛 매매 결과는 성과를 자료 전체(period)로 계산했다(봉인 구간은 현금으로 두고 보유 전략과 비교) → 자료 끝까지 채점한 것으로 본다
        ev = r.get("evaluation_period") or (r.get("period") if mode == "trading" else None) or r.get("prediction_period")
        if ev and str(ev[1])[:10] >= str(sf)[:10]:
            out.append({"result": p.stem, "where": "history" if p.parent.name == "history" else "results", "evaluated_to": str(ev[1])[:10],
                        "sealed_from_in_config": (c.get("validation") or {}).get("sealed_from"),
                        "override_reason": (r.get("seal_policy") or {}).get("reason")})
    return out


def preview(r: dict, stem: str, dv, reg, log: list[dict], latest_data_version: str | None, res_dir: str | Path | None = None) -> dict:
    """봉인 평가 미리보기. blockers 가 있으면 실행하지 않는다."""
    blockers, warnings = [], []
    if "error" in r:
        blockers.append("이 결과는 실패한 실험입니다")
    if r.get("sealed_evaluation"):
        blockers.append("이 결과 자체가 봉인 평가 결과입니다 — 봉인은 개발 결과(후보)에서 엽니다")
    cfg_dict, pin_notes = candidate_config(r)
    sf = (cfg_dict.get("validation") or {}).get("sealed_from")
    if not sf:
        blockers.append("이 후보의 설정에 봉인 시작일이 없습니다 — 개발 때 봉인 구간을 정하지 않았으므로 '처음 보는 기간'이 없습니다. "
                        "새 실험 화면에서 봉인 시작일을 정하고 개발 실험을 다시 돌린 뒤 그 결과에서 봉인을 여세요")
    try:
        h = ExperimentConfig(**cfg_dict).config_hash()
    except Exception as e:  # noqa: BLE001
        blockers.append(f"설정을 지금 코드로 읽을 수 없습니다: {e}")
        h = r.get("config_hash")
    if not blockers:
        try:  # 봉인 정책(대상별 봉인 시작일)과 맞는지 — 실행과 같은 함수
            from engine import sealpolicy
            sealpolicy.apply(ExperimentConfig(**cfg_dict), sealed=True)
        except ValueError as e:
            blockers.append(str(e))
    exposures = seal_exposures(res_dir, cfg_dict)
    if exposures:
        ex = ", ".join(f"{x['result']}(~{x['evaluated_to']})" for x in exposures[:3])
        warnings.append(f"이 대상의 봉인 구간({sf} 이후)을 이미 채점한 개발 결과가 {len(exposures)}건 있습니다(예: {ex}). 그 결과를 본 뒤라면 봉인 구간은 "
                        "'한 번도 보지 않은 기간'이 아닙니다 — 봉인 결과를 보고할 때 이 사실을 함께 적어야 합니다")
    pl = None
    if not blockers:
        try:
            pl = plan(cfg_dict, dv, reg)
        except KeyError as e:
            blockers.append(f"이 결과의 설정(파라미터 이름 또는 변수 ID)이 지금 코드·변수 사전과 맞지 않습니다({e}) — 같은 가설을 지금 코드로 다시 돌려 새 후보로 만든 뒤 봉인을 여세요")
        except ValueError as e:
            blockers.append(str(e))
    hist = history(log, h, sf)
    if hist["same_config"]:
        warnings.append(f"이 설정(해시 {h})은 이미 {len(hist['same_config'])}회 봉인 평가했습니다 — 다시 열면 그 결과를 본 뒤 고른 것이 되어 봉인의 의미가 없습니다")
    if hist["same_seal_date"] and not hist["same_config"]:
        warnings.append(f"같은 봉인 시작일({sf})로 다른 후보를 {len(hist['same_seal_date'])}회 열었습니다. 후보를 여러 개 열면 그중 좋은 것을 고르는 순간 "
                        "봉인 구간에서도 다중검정이 생깁니다 — 열어 본 후보 수를 논문에 함께 보고하세요")
    if pl is not None:
        recorded = r["model"].get("params") or {}
        pinned = set(cfg_dict.get("model_params") or {}) - set(r["config"].get("model_params") or {})
        diff = {k_: (recorded.get(k_, "(기록 없음)"), v_) for k_, v_ in pl["params_now"].items() if k_ not in pinned and recorded.get(k_, "(기록 없음)") != v_}
        if diff:
            warnings.append("개발 결과에 기록된 파라미터와 지금 코드가 채운 파라미터가 다릅니다 — " + ", ".join(f"{k_}: {a} → {b}" for k_, (a, b) in diff.items())
                            + ". 같은 후보가 아닐 수 있으니 확인하세요")
        pl.pop("params_now", None)
    for n in pin_notes:
        warnings.append("옛 기본값 고정: " + n)
    cv_now, cv_r = _code_version(), r.get("code_version")
    if cv_r in (None, "nogit") or cv_now == "nogit":
        warnings.append("코드 버전(git 커밋)이 기록되지 않습니다(nogit) — 이 봉인 결과를 어떤 코드로 냈는지 증명할 수 없습니다. "
                        "열기 전에 git 첫 커밋을 만드는 것을 권합니다(git_first_push.bat)")
    elif cv_now != cv_r:
        warnings.append(f"개발 결과의 코드({cv_r})와 지금 코드({cv_now})가 다릅니다 — 그 사이 엔진·모델이 바뀌었으면 같은 후보가 아닙니다")
    if str(cv_now).endswith("-dirty"):
        warnings.append("지금 코드에 커밋하지 않은 수정이 있습니다(-dirty) — 커밋만으로는 이 봉인 결과를 재현할 수 없습니다")
    if r.get("lag_unconfirmed"):
        warnings.append(f"발표시차(lag)가 확인되지 않은 변수 {len(r['lag_unconfirmed'])}개를 씁니다: {', '.join(r['lag_unconfirmed'][:8])}"
                        + (" …" if len(r["lag_unconfirmed"]) > 8 else "") + " — 봉인 결과도 이 가정에 기대게 됩니다")
    if r.get("leak_warning"):
        warnings.append(f"개발 결과에 누출 경고가 있습니다: {r['leak_warning']}")
    if r.get("degenerate_model"):
        warnings.append("개발 결과가 퇴화(모든 입력의 중요도 0 → 상수 예측)였습니다")
    dvv = r.get("data_version")
    if latest_data_version and dvv and dvv != latest_data_version:
        warnings.append(f"후보는 데이터 버전 {dvv} 로 평가합니다(같은 후보 = 같은 데이터). 최신 버전은 {latest_data_version} — 봉인 구간 끝이 다를 수 있습니다")
    est = None
    if pl is not None and r.get("seconds") and r.get("n_folds"):
        est = round(float(r["seconds"]) * max(pl["n_folds"], 1) / max(int(r["n_folds"]), 1), 1)
    return {"stem": stem, "config": cfg_dict, "config_hash": h, "plan": pl, "history": hist, "warnings": warnings, "blockers": blockers,
            "criteria": CRITERIA, "estimate_seconds": est, "code_version_now": cv_now, "code_version_result": cv_r, "exposures": exposures,
            "dev": {"evaluation_period": r.get("evaluation_period") or r.get("prediction_period"), "n_folds": r.get("n_folds"),
                    "prediction_metrics": r.get("prediction_metrics") or {}, "performance": r.get("performance") or {},
                    "periods": (r.get("periods") or {}).get("test"), "data_version": dvv, "model": r["model"], "mode": r["config"].get("mode", "trading")}}


def read_log(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_log(path: Path, hist: list[dict]) -> None:
    """봉인 기록을 임시 파일에 쓰고 한 번에 바꾼다(쓰는 도중 끊겨 기록이 깨지지 않게)."""
    import os

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(hist, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)
