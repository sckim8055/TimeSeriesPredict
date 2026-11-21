"""결과 꾸러미(2026-11-21) — 결과 하나에 대한 모든 것을 zip 하나로: 이것만 받으면 시스템에 다시 들어오지 않아도
'언제, 어떤 데이터로, 어떤 방법(설정·모델)으로, 어떻게 진행되어, 어떤 결과가 나왔는지'를 사전·사중·사후로 알 수 있게.

구성(zip 안):
  README.txt                       파일 목록과 읽는 순서
  보고서.docx                      위 내용을 사람이 읽게 정리한 Word 보고서 (python-docx 가 없으면 보고서.html)
  1_사전_입력/                     설정·모델·데이터(실제로 쓴 입력 자료와 열 설명)·변수 사전·검증 계획
  2_사중_진행/                     실행 기록(시각·장치·라이브러리)·폴드별 기간·작업 출력·개발 로그·실행 기록·접속 기록
  3_사후_결과/                     결과 JSON·지표와 뜻·일별 결과(예측·수익)와 열 설명·변수 중요도
원칙: 숫자는 결과 파일·데이터 버전·로그에 있는 값 그대로 옮긴다(여기서 다시 계산하지 않음). 입력 자료는 결과에 적힌 데이터 버전과 설정으로
지금 코드가 다시 만든 것이라, 결과의 입력 변수 목록과 같은지 확인해 README·보고서에 적는다. 로그·결과 파일은 읽기만 한다.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------- 뜻 사전
# 지표 이름 → 뜻. 정의는 engine/experiment.py(forecast_metrics·예측 지표)·engine/backtest.py(performance)의 계산 그대로.
METRIC_DOC = {
    "rmse": "예측 오차의 제곱평균제곱근(정답과 같은 단위). 작을수록 좋음",
    "mae": "예측 오차 절댓값의 평균. 작을수록 좋음",
    "rmse_zero": "늘 0(변화 없음)이라고 예측했을 때의 RMSE — 기준선",
    "rmse_naive": "naive 기준선의 RMSE — 예측 연구: 직전에 끝난 k기간 변화율이 그대로 이어진다고 예측(y[t−k]) / 변동성: 최근 k기간 변동성이 이어진다",
    "dir_acc": "방향 적중률 — 예측과 정답의 부호(오름·내림)가 같은 비율(분류는 0.5 기준)",
    "dir_acc_naive": "naive 기준선의 방향 적중률",
    "dm_vs_zero": "Diebold–Mariano 통계량(모델 vs 0 예측, 제곱오차). 음수면 모델이 덜 틀림",
    "dm_vs_zero_p": "위 DM 검정의 p값 — 작을수록(예: < 0.05) 차이가 우연이 아님",
    "dm_vs_naive": "Diebold–Mariano 통계량(모델 vs naive, 제곱오차) — 사전 등록한 판정 기준. 음수면 모델이 덜 틀림",
    "dm_vs_naive_p": "위 DM 검정의 p값",
    "cw_vs_zero": "Clark–West 통계량(0 예측을 포함하는 비교용 보정 검정). 양수면 모델이 나음",
    "cw_vs_zero_p": "위 Clark–West 검정의 p값",
    "pred_sd_ratio": "예측의 흩어짐 ÷ 정답의 흩어짐. 0 에 가까우면 거의 상수 예측",
    "corr": "예측과 정답의 상관계수",
    "level_rmse": "변화율 예측을 수준값(가격·지수 자체)으로 되돌린 RMSE",
    "level_rmse_naive": "naive 의 수준값 RMSE", "level_rmse_zero": "0 예측(수준 그대로)의 수준값 RMSE",
    "level_mape": "수준값 평균 절대 백분율 오차(MAPE)", "level_mape_naive": "naive 의 수준값 MAPE", "level_mape_zero": "0 예측의 수준값 MAPE",
    "skill_vs_naive": "1 − RMSE/RMSE_naive. 양수면 naive 보다 나음(0.1 = 오차 10% 줄임)",
    "skill_mape_vs_naive": "1 − MAPE/MAPE_naive (참고용, 판정은 dm_vs_naive)",
    "dm_ape_vs_naive": "절대 백분율 오차로 한 DM 통계량(참고용)", "dm_ape_vs_naive_p": "위 검정의 p값",
    "brier": "확률 예측의 제곱오차(분류). 작을수록 좋음", "auc": "ROC 곡선 아래 넓이(분류). 0.5 = 무작위, 1 = 완벽",
    "n_periods": "평가 구간의 결정일 수",
    "total_return": "평가 구간 전략 총수익률(비용 뺀 뒤)", "cagr": "연평균 복리 수익률",
    "ann_vol": "연환산 변동성(수익률 표준편차 × √연간 결정일 수)",
    "sharpe": "샤프 비율 = 평균 수익 ÷ 표준편차 × √연간 결정일 수(무위험 이자율 0)",
    "sortino": "소르티노 비율 = 평균 수익 ÷ 손실 난 날의 표준편차 × √연간 결정일 수",
    "mdd": "최대 낙폭 — 그때까지의 최고점 대비 가장 깊게 떨어진 비율",
    "win_rate": "수익이 난 날 ÷ 수익이 0 이 아닌 날",
    "benchmark_total_return": "같은 구간 사서 들고 있기(B&H) 총수익률", "benchmark_cagr": "B&H 연평균 복리 수익률", "benchmark_mdd": "B&H 최대 낙폭",
    "excess_total_return": "전략 총수익률 − B&H 총수익률", "excess_mean_per_period": "결정일 하나당 평균 초과수익(전략 − B&H)",
    "n_trades": "사거나 판 날의 수", "turnover": "사고판 양의 합(포지션 변화량, 비율)", "exposure": "평균 투자 비율(1 = 늘 100% 투자)",
    "total_cost": "수수료·슬리피지 비용 합계(수익률 단위)",
    "bootstrap_p_excess": "블록 부트스트랩 p값 — 정말로는 B&H 보다 나을 게 없는데 우연히 이만큼 초과수익이 나올 확률",
    "deflated_sharpe_p": "Deflated Sharpe p값 — 지금까지 시도한 횟수를 감안해, 이 샤프가 운으로 나왔을 확률",
    "n_trials_so_far": "이 결과까지의 실험 시도 횟수(카운터)", "trial_sr_var": "시도들 샤프의 분산(Deflated Sharpe 계산에 씀)",
    "return_skew": "수익률 왜도(치우침)", "return_kurt": "수익률 첨도(꼬리 두께)",
    "weight": "섀넌 기준의 투자 비율", "every": "섀넌 기준의 재조정 주기",
}
DAILY_DOC = {
    "open": "그날 시가", "close": "그날 종가", "level": "대상 시계열 값(예측 연구)",
    "signal": "모델 예측을 매매 규칙에 넣어 나온 신호", "position": "그날 투자 비율(1 = 100%)",
    "hold_return": "그날 B&H 수익률", "gross_return": "비용 빼기 전 전략 수익률", "cost": "그날 비용(수익률 단위)",
    "net_return": "비용 뺀 전략 수익률", "equity": "전략 자산(시작 1)", "benchmark_equity": "B&H 자산(시작 1)",
    "prediction": "모델 예측(테스트 구간만, 정답과 같은 단위)", "target": "정답(결정일 t 기준 k기간 뒤의 값으로 만든 것)",
    "naive": "naive 기준선 예측", "actual_future_level": "k기간 뒤 실제 수준값", "pred_future_level": "예측을 수준값으로 되돌린 값",
    "shannon_return": "섀넌 50:50 기준 수익률(평가 구간만)", "turnover": "그날 사고판 양(비율)",
}


def config_fields() -> list[dict]:
    """설정 JSON 의 모든 칸(안쪽 칸 포함): 경로·뜻·기본값·선택지 — ExperimentConfig 의 Field description 에서."""
    from engine.experiment import ExperimentConfig
    sch = ExperimentConfig.model_json_schema()
    defs = sch.get("$defs", {})
    out: list[dict] = []

    def ref_of(v: dict):
        r = v.get("$ref") or next((x.get("$ref") for x in v.get("allOf", []) + v.get("anyOf", []) if isinstance(x, dict) and x.get("$ref")), None)
        return r.split("/")[-1] if r else None

    def choices(v: dict):
        if "enum" in v:
            return v["enum"]
        for x in v.get("anyOf", []):
            if isinstance(x, dict) and "enum" in x:
                return x["enum"]
        return None

    def walk(props: dict, prefix: str):
        for k, v in props.items():
            name = ref_of(v)
            out.append({"path": prefix + k, "desc": v.get("description", ""), "default": v.get("default"), "choices": choices(v), "group": bool(name)})
            if name:
                walk(defs[name]["properties"], prefix + k + ".")
    walk(sch["properties"], "")
    return out


def _get(o, path: str):
    for k in path.split("."):
        if not isinstance(o, dict):
            return None
        o = o.get(k)
    return o


def _show(v) -> str:
    return "(비움)" if v is None else json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, bool)) else str(v)


def _csv(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")   # 엑셀에서 한글이 깨지지 않게 BOM


def _local(ts: str | None):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        return None


def collect_logs(result_path: Path, r: dict, *, log_dir: Path, jobs_dir: Path, config_dir: Path) -> tuple[dict[str, str], list[str], dict]:
    """이 실행의 로그들: 작업 설정·작업 출력·개발 로그(이 실행 태그 줄)·실행 기록 줄·실행 시간대 접속 기록. (파일 이름 → 내용, 빠진 것, 정보)"""
    rr = r.get("run_record") or {}
    run_id, job_id = rr.get("run_id"), rr.get("job_id")
    if not job_id and jobs_dir.exists():   # 옛 결과: 작업 기록의 result_path 로 찾는다
        for jp in jobs_dir.glob("*.json"):
            try:
                j = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if str(j.get("result_path") or "").replace("\\", "/").endswith(result_path.name):
                job_id = j.get("id") or jp.stem
                break
    t0 = _local(rr.get("started_utc"))
    t1 = _local(rr.get("finished_utc")) or t0
    days: list[str] = []
    if t0:
        d = t0.date() - timedelta(days=1)
        while d <= (t1 or t0).date() + timedelta(days=1):
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
    tags = [f"[run={x}]" for x in (job_id, run_id) if x]
    files: dict[str, str] = {}
    missing: list[str] = []
    if job_id and (config_dir / f"{job_id}.json").exists():
        files["job_config.json"] = (config_dir / f"{job_id}.json").read_text(encoding="utf-8", errors="replace")
    else:
        missing.append("job_config.json — 작업 설정 파일 없음(명령줄로 직접 실행했거나 옛 결과)")
    if job_id and (jobs_dir / f"{job_id}.log").exists():
        files["job_output.log"] = (jobs_dir / f"{job_id}.log").read_text(encoding="utf-8", errors="replace")
    else:
        missing.append("job_output.log — 작업 출력 로그 없음")
    dev, acc = [], []
    for day in days:
        fp = log_dir / f"tsp_{day}.log"
        if fp.exists() and tags:
            keep = False
            for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                if line[:4].isdigit():
                    keep = any(t in line for t in tags)
                if keep:
                    dev.append(line)
        ap = log_dir / f"access_{day}.log"
        if ap.exists() and t0:
            lo, hi = t0 - timedelta(minutes=1), (t1 or t0) + timedelta(minutes=1)
            for line in ap.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if lo <= ts <= hi:
                    acc.append(line)
    files["dev_log_this_run.log"] = "\n".join(dev) + ("\n" if dev else "")
    if not dev:
        missing.append("dev_log_this_run.log 이 비어 있음 — 개발 로그가 생기기 전(2026-11-06) 결과이거나 로그 파일이 지워짐")
    files["access_during_run.log"] = "\n".join(acc) + ("\n" if acc else "")
    rl = []
    rp = result_path.parent / "run_log.jsonl"
    if rp.exists() and run_id:
        rl = [ln for ln in rp.read_text(encoding="utf-8", errors="replace").splitlines() if run_id in ln]
    files["run_log.jsonl"] = "\n".join(rl) + ("\n" if rl else "")
    info = {"run_id": run_id, "job_id": job_id, "start": t0, "end": t1,
            "n_warn": sum(" WARNING " in ln for ln in dev), "n_error": sum((" ERROR " in ln) or (" CRITICAL " in ln) for ln in dev),
            "warn_lines": [ln for ln in dev if (" WARNING " in ln) or (" ERROR " in ln) or (" CRITICAL " in ln)][-20:]}
    return files, missing, info


def _metrics_table(r: dict) -> pd.DataFrame:
    rows = []
    for sect, d in (("예측 지표", r.get("prediction_metrics") or {}), ("매매 성과", r.get("performance") or {})):
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"구분": f"{sect} · {k}", "지표": kk, "값": vv, "뜻": ("섀넌 50:50 기준 — " if k == "shannon" else "") + METRIC_DOC.get(kk, "")})
            else:
                rows.append({"구분": sect, "지표": k, "값": v, "뜻": METRIC_DOC.get(k, "(설명 없음 — 코드의 이름 그대로)")})
    return pd.DataFrame(rows, columns=["구분", "지표", "값", "뜻"])


MADE_AT_RUN = "실험 직후 자동으로 만들어 저장한 것 — 실행 당시의 데이터·코드·로그 그대로"
MADE_ON_DOWNLOAD = ("내려받을 때 만든 것 — 이 결과에는 실험 직후 저장한 꾸러미가 없습니다(2026-11-21 이전 결과이거나 명령줄로 실행). "
                    "숫자(결과·지표·폴드)는 결과 파일 그대로지만, 입력 자료·단계표·설명은 내려받은 때의 데이터 버전·코드로 다시 만든 것입니다")


def package_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_package.zip")


def save_package(result_path: Path, **kw) -> Path:
    """실험 직후: 꾸러미를 결과 옆(<stem>_package.zip)에 저장. 임시 파일에 쓴 뒤 이름을 바꿔 반쯤 쓴 파일이 남지 않게."""
    import os
    out = package_path(result_path)
    data = build_package(result_path.stem, result_path, made=MADE_AT_RUN, **kw)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    return out


def build_package(stem: str, result_path: Path, *, data_root: Path, registry, log_dir: Path, jobs_dir: Path, config_dir: Path,
                  made: str = MADE_ON_DOWNLOAD) -> bytes:
    """결과 꾸러미 zip 의 바이트. made = 언제 만든 것인지(README·보고서 맨 위에 적힘)."""
    from engine.experiment import ExperimentConfig, load_plugin, prepare
    from engine.features import describe_feature
    from engine.store import DataVersion

    raw = result_path.read_text(encoding="utf-8")
    r = json.loads(raw)
    cfg = r.get("config") or {}
    rr = r.get("run_record") or {}
    mode = cfg.get("mode", "trading")
    notes: list[str] = []
    files: dict[str, bytes] = {}

    # ------------------------------------------------------------------ 1. 사전(입력)
    P1, P2, P3 = "1_사전_입력/", "2_사중_진행/", "3_사후_결과/"
    fields = config_fields()
    cfg_rows = []
    for f in fields:
        if f["group"]:
            continue
        v = _get(cfg, f["path"])
        cfg_rows.append({"칸": f["path"], "이 결과의 값": _show(v), "기본값": _show(f["default"]),
                         "바꿈": "예" if v is not None and json.dumps(v, sort_keys=True, default=str) != json.dumps(f["default"], sort_keys=True, default=str) else "",
                         "뜻": f["desc"], "선택지": " · ".join(map(str, f["choices"])) if f["choices"] else ""})
    cfg_df = pd.DataFrame(cfg_rows)
    files[P1 + "설정_전체와_뜻.csv"] = _csv(cfg_df)
    files[P1 + "설정.json"] = json.dumps(cfg, ensure_ascii=False, indent=1).encode("utf-8")
    model_file = cfg.get("model", "")
    params = (r.get("model") or {}).get("params") or {}
    try:
        plugin = load_plugin(model_file)
        pdefs, model_name, model_doc = plugin.PARAMS, plugin.NAME, (plugin.__doc__ or "").strip()
    except Exception as e:  # noqa: BLE001
        pdefs, model_name, model_doc = {}, model_file, ""
        notes.append(f"모델 코드를 불러오지 못해 파라미터 뜻을 비웠습니다: {e}")
    par_df = pd.DataFrame([{"파라미터": k, "실제로 쓰인 값": _show(v),
                            "기본값": _show((pdefs.get(k) or {}).get("default_by_freq") or (pdefs.get(k) or {}).get("default")),
                            "뜻": (pdefs.get(k) or {}).get("desc", "")} for k, v in params.items()], columns=["파라미터", "실제로 쓰인 값", "기본값", "뜻"])
    files[P1 + "모델_파라미터와_뜻.csv"] = _csv(par_df)
    files[P1 + "모델_설명.txt"] = (f"모델: {model_name} (models/{model_file}.py)\n입력 모양: {(r.get('model') or {}).get('input_shape', '')}\n\n{model_doc}\n").encode("utf-8")

    # 변수 사전(이 결과가 쓴 변수)
    used = (rr.get("variables") or {}).get("used") or cfg.get("variable_ids") or []
    var_rows = []
    for vid in used:
        try:
            v = registry.get(vid)
        except Exception:  # noqa: BLE001
            var_rows.append({"변수 ID": vid, "이름": "(지금 변수 사전에 없음)"})
            continue
        var_rows.append({"변수 ID": vid, "이름": v.name, "설명": v.description, "출처": v.source, "조회 코드": v.source_code, "주기": v.freq,
                         "발표 시차(일)": (r.get("lag_days_used") or {}).get(vid, v.lag_days), "시차 확인됨": v.lag_confirmed,
                         "변환": (r.get("transforms_used") or {}).get(vid, v.transform), "단위": v.unit, "첫 날짜(이 실험)": (r.get("variable_first_date") or {}).get(vid, ""),
                         "비고": v.note})
    files[P1 + "변수_사전.csv"] = _csv(pd.DataFrame(var_rows))

    # 실제로 쓴 입력 자료 — 결과의 데이터 버전·설정으로 다시 만들어 결과의 입력 목록과 대조
    ds = None
    data_check = ""
    var_sum, n_tech = pd.DataFrame(), 0
    dv_id = r.get("data_version") or cfg.get("data_version")
    try:
        dv = DataVersion(data_root, dv_id)
        if not dv.dir.exists():
            raise FileNotFoundError(f"데이터 버전 {dv_id} 폴더가 없습니다")
        ec = ExperimentConfig(**{**cfg, "data_version": dv_id})
        ec2, _p, _pa, _tt, ds = prepare(ec, dv, registry)
        same = list(ds.feature_names) == list(r.get("feature_names") or [])
        data_check = ("입력 열이 결과의 입력 목록과 같습니다(" + str(len(ds.feature_names)) + "개)." if same else
                      f"주의: 다시 만든 입력 열({len(ds.feature_names)}개)이 결과의 목록({len(r.get('feature_names') or [])}개)과 다릅니다 — "
                      "결과 뒤에 코드나 변수 사전이 바뀐 것. 결과 JSON 의 feature_names 가 실제로 쓴 목록입니다.")
        df = ds.X.copy()
        df.insert(0, "정답_target", ds.y)
        pr = ds.prices if isinstance(ds.prices, pd.DataFrame) else ds.prices.to_frame("level")
        for c in pr.columns[::-1]:
            df.insert(0, f"가격_{c}", pr[c].reindex(df.index))
        df.index.name = "결정일"
        files[P1 + "입력_자료.csv"] = _csv(df, index=True)
        col_rows = [{"열": "결정일", "뜻": "판단하는 날(이 날까지 알 수 있는 값만 입력에 들어감)"}]
        col_rows += [{"열": f"가격_{c}", "뜻": f"대상의 {c} (정답·체결 계산용, 입력 아님)"} for c in pr.columns]
        col_rows.append({"열": "정답_target", "뜻": f"정답 — {cfg.get('target', {}).get('kind', 'return')}, k={cfg.get('target', {}).get('k', 1)} 기간 뒤 값으로 만든 것(입력 아님)"})
        col_rows += [{"열": c, "뜻": describe_feature(c, registry)} for c in ds.feature_names]
        files[P1 + "입력_자료_열설명.csv"] = _csv(pd.DataFrame(col_rows))
        try:   # 데이터 변화 단계: 변수마다 원자료 → 시차·채움 → 변환 → 모델 입력 (실제 입력과 대조)
            from engine.datatrace import technical_trace, variable_trace
            var_sum, var_tabs = variable_trace(ec2, dv, registry, ds)
            T = P1 + "데이터_변화_단계/"
            if len(var_sum):
                files[T + "요약.csv"] = _csv(var_sum)
                for nm, tb in var_tabs.items():
                    files[T + f"{nm}.csv"] = _csv(tb)
                bad = var_sum[var_sum["실제 입력과 일치"] != "일치"]
                if len(bad):
                    notes.append(f"데이터 변화 단계표가 실제 입력과 다른 변수 {len(bad)}개: {', '.join(bad['변수 ID'])} — 단계표를 만드는 규칙이 엔진과 다릅니다(알려 주세요)")
            tech = technical_trace(ds, list(var_tabs), registry)
            n_tech = len(tech)
            if n_tech:
                files[T + "기술지표.csv"] = _csv(tech)
        except Exception as e:  # noqa: BLE001
            notes.append(f"데이터 변화 단계표를 만들지 못했습니다: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        data_check = f"입력 자료를 다시 만들지 못했습니다: {type(e).__name__}: {e}"
        notes.append(data_check)
    files[P1 + "데이터_버전.json"] = json.dumps({"data_version": dv_id, "기록": rr.get("data_version"), "변수 사전": rr.get("master"),
                                                 "입력 자료 확인": data_check}, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    plan = None
    try:
        from engine.splitplan import validation_plan
        plan = validation_plan({**cfg, "data_version": dv_id}, DataVersion(data_root, dv_id), registry)
        files[P1 + "검증_계획.json"] = json.dumps(plan, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        notes.append(f"검증 계획을 다시 계산하지 못했습니다: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ 2. 사중(진행)
    logs, missing, linfo = collect_logs(result_path, r, log_dir=log_dir, jobs_dir=jobs_dir, config_dir=config_dir)
    for k, v in logs.items():
        files[P2 + k] = v.encode("utf-8")
    files[P2 + "실행_기록.json"] = json.dumps(rr, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    per = r.get("periods") or {}
    folds = per.get("folds") or [x for x in (per.get("first_fold"), per.get("last_fold")) if x]
    fold_df = pd.DataFrame([{"폴드": (i + 1) if per.get("folds") else ("첫" if i == 0 else "마지막"),
                             "학습 시작": (f.get("train") or [None, None])[0], "학습 끝": (f.get("train") or [None, None])[1], "학습 개수": f.get("n_train"),
                             "검증 시작": (f.get("val") or [None, None])[0], "검증 끝": (f.get("val") or [None, None])[1], "검증 개수": f.get("n_val"),
                             "테스트 시작": (f.get("test") or [None, None])[0], "테스트 끝": (f.get("test") or [None, None])[1], "테스트 개수": f.get("n_test"),
                             "빈칸(입력·정답)으로 뺀 학습 행": f.get("n_train_dropped_blank"), "입력 빈칸으로 예측 못한 테스트 행": f.get("n_test_not_predicted"),
                             "평균으로 채운 빈칸 수(창 모델)": f.get("n_blank_filled"), "학습 창 수(창 모델)": f.get("n_windows"),
                             "정답 없거나 빈칸이라 뺀 창 수": f.get("n_windows_dropped")}
                            for i, f in enumerate(folds)])
    if len(fold_df):   # 새로 기록하는 칸(2026-11-21~)은 이 결과·모델에 없으면 뺀다. 기간 칸은 늘 둔다
        extra = [c for c in fold_df.columns[10:] if fold_df[c].isna().all()]
        fold_df = fold_df.drop(columns=extra)
    sc_rows = []
    for i, f in enumerate(folds):
        sc = f.get("scale")
        if not sc:
            continue
        a, b = ("mean", "std") if sc["kind"] == "standard" else ("min", "max")
        for j, c in enumerate(sc["features"]):
            sc_rows.append({"폴드": i + 1, "학습 시작": (f.get("train") or [None])[0], "입력 열": c, "방식": "표준화 (x−평균)÷표준편차" if sc["kind"] == "standard" else "최소-최대 (x−최소)÷(최대−최소)",
                            "평균" if a == "mean" else "최소": sc[a][j], "표준편차" if b == "std" else "최대": sc[b][j]})
    if sc_rows:
        files[P2 + "폴드별_눈금.csv"] = _csv(pd.DataFrame(sc_rows))
    elif (r.get("model") or {}).get("input_shape") in ("table", "window"):
        notes.append("폴드별 눈금(표준화 기준값)은 2026-11-21 이후 결과부터 기록됩니다 — 이 결과에는 없습니다.")
    expl = _explain_block(r, cfg, dv_id, data_root, registry, files, P2 + "계산_따라가기/", notes)
    files[P2 + "폴드별_기간.csv"] = _csv(fold_df)
    if not per.get("folds"):
        notes.append("이 결과는 모든 폴드 기록이 생기기 전(2026-11-20 이전)에 만들어져 폴드별 기간에 첫·마지막 폴드만 있습니다.")

    # ------------------------------------------------------------------ 3. 사후(결과)
    files[P3 + "result.json"] = raw.encode("utf-8")
    met_df = _metrics_table(r)
    files[P3 + "지표와_뜻.csv"] = _csv(met_df)
    fi = r.get("feature_importance")
    if fi and r.get("feature_names"):
        files[P3 + "변수_중요도.csv"] = _csv(pd.DataFrame({"입력": r["feature_names"], "중요도(폴드 평균)": fi,
                                                          "뜻": [describe_feature(c, registry) for c in r["feature_names"]]}).sort_values("중요도(폴드 평균)", key=abs, ascending=False))
    dp = result_path.with_name(f"{stem}_daily.parquet")
    if dp.exists():
        daily = pd.read_parquet(dp)
        daily.index.name = "결정일"
        files[P3 + "일별_결과.csv"] = _csv(daily, index=True)
        files[P3 + "일별_결과_열설명.csv"] = _csv(pd.DataFrame([{"열": c, "뜻": DAILY_DOC.get(c, "")} for c in daily.columns]))
        if expl is not None:
            expl["_compare"] = _compare_explain(expl, daily, int(cfg.get("n_seeds", 1) or 1))
            c = expl["_compare"]
            if c and not c["ok"]:
                notes.append(f"계산 따라가기로 다시 계산한 폴드 예측 {c['n']}개가 결과 파일의 예측과 최대 {c['max_abs_diff']:.3g} 다릅니다"
                             + (" — 시드 여러 개를 평균한 실험이라 첫 시드만 따라간 값과 다른 것이 정상입니다." if c["n_seeds"] > 1 else
                                " — 결과 뒤에 코드·데이터가 바뀌었거나 재현이 안 되는 것. 확인이 필요합니다."))
    else:
        notes.append("일별 결과 파일(*_daily.parquet)이 없습니다.")

    # ------------------------------------------------------------------ 보고서·README
    from engine.datatrace import pipeline_steps
    steps_all = pipeline_steps(r, cfg, ds, var_sum, n_tech)
    files[P1 + "데이터_흐름_전체.csv"] = _csv(pd.DataFrame(steps_all, columns=["단계", "무엇을 했나", "이 결과에서", "확인할 파일"]))
    ctx = dict(made=made, steps_all=steps_all, var_sum=var_sum, expl=expl, stem=stem, r=r, cfg=cfg, rr=rr, mode=mode, model_name=model_name, model_doc=model_doc, cfg_df=cfg_df, par_df=par_df,
               var_rows=var_rows, data_check=data_check, ds=ds, plan=plan, fold_df=fold_df, linfo=linfo, missing=missing, met_df=met_df, notes=notes)
    try:
        files["보고서.docx"] = _report_docx(ctx)
    except ImportError:
        files["보고서.html"] = _report_html(ctx).encode("utf-8")
        notes.append("python-docx 가 설치되어 있지 않아 Word 대신 보고서.html 을 만들었습니다(Word 로 열 수 있음). 설치: pip install python-docx")
    files["README.txt"] = _readme(ctx, sorted(files)).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["README.txt"] + [n for n in files if n != "README.txt"]:
            z.writestr(name, files[name])
    return buf.getvalue()


def _explain_block(r, cfg, dv_id, data_root, registry, files, folder, notes) -> dict | None:
    """결정일 하루 계산 따라가기(engine/explain.py — 실험과 같은 코드로 폴드 하나를 다시 돌림). 결과의 데이터 버전으로,
    그 폴드의 예측이 결과 파일의 예측과 같은지 확인한다. 창 모델(LSTM·GRU·BIVA)은 다시 학습하는 데 오래 걸려 넣지 않는다."""
    shape = (r.get("model") or {}).get("input_shape")
    if shape == "window":
        notes.append("계산 따라가기는 창 모델(LSTM·GRU·BIVA)이라 꾸러미에 넣지 않았습니다(폴드를 다시 학습해야 해서 내려받기가 오래 걸림) — "
                     "실험 설정 화면의 '실제 데이터로 계산 따라가기'에서 이 결과의 데이터 버전으로 확인하세요.")
        return None
    try:
        from engine.explain import explain
        from engine.store import DataVersion
        ex = explain({**cfg, "data_version": dv_id}, DataVersion(data_root, dv_id), registry)
    except Exception as e:  # noqa: BLE001
        notes.append(f"계산 따라가기를 만들지 못했습니다: {type(e).__name__}: {e}")
        return None
    ex.pop("_y", None)
    files[folder + "전체.json"] = json.dumps(ex, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    lines = [f"결정일 {ex.get('decision_date')} · 폴드 {(ex.get('fold') or {}).get('index')}/{(ex.get('fold') or {}).get('count')} · 모델 {ex.get('name')}", ""]
    for i, st in enumerate(ex.get("steps") or [], 1):
        lines.append(f"[{i}] {st.get('title', '')}")
        if st.get("plain"):
            lines.append("    " + str(st["plain"]))
        if st.get("formula"):
            lines.append("    식: " + str(st["formula"]))
        if st.get("values"):
            lines.append("    값: " + json.dumps(st["values"], ensure_ascii=False, default=str)[:2000])
        tb = st.get("table")
        if tb and tb.get("rows"):
            fn = f"{i:02d}_{st.get('id', 'step')}.csv"
            files[folder + fn] = _csv(pd.DataFrame(tb["rows"], columns=tb["columns"]))
            lines.append(f"    표: {fn} ({len(tb['rows'])}행)")
        lines.append("")
    for n in ex.get("notes") or []:
        lines.append("알림: " + str(n))
    files[folder + "단계.txt"] = ("\n".join(lines) + "\n").encode("utf-8")
    checks = list(ex.get("checks") or [])
    # 결과 파일의 예측과 대조: 같은 폴드 예측 전부
    fp = ex.get("fold_predictions") or {}
    cmp = None
    if fp.get("rows") and fp.get("columns", [None, None])[1] == "예측":
        cmp = {"rows": fp["rows"]}
    ex["_compare_rows"] = cmp
    files[folder + "확인.csv"] = _csv(pd.DataFrame([{"확인": c.get("what"), "통과": "예" if c.get("ok") else "아니오", "최대 차이": c.get("max_abs_diff")} for c in checks]))
    return ex


def _compare_explain(ex, daily: pd.DataFrame | None, n_seeds: int) -> dict | None:
    """계산 따라가기의 폴드 예측 vs 결과 파일(일별 결과)의 예측."""
    if not ex or daily is None or not ex.get("_compare_rows") or "prediction" not in daily:
        return None
    rows = ex["_compare_rows"]["rows"]
    d = pd.DataFrame(rows, columns=["date", "pred", "actual"]).dropna(subset=["pred"])
    d["date"] = pd.to_datetime(d["date"])
    res = daily["prediction"].reindex(d["date"]).values.astype(float)
    diff = np.abs(d["pred"].values.astype(float) - res)
    ok = bool(np.nanmax(diff) <= 1e-9) if len(diff) and np.isfinite(diff).any() else False
    return {"n": int(len(d)), "max_abs_diff": float(np.nanmax(diff)) if len(diff) and np.isfinite(diff).any() else None, "ok": ok, "n_seeds": n_seeds}


# ---------------------------------------------------------------------------------------------- 보고서 내용(Word·HTML 공용)
def _summary_rows(ctx) -> list[tuple[str, str]]:
    r, cfg, rr, li = ctx["r"], ctx["cfg"], ctx["rr"], ctx["linfo"]
    v = cfg.get("validation") or {}
    t = cfg.get("target") or {}
    rows = [("결과 이름", ctx["stem"]),
            ("모드", "매매 시뮬레이션" if ctx["mode"] == "trading" else "예측 연구"),
            ("대상", f"{cfg.get('ticker', '')}" if ctx["mode"] == "trading" else f"{cfg.get('target_variable') or cfg.get('ticker', '')}"),
            ("맞힌 것(정답)", f"{t.get('kind')} · {t.get('k')}기간 뒤 · 주기 {cfg.get('frequency')}"),
            ("모델", f"{ctx['model_name']} ({cfg.get('model')})"),
            ("검증 방식", ("워크포워드 — 학습 창 %s · 재학습 %s" % (v.get("train_len"), v.get("test_len"))) if v.get("mode", "walkforward") == "walkforward"
             else f"1회 분할 — 테스트 시작 {v.get('test_start') or v.get('split_date')}"),
            ("봉인 시작", str(v.get("sealed_from") or "없음")),
            ("평가 구간", " ~ ".join(map(str, r.get("evaluation_period") or r.get("prediction_period") or []))),
            ("폴드 수", str(r.get("n_folds"))),
            ("데이터 버전", str(r.get("data_version"))),
            ("실행", f"{li.get('start')} ~ {li.get('end')} (서버 시각) · {r.get('seconds', '')}초 · 장치 {r.get('device', '')}"),
            ("코드 · 해시", f"{r.get('code_version')} · {r.get('config_hash')}"),
            ("run_id · job_id", f"{li.get('run_id')} · {li.get('job_id')}")]
    return rows


def _key_metrics(ctx) -> list[tuple[str, str, str]]:
    m = ctx["met_df"]
    keys = ["dm_vs_naive", "dm_vs_naive_p", "skill_vs_naive", "rmse", "rmse_naive", "dir_acc", "level_mape", "level_mape_naive",
            "sharpe", "total_return", "benchmark_total_return", "mdd", "bootstrap_p_excess", "deflated_sharpe_p"]
    out = []
    for k in keys:
        row = m[(m["지표"] == k) & (~m["구분"].str.contains("·"))]
        if len(row):
            v = row.iloc[0]["값"]
            out.append((k, f"{v:.4g}" if isinstance(v, float) else str(v), row.iloc[0]["뜻"]))
    return out


def _sections(ctx) -> list[tuple[str, list]]:
    """보고서의 절: (제목, [('p', 글) | ('table', 머리, 행들)])."""
    S = []
    S.append(("요약", [("table", ["항목", "내용"], _summary_rows(ctx)),
                       ("p", "핵심 지표(전체는 3장·3_사후_결과/지표와_뜻.csv):"),
                       ("table", ["지표", "값", "뜻"], _key_metrics(ctx))]))
    pre = [("p", "실험에 들어간 것 — 설정, 모델, 데이터, 나누는 방법. 이 장의 표는 1_사전_입력/ 폴더의 파일과 같습니다.")]
    ch = ctx["cfg_df"]
    pre.append(("p", "설정 전체('바꿈'=예 는 기본값에서 바꾼 칸):"))
    pre.append(("table", ["칸", "값", "기본값", "바꿈", "뜻"], [tuple(x) for x in ch[["칸", "이 결과의 값", "기본값", "바꿈", "뜻"]].values.tolist()]))
    pre.append(("p", f"모델 {ctx['model_name']}: {ctx['model_doc'].splitlines()[0] if ctx['model_doc'] else ''}"))
    pre.append(("table", ["파라미터", "실제 값", "기본값", "뜻"], [tuple(x) for x in ctx["par_df"].values.tolist()]))
    if ctx["var_rows"]:
        pre.append(("p", "입력 변수(변수 사전):"))
        pre.append(("table", ["ID", "이름", "출처", "주기", "발표 시차(일)", "변환", "설명"],
                    [(v.get("변수 ID"), v.get("이름"), v.get("출처", ""), v.get("주기", ""), v.get("발표 시차(일)", ""), v.get("변환", ""), v.get("설명", "")) for v in ctx["var_rows"]]))
    ds = ctx["ds"]
    if ds is not None:
        pre.append(("p", f"입력 자료: 결정일 {len(ds.X)}개({ds.X.index.min().date()} ~ {ds.X.index.max().date()}), 입력 열 {len(ds.feature_names)}개. {ctx['data_check']} "
                         "실제 값은 1_사전_입력/입력_자료.csv, 열의 뜻은 입력_자료_열설명.csv."))
    else:
        pre.append(("p", ctx["data_check"]))
    plan = ctx["plan"]
    if plan:
        steps = (plan.get("folds") or [{}])[0].get("steps") if plan.get("mode") == "walkforward" else plan.get("steps")
        if steps:
            pre.append(("p", "학습 창을 자르는 순서(첫 폴드, 지금 코드로 다시 센 값):"))
            pre.append(("table", ["무엇", "빼는 수", "남는 수"], [(s["what"], f"−{s['minus']}" if s.get("minus") is not None else "", s["n"]) for s in steps]))
        for n in plan.get("notes", []):
            pre.append(("p", "· " + n))
    S.append(("1. 사전 — 입력과 방법", pre))
    flow = [("p", "데이터가 원자료에서 결과가 되기까지 거친 단계 전체입니다. 표의 '확인할 파일'에 단계마다 실제 값이 있습니다."),
            ("table", ["단계", "무엇을 했나", "이 결과에서", "확인할 파일"], ctx["steps_all"])]
    vs = ctx["var_sum"]
    if len(vs):
        flow.append(("p", "변수마다 거친 순서와 확인 결과(결정일마다의 값은 1_사전_입력/데이터_변화_단계/<입력 열>.csv — 사용한 원자료 기준일·공개일·원래 값·"
                          "시차·채움 후 값·변환 후 값(=모델 입력)·새 값 공개 여부·빈칸 이유):"))
        flow.append(("table", ["변수", "모델 입력 열", "거친 순서", "첫 사용 결정일", "빈칸 결정일", "새 값 공개", "실제 입력과"],
                     [(x["변수 ID"], x["모델 입력 열"], x["거친 순서"], x["첫 사용 결정일"], x["빈칸 결정일 수"], x["새 값 공개 횟수"], x["실제 입력과 일치"]) for _, x in vs.iterrows()]))
    S.append(("데이터가 거친 단계", flow))
    mid = [("p", "실험이 어떻게 진행됐는지 — 실행 기록, 폴드별 기간, 로그. 2_사중_진행/ 폴더."), ("table", ["항목", "내용"], [
        ("실행 시각", f"{ctx['linfo'].get('start')} ~ {ctx['linfo'].get('end')} (서버 시각)"),
        ("장치·파이썬", f"{(ctx['rr'].get('host') or {}).get('device', '')} · Python {(ctx['rr'].get('host') or {}).get('python', '')} · {(ctx['rr'].get('host') or {}).get('platform', '')}"),
        ("라이브러리", ", ".join(f"{k} {v}" for k, v in (ctx["rr"].get("libraries") or {}).items())),
        ("개발 로그", f"이 실행의 경고 {ctx['linfo'].get('n_warn', 0)}줄 · 오류 {ctx['linfo'].get('n_error', 0)}줄")])]
    fd = ctx["fold_df"]
    if len(fd):
        mid.append(("p", f"폴드별 기간({len(fd)}개):"))
        mid.append(("table", ["폴드", "학습", "개수", "검증", "개수", "테스트", "개수"],
                    [(x["폴드"], f"{x['학습 시작']} ~ {x['학습 끝']}", x["학습 개수"], f"{x['검증 시작']} ~ {x['검증 끝']}" if x["검증 시작"] else "없음",
                      x["검증 개수"] if x["검증 시작"] else "", f"{x['테스트 시작']} ~ {x['테스트 끝']}", x["테스트 개수"]) for _, x in fd.iterrows()]))
    md = ctx["r"].get("model_diag")
    if md:
        mid.append(("p", "학습 진단: " + json.dumps(md, ensure_ascii=False, default=str)[:1500]))
    if ctx["linfo"].get("warn_lines"):
        mid.append(("p", "개발 로그의 경고·오류(마지막 20줄):"))
        for ln in ctx["linfo"]["warn_lines"]:
            mid.append(("p", ln))
    ex = ctx.get("expl")
    if ex:
        mid.append(("p", f"계산 따라가기 — 결정일 {ex.get('decision_date')} 하루를 실험과 같은 코드로 다시 계산한 단계(폴드 {(ex.get('fold') or {}).get('index')}/{(ex.get('fold') or {}).get('count')}, "
                         "자세한 값은 2_사중_진행/계산_따라가기/):"))
        mid.append(("table", ["#", "단계", "쉬운 설명", "식"], [(i, st.get("title", ""), st.get("plain", ""), st.get("formula", "")) for i, st in enumerate(ex.get("steps") or [], 1)]))
        chk = [(c.get("what"), "통과" if c.get("ok") else "불일치", c.get("max_abs_diff")) for c in ex.get("checks") or []]
        c = ex.get("_compare")
        if c:
            chk.append((f"다시 계산한 이 폴드의 예측 {c['n']}개 = 결과 파일의 예측", "통과" if c["ok"] else ("다름(시드 평균 실험)" if c["n_seeds"] > 1 else "불일치"), c["max_abs_diff"]))
        mid.append(("table", ["확인", "결과", "최대 차이"], chk))
    S.append(("2. 사중 — 진행", mid))
    post = [("p", "결과 — 지표 전체와 뜻. 3_사후_결과/ 폴더(일별 결과·변수 중요도 포함)."),
            ("table", ["구분", "지표", "값", "뜻"], [(a, b, f"{c:.6g}" if isinstance(c, float) else str(c), d) for a, b, c, d in ctx["met_df"].values.tolist()])]
    r = ctx["r"]
    flags = []
    if r.get("leak_warning"):
        flags.append("누출 경고가 켜져 있습니다(미래 값이 섞였을 수 있는 설정).")
    if r.get("degenerate_model"):
        flags.append("퇴화 모델: 변수 중요도가 모두 0 — 입력을 쓰지 않은 상수 예측.")
    if r.get("sealed_evaluation"):
        flags.append("봉인 평가 결과입니다.")
    for f in flags:
        post.append(("p", "· " + f))
    S.append(("3. 사후 — 결과", post))
    if ctx["notes"] or ctx["missing"]:
        S.append(("알림", [("p", "· " + n) for n in ctx["notes"] + ctx["missing"]]))
    return S


def _report_docx(ctx) -> bytes:
    from docx import Document          # python-docx — 없으면 ImportError → HTML
    from docx.oxml.ns import qn
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Malgun Gothic"
    st.font.size = Pt(9.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")   # 한글 글꼴
    doc.add_heading(f"결과 보고서 — {ctx['stem']}", 0)
    doc.add_paragraph(f"만든 시각 {datetime.now():%Y-%m-%d %H:%M} · TimeSeriesPredict 결과 꾸러미. 숫자는 결과 파일·데이터·로그의 값 그대로입니다.")
    doc.add_paragraph(f"만든 때: {ctx.get('made', '')}")
    for title, blocks in _sections(ctx):
        doc.add_heading(title, 1)
        for b in blocks:
            if b[0] == "p":
                doc.add_paragraph(str(b[1]))
            else:
                head, rows = b[1], b[2]
                t = doc.add_table(rows=1, cols=len(head))
                t.style = "Table Grid"
                for i, h in enumerate(head):
                    t.rows[0].cells[i].text = str(h)
                    for run in t.rows[0].cells[i].paragraphs[0].runs:
                        run.bold = True
                for row in rows:
                    cells = t.add_row().cells
                    for i, v in enumerate(row):
                        cells[i].text = "" if v is None or (isinstance(v, float) and v != v) else str(v)
                doc.add_paragraph()
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _report_html(ctx) -> str:
    import html
    e = lambda x: html.escape("" if x is None or (isinstance(x, float) and x != x) else str(x))  # noqa: E731
    parts = [f"<html><head><meta charset='utf-8'><title>결과 보고서 {e(ctx['stem'])}</title>"
             "<style>body{font-family:'Malgun Gothic',sans-serif;font-size:13px}table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #999;padding:3px 6px;vertical-align:top}</style></head><body>",
             f"<h1>결과 보고서 — {e(ctx['stem'])}</h1>", f"<p>만든 때: {e(ctx.get('made', ''))}</p>"]
    for title, blocks in _sections(ctx):
        parts.append(f"<h2>{e(title)}</h2>")
        for b in blocks:
            if b[0] == "p":
                parts.append(f"<p>{e(b[1])}</p>")
            else:
                parts.append("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in b[1]) + "</tr>"
                             + "".join("<tr>" + "".join(f"<td>{e(v)}</td>" for v in row) + "</tr>" for row in b[2]) + "</table>")
    parts.append("</body></html>")
    return "".join(parts)


def _readme(ctx, names: list[str]) -> str:
    DOC = {
        "보고서.docx": "사람이 읽는 보고서 — 요약 · 1.사전(입력과 방법) · 2.사중(진행) · 3.사후(결과). 먼저 이것을 여세요",
        "보고서.html": "사람이 읽는 보고서(HTML, Word 로도 열림)",
        "1_사전_입력/설정_전체와_뜻.csv": "설정 JSON 의 모든 칸: 이 결과의 값·기본값·바꿈·뜻·선택지",
        "1_사전_입력/설정.json": "설정 원본",
        "1_사전_입력/모델_파라미터와_뜻.csv": "실제로 쓰인 모델 파라미터·기본값·뜻",
        "1_사전_입력/모델_설명.txt": "모델 코드의 설명",
        "1_사전_입력/변수_사전.csv": "입력 변수의 이름·설명·출처·조회 코드·주기·발표 시차·변환",
        "1_사전_입력/입력_자료.csv": "실제로 모델에 들어간 입력 값(결정일 × 입력 열) + 정답 + 가격. 결과의 데이터 버전·설정으로 다시 만든 것",
        "1_사전_입력/입력_자료_열설명.csv": "입력 자료 각 열의 뜻",
        "1_사전_입력/데이터_버전.json": "데이터 버전·수집 기록·변수 사전 지문·입력 자료 확인 결과",
        "1_사전_입력/검증_계획.json": "학습·검증·테스트를 자르는 순서와 개수(지금 코드로 다시 센 값)",
        "1_사전_입력/데이터_흐름_전체.csv": "원자료 → 시차·변환 → 정답 → 나누기 → 빈칸 처리 → 눈금 → 학습·예측 → 평가: 단계마다 무엇을, 이 결과에서 어떻게, 어느 파일",
        "1_사전_입력/데이터_변화_단계/요약.csv": "변수마다 거친 순서·첫 사용일·빈칸·새 값 공개 횟수·실제 입력과 일치 여부",
        "1_사전_입력/데이터_변화_단계/기술지표.csv": "가격에서 계산한 기술지표 열의 뜻·첫 값·빈칸",
        "2_사중_진행/폴드별_눈금.csv": "폴드마다 학습 구간으로 정한 표준화(평균·표준편차) 또는 최소-최대 기준값",
        "2_사중_진행/계산_따라가기/단계.txt": "결정일 하루의 계산 단계(쉬운 설명·식·값)",
        "2_사중_진행/계산_따라가기/확인.csv": "손으로 다시 계산한 값 = 모델 출력 확인",
        "2_사중_진행/계산_따라가기/전체.json": "계산 따라가기 원본(모든 중간값)",
        "2_사중_진행/실행_기록.json": "실행 시각·장치·파이썬·라이브러리 버전·변수 치환 기록",
        "2_사중_진행/폴드별_기간.csv": "폴드마다 학습·검증·테스트 날짜와 개수",
        "2_사중_진행/job_config.json": "화면이 보낸 실험 설정(작업)",
        "2_사중_진행/job_output.log": "실험 프로세스의 화면 출력 전체",
        "2_사중_진행/dev_log_this_run.log": "개발 로그에서 이 실행의 줄([run=…])",
        "2_사중_진행/run_log.jsonl": "실행 기록(시작·끝·실패)에서 이 실행의 줄",
        "2_사중_진행/access_during_run.log": "실행 시간대의 접속 기록",
        "3_사후_결과/result.json": "결과 원본(모든 숫자)",
        "3_사후_결과/지표와_뜻.csv": "예측 지표·매매 성과 전체와 뜻",
        "3_사후_결과/일별_결과.csv": "결정일마다 예측·정답·(매매) 포지션·수익·비용",
        "3_사후_결과/일별_결과_열설명.csv": "일별 결과 각 열의 뜻",
        "3_사후_결과/변수_중요도.csv": "입력별 중요도(폴드 평균)와 뜻",
    }
    lines = [f"TimeSeriesPredict 결과 꾸러미 — {ctx['stem']}", f"만든 시각 {datetime.now():%Y-%m-%d %H:%M:%S}", f"만든 때: {ctx.get('made', '')}", "",
             "이 zip 하나로 이 결과가 언제, 어떤 데이터로, 어떤 방법으로, 어떻게 진행되어, 무엇이 나왔는지 알 수 있게 모았습니다.",
             "읽는 순서: 보고서 → 1_사전_입력 → 2_사중_진행 → 3_사후_결과. CSV 는 엑셀에서 바로 열립니다(UTF-8).", "",
             "파일:"]
    def doc(n):
        if n in DOC:
            return DOC[n]
        if n.startswith("1_사전_입력/데이터_변화_단계/"):
            return "이 입력 열이 결정일마다 거친 단계(원자료 기준일·공개일·원래 값·시차·채움 후·변환 후=모델 입력·빈칸 이유)"
        if n.startswith("2_사중_진행/계산_따라가기/"):
            return "계산 따라가기 단계의 표"
        return ""
    lines += [f"  {n:<40} {doc(n)}" for n in names if n != "README.txt"]
    if ctx["notes"] or ctx["missing"]:
        lines += ["", "알림:"] + ["  - " + n for n in ctx["notes"] + ctx["missing"]]
    return "\n".join(lines) + "\n"
