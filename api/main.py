"""FastAPI — 엔진의 뷰어. 계산은 하지 않는다(절대규칙: 화면은 계산하지 않는다 → 지표 정의는 engine/backtest.py).

실행(프로젝트 루트):  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
문서:                http://127.0.0.1:8000/docs
의존 방향: api → engine → models/adapters (모델·어댑터는 api 를 모른다)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from api.jobs import JobStore  # noqa: E402
from engine.experiment import ENGINE_REV, ExperimentConfig, default_device, list_plugins, results_dir  # noqa: E402
from engine.features import describe_feature  # noqa: E402
from engine.registry import LAG_DEFAULT, Registry  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
WEB_DIST = ROOT / "web" / "dist"

class SafeJSON(JSONResponse):
    """NaN/inf 는 JSON 에 없다 → null. 결과 json 에 NaN 이 들어 있어도(샤프 계산 불가 등) 응답이 깨지지 않게 한다."""

    def render(self, content) -> bytes:
        def clean(o):
            if isinstance(o, float):
                return None if (o != o or o in (float("inf"), float("-inf"))) else o
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


app = FastAPI(title="TimeSeriesPredict API", version="0.1", default_response_class=SafeJSON)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from engine import logs as _logs  # noqa: E402

alog = _logs.setup("api", "api")


@app.middleware("http")
async def _log_requests(request, call_next):
    """요청마다 한 줄: 방법·경로·쿼리·본문 크기 → 상태·걸린 시간. POST 본문은 DEBUG 로 요약(실험 설정이 그대로 보임)."""
    import time as _t
    t0 = _t.perf_counter()
    body = b""
    if request.method in ("POST", "PUT", "DELETE"):
        body = await request.body()
        if body:
            alog.debug("요청 본문 %s %s: %s", request.method, request.url.path, body[:2000].decode("utf-8", "replace"))
    try:
        resp = await call_next(request)
    except Exception:
        alog.exception("요청 처리 실패 %s %s", request.method, request.url.path)
        raise
    ms = (_t.perf_counter() - t0) * 1000
    lvl = 20 if (request.url.path.startswith("/api/") and request.method != "GET") or resp.status_code >= 400 else 10
    alog.log(lvl, "%s %s%s → %d (%.0fms, 본문 %dB)", request.method, request.url.path, f"?{request.url.query}" if request.url.query else "", resp.status_code, ms, len(body))
    return resp


jobs = JobStore()


# ---------------------------------------------------------------------------
# 메타: 변수·모델·데이터 버전
# ---------------------------------------------------------------------------
def _registry() -> tuple[Registry, list[str]]:
    master = DATA / "variable_master.xlsx"
    if master.exists():
        return Registry.load_master(master)
    return Registry.load_csv(DATA / "registry.csv"), []


def _versions() -> list[dict]:
    out = []
    for p in sorted((DATA / "versions").glob("*")):
        m = p / "manifest.json"
        if m.exists():
            j = json.loads(m.read_text(encoding="utf-8"))
            out.append({"id": p.name, "created_utc": j.get("created_utc"), "tickers": j.get("tickers", []), "variables": j.get("variables_collected"),
                        "start": j.get("start"), "end": j.get("end"), "migrated_from": j.get("migrated_from")})
    return out


@app.get("/api/meta")
def meta():
    reg, warns = _registry()
    available: set[str] = set()
    stored: dict[str, dict] = {}
    versions = _versions()
    if versions:
        vdir = DATA / "versions" / versions[-1]["id"]
        for p in vdir.glob("variables_*.parquet"):
            df = pd.read_parquet(p, columns=["variable_id", "date"])
            available |= set(df["variable_id"].unique())
            g = df.groupby("variable_id")["date"].agg(["min", "max", "count"])
            for vid, r in g.iterrows():  # 어느 파일에 몇 개, 언제부터 언제까지 있는지 (화면 '출처·위치' 표시용)
                stored[vid] = {"file": f"data/versions/{versions[-1]['id']}/{p.name}", "first": str(pd.Timestamp(r["min"]).date()),
                               "last": str(pd.Timestamp(r["max"]).date()), "n": int(r["count"]),
                               "upload": (("data/variable_master.xlsx" if p.stem == "variables_xlsx_master" else f"data/uploads/{p.stem.removeprefix('variables_xlsx_')}.xlsx (2026-11-09 전 데이터 버전 — 지금은 variable_master.xlsx 의 데이터 시트)") if p.stem.startswith("variables_xlsx_") else None)}
    from engine.api_spec import price_specs, request_spec
    man = json.loads((DATA / "versions" / versions[-1]["id"] / "manifest.json").read_text(encoding="utf-8")) if versions else {}
    c_start, c_end, c_at = man.get("start", "2006-01-01"), man.get("end"), man.get("created_utc")
    variables = []
    for vid in reg.ids(usable_only=False):
        v = reg.get(vid)
        variables.append({"id": v.id, "name": v.name, "description": v.description, "group": v.group, "domain": v.domain, "freq": v.freq, "lag_days": v.lag_days,
                          "lag_confirmed": v.lag_confirmed, "source": v.source, "transform": v.transform, "active": v.active,
                          "duplicate_of": v.duplicate_of, "usable": v.usable, "has_data": vid in available, "note": v.note,
                          "source_code": v.source_code, "stored": stored.get(vid),
                          "api": request_spec(v.source, v.source_code, c_start, c_end, c_at)})
    counters, sealed = {}, {}
    for mode in ("trading", "forecast"):
        d = results_dir(RESULTS, mode)
        sl = d / "sealed_log.json"
        from engine import ledger
        counters[mode] = ledger.count(d)  # 시도 원장 기준(원장이 없으면 옛 카운터 값) — 화면을 여는 것만으로 원장을 만들지는 않는다
        sealed[mode] = json.loads(sl.read_text(encoding="utf-8")) if sl.exists() else []
    n = counters["trading"]
    # 예측 연구 대상 후보: 데이터가 있고 양수인 D/W/M 변수 (지수·가격·수준값). 변화율 변수는 제외 권장이지만 판단은 사용자에게
    forecast_targets = [{"id": v["id"], "name": v["name"], "freq": v["freq"], "group": v["group"], "domain": v["domain"]}
                        for v in variables if v["has_data"] and v["freq"] in ("D", "W", "M") and v["usable"]]
    return {
        "models": list_plugins(),
        "variables": variables,
        "groups": [g for g in reg.groups() if g],
        "master_warnings": warns,
        "lag_defaults": LAG_DEFAULT,
        "data_versions": versions,
        "tickers": [{"id": "091160", "name": "KODEX 반도체"}, {"id": "091230", "name": "TIGER 반도체"}],
        "price_api": {t: price_specs(t, c_start, c_end, c_at) for t in ("091160", "091230")},
        "data_version_collect": {"start": c_start, "end": c_end, "created_utc": c_at},
        "defaults": json.loads(ExperimentConfig(model="ridge").model_dump_json()),
        "experiment_counter": n,
        "experiment_counters": counters,
        "sealed_evaluations": sealed["trading"],
        "sealed_evaluations_by_mode": sealed,
        "forecast_targets": forecast_targets,
        "device": default_device(),
        "transforms": ["none", "pct", "pct5", "pct20", "diff", "diff5", "diff20", "log", "logdiff"],
        "variable_sets": _variable_sets(),
    }


SETS_PATH = DATA / "variable_sets.json"


def _variable_sets() -> list[dict]:
    return json.loads(SETS_PATH.read_text(encoding="utf-8"))["sets"] if SETS_PATH.exists() else []


class VariableSet(BaseModel):
    id: str
    name: str
    description: str = ""
    variable_ids: list[str]
    transforms: dict[str, str] = {}
    lag_override: dict[str, int] = {}
    recommended_for: list[str] = []


@app.post("/api/variable_sets")
def save_variable_set(body: VariableSet):
    """변수 세트 저장(같은 id 면 덮어씀). 세트는 모델과 독립 — 어떤 모델에나 붙일 수 있다."""
    sets = [x for x in _variable_sets() if x["id"] != body.id] + [body.model_dump()]
    SETS_PATH.write_text(json.dumps({"sets": sets}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"saved": body.id, "count": len(sets)}


@app.delete("/api/variable_sets/{set_id}")
def delete_variable_set(set_id: str):
    sets = [x for x in _variable_sets() if x["id"] != set_id]
    SETS_PATH.write_text(json.dumps({"sets": sets}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"count": len(sets)}


# ---------------------------------------------------------------------------
# 실험 제출 / 작업
# ---------------------------------------------------------------------------
class SubmitBody(BaseModel):
    config: dict
    sealed: bool = False


class EnsembleReq(BaseModel):
    stems: list[str]
    method: str = "mean"


@app.post("/api/ensemble")
def create_ensemble(req: EnsembleReq):
    from engine.ensemble import save_ensemble

    try:
        out = save_ensemble(req.stems, RESULTS, req.method)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    return {"saved": Path(out).stem}


@app.post("/api/explain")
def explain_api(body: SubmitBody, date: str | None = None, epochs: int | None = None):
    """계산 따라가기: 실제 데이터·실제 코드로 폴드 하나를 돌려 결정일 하루의 중간값을 전부 돌려준다(engine/explain.py)."""
    from engine.explain import explain
    from engine.store import DataVersion
    try:
        cfg = dict(body.config)
        if cfg.get("preset"):
            name = Path(str(cfg.pop("preset"))).name
            cfg = {**json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")), **cfg}
        vs = _versions()
        if not vs:
            raise HTTPException(400, "데이터 버전이 없습니다")
        vid = cfg.get("data_version") or vs[-1]["id"]
        reg, _ = _registry()
        return explain(cfg, DataVersion(DATA, vid), reg, date=date, epochs=epochs)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"계산 따라가기 오류: {e}")


@app.post("/api/flow")
def flow(body: SubmitBody):
    """모델 흐름도용 실제 모양(학습 없음). 화면이 설정을 바꿀 때마다 부른다."""
    from engine.flow import model_flow
    from engine.store import DataVersion
    try:
        cfg = dict(body.config)
        if cfg.get("preset"):  # 도움말 그림용: configs/<preset>.json 을 그대로 쓴다(예: BIVA 원본 재현 설정)
            name = Path(str(cfg.pop("preset"))).name
            cfg = {**json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")), **cfg}
        vs = _versions()
        if not vs:
            raise HTTPException(400, "데이터 버전이 없습니다")
        vid = cfg.get("data_version") or vs[-1]["id"]
        reg, _ = _registry()
        return model_flow(cfg, DataVersion(DATA, vid), reg)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"흐름 계산 오류: {e}")


@app.post("/api/experiments")
def submit(body: SubmitBody):
    try:
        cfg = ExperimentConfig(**body.config)  # 검증만 (엔진 스키마가 유일한 기준)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"설정 오류: {e}")
    if not cfg.data_version:
        vs = _versions()
        if not vs:
            raise HTTPException(400, "데이터 버전이 없습니다. collect.py 를 먼저 실행하세요")
        cfg.data_version = vs[-1]["id"]
    if body.sealed:  # 봉인 평가는 전용 절차(/api/sealed/run: 사전 점검·중복 검사·기록)로만 — 이 길로는 점검 없이 봉인 구간을 열 수 있었다(2026-11-10)
        raise HTTPException(400, "봉인 평가는 '봉인 평가' 화면에서만 실행할 수 있습니다")
    job = jobs.submit(json.loads(cfg.model_dump_json()), sealed=False)
    return {"job": job, "config_hash": cfg.config_hash()}


class SealedRunBody(BaseModel):
    stem: str
    note: str = ""
    confirm: bool = False
    allow_reopen: bool = False


def _sealed_log(mode: str) -> list[dict]:
    from engine.sealed import read_log
    return read_log(results_dir(RESULTS, mode) / "sealed_log.json")


@app.get("/api/sealed/preview/{stem}")
def sealed_preview(stem: str):
    """봉인 평가 미리보기(학습 없음, 저장 없음): 실행과 같은 함수로 봉인 폴드·날짜를 계산하고, 봉인 기록·경고·막는 이유를 돌려준다."""
    from engine.sealed import preview
    from engine.store import DataVersion
    p = _find_result(stem)
    if p is None:
        raise HTTPException(404, "결과 없음")
    r = json.loads(p.read_text(encoding="utf-8"))
    mode = r["config"].get("mode", "trading")
    vs = _versions()
    vid = r.get("data_version") or (vs[-1]["id"] if vs else None)
    if not vid:
        raise HTTPException(400, "데이터 버전이 없습니다")
    reg, _ = _registry()
    try:
        return preview(r, stem, DataVersion(DATA, vid), reg, _sealed_log(mode), vs[-1]["id"] if vs else None)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"봉인 평가 미리보기 오류: {e}")


@app.post("/api/sealed/run")
def sealed_run(body: SealedRunBody):
    """봉인 평가 실행: 미리보기를 다시 계산해 막는 이유가 없을 때만, 후보 설정 그대로 봉인 모드 작업을 큐에 넣는다.
    같은 설정을 이미 열었으면 allow_reopen 없이는 거절한다(재개봉은 기록에 남고 결과에 '재개봉' 번호가 붙는다)."""
    if not body.confirm:
        raise HTTPException(400, "확인(confirm)이 필요합니다 — 봉인 평가는 후보당 한 번입니다")
    pv = sealed_preview(body.stem)
    if pv["blockers"]:
        raise HTTPException(409, "봉인 평가를 할 수 없습니다: " + " / ".join(pv["blockers"]))
    if pv["history"]["same_config"] and not body.allow_reopen:
        raise HTTPException(409, f"이 설정은 이미 {len(pv['history']['same_config'])}회 봉인 평가했습니다 — 다시 열려면 '재개봉' 확인이 필요합니다")
    cfg = ExperimentConfig(**pv["config"])
    # 대기·실행 중인 봉인 작업도 본다 — 봉인 기록은 실행이 시작돼야 써지므로, 연달아 두 번 누르면 둘 다 통과하던 틈(2026-11-10 점검 S4)
    pending = [j for j in jobs.list(limit=100000) if j.get("sealed") and j.get("status") in ("queued", "running") and j.get("sealed_source") == body.stem]
    if pending:
        raise HTTPException(409, f"이 후보의 봉인 평가가 이미 접수되어 {('실행 중' if pending[0]['status'] == 'running' else '대기 중')}입니다(작업 {pending[0]['id']}) — 끝난 뒤 결과를 확인하세요")
    job = jobs.submit(json.loads(cfg.model_dump_json()), sealed=True, sealed_note=body.note[:2000], sealed_source=body.stem)
    return {"job": job, "config_hash": cfg.config_hash()}


@app.get("/api/sealed/log")
def sealed_log(mode: str = "trading"):
    """봉인 평가 기록(오래된 순). 지우는 기능은 없다 — 몇 번 열었는지는 사실이므로."""
    return _sealed_log(mode)


@app.get("/api/jobs")
def list_jobs(limit: int = 100):
    return jobs.list(limit)


@app.get("/api/logs")
def get_logs(date: str | None = None, tail: int = 400, run: str | None = None, level: str | None = None, q: str | None = None):
    """개발 로그 보기. date=YYYYMMDD(기본 오늘), run=작업 ID 일부, level=DEBUG/INFO/WARNING/ERROR(이상), q=포함 문자열."""
    from engine.logs import LOG_DIR
    files = sorted(p.name for p in LOG_DIR.glob("tsp_*.log")) if LOG_DIR.exists() else []
    name = f"tsp_{date}.log" if date else (files[-1] if files else None)
    lines: list[str] = []
    if name and (LOG_DIR / name).exists():
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        lv = order.index(level.upper()) if level and level.upper() in order else 0
        with open(LOG_DIR / name, encoding="utf-8", errors="replace") as f:
            buf = []
            for line in f:
                if line[:4].isdigit() and len(line) > 30:  # 새 줄(시각으로 시작). 예외 추적 줄은 앞 줄에 붙인다
                    buf.append(line)
                elif buf:
                    buf[-1] += line
        for ln in buf:
            parts = ln.split(None, 3)
            lvn = parts[2] if len(parts) > 2 else "DEBUG"
            if lvn in order and order.index(lvn) < lv:
                continue
            if run and f"[run={run}" not in ln and run not in ln[:80]:
                continue
            if q and q not in ln:
                continue
            lines.append(ln.rstrip("\n"))
    return {"files": files, "file": name, "lines": lines[-max(1, min(tail, 5000)):], "total": len(lines)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "작업 없음")
    j["log_tail"] = jobs.log(job_id, 80)
    return j


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"cancelled": jobs.cancel(job_id)}


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
def _result_files(mode: str = "trading") -> list[Path]:
    d = results_dir(RESULTS, mode)
    return sorted((p for p in d.glob("*.json") if p.name not in ("experiment_counter.json", "sealed_log.json") and not p.name.startswith("ablation_")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


_NOT_RESULTS = ("experiment_counter", "sealed_log")


def _find_result(stem: str) -> Path | None:
    """결과 이름 → 파일. 카운터·봉인 기록·ablation 표는 결과가 아니다(전에는 결과처럼 조회·삭제됐다 — 2026-11-10 점검 S2).
    이름에 경로 문자(/ \\ : ..)가 있으면 찾지 않는다."""
    if not stem or stem in _NOT_RESULTS or stem.startswith(("ablation_", ".")) or any(c in stem for c in "/\\:") or ".." in stem:
        return None
    for mode in ("trading", "forecast"):
        p = results_dir(RESULTS, mode) / f"{stem}.json"
        if p.exists():
            return p
    return None


def _summary(p: Path) -> dict:
    r = json.loads(p.read_text(encoding="utf-8"))
    c, pm, pf = r["config"], r.get("prediction_metrics", {}), r.get("performance", {})
    return {
        "file": p.stem, "mode": c.get("mode", "trading"), "target_variable": c.get("target_variable"),
        "name": c.get("name") or r["model"]["file"], "model": r["model"]["name"], "model_file": r["model"]["file"],
        "ticker": c["ticker"], "frequency": c["frequency"], "k": c["target"]["k"], "target_kind": c["target"]["kind"],
        "variables": c.get("variable_ids", []), "use_technical": c.get("use_technical"), "n_features": r.get("n_features"),
        "data_version": r.get("data_version"), "evaluation_period": r.get("evaluation_period") or r.get("prediction_period"),
        "n_predictions": r.get("n_predictions"), "n_folds": r.get("n_folds"),
        "dir_acc": pm.get("dir_acc"), "rmse": pm.get("rmse"), "dm_vs_zero": pm.get("dm_vs_zero"), "dm_p": pm.get("dm_vs_zero_p"), "auc": pm.get("auc"),
        "cw_vs_zero": pm.get("cw_vs_zero"), "cw_p": pm.get("cw_vs_zero_p"),
        # stale: 지금 엔진과 계산 방식이 다른 옛 결과(engine_rev 불일치) — 순위·다중비교에서 뺀다. sealed_inclusive: 평가 구간에 봉인 구간이 섞인 옛 매매 결과
        "engine_rev": r.get("engine_rev"), "stale": r.get("engine_rev") != ENGINE_REV,
        "sealed_inclusive": c.get("mode", "trading") != "forecast" and not r.get("evaluation_period") and not r.get("sealed_evaluation"),
        "rmse_naive": pm.get("rmse_naive"), "rmse_zero": pm.get("rmse_zero"), "dir_acc_naive": pm.get("dir_acc_naive"),
        "dm_vs_naive": pm.get("dm_vs_naive"), "dm_naive_p": pm.get("dm_vs_naive_p"), "level_mape": pm.get("level_mape"),
        "level_mape_naive": pm.get("level_mape_naive"), "level_mape_zero": pm.get("level_mape_zero"), "skill_vs_naive": pm.get("skill_mape_vs_naive"),
        "info_time": c.get("forecast_info_time", "stamp"),
        "sharpe": pf.get("sharpe"), "cagr": pf.get("cagr"), "mdd": pf.get("mdd"), "sortino": pf.get("sortino"), "win_rate": pf.get("win_rate"),
        "bh_cagr": pf.get("benchmark_cagr"), "bh_mdd": pf.get("benchmark_mdd"), "excess_total": pf.get("excess_total_return"),
        "n_trades": pf.get("n_trades"), "exposure": pf.get("exposure"), "p_excess": pf.get("bootstrap_p_excess"), "deflated_p": pf.get("deflated_sharpe_p"),
        "trials": pf.get("n_trials_so_far"), "degenerate": r.get("degenerate_model"), "leak_warning": r.get("leak_warning"),
        "lag_unconfirmed": r.get("lag_unconfirmed") or [], "sealed": r.get("sealed_evaluation", False), "device": r.get("device"), "seconds": r.get("seconds"),
        "mtime": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat(),
    }


@app.get("/api/results")
def results(mode: str = "trading"):
    out = []
    for p in _result_files(mode):
        try:
            out.append(_summary(p))
        except Exception as e:  # noqa: BLE001
            out.append({"file": p.stem, "error": str(e)})
    if mode == "forecast":
        _holm_by_family(out)
    return out


def _holm_by_family(rows: list[dict]) -> None:
    """다중비교 보정(Holm–Bonferroni). '같은 문제'(대상·k·맞히는 방식·정보 시점·평가 구간)를 푼 결과들을 한 가족으로 보고,
    각 모델의 'naive 대비' p값을 가족 크기 m 에 맞춰 보정한다. 봉인 평가 결과는 제외(1회용이라 따로 본다).
    p_(1) ≤ … ≤ p_(m) 에서 adj_(i) = max_{j≤i} min(1, (m−j+1)·p_(j)).  (Holm 1979)"""
    fam: dict[tuple, list[dict]] = {}
    for r in rows:
        if "error" in r or r.get("sealed") or r.get("dm_naive_p") is None or r.get("stale"):
            continue  # 옛 계산 방식의 결과는 가족에 넣지 않는다 — 같은 실험의 옛 판·새 판이 따로 세어져 m 이 부풀던 문제(2026-11-10 점검 S6)
        key = (r.get("target_variable"), r.get("k"), r.get("target_kind"), r.get("info_time"), tuple(r.get("evaluation_period") or []))
        fam.setdefault(key, []).append(r)
    for members in fam.values():
        m = len(members)
        members.sort(key=lambda r: r["dm_naive_p"])
        run = 0.0
        for i, r in enumerate(members):
            run = max(run, min(1.0, (m - i) * r["dm_naive_p"]))
            r["p_holm"] = run
            r["family_size"] = m


@app.get("/api/results/{stem}")
def result_detail(stem: str):
    p = _find_result(stem)
    if p is None:
        raise HTTPException(404, "결과 없음")
    r = json.loads(p.read_text(encoding="utf-8"))
    reg, _ = _registry()
    r["feature_descriptions"] = {n: describe_feature(n, reg) for n in r.get("feature_names", [])}
    return r


@app.get("/api/results/{stem}/series")
def result_series(stem: str):
    """에쿼티 곡선·포지션·예측 (평가 구간만). 화면은 이 값을 그리기만 한다."""
    rp = _find_result(stem)
    p = rp.with_name(f"{stem}_daily.parquet") if rp else None
    if p is None or not p.exists():
        raise HTTPException(404, "일별 결과 없음")
    r = json.loads(rp.read_text(encoding="utf-8"))
    full = pd.read_parquet(p)
    if r["config"].get("mode") == "forecast":
        return _forecast_series(r, full)
    ev = r.get("evaluation_period")
    d = full.loc[ev[0]: ev[1]] if ev else full
    d = d.dropna(subset=["net_return"])
    eq = (1 + d["net_return"]).cumprod()
    bh = (1 + d["hold_return"]).cumprod()
    dd = eq / eq.cummax() - 1
    # 실제 vs 예측 (전체 기간): 예측이 없는 앞부분 = 학습 워밍업, 봉인 이후 = 봉인 구간
    k = int(r["config"]["target"]["k"])
    is_rule = r["model"]["input_shape"] == "rule"
    pred = full["prediction"] if not is_rule else pd.Series(float("nan"), index=full.index)
    pred_price = (full["open"].shift(-1) * (1 + pred)).where(pred.notna())  # 진입가(t+1 시가) 기준 예측 청산가
    actual_price = full["open"].shift(-1 - k)  # 실제 청산가(t+1+k 시가)
    sealed_from = r["config"].get("validation", {}).get("sealed_from")
    first_pred = pred.first_valid_index()

    def _l(x: pd.Series, nd: int):  # NaN → null (JSON 호환)
        return [None if pd.isna(v) else round(float(v), nd) for v in x]

    return {
        "date": [x.strftime("%Y-%m-%d") for x in d.index],
        "equity": eq.round(5).tolist(), "benchmark": bh.round(5).tolist(), "drawdown": dd.round(4).tolist(),
        "position": d["position"].fillna(0).tolist(),
        "full": {
            "date": [x.strftime("%Y-%m-%d") for x in full.index],
            "close": _l(full["close"], 2),
            "actual_return": _l(full["target"], 5),
            "pred_return": _l(pred, 5),
            "actual_exit_price": _l(actual_price, 2),
            "pred_exit_price": _l(pred_price, 2),
            "fit_return": _l(full["fit"], 5) if "fit" in full.columns else None,
            "fit_exit_price": _l((full["open"].shift(-1) * (1 + full["fit"])).where(full["fit"].notna()), 2) if "fit" in full.columns else None,
            "split_date": r["config"].get("validation", {}).get("split_date"),
            "mode": r["config"].get("validation", {}).get("mode", "walkforward"),
            "first_prediction": first_pred.strftime("%Y-%m-%d") if first_pred is not None else None,
            "sealed_from": sealed_from,
            "is_rule": is_rule,
            "k": k,
        },
    }


def _forecast_series(r: dict, full: pd.DataFrame) -> dict:
    """예측 연구 모드: 수준값(실제/예측/naive)과 변화율(실제/예측/naive). 체결·자산곡선 없음."""
    k = int(r["config"]["target"]["k"])
    pred = full["prediction"]
    naive_level = (full["level"] * (1 + full["naive"])).where(full["naive"].notna())
    first_pred = pred.first_valid_index()

    def _l(x: pd.Series, nd: int):
        return [None if pd.isna(v) else round(float(v), nd) for v in x]

    return {
        "mode": "forecast",
        "full": {
            "date": [x.strftime("%Y-%m-%d") for x in full.index],
            # 수준값 차트: t 에서 내다본 t+k 값을 t+k 날짜에 그린다(실제 수준선과 같은 위치에서 비교되도록 k 행 이동)
            "level": _l(full["level"], 4),
            "actual_future_level": _l(full["actual_future_level"], 4),
            "pred_future_level": _l(full["pred_future_level"].shift(k), 4),
            "naive_future_level": _l(naive_level.shift(k), 4),
            "actual_return": _l(full["target"], 5), "pred_return": _l(pred, 5), "naive_return": _l(full["naive"], 5),
            "fit_return": _l(full["fit"], 5) if "fit" in full.columns else None,
            "fit_future_level": _l((full["level"] * (1 + full["fit"])).where(full["fit"].notna()).shift(k), 4) if "fit" in full.columns else None,
            "first_prediction": first_pred.strftime("%Y-%m-%d") if first_pred is not None else None,
            "sealed_from": r["config"].get("validation", {}).get("sealed_from"),
            "split_date": r["config"].get("validation", {}).get("split_date"),
            "mode": r["config"].get("validation", {}).get("mode", "walkforward"),
            "is_rule": False, "k": k, "target_variable": r["config"].get("target_variable"),
        },
    }


@app.delete("/api/results/{stem}")
def delete_result(stem: str):
    """결과를 목록에서 뺀다 — 파일은 지우지 않고 history/ 로 옮기며(되살릴 수 있음) 실행 이력·시도 원장에 남긴다.
    시도 횟수는 줄지 않는다(시도는 사실이므로). 2026-11-10 전에는 파일을 바로 지웠다."""
    from engine import ledger, runlog

    rp = _find_result(stem)
    if rp is None:
        return {"deleted": 0}
    moved = runlog.archive_existing(rp.parent, stem, tag="deleted")
    runlog.append_log(rp.parent, {"event": "deleted", "result": stem, "moved_to": moved})
    ledger.record(rp.parent, "deleted", result=stem, moved_to=moved)
    return {"deleted": len(moved), "moved_to": moved}


@app.get("/api/ablations")
def ablations():
    out = []
    for p in sorted(results_dir(RESULTS, "trading").glob("ablation_*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({"file": p.stem, "markdown": p.read_text(encoding="utf-8")})
    return out


# ---------------------------------------------------------------------------
# 가이드 문서 (docs/guide → 코드에서 표를 채워 docs/guide_build)
# ---------------------------------------------------------------------------
GUIDE_BUILD = ROOT / "docs" / "guide_build"


def _ensure_guide() -> None:
    try:
        from tools.build_guide import build

        build()
    except Exception as e:  # noqa: BLE001
        print("가이드 생성 실패:", e)


_ensure_guide()


@app.get("/api/guide")
def guide_list():
    docs = []
    for p in sorted(GUIDE_BUILD.glob("*.md")):
        title = next((ln.lstrip("# ").strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("# ")), p.stem)
        docs.append({"id": p.stem, "title": title})
    return docs


@app.get("/api/guide/fig/{name}")
def guide_fig(name: str):
    """수식 읽는 법 그림(SVG). tools/formula_figs.py 가 가이드 생성 때 만든다."""
    from fastapi.responses import Response
    p = GUIDE_BUILD / "figs" / Path(name).name
    if p.suffix != ".svg" or not p.exists():
        raise HTTPException(404, "그림 없음")
    return Response(p.read_text(encoding="utf-8"), media_type="image/svg+xml")


@app.get("/api/guide/walk")
def guide_walk_list():
    """코드 따라가기(실제 실행 기록으로 만든 단계별 설명 화면·영상). tools/codewalk/*_script.py 가 만든다."""
    d = GUIDE_BUILD / "walk"
    out = []
    for p in sorted(d.glob("*.html")) if d.exists() else []:
        mp4 = p.with_suffix(".mp4")
        out.append({"name": p.stem, "html": f"/api/guide/walk/{p.name}", "mp4": f"/api/guide/walk/{mp4.name}" if mp4.exists() else None})
    return out


@app.get("/api/guide/walk/audio/{walk}/{name}")
def guide_walk_audio(walk: str, name: str):
    """코드 따라가기 설명 음성(tools/narrate.py 가 만든 단계별 mp3·자막 시각 json)."""
    p = GUIDE_BUILD / "walk" / "audio" / Path(walk).name / Path(name).name
    if p.suffix not in (".mp3", ".json") or not p.exists():
        raise HTTPException(404, "설명 음성 없음 — narrate.bat <화면이름> 으로 만듭니다")
    return FileResponse(p, media_type="audio/mpeg" if p.suffix == ".mp3" else "application/json")


@app.get("/api/guide/walk/{name}")
def guide_walk(name: str):
    from fastapi.responses import FileResponse
    p = GUIDE_BUILD / "walk" / Path(name).name
    if p.suffix not in (".html", ".mp4") or not p.exists():
        raise HTTPException(404, "코드 따라가기 파일 없음 — python -m tools.codewalk.biva_script 로 만듭니다")
    return FileResponse(p, media_type="text/html; charset=utf-8" if p.suffix == ".html" else "video/mp4")


@app.get("/api/guide/{doc_id}")
def guide_doc(doc_id: str):
    p = GUIDE_BUILD / f"{doc_id}.md"
    if not p.exists() or ".." in doc_id:
        raise HTTPException(404, "문서 없음")
    return {"id": doc_id, "markdown": p.read_text(encoding="utf-8")}


# 정적 화면 (web/dist 가 있으면 같은 포트에서 서빙)
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
