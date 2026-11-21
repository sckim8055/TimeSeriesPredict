"""FastAPI — 엔진의 뷰어. 계산은 하지 않는다(절대규칙: 화면은 계산하지 않는다 → 지표 정의는 engine/backtest.py).

실행(프로젝트 루트):  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
문서:                http://127.0.0.1:8000/docs
의존 방향: api → engine → models/adapters (모델·어댑터는 api 를 모른다)
"""
from __future__ import annotations

import json
import os
from typing import Literal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:  # 접속 키(.env)를 서버도 읽는다 — 화면의 '변수 등록 전 확인'이 출처에 직접 물어보기 때문(2026-11-12). 키는 화면·로그에 내보내지 않는다
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # noqa: BLE001
    pass
from api.jobs import JobStore  # noqa: E402
from engine.experiment import ENGINE_REV, ExperimentConfig, default_device, list_plugins, results_dir  # noqa: E402
from engine.features import describe_feature  # noqa: E402
from engine.registry import LAG_DEFAULT, Registry  # noqa: E402
from engine import tickers as tickers_mod  # noqa: E402
from adapters.base import scrub_secrets  # noqa: E402

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
# CORS 허용은 두지 않는다(2026-11-13 독립 검토 13): 화면은 이 서버가 직접 주고, 개발 화면(vite)도 프록시로 같은 출처에서 부른다.
# 전에는 모든 출처를 허용해, 암호를 정하지 않은 노트북에서 다른 웹사이트가 브라우저를 통해 실험 실행·삭제 요청을 보낼 수 있었다.
from engine import logs as _logs  # noqa: E402

alog = _logs.setup("api", "api")
acclog = _logs.access_logger()   # 접속 기록 logs/access_날짜.log (2026-11-17)


def _client_info(request) -> tuple[str, str, str]:
    """(보낸 주소, 경로, 브라우저 정보). 주소: uvicorn 이 앞단 프록시(Tailscale)가 붙인 X-Forwarded-For 를 믿고(127.0.0.1 에서 온 것만) 원래 주소로 바꿔 준 값.
    경로: Tailscale 이 붙이는 머리글이 있으면 '외부(Funnel)', 없으면 '노트북 직접'. 주소는 기록용일 뿐 막는 데 쓰지 않는다(막는 것은 암호)."""
    ip = request.client.host if request.client else "-"
    via = "외부(Funnel)" if _is_external(request) else "노트북 직접"
    ua = (request.headers.get("user-agent") or "-").replace("\n", " ")[:160]
    return ip, via, ua

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
    ip, via, ua = _client_info(request)
    q = f"?{request.url.query}" if request.url.query else ""
    alog.log(lvl, "%s %s%s → %d (%.0fms, 본문 %dB) · %s %s", request.method, request.url.path, q, resp.status_code, ms, len(body), ip, via)
    # 접속 기록: 한 줄 = 시각 | 보낸 주소 | 경로 | 방법 주소 | 결과 | 걸린 시간 | 본문 크기 | 브라우저 정보 (암호 머리글 등 머리글 값은 남기지 않음)
    acclog.info("%s | %s | %s %s%s | %d | %.0fms | %dB | %s", ip, via, request.method, request.url.path, q, resp.status_code, ms, len(body), ua)
    return resp


# ---------------------------------------------------------------------------
# 바꾸는 동작에 암호 (2026-11-12) — 외부 공개 주소(Tailscale Funnel)로 열어 둘 때
#   .env 의 TSP_WRITE_PASSWORD 가 있으면: /api 의 GET 이 아닌 요청(실험 실행·데이터 가져오기·변수/종목 추가·삭제·계산 따라가기 등)은
#     머리글 X-TSP-Password 가 맞아야 통과. 보기(GET)는 열려 있다.
#   없으면: 노트북에서 직접 하는 요청은 지금처럼 통과, Funnel 로 들어온 요청(머리글 Tailscale-Funnel-Request — Tailscale 이 붙인다)은 막는다.
#   Funnel 로 들어온 요청은 연결상 127.0.0.1 에서 온다(uvicorn 이 X-Forwarded-For 로 원래 주소를 보여 주지만, 그 값은 접속 기록용으로만 쓴다).
#   주소로 막지 않고 암호가 기본 방어(2026-11-17 설명 고침 — 전에는 '주소로는 가릴 수 없다'고만 적혀 있었다).
# ---------------------------------------------------------------------------
_WRONG_DELAY = 1.0   # 틀린 암호마다 기다리는 초(마구 넣어 보기를 늦춘다). 서버 전체에서 한 번에 하나씩(아래 _WRONG_LOCK)
import threading as _threading  # noqa: E402
_WRONG_LOCK = _threading.Lock()   # 동시에 여러 개를 넣어도 '초당 한 번'보다 빨라지지 않게(2026-11-13 검토 13 — 전에는 요청마다 따로 기다렸다)


def _wrong_wait() -> None:
    import time as _t
    with _WRONG_LOCK:
        _t.sleep(_WRONG_DELAY)


def _origin_problem(request) -> str:
    """바꾸는 요청의 Origin 점검(2026-11-13 검토 13, CSRF). 브라우저는 다른 사이트에서 보낸 요청에 그 사이트 주소를 Origin 으로 붙인다.
    Origin 이 없으면(화면 밖의 도구·같은 출처의 일부 요청) 통과. 있으면 그 '호스트:포트'가 이 요청의 Host 와 같아야 한다.
    외부(Funnel) 요청은 여기서 보지 않는다 — 암호가 지킨다(Funnel 이 Host 를 어떻게 넘기는지 노트북에서 확인하지 못해, 막아 버리지 않으려고)."""
    from urllib.parse import urlsplit
    origin = request.headers.get("origin")
    if origin is None or _is_external(request):
        return ""
    host = (request.headers.get("host") or "").lower()
    fwd = (request.headers.get("x-forwarded-host") or "").lower()   # 앞단 프록시(Tailscale 등)가 원래 주소를 적는 머리글. 다른 사이트는 이것을 붙여 보낼 수 없다
    o = urlsplit(origin.lower()) if origin != "null" else None     # (사용자 지정 머리글은 사전 확인(preflight)을 부르고, CORS 허용이 없어 브라우저가 막는다)
    if o and o.netloc and o.netloc in {host, fwd} - {""}:
        return ""
    return f"다른 사이트({origin})에서 보낸 요청이라 막았습니다 — 이 화면은 {host or '이 서버'} 에서 직접 열어 쓰세요"
_OPEN_METHODS = ("GET", "HEAD", "OPTIONS")
# POST 이지만 아무것도 저장하지 않는 '보기' — 화면을 열면 저절로 불리는 설명 계산(2026-11-13: 화면을 열자마자 암호 창이 떠서 뺌)
# 2026-11-21: /api/split-plan(검증 계획 표, 학습 없음·저장 없음)을 11-18 에 만들면서 여기에 넣지 않아 화면을 열자마자 다시 암호 창이 떴다 — 추가
_OPEN_POST_PATHS = ("/api/flow", "/api/explain", "/api/split-plan")


def _write_password() -> str:
    return (os.environ.get("TSP_WRITE_PASSWORD") or "").strip()


def _is_external(request) -> bool:
    return bool(request.headers.get("tailscale-funnel-request"))


@app.middleware("http")
async def _guard_writes(request, call_next):
    import asyncio
    import hmac
    if request.url.path.startswith("/api/") and request.method not in _OPEN_METHODS:
        why = _origin_problem(request)          # 보기용 POST(_OPEN_POST_PATHS)도 다른 사이트에서는 받지 않는다(계산이 무거워 서버를 붙잡을 수 있다)
        if why:
            alog.warning("다른 출처의 요청을 막음: %s %s (Origin %s)", request.method, request.url.path, request.headers.get("origin"))
            return JSONResponse({"detail": why}, status_code=403)
    if request.url.path.startswith("/api/") and request.method not in _OPEN_METHODS and request.url.path not in _OPEN_POST_PATHS:
        pw = _write_password()
        if pw:
            given = request.headers.get("x-tsp-password") or ""
            if not given:
                return JSONResponse({"detail": "바꾸는 동작(실험 실행·데이터 가져오기·추가·삭제)에는 암호가 필요합니다.", "need_password": True}, status_code=401)
            if not hmac.compare_digest(given.encode(), pw.encode()):
                alog.warning("암호가 틀린 요청: %s %s", request.method, request.url.path)
                await asyncio.to_thread(_wrong_wait)
                return JSONResponse({"detail": "암호가 틀렸습니다.", "need_password": True}, status_code=401)
        elif _is_external(request):
            return JSONResponse({"detail": "외부 주소에서는 바꾸는 동작이 막혀 있습니다 — 노트북의 .env 파일에 TSP_WRITE_PASSWORD=암호 를 넣고 서버를 다시 시작하면 "
                                           "암호를 넣고 쓸 수 있습니다(노트북에서 직접 열면 지금도 됩니다)."}, status_code=403)
    return await call_next(request)


@app.get("/api/auth")
def auth_status(request: Request) -> dict:
    return {"password_set": bool(_write_password()), "external": _is_external(request)}


@app.post("/api/auth/check")
def auth_check() -> dict:
    return {"ok": True}   # 여기까지 왔으면 암호가 맞았거나(있을 때) 노트북에서 직접 연 것


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
    """데이터 버전 목록. 맨 끝이 '기본으로 쓰는 버전'이 되게 놓는다: 일부를 받지 못한 버전(incomplete)들 → 온전한 버전들(각각 오래된 순).
    그래서 화면·실행의 기본값(목록의 마지막)은 가장 새 '온전한' 버전이다(engine.store.latest_version_id 와 같은 규칙)."""
    from engine.store import list_versions
    out = []
    for v in list_versions(DATA):
        j = v["manifest"]
        out.append({"id": v["id"], "created_utc": j.get("created_utc"), "tickers": j.get("tickers", []), "variables": j.get("variables_collected"),
                    "start": j.get("start"), "end": j.get("end"), "migrated_from": j.get("migrated_from"),
                    "incomplete": v["incomplete"], "failed_items": j.get("failed_items") or []})
    return [v for v in out if v["incomplete"]] + [v for v in out if not v["incomplete"]]


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
        "tickers": tickers_mod.load(DATA),
        "price_api": {t: price_specs(t, c_start, c_end, c_at) for t in tickers_mod.ids(DATA)},
        "data_version_collect": {"start": c_start, "end": c_end, "created_utc": c_at},
        "defaults": json.loads(ExperimentConfig(model="ridge").model_dump_json()),
        "experiment_counter": n,
        "experiment_counters": counters,
        "sealed_evaluations": sealed["trading"],
        "sealed_evaluations_by_mode": sealed,
        "forecast_targets": forecast_targets,
        "device": default_device(),
        # 변환 선택지는 engine.experiment.TRANSFORM_OPTIONS 한 곳에서(2026-11-14 — 화면 세 곳과 도움말이 같은 목록·설명을 쓰게)
        "transforms": [o["value"] for o in _TRANSFORM_OPTIONS],
        "transform_options": _TRANSFORM_OPTIONS,
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


# ---------------------------------------------------------------------------
# 화면에서 변수 등록 · 데이터 가져오기 (2026-11-12)
# ---------------------------------------------------------------------------
# 지원하는 출처만: 출처마다 응답 형식·인증이 달라 연결 코드(adapters/)가 있는 것만 받을 수 있다. 접속 키는 화면에서 받지 않고 .env 에만 둔다.
_SOURCES = {
    "YF": {"label": "야후 파이낸스 (주가·지수·환율·원자재)", "code_help": "야후 종목 기호", "example": "^SOX · 005930.KS · KRW=X · CL=F", "key_env": None},
    "FRED": {"label": "미국 세인트루이스 연준 FRED (미국 금리·물가·경기)", "code_help": "FRED 시리즈 ID", "example": "DGS10 · CPIAUCSL · VIXCLS", "key_env": "FRED_API_KEY"},
    "ECOS": {"label": "한국은행 ECOS (한국 금리·물가·수출입)", "code_help": "통계표코드/항목코드/주기 (주기는 D·M·Q·A)", "example": "722Y001/0101000/M", "key_env": "ECOS_API_KEY"},
}


class VariableCheckBody(BaseModel):
    source: Literal["YF", "FRED", "ECOS"]
    source_code: str


class TickerCheckBody(BaseModel):
    id: str


class TickerBody(BaseModel):
    id: str
    name: str = ""
    kind: Literal["etf", "stock"] | None = None   # 비우면 자동 판별(2026-11-13 — 사람이 고르지 않는다)


_KIND_PROBE_DAYS = 120


def _probe_kind(code: str) -> tuple[str, str]:
    """공식 ETF 시세(data.go.kr)에 그 코드가 있으면 ETF, '응답 없음'이면 주식. 키가 없거나 연결이 안 되면 판별 못 함 → 주식으로 둔다
    (공식 출처를 묻지 않으니 수집 실패가 생기지 않는다). 돌려주는 값: (종류, 근거 문장)."""
    from adapters import datago_adapter
    if not os.environ.get("DATAGO_API_KEY"):
        return "stock", "공식 ETF 시세 키(DATAGO_API_KEY)가 없어 ETF 인지 판별하지 못했습니다 — 주식으로 두고 야후 가격으로 받습니다"
    # 최근 120일을 묻는다(2026-11-13 검토 13 — 30일이면 공식 시세가 잠시 늦거나 비는 때 ETF 를 주식으로 잘못 볼 수 있었다)
    start = (pd.Timestamp.today() - pd.Timedelta(days=_KIND_PROBE_DAYS)).strftime("%Y-%m-%d")
    try:
        df = datago_adapter.fetch_prices(code, start, None)
    except RuntimeError as e:
        if "응답 없음" in str(e):
            return "stock", f"공식 ETF 시세(최근 {_KIND_PROBE_DAYS}일)에 없는 코드라 주식으로 봅니다(가격은 야후, 빠진 날은 네이버로 메움)"
        return "stock", f"공식 ETF 시세를 확인하지 못해 ETF 인지 판별하지 못했습니다({scrub_secrets(str(e))[:120]}) — 주식으로 두고 야후 가격으로 받습니다"
    except Exception as e:  # noqa: BLE001 — 연결 오류 등
        return "stock", f"공식 ETF 시세에 연결하지 못해 ETF 인지 판별하지 못했습니다({type(e).__name__}) — 주식으로 두고 야후 가격으로 받습니다"
    if df is not None and len(df):
        return "etf", f"공식 ETF 시세(최근 {_KIND_PROBE_DAYS}일)에 있는 코드라 ETF 로 봅니다(가격은 공식 2020~ + 야후 그 이전, 빠진 날은 네이버로 메움)"
    return "stock", f"공식 ETF 시세(최근 {_KIND_PROBE_DAYS}일)에 없는 코드라 주식으로 봅니다(가격은 야후, 빠진 날은 네이버로 메움)"


def _probe_prices(code: str) -> pd.DataFrame:
    """야후에서 그 종목의 최근 3년 가격을 실제로 받아 본다(등록 전 확인용 — 캐시·데이터 버전에 쓰지 않는다)."""
    from adapters import yf_adapter
    start = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    return yf_adapter.fetch_prices(code, start, None)


@app.get("/api/tickers")
def list_tickers() -> list[dict]:
    return tickers_mod.load(DATA)


@app.post("/api/tickers/check")
def check_ticker(body: TickerCheckBody) -> dict:
    tid = (body.id or "").strip()
    if not tid.isdigit() or len(tid) != 6:
        return {"ok": False, "error": "종목 코드는 한국거래소 6자리 숫자여야 합니다(예: 005930)"}
    try:
        px = _probe_prices(tid)
    except Exception as e:  # noqa: BLE001 — 출처 오류 문구를 그대로(키는 가린다) 보여 준다
        return {"ok": False, "error": f"야후에서 {tid}.KS 의 가격을 받지 못했습니다: {scrub_secrets(str(e))[:300]}"}
    if px is None or len(px) == 0:
        return {"ok": False, "error": f"야후에서 {tid}.KS 의 가격을 받지 못했습니다: 빈 응답"}
    tail = px["close"].tail(5)
    return {"ok": True, "rows": int(len(px)), "first": str(px.index.min().date()), "last": str(px.index.max().date()),
            "sample": [{"date": str(d.date()), "close": float(v)} for d, v in tail.items()]}


@app.post("/api/tickers")
def add_ticker(body: TickerBody) -> dict:
    why = tickers_mod.problem(DATA, body.id, body.kind or "etf")
    if why:
        raise HTTPException(400, why)
    chk = check_ticker(TickerCheckBody(id=body.id))
    if not chk["ok"]:
        raise HTTPException(400, chk["error"])
    if body.kind:
        kind, basis = body.kind, "종류를 직접 지정했습니다"
    else:
        kind, basis = _probe_kind(body.id.strip())
    row = tickers_mod.add(DATA, body.id, body.name, kind)
    return {"added": row["id"], "ticker": row, "check": chk, "kind_basis": basis,
            "note": "목록에 넣었습니다. 아래 [데이터 가져오기]를 눌러야 이 종목의 가격이 데이터에 들어옵니다(그 뒤 실험에서 고를 수 있습니다)."}


class VariableBody(BaseModel):
    id: str
    name: str = ""
    source: Literal["YF", "FRED", "ECOS"]
    source_code: str
    freq: Literal["D", "W", "M", "Q", "H", "Y"] | None = None   # 비우면 받아 본 자료로 정한다(2026-11-13 검토 1) — 넣으면 자료와 같아야 한다
    group: str = ""
    transform: Literal["pct", "diff", "none"] = "pct"
    lag_days: int | None = Field(None, ge=0, description="기준일→공개일 일수. 비우면 주기별 보수적 기본값(미확인으로 표시)")
    unit: str = ""
    note: str = ""


def _probe_series(source: str, code: str, freq: str = "D") -> pd.Series:
    """출처에서 최근 3년치를 실제로 받아 본다(등록 전 확인용 — 캐시·데이터 버전에 쓰지 않는다)."""
    start = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    if source == "YF":
        from adapters import yf_adapter
        return yf_adapter.fetch_series(code, start, None)
    if source == "FRED":
        from adapters import fred_adapter
        return fred_adapter.fetch_series(code, start, None)
    from adapters import ecos_adapter
    from engine.api_spec import ecos_period
    st, it, fq = code.split("/")
    return ecos_adapter.fetch_series(st, it, fq, ecos_period(pd.Timestamp(start), fq), ecos_period(pd.Timestamp.today(), fq))


_FREQ_KO = {"D": "일", "W": "주", "M": "월", "Q": "분기", "H": "반기", "Y": "연"}
_ECOS_FREQ = {"D": "D", "M": "M", "Q": "Q", "A": "Y"}   # 한국은행 코드 끝 주기 → 우리 주기


def _infer_freq(source: str, code: str, idx) -> tuple[str | None, str]:
    """받아 본 자료의 주기를 정한다(2026-11-13 독립 검토 1).
    왜: 주기는 발표시차 기본값(일 1일, 월 45일 …)과 빈 기간 점검의 기준이다. 화면 기본값 '일'로 월 자료를 등록하면
        발표시차가 1일이 되어, 실제로는 한 달 반 뒤에 나오는 값을 미리 쓰는 누출이 생긴다. 그래서 사람이 고르지 않고 자료로 정한다.
    어떻게: 한국은행은 코드 끝 주기(요청 자체가 그 주기). 나머지는 날짜 간격의 중앙값(일):
        ≤3 일 · ≤10 주 · ≤45 월 · ≤120 분기 · ≤250 반기 · 그보다 길면 연. 간격이 둘 미만(값 3개 미만)이면 판별하지 않는다(None)."""
    if source == "ECOS":
        fq = _ECOS_FREQ.get(code.split("/")[-1])
        return fq, f"한국은행 코드 끝의 주기 '{code.split('/')[-1]}' 를 따랐습니다({_FREQ_KO.get(fq, '?')})"
    d = pd.Series(pd.DatetimeIndex(idx)).sort_values().diff().dt.days.dropna()
    if len(d) < 2:
        return None, "값이 너무 적어(3개 미만) 날짜 간격으로 주기를 판별하지 못했습니다"
    med = float(d.median())
    fq = "D" if med <= 3 else "W" if med <= 10 else "M" if med <= 45 else "Q" if med <= 120 else "H" if med <= 250 else "Y"
    return fq, f"받아 본 값들의 날짜 간격 중앙값이 {med:g}일이라 {_FREQ_KO[fq]} 자료로 봅니다"


def _code_problem(source: str, code: str) -> str:
    code = (code or "").strip()
    if not code:
        return "출처 코드를 넣어 주세요"
    if source == "ECOS" and (len(code.split("/")) != 3 or code.split("/")[2] not in ("D", "M", "Q", "A")):
        return "한국은행 코드는 '통계표코드/항목코드/주기' 꼴이어야 합니다(예: 722Y001/0101000/M)"
    env = _SOURCES[source]["key_env"]
    if env and not os.environ.get(env):
        return f"이 출처는 접속 키가 필요합니다 — 프로젝트 폴더의 .env 파일에 {env} 를 넣고 서버를 다시 시작하세요(키는 화면에서 받지 않습니다)"
    return ""


@app.get("/api/variables/sources")
def variable_sources() -> list[dict]:
    return [{"source": k, **v, "key_present": (v["key_env"] is None) or bool(os.environ.get(v["key_env"]))} for k, v in _SOURCES.items()]


@app.post("/api/variables/check")
def check_variable(body: VariableCheckBody) -> dict:
    """출처·코드로 실제 자료를 받아 본다. 받아지면 줄 수·기간·최근 값 5개를, 아니면 이유를 쉬운 말로."""
    why = _code_problem(body.source, body.source_code)
    if why:
        return {"ok": False, "error": why}
    try:
        sr = _probe_series(body.source, body.source_code.strip(), "D").dropna()
    except Exception as e:  # noqa: BLE001
        from adapters.base import scrub_secrets  # 오류 문구의 요청 주소에 접속 키가 들어 있다 → 지우고 보여 준다
        return {"ok": False, "error": f"이 코드로 자료를 받지 못했습니다 — 코드가 맞는지, 인터넷이 되는지 확인하세요. (원인: {scrub_secrets(f'{type(e).__name__}: {e}')[:200]})"}
    if not len(sr):
        return {"ok": False, "error": "이 코드로 자료를 받지 못했습니다 — 받은 값이 하나도 없습니다"}
    fq, basis = _infer_freq(body.source, body.source_code.strip(), sr.index)
    return {"ok": True, "rows": int(len(sr)), "first": str(pd.Timestamp(sr.index[0]).date()), "last": str(pd.Timestamp(sr.index[-1]).date()),
            "sample": [{"date": str(pd.Timestamp(d).date()), "value": float(v)} for d, v in sr.tail(5).items()],
            "freq": fq, "freq_basis": basis}


@app.post("/api/variables")
def add_variable(body: VariableBody) -> dict:
    """변수 사전(variable_master.xlsx 의 variables 시트)에 한 줄을 추가한다. 자료는 다음 '데이터 가져오기' 때 받는다.
    추가 전에 출처에서 실제로 받아 보고(받아지지 않으면 거절), 사전을 고치기 전의 사본을 data/master_history/ 에 남긴다."""
    from engine import runlog
    from engine.master_check import _bad_id
    from engine.registry import append_master_rows

    master = DATA / "variable_master.xlsx"
    if not master.exists():
        raise HTTPException(400, "변수 사전(data/variable_master.xlsx)이 없습니다")
    vid = body.id.strip()
    bad = _bad_id(vid) if vid else "변수 이름(id)을 넣어 주세요"
    if bad:
        raise HTTPException(400, f"변수 이름(id) '{vid}': {bad}")
    reg, _ = Registry.load_master(master)
    if vid.lower() in {i.lower() for i in reg.ids(usable_only=False)}:
        raise HTTPException(400, f"'{vid}' 는 이미 있는 변수 이름입니다(대소문자만 다른 것도 같은 이름으로 봅니다)")
    same = [v.id for v in (reg.get(i) for i in reg.ids(usable_only=False)) if v.source == body.source and v.source_code.strip() == body.source_code.strip()]
    if same:
        raise HTTPException(400, f"같은 출처·같은 코드의 변수가 이미 있습니다: {same[0]}")
    chk = check_variable(VariableCheckBody(source=body.source, source_code=body.source_code))
    if not chk["ok"]:
        raise HTTPException(400, chk["error"])
    fq = chk["freq"]                       # 주기는 자료로 정한다(검토 1) — 사람이 넣은 값은 '맞는지 확인'에만 쓴다
    if fq is None:
        raise HTTPException(400, f"주기를 정하지 못해 등록하지 않았습니다 — {chk['freq_basis']}. 이 변수는 엑셀(변수 사전)로 등록해 주세요")
    if body.freq is not None and body.freq != fq:
        raise HTTPException(400, f"넣은 주기({_FREQ_KO[body.freq]})가 자료의 주기({_FREQ_KO[fq]})와 다릅니다 — {chk['freq_basis']}. "
                                 "주기가 틀리면 발표시차가 잘못 잡혀 미래 값이 섞일 수 있어 등록하지 않았습니다. 주기를 비우거나 맞게 넣어 주세요")
    runlog.snapshot_master(master, DATA)  # 고치기 전 사본
    row = {"id": vid, "name": body.name.strip(), "description": "", "group": body.group.strip(), "freq": fq, "lag_days": "" if body.lag_days is None else body.lag_days,
           "lag_confirmed": "N" if body.lag_days is None else "Y", "source": body.source, "source_code": body.source_code.strip(), "default_fill": "ffill",
           "transform": body.transform, "unit": body.unit.strip(), "active": "Y",
           "note": (body.note.strip() + " " if body.note.strip() else "") + f"화면에서 등록 {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}"}
    if append_master_rows(master, [row]) != 1:
        raise HTTPException(500, "변수 사전에 쓰지 못했습니다 — 엑셀에서 파일을 열어 두었다면 닫고 다시 해 보세요")
    import logging
    logging.getLogger("tsp.api").info("변수 등록: %s (%s %s)", vid, body.source, body.source_code)
    from engine.registry import LAG_DEFAULT
    return {"added": vid, "checked": chk, "freq": fq,
            "note": f"변수 사전에 추가했습니다(주기: {_FREQ_KO[fq]} — {chk['freq_basis']}). '데이터 가져오기'를 누르면 이 변수의 자료를 받아 새 데이터 버전을 만듭니다."
                    + ("" if body.lag_days is not None else f" 발표시차(lag)는 {_FREQ_KO[fq]} 자료 기본값 {LAG_DEFAULT[fq]}일을 쓰고 '미확인'으로 표시됩니다 — 실제 발표일을 확인해 [값 보기]에서 고쳐 주세요.")}


# ---------------------------------------------------------------------------
# 변수 값 보기 · CSV · 발표시차·변환 고치기 (2026-11-13)
# ---------------------------------------------------------------------------
# 빈 기간·되돌림 점검 규칙은 engine/datacheck.py 한 곳에 둔다(2026-11-14 — 자료 점검표와 [값 보기]가 같은 규칙을 쓰게)
from engine.experiment import TRANSFORM_OPTIONS as _TRANSFORM_OPTIONS  # noqa: E402
from engine.datacheck import GAP_DAYS as _GAP_DAYS, OUTLIER_Z as _OUTLIER_Z, gap_and_reversal as _gap_and_reversal_checks  # noqa: E402


def _pick_version(version: str | None) -> str:
    vs = _versions()
    if not vs:
        raise HTTPException(404, "데이터 버전이 없습니다 — 먼저 '데이터 가져오기'를 하세요")
    if version is None:
        return vs[-1]["id"]                                              # 화면·실행의 기본과 같은 '가장 새 온전한 버전'
    if version not in {v["id"] for v in vs}:
        raise HTTPException(404, f"데이터 버전 {version} 이 없습니다")
    return version


def _series_of(vid: str, version: str | None):
    from engine.store import DataVersion
    reg, _ = _registry()
    if vid not in set(reg.ids(usable_only=False)):
        raise HTTPException(404, f"변수 {vid} 가 변수 사전에 없습니다")
    v = reg.get(vid)
    ver = _pick_version(version)
    long = DataVersion(DATA, ver).read_variables([vid])
    return v, ver, long


@app.get("/api/technical-info")
def technical_info(freq: str = "D") -> dict:
    """기술지표 설명(2026-11-14): 변수 사전에 없는 A_ 입력의 이름·묶음·계산식과 묶음별 '왜 있나·읽는 법', 엔진이 계산한 예시.
    정의는 engine/features.py 한 곳(TECH_GROUP_DOCS·PRESETS·technical_block) — 도움말 20장 5절과 같은 내용."""
    from engine.features import PRESETS, TECH_GROUP_DOCS, TECH_ORIGIN, feature_description, preset_calendar_weeks, technical_columns, technical_example, technical_group
    if freq not in PRESETS:
        raise HTTPException(400, f"주기는 {', '.join(PRESETS)} 중 하나")
    cols = technical_columns(freq)
    return {"freq": freq, "preset": {k: list(v) if isinstance(v, tuple) else v for k, v in PRESETS[freq].items()},
            "columns": [{"name": c, "group": technical_group(c), "desc": feature_description(c)} for c in cols],
            "groups": TECH_GROUP_DOCS, "example": technical_example(),
            "origin": TECH_ORIGIN, "calendar_weeks": preset_calendar_weeks()}  # 출처·주기별 창의 대략 주 수(2026-11-14)


_DC_CACHE: dict = {}


@app.get("/api/data-checks")
def data_checks(version: str | None = None) -> dict:
    """자료 점검표(2026-11-14): 사용 가능한 변수 전부의 정상성·계절성·계절조정 표시·빈 기간·확인해 볼 날과 신호.
    값을 바꾸지 않는다. 같은 데이터 버전·같은 변수 사전이면 다시 계산하지 않는다(서버 메모리에만 둠)."""
    from engine.datacheck import CROSS_RULE, RULE, check_table, cross_table
    from engine.store import DataVersion
    reg, _ = _registry()
    ver = _pick_version(version)
    mp = DATA / "variable_master.xlsx"
    key = (ver, mp.stat().st_mtime if mp.exists() else None, tuple(reg.ids()))
    if key not in _DC_CACHE:
        dv = DataVersion(DATA, ver)
        rows = check_table(dv, reg)
        cross = cross_table(dv, reg)       # 다른 출처와 대조(2026-11-14, 표시만) — duplicate_of 짝 전부
        _DC_CACHE.clear()
        _DC_CACHE[key] = {"version": ver, "rows": rows, "rule": RULE,
                          "counts": {c: sum(1 for r in rows if r["signal"] == c) for c in ("초록", "노랑", "회색")},
                          "cross": cross, "cross_rule": CROSS_RULE,
                          "cross_counts": {c: sum(1 for r in cross if r["signal"] == c) for c in ("초록", "노랑", "회색")}}
    return _DC_CACHE[key]


@app.get("/api/variables/{vid}/values")
def variable_values(vid: str, version: str | None = None) -> dict:
    """한 변수의 값(그 데이터 버전에 저장된 그대로)과 점검 결과. 점검은 알림일 뿐 값을 바꾸지 않는다.
    빈 기간 = 주기별 기준보다 긴 날짜 간격, 중복 = 같은 날짜가 두 번 이상, 확인해 볼 날 = 하루 크게(강건 z > 8) 움직였다가 다음 날 되돌아온 날,
    가장 큰 움직임 5개(참고),
    0 이하 = 변화율·로그 변환인데 0 이하 값(계산 불가)."""
    from engine.experiment import apply_transform
    v, ver, long = _series_of(vid, version)
    base = {"id": v.id, "name": v.name, "freq": v.freq, "transform": v.transform, "lag_days": v.lag_days, "lag_confirmed": v.lag_confirmed,
            "source": v.source, "source_code": v.source_code, "version": ver}
    if long.empty:
        return {**base, "n": 0, "values": [], "checks": None,
                "message": f"이 데이터 버전({ver})에는 {vid} 의 값이 없습니다 — 새로 추가한 변수라면 '데이터 가져오기'가 끝나야 생깁니다"}
    long = long.sort_values("date", kind="stable")
    dates = pd.to_datetime(long["date"])
    dup = int(dates.duplicated(keep="first").sum())
    s = pd.Series(long["value"].astype(float).to_numpy(), index=dates)
    s = s[~s.index.duplicated(keep="last")]
    msg = None
    try:
        tr = apply_transform(s, v.transform)
    except Exception as e:  # noqa: BLE001
        tr, msg = pd.Series(np.nan, index=s.index), f"변환 '{v.transform}' 을 계산하지 못했습니다: {e}"
    # '하루 크게 움직였다가 다음 날 되돌아온 날' = 자료 오류의 흔한 모양(단위·자릿수 잘못, 한 칸 밀림). 실제 급변도 걸릴 수 있어 '확인해 볼 날'로만 알린다.
    # 값이 0 근처를 오가는 흐름 자료(변환 none 이고 음수 있음, 예: 순매수대금)는 매일 크게 오가는 게 정상이라 되돌림 검사를 하지 않는다.
    flow_like = (v.transform or "none").lower() == "none" and not bool((s.dropna() > 0).all())
    chk = _gap_and_reversal_checks(s, v.freq, "0 근처를 오가는 자료라 되돌림 검사를 하지 않습니다" if flow_like else None)
    t_low = (v.transform or "none").lower()
    nonpos = int((s <= 0).sum()) if (t_low.startswith("pct") or t_low.startswith("log")) else 0
    vals = [{"date": str(t.date()), "value": (None if not np.isfinite(s[t]) else float(s[t])),
             "transformed": (None if pd.isna(tr[t]) or not np.isfinite(tr[t]) else float(tr[t]))} for t in s.index]
    # 바꾼 값의 뜻(2026-11-13 검토 13): 여기 값은 '변수의 원래 주기에서' 변환한 것이다. 엔진은 그 뒤 발표시차·날짜 맞추기를 하고,
    # 변수가 모델보다 촘촘하면(일간 변수 → 주·월간 모델, 간격을 적지 않은 pct·diff·logdiff) 결정일 사이의 변화를 다시 구한다(experiment.build_dataset).
    tr_note = ("원래 주기에서 바꾼 값입니다. 실험에서는 여기에 발표시차·날짜 맞추기를 더 하고, 이 변수보다 느린 주기의 모델(예: 일간 변수를 월간 모델에)에서는 "
               "결정일 사이의 변화를 다시 구하므로 모델에 들어가는 값과 다를 수 있습니다") if t_low not in ("", "none") else None
    from engine.datacheck import CROSS_RULE, check_variable, cross_table
    from engine.store import DataVersion
    dc = check_variable(v, long)          # 자료 점검표의 이 변수 줄(정상성·계절성·계절조정 표시, 2026-11-14)
    reg, _ = _registry()
    cross = cross_table(DataVersion(DATA, ver), reg, ids=[vid])   # 이 변수가 들어간 다른 출처 짝(2026-11-14, 표시만)
    return {**base, "n": int(len(s)), "first": str(s.index.min().date()), "last": str(s.index.max().date()), "values": vals, "message": msg, "transform_note": tr_note,
            "datacheck": dc, "cross": cross, "cross_rule": CROSS_RULE,
            "checks": {**chk, "duplicate_dates": dup, "nonpositive": nonpos, "n_nan": int(s.isna().sum())}}


@app.get("/api/variables/{vid}/values.csv")
def variable_values_csv(vid: str, version: str | None = None):
    """엑셀에서 바로 열리는 CSV(UTF-8 BOM — 한글 깨짐 방지). 날짜·저장된 값·모델에 넣는 꼴로 바꾼 값."""
    from fastapi.responses import Response
    r = variable_values(vid, version)
    lines = [f"date,value,변환값({r['transform']})"]
    for row in r["values"]:
        lines.append(f"{row['date']},{'' if row['value'] is None else repr(row['value'])},{'' if row['transformed'] is None else repr(row['transformed'])}")   # 빈 값은 빈칸('nan' 글자 아님)
    body = ("\ufeff" + "\r\n".join(lines) + "\r\n").encode("utf-8")
    return Response(content=body, media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": f'attachment; filename="{vid}_{r["version"]}.csv"'})


@app.get("/api/tickers/{tid}/values")
def ticker_values(tid: str, version: str | None = None) -> dict:
    """종목 가격(실험이 읽는 조립된 prices_<코드>) 그대로와 점검. 점검은 종가 기준 빈 기간·중복·하루 튀었다 되돌아온 날,
    0 이하 가격, 고가/저가 앞뒤 안 맞음, 네이버로 메운 날(repair_log). 값을 바꾸지 않는다."""
    ver = _pick_version(version)
    vdir = DATA / "versions" / ver
    known = set(tickers_mod.ids(DATA))
    f = vdir / f"prices_{tid}.parquet"
    if tid not in known and not f.exists():
        raise HTTPException(404, f"종목 {tid} 가 종목 목록에 없습니다")
    base = {"id": tid, "name": next((t["name"] for t in tickers_mod.load(DATA) if t["id"] == tid), ""), "version": ver}
    if not f.exists():
        return {**base, "n": 0, "values": [], "checks": None,
                "message": f"이 데이터 버전({ver})에는 {tid} 의 가격이 없습니다 — 새로 추가한 종목이라면 '데이터 가져오기'가 끝나야 생깁니다"}
    p = pd.read_parquet(f).sort_index(kind="stable")
    p.index = pd.to_datetime(p.index)
    dup = int(p.index.duplicated(keep="first").sum())
    p = p[~p.index.duplicated(keep="last")]
    close = p["close"].astype(float)
    chk = _gap_and_reversal_checks(close, "D")
    o, h, l, c = (p[k].astype(float) for k in ("open", "high", "low", "close"))
    bad = (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (h < l)
    ohlc_bad = [str(t.date()) for t in p.index[bad.fillna(False).to_numpy()]]
    rl = vdir / f"repair_log_{tid}.csv"
    repaired = 0
    if rl.exists():
        try:
            repaired = int(len(pd.read_csv(rl, encoding="utf-8-sig")))
        except Exception:  # noqa: BLE001 — 빈 파일
            repaired = 0
    nonpos = int((p[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    vals = [{"date": str(t.date()), **{k: (None if pd.isna(p.at[t, k]) else float(p.at[t, k])) for k in ("open", "high", "low", "close", "volume")}} for t in p.index]
    return {**base, "n": int(len(p)), "first": str(p.index.min().date()), "last": str(p.index.max().date()), "values": vals, "message": None,
            "checks": {**chk, "duplicate_dates": dup, "nonpositive": nonpos, "ohlc_bad": len(ohlc_bad), "ohlc_bad_dates": ohlc_bad[:20],
                       "repaired": repaired, "n_nan": int(close.isna().sum())}}


@app.get("/api/tickers/{tid}/values.csv")
def ticker_values_csv(tid: str, version: str | None = None):
    from fastapi.responses import Response
    r = ticker_values(tid, version)
    cols = ("open", "high", "low", "close", "volume")
    lines = ["date," + ",".join(cols)] + [row["date"] + "," + ",".join("" if row[k] is None else repr(row[k]) for k in cols) for row in r["values"]]
    body = ("\ufeff" + "\r\n".join(lines) + "\r\n").encode("utf-8")
    return Response(content=body, media_type="text/csv; charset=utf-8",
                    headers={"content-disposition": f'attachment; filename="prices_{tid}_{r["version"]}.csv"'})


class VariableEditBody(BaseModel):
    lag_days: int | None = Field(None, ge=0, le=800)
    lag_confirmed: bool | None = None
    transform: str | None = None
    basis: str = ""                       # 고친 근거(예: 공표일정 주소) — 비고에 날짜와 함께 남긴다


@app.patch("/api/variables/{vid}")
def update_variable(vid: str, body: VariableEditBody) -> dict:
    """변수 사전에서 발표시차(lag_days)·확인 여부·변환(transform)을 고친다. 고치기 전 사본을 data/master_history/ 에 남긴다.
    다음 실험부터 적용된다. 지난 결과는 그때 쓴 값(lag_days_used·transforms_used)과 사전 sha256 이 기록돼 있어 그대로 재현된다."""
    from engine import runlog
    from engine.experiment import _TR_RE
    from engine.registry import update_master_row

    master = DATA / "variable_master.xlsx"
    if not master.exists():
        raise HTTPException(400, "변수 사전(data/variable_master.xlsx)이 없습니다")
    reg, _ = Registry.load_master(master)
    if vid not in set(reg.ids(usable_only=False)):
        raise HTTPException(404, f"변수 {vid} 가 변수 사전에 없습니다")
    v = reg.get(vid)
    changes: dict = {}
    if body.transform is not None:
        t = body.transform.strip().lower()
        if not (t in ("none", "log", "logdiff") or _TR_RE.match(t)):
            raise HTTPException(400, f"알 수 없는 변환 '{body.transform}' — none, pct, pct5 처럼 pct+숫자, diff, diff+숫자, log, logdiff 중 하나")
        if t != (v.transform or "none").lower():
            changes["transform"] = t
    if body.lag_days is not None and (body.lag_days != v.lag_days or not v.lag_confirmed and body.lag_confirmed):
        changes["lag_days"] = int(body.lag_days)
    if body.lag_confirmed is not None and body.lag_confirmed != v.lag_confirmed:
        changes["lag_confirmed"] = "Y" if body.lag_confirmed else "N"
        if body.lag_confirmed and "lag_days" not in changes:
            changes["lag_days"] = int(v.lag_days)       # 기본값(빈칸)을 확인한 값으로 적어 둔다 — 빈칸이면 다시 '미확인'으로 읽힌다
    if not changes:
        raise HTTPException(400, "바뀐 것이 없습니다")
    label = {"lag_days": "발표시차", "lag_confirmed": "확인", "transform": "변환"}
    what = ", ".join(f"{label[k]} {getattr(v, k)}→{changes[k]}" for k in ("lag_days", "lag_confirmed", "transform") if k in changes)
    note = (v.note + " | " if v.note else "") + f"화면에서 고침 {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}: {what}" + (f"; 근거: {body.basis.strip()}" if body.basis.strip() else "")
    runlog.snapshot_master(master, DATA)  # 고치기 전 사본
    try:
        update_master_row(master, vid, {**changes, "note": note})
    except PermissionError:
        raise HTTPException(500, "변수 사전에 쓰지 못했습니다 — 엑셀에서 파일을 열어 두었다면 닫고 다시 해 보세요")
    import logging
    logging.getLogger("tsp.api").info("변수 고침: %s %s", vid, what)
    return {"updated": vid, "changed": [k for k in ("lag_days", "lag_confirmed", "transform") if k in changes],
            "note": f"{vid}: {what}. 다음 실험부터 적용됩니다(지난 결과는 그때 쓴 값이 기록돼 있어 바뀌지 않습니다)."}


_COLLECT: dict = {}


def _collect_cmd() -> list[str]:
    return [sys.executable, str(ROOT / "collect.py")]


@app.get("/api/collect")
def collect_status() -> dict:
    """데이터 가져오기의 상태: none(한 적 없음) / running / done(종료 코드와 뜻, 로그 끝 40줄)."""
    if not _COLLECT:
        return {"state": "none"}
    rc = _COLLECT["proc"].poll()
    log = Path(_COLLECT["log"])
    from adapters.base import scrub_secrets
    tail = [scrub_secrets(ln) for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]] if log.exists() else []
    out = {"state": "running" if rc is None else "done", "started": _COLLECT["started"], "log": log.name, "log_tail": tail}
    if rc is not None:
        out["returncode"] = rc
        out["message"] = {0: "모두 받았습니다. 새 데이터 버전이 만들어졌습니다.",
                          1: "종목 가격을 새로 받지 못했습니다(받은 것이 아예 없으면 새 데이터 버전도 만들지 않았습니다). 아래 기록에서 [fail] 줄을 확인하세요.",
                          3: "시작하기 전에 멈췄습니다 — 아무것도 받지 않았고 새 데이터 버전도 만들지 않았습니다. 아래 기록의 이유(변수 사전의 고칠 곳 등)를 확인하세요.",
                          2: "받지 못한 항목이 있습니다 — 그 항목은 직전 값 그대로 두었고, 이 버전은 '일부 받지 못함'으로 표시됩니다. 아래 기록에서 [fail] 줄을 확인하세요."
                          }.get(rc, f"가져오기가 오류로 끝났습니다(종료 코드 {rc}). 아래 기록을 확인하세요.")
    return out


@app.post("/api/collect")
def collect_start() -> dict:
    """collect.py 를 따로 띄운다(한 번에 하나). 실험이 도는 중이면 시작하지 않는다 — 실험 도중에 '최신 데이터 버전'이 바뀌지 않게."""
    import subprocess

    if _COLLECT and _COLLECT["proc"].poll() is None:
        raise HTTPException(409, "데이터 가져오기가 이미 진행 중입니다")
    if any(j_.get("status") in ("queued", "running") for j_ in jobs.list(limit=100000)):
        raise HTTPException(409, "실험이 실행 중이거나 대기 중입니다 — 끝난 뒤에 데이터를 가져오세요")
    d = RESULTS / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log = d / f"collect_{started}.log"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(_collect_cmd(), cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
    _COLLECT.clear(); _COLLECT.update(proc=proc, log=str(log), started=started)
    return collect_status()


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


class SplitPlanBody(BaseModel):
    config: dict
    ratios: list[float] | None = None   # [학습, 검증, 테스트](%) — 주면 날짜를 계산, 없으면 설정의 날짜로 계획만


@app.post("/api/split-plan")
def split_plan_api(body: SplitPlanBody):
    """검증 계획(학습 없음): 워크포워드는 폴드 수·첫/마지막 폴드의 자르는 순서·개수, 1회 분할은 비율이나 날짜로 나눈 구간·개수, 빈 칸 처리."""
    from engine.splitplan import validation_plan
    from engine.store import DataVersion
    try:
        cfg = dict(body.config)
        vs = _versions()
        if not vs:
            raise HTTPException(400, "데이터 버전이 없습니다")
        vid = cfg.get("data_version") or vs[-1]["id"]
        reg, _ = _registry()
        if body.ratios is not None and len(body.ratios) != 3:
            raise HTTPException(422, "비율은 [학습, 검증, 테스트] 세 개입니다")
        return validation_plan(cfg, DataVersion(DATA, vid), reg, ratios=body.ratios)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"분할 계획 오류: {e}")


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
        return preview(r, stem, DataVersion(DATA, vid), reg, _sealed_log(mode), vs[-1]["id"] if vs else None, res_dir=results_dir(RESULTS, mode))
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
    # 같은 '설정'으로 비교한다(해시). 후보 이름으로만 비교하면 같은 설정이 다른 이름의 결과로도 저장돼 있을 때(다시 돌린 결과 등) 둘 다 접수됐다(독립 검토 F10)
    def _job_hash(j: dict) -> str | None:
        try:
            return ExperimentConfig(**json.loads(Path(j["config_path"]).read_text(encoding="utf-8"))).config_hash()
        except Exception:  # noqa: BLE001
            return None
    h_now = cfg.config_hash()
    pending = [j for j in jobs.list(limit=100000) if j.get("sealed") and j.get("status") in ("queued", "running")
               and (j.get("sealed_source") == body.stem or _job_hash(j) == h_now)]
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
def get_logs(date: str | None = None, tail: int = 400, run: str | None = None, level: str | None = None, q: str | None = None, kind: str = "tsp"):
    """개발 로그 보기. date=YYYYMMDD(기본 최신), run=작업 ID 일부, level=DEBUG/INFO/WARNING/ERROR(이상), q=포함 문자열.
    kind=tsp(개발 로그, 기본) / access(접속 기록 — 2026-11-17)."""
    from engine.logs import LOG_DIR
    if kind not in ("tsp", "access"):
        raise HTTPException(400, "kind 는 tsp 또는 access")
    if date is not None and not (len(date) == 8 and date.isdigit()):
        raise HTTPException(400, "date 는 YYYYMMDD")
    files = sorted(p.name for p in LOG_DIR.glob(f"{kind}_*.log")) if LOG_DIR.exists() else []
    name = f"{kind}_{date}.log" if date else (files[-1] if files else None)
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
            lvn = parts[2] if len(parts) > 2 and kind == "tsp" else "INFO"
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
        # 학습 창 평균 기준(2026-11-10): '평균을 아는 것 말고 예측 정보가 더 있는가'(CW, 한쪽 p) · 표본 밖 R²
        "cw_vs_mean": pm.get("cw_vs_mean"), "cw_mean_p": pm.get("cw_vs_mean_p"), "dm_vs_mean": pm.get("dm_vs_mean"), "dm_mean_p": pm.get("dm_vs_mean_p"),
        "r2_oos": pm.get("r2_oos_vs_mean"), "pred_sd_ratio": pm.get("pred_sd_ratio"),
        "n_seeds": ((r.get("model_diag") or {}).get("seeds") or {}).get("n_seeds"),
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
    """다중비교 보정(Holm–Bonferroni) — 가족 정의와 계산은 engine/multitest.py(통계 재계산 도구와 같은 함수).
    'naive 보다 낫다'의 한쪽 p(dm_naive_p_better)에 적용한다: 양측 p 는 '더 나쁘다'일 때도 작아서, 보정값만 보고 '유의하게 좋다'로 읽을 수 있었다(2026-11-10 F13).
    p_holm = 가족(같은 대상·k·타겟 종류·정보 시점) 안 보정, p_holm_all = 이 모드의 지금 엔진 결과 전부를 한 가족으로 본 보정. 봉인 평가·옛 계산 방식 결과는 제외."""
    from engine.multitest import better_one_sided, holm

    for r in rows:
        if "error" not in r:
            r["dm_naive_p_better"] = better_one_sided(r.get("dm_vs_naive"), r.get("dm_naive_p"))
    holm(rows, "dm_naive_p_better", "p_holm", skip=lambda r: "error" in r or r.get("sealed") or r.get("stale"))
    for r in rows:
        if "p_holm_m" in r:
            r["family_size"] = r["p_holm_m"]


@app.get("/api/config-guide")
def config_guide():
    """실험 설정(JSON)의 모든 칸과 뜻·기본값·선택지 — 설정 스키마(ExperimentConfig 의 Field description)에서 그대로 만든다(2026-11-20)."""
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
    return {"fields": out}


@app.get("/api/results/{stem}/logs.zip")
def result_logs_zip(stem: str):
    """이 결과에 관한 로그를 한 번에 내려받기(2026-11-20): 결과 JSON, 작업 설정·작업 출력 로그, 개발 로그(tsp_*.log)에서 이 실행의 줄,
    실행 기록(run_log.jsonl)의 줄, 실행 시간대의 접속 기록. 로그 파일은 바꾸지 않고 읽기만 한다."""
    import io
    import zipfile
    from datetime import datetime, timedelta
    from engine.logs import LOG_DIR
    from api.jobs import CONFIG_DIR, JOBS_DIR
    p = _find_result(stem)
    if p is None:
        raise HTTPException(404, "결과 없음")
    r = json.loads(p.read_text(encoding="utf-8"))
    rr = r.get("run_record") or {}
    run_id, job_id = rr.get("run_id"), rr.get("job_id")
    if not job_id and JOBS_DIR.exists():   # 옛 결과: 작업 기록의 result_path 로 찾는다
        for jp in JOBS_DIR.glob("*.json"):
            try:
                j = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if str(j.get("result_path") or "").replace("\\", "/").endswith(p.name):
                job_id = j.get("id") or jp.stem
                break

    def local(ts: str | None):
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        except Exception:  # noqa: BLE001
            return None
    t0 = local(rr.get("started_utc"))
    t1 = local(rr.get("finished_utc")) or t0
    days: list[str] = []
    if t0:
        d = t0.date() - timedelta(days=1)
        while d <= (t1 or t0).date() + timedelta(days=1):
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
    tags = [f"[run={x}]" for x in (job_id, run_id) if x]
    readme = [f"결과 {stem} 의 로그 묶음 — 만든 시각 {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"run_id={run_id} · job_id={job_id} · 실행 {t0} ~ {t1} (서버 시각)", "",
              "result.json            결과 전체(설정·기간·폴드·지표·실행 기록)",
              "job_config.json        화면이 보낸 실험 설정(작업)",
              "job_output.log         실험 프로세스 화면 출력 전체(results/jobs/<job_id>.log)",
              "dev_log_this_run.log   개발 로그(logs/tsp_*.log)에서 이 실행 태그([run=…])가 붙은 줄",
              "run_log.jsonl          실행 기록(results/<mode>/run_log.jsonl)에서 이 run_id 의 줄",
              "access_during_run.log  실행 시간대의 접속 기록(logs/access_*.log)", ""]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("result.json", p.read_text(encoding="utf-8"))
        missing = []
        if job_id and (CONFIG_DIR / f"{job_id}.json").exists():
            z.writestr("job_config.json", (CONFIG_DIR / f"{job_id}.json").read_text(encoding="utf-8", errors="replace"))
        else:
            missing.append("job_config.json — 작업 설정 파일 없음(CLI 로 직접 실행했거나 옛 결과)")
        if job_id and (JOBS_DIR / f"{job_id}.log").exists():
            z.writestr("job_output.log", (JOBS_DIR / f"{job_id}.log").read_text(encoding="utf-8", errors="replace"))
        else:
            missing.append("job_output.log — 작업 출력 로그 없음")
        dev, acc = [], []
        for day in days:
            fp = LOG_DIR / f"tsp_{day}.log"
            if fp.exists() and tags:
                keep = False
                for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line[:4].isdigit():
                        keep = any(t in line for t in tags)
                    if keep:
                        dev.append(line)
            ap = LOG_DIR / f"access_{day}.log"
            if ap.exists() and t0:
                lo, hi = t0 - timedelta(minutes=1), (t1 or t0) + timedelta(minutes=1)
                for line in ap.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    if lo <= ts <= hi:
                        acc.append(line)
        z.writestr("dev_log_this_run.log", "\n".join(dev) + ("\n" if dev else ""))
        if not dev:
            missing.append("dev_log_this_run.log 이 비어 있음 — 개발 로그가 생기기 전(2026-11-06) 결과이거나 로그 파일이 지워짐")
        z.writestr("access_during_run.log", "\n".join(acc) + ("\n" if acc else ""))
        rl = []
        rp = p.parent / "run_log.jsonl"
        if rp.exists() and run_id:
            rl = [ln for ln in rp.read_text(encoding="utf-8", errors="replace").splitlines() if run_id in ln]
        z.writestr("run_log.jsonl", "\n".join(rl) + ("\n" if rl else ""))
        z.writestr("README.txt", "\n".join(readme + (["빠진 것:"] + ["- " + m for m in missing] if missing else [])) + "\n")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_logs.zip"'})


@app.get("/api/results/{stem}/package.zip")
def result_package_zip(stem: str):
    """결과 꾸러미(2026-11-21): 사전(설정·모델·입력 자료와 뜻)·사중(실행 기록·폴드·로그)·사후(결과·지표와 뜻·일별 결과)와
    Word 보고서를 zip 하나로. engine/report_pack.py. 결과·로그는 읽기만 한다."""
    from engine.logs import LOG_DIR
    from engine.report_pack import build_package
    from api.jobs import CONFIG_DIR, JOBS_DIR
    p = _find_result(stem)
    if p is None:
        raise HTTPException(404, "결과 없음")
    reg, _ = _registry()
    data = build_package(stem, p, data_root=DATA, registry=reg, log_dir=LOG_DIR, jobs_dir=JOBS_DIR, config_dir=CONFIG_DIR)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_package.zip"'})


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
    sh = (1 + d["shannon_return"].fillna(0)).cumprod() if "shannon_return" in d else None   # 비교 기준 섀넌 50:50(2026-11-15 이후 결과만)
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
        "shannon": None if sh is None else sh.round(5).tolist(),
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
    # 이름만 받는다(2026-11-14): 전에는 '..' 만 막아, 윈도우에서 'F:\\…' 같은 드라이브 경로나 절대 경로를 주면 도움말 폴더 밖의 .md 를 읽을 수 있었다.
    if not doc_id or any(c in doc_id for c in "/\\:") or ".." in doc_id:
        raise HTTPException(404, "문서 없음")
    p = GUIDE_BUILD / f"{doc_id}.md"
    if not p.exists() or p.resolve().parent != GUIDE_BUILD.resolve():
        raise HTTPException(404, "문서 없음")
    return {"id": doc_id, "markdown": p.read_text(encoding="utf-8")}


# 정적 화면 (web/dist 가 있으면 같은 포트에서 서빙)
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
