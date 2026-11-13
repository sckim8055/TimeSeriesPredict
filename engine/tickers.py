"""종목 목록 — data/tickers.json (2026-11-12).

수집기(collect.py)가 가격을 받을 종목과, 화면에서 고를 수 있는 대상·묶어 학습 종목의 목록.
파일이 없으면 기본 둘(KODEX 반도체·TIGER 반도체). 화면의 '종목 추가'가 이 파일에 한 줄을 보탠다(고치기 전 사본: data/tickers_history/).

kind: "etf"   — 공식 출처(data.go.kr ETF 시세)·야후·pykrx 로 받는다
      "stock" — 공식 출처는 ETF 전용이라 묻지 않고 야후(+pykrx 교차검증)로 받는다
코드는 한국거래소 6자리 숫자만(야후 기호는 '<코드>.KS' — 코스피 종목. 코스닥 종목(.KQ)은 아직 지원하지 않는다).
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT = [{"id": "091160", "name": "KODEX 반도체", "kind": "etf"},
           {"id": "091230", "name": "TIGER 반도체", "kind": "etf"}]
KINDS = ("etf", "stock")
FILE = "tickers.json"


def load(data_dir: str | Path) -> list[dict]:
    p = Path(data_dir) / FILE
    if not p.exists():
        return [dict(t) for t in DEFAULT]
    rows = json.loads(p.read_text(encoding="utf-8"))
    return [{"id": str(r["id"]), "name": str(r.get("name", "")), "kind": str(r.get("kind", "etf"))} for r in rows]


def ids(data_dir: str | Path) -> list[str]:
    return [t["id"] for t in load(data_dir)]


def problem(data_dir: str | Path, tid: str, kind: str = "etf") -> str:
    """추가하면 안 되는 이유(없으면 빈 문자열)."""
    tid = (tid or "").strip()
    if not re.fullmatch(r"\d{6}", tid):
        return "종목 코드는 한국거래소 6자리 숫자여야 합니다(예: 005930)"
    if kind not in KINDS:
        return f"종류는 {', '.join(KINDS)} 중 하나여야 합니다"
    if tid in ids(data_dir):
        return f"{tid} 은(는) 이미 목록에 있습니다"
    return ""


def add(data_dir: str | Path, tid: str, name: str = "", kind: str = "etf") -> dict:
    why = problem(data_dir, tid, kind)
    if why:
        raise ValueError(why)
    d = Path(data_dir); d.mkdir(parents=True, exist_ok=True)
    p = d / FILE
    if p.exists():
        h = d / "tickers_history"; h.mkdir(exist_ok=True)
        shutil.copy2(p, h / f"tickers_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
    rows = load(d) + [{"id": tid.strip(), "name": (name or "").strip(), "kind": kind}]
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)
    return rows[-1]
