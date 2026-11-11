"""개발용 로그 — 값이 어디서 어디로 넘어가는지 단계별로 남긴다.

어디에 남나
  logs/tsp_YYYYMMDD.log    모든 프로세스(API 서버, 실험 하위 프로세스, CLI)의 로그를 한 파일에(날짜별). 기본 DEBUG 까지.
  results/jobs/<id>.log    작업 하나의 화면 출력(INFO 이상 + 기존 print). 실행 화면의 '로그' 와 같다.
  콘솔                      INFO 이상.
  환경변수 TSP_LOG_LEVEL=DEBUG|INFO|WARNING 으로 파일 수준을, TSP_LOG_CONSOLE=DEBUG 로 콘솔 수준을 바꾼다.

형식(한 줄): 2026-11-06 10:12:33.123 DEBUG [run=20261106_101230_ab12] tsp.engine.data | 메시지
  - run=: 작업 ID(화면 작업이면 jobs 의 id, CLI 직접 실행이면 cli-시각). 같은 실험의 줄을 grep 으로 모을 수 있다.
  - 이름(tsp.api / tsp.jobs / tsp.cli / tsp.engine.* / tsp.model.*)으로 어느 단계인지 안다.

값 요약: describe(x) — 배열·표·시리즈를 '모양·자료형·빈 칸 수·최소/평균/최대·날짜 범위' 한 줄로. 값 자체를 통째로 찍지 않는다(용량·개인정보).
단계 표시: with step("이름", **값): ... — 시작·끝·걸린 시간과 넘긴 값 요약을 남긴다. 예외가 나면 그 단계 이름과 함께 기록.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
_RUN = {"id": "-"}
_CONFIGURED = {"done": False}


class _RunFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run = _RUN["id"]
        return True


def set_run(run_id: str | None) -> None:
    _RUN["id"] = run_id or "-"


def setup(component: str, run_id: str | None = None) -> logging.Logger:
    """프로세스마다 한 번. component: api / job / cli / test."""
    if run_id:
        set_run(run_id)
    root = logging.getLogger("tsp")
    if _CONFIGURED["done"]:
        return logging.getLogger(f"tsp.{component}")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s [run=%(run)s] %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    flt = _RunFilter()
    try:
        LOG_DIR.mkdir(exist_ok=True)
        fh = logging.FileHandler(LOG_DIR / f"tsp_{datetime.now():%Y%m%d}.log", encoding="utf-8")
        fh.setLevel(getattr(logging, os.environ.get("TSP_LOG_LEVEL", "DEBUG").upper(), logging.DEBUG))
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        root.addHandler(fh)
    except OSError:  # 읽기 전용 폴더 등 — 콘솔만
        pass
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, os.environ.get("TSP_LOG_CONSOLE", "INFO").upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
    ch.addFilter(flt)
    root.addHandler(ch)
    _CONFIGURED["done"] = True
    lg = logging.getLogger(f"tsp.{component}")
    lg.debug("로그 시작: 프로세스 %s pid=%d 파일수준=%s", component, os.getpid(), os.environ.get("TSP_LOG_LEVEL", "DEBUG"))
    return lg


def describe(x: Any, name: str | None = None) -> str:
    """값 한 줄 요약. numpy 배열 / pandas Series·DataFrame / dict / list / 스칼라."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # pragma: no cover
        return repr(x)[:200]
    pre = f"{name}: " if name else ""
    try:
        if isinstance(x, pd.DataFrame):
            idx = f" {x.index[0]:%Y-%m-%d}~{x.index[-1]:%Y-%m-%d}" if len(x) and isinstance(x.index, pd.DatetimeIndex) else ""
            v = x.select_dtypes("number")
            nn = int(v.isna().sum().sum()) if v.size else 0
            cols = list(x.columns)
            colstr = ",".join(map(str, cols[:8])) + (f",…(+{len(cols) - 8})" if len(cols) > 8 else "")
            return f"{pre}DataFrame{x.shape}{idx} 빈칸 {nn} 열[{colstr}]"
        if isinstance(x, pd.Series):
            idx = f" {x.index[0]:%Y-%m-%d}~{x.index[-1]:%Y-%m-%d}" if len(x) and isinstance(x.index, pd.DatetimeIndex) else ""
            s = pd.to_numeric(x, errors="coerce")
            ok = s.dropna()
            st = f" 최소 {ok.min():.6g} 평균 {ok.mean():.6g} 최대 {ok.max():.6g}" if len(ok) else ""
            return f"{pre}Series[{len(x)}]{idx} 빈칸 {int(s.isna().sum())}{st}"
        if isinstance(x, np.ndarray):
            if x.dtype.kind in "fiub" and x.size:
                f = x.astype(float)
                fin = f[np.isfinite(f)]
                st = f" 최소 {fin.min():.6g} 평균 {fin.mean():.6g} 최대 {fin.max():.6g}" if fin.size else ""
                return f"{pre}ndarray{x.shape} {x.dtype} 빈칸 {int(np.isnan(f).sum())}{st}"
            return f"{pre}ndarray{x.shape} {x.dtype}"
        if isinstance(x, dict):
            s = json.dumps(x, ensure_ascii=False, default=str)
            return pre + (s if len(s) <= 600 else s[:600] + f"…(+{len(s) - 600}자)")
        if isinstance(x, (list, tuple)):
            return f"{pre}{type(x).__name__}[{len(x)}] {str(list(x)[:10])[:300]}"
    except Exception as e:  # noqa: BLE001 — 요약 실패가 실행을 멈추면 안 된다
        return f"{pre}<요약 실패 {type(e).__name__}>"
    return f"{pre}{x!r}"[:300]


@contextlib.contextmanager
def step(logger: logging.Logger, name: str, level: int = logging.DEBUG, **values: Any):
    """with step(log, "데이터셋 만들기", cfg=cfg): ... → 시작(넘긴 값 요약)·끝(걸린 시간) 두 줄. 예외는 단계 이름과 함께 ERROR."""
    if values:
        logger.log(level, "▶ %s | %s", name, " · ".join(describe(v, k) for k, v in values.items()))
    else:
        logger.log(level, "▶ %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as e:
        logger.error("✖ %s 실패 (%.2fs): %s: %s", name, time.perf_counter() - t0, type(e).__name__, e, exc_info=True)
        raise
    logger.log(level, "◀ %s (%.2fs)", name, time.perf_counter() - t0)
