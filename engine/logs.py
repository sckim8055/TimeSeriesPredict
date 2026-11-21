"""개발용 로그 — 값이 어디서 어디로 넘어가는지 단계별로 남긴다.

어디에 남나
  logs/tsp_YYYYMMDD.log    모든 프로세스(API 서버, 실험 하위 프로세스, CLI)의 로그를 한 파일에(날짜별). 기본 DEBUG 까지.
                           2026-11-17: 노트북 시각으로 날짜가 바뀌면(자정 뒤 첫 기록) 새 날짜 파일에 쓴다 — 전에는 서버를 켠 날 파일에 계속 썼다.
                           쓰기가 실패하면(드라이브가 잠깐 끊겼다 붙는 등) 파일을 다시 열어 한 번 더 쓴다 — 전에는 서버를 다시 켤 때까지 계속 실패했다.
  logs/access_YYYYMMDD.log 접속 기록(2026-11-17, API 서버만): 요청마다 한 줄 — 시각·보낸 주소·경로(노트북 직접 / 외부 Funnel)·방법·주소·결과·걸린 시간·브라우저 정보.
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
LOG_DIR = Path(os.environ.get("TSP_LOG_DIR") or (ROOT / "logs"))   # 시험에서 임시 폴더로 바꿀 수 있게(2026-11-17)
_RUN = {"id": "-"}
_CONFIGURED = {"done": False}


class _RunFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run = _RUN["id"]
        return True


def set_run(run_id: str | None) -> None:
    _RUN["id"] = run_id or "-"


class DailyFileHandler(logging.Handler):
    """날짜별 파일 + 실패하면 다시 열기(2026-11-17).
    · 기록할 때마다 지금 날짜(노트북 현지 시각)로 파일 이름을 정한다 — 날짜가 바뀌면 앞 파일을 닫고 새 파일을 연다(자정 뒤 첫 기록 때 생김).
    · 쓰기·비우기(flush)가 OSError 로 실패하면 파일을 닫고 다시 열어 한 번 더 쓴다. 그래도 실패하면 콘솔(stderr)에 10분에 한 번만 짧게 알린다
      (전에는 기록할 때마다 긴 예외 추적이 콘솔을 덮었다 — 2026-11-17 사용자 화면)."""

    def __init__(self, directory: Path, prefix: str, clock=None):
        super().__init__()
        self.directory, self.prefix = Path(directory), prefix
        self.clock = clock or datetime.now
        self._day: str | None = None
        self._f = None
        self._last_warn: float | None = None   # 마지막 알림 시각(처음엔 없음 — monotonic 은 부팅 뒤 초라 0 과 비교하면 안 됨)
        self.n_failed = 0

    def path_for(self, day: str) -> Path:
        return self.directory / f"{self.prefix}_{day}.log"

    def _open(self, day: str) -> None:
        self._close()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path_for(day), "a", encoding="utf-8")
        self._day = day

    def _close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
        self._f = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
        except Exception:
            self.handleError(record)
            return
        day = self.clock().strftime("%Y%m%d")
        with self.lock:
            for attempt in (1, 2):
                try:
                    if self._f is None or day != self._day:
                        self._open(day)
                    self._f.write(msg)
                    self._f.flush()
                    return
                except OSError as e:
                    self._close()
                    if attempt == 2:
                        self.n_failed += 1
                        now = time.monotonic()
                        if self._last_warn is None or now - self._last_warn > 600:
                            self._last_warn = now
                            import sys
                            try:
                                sys.stderr.write(f"[로그] {self.path_for(day)} 에 쓰지 못함({e.__class__.__name__}: {e}) — 드라이브 연결을 확인하세요. 실패 {self.n_failed}번째(10분에 한 번만 알림)\n")
                            except Exception:
                                pass

    def close(self) -> None:
        with self.lock:
            self._close()
        super().close()


def access_logger() -> logging.Logger:
    """접속 기록(logs/access_날짜.log). 개발 로그(tsp_날짜.log)와 따로 — 위쪽 로거로 올려 보내지 않는다."""
    lg = logging.getLogger("tsp_access")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        lg.propagate = False
        h = DailyFileHandler(LOG_DIR, "access")
        h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
    return lg


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
        fh = DailyFileHandler(LOG_DIR, "tsp")   # 날짜가 바뀌면 새 파일, 실패하면 다시 열기(2026-11-17)
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
