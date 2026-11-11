"""작업 큐(파일 기반, 1단계 본인용). 실험 1건 = 작업 1개 = 하위 프로세스 `python cli.py run <config>`.

- 큐는 메모리 + results/jobs/<job_id>.json(상태 영속). 서버 재시작 시 'queued'/'running' 이던 작업은 'interrupted' 로 바뀐다.
- 동시에 1개만 실행(GPU·CPU 경합 방지). 로그는 results/jobs/<job_id>.log.
- 공개 서비스 단계에서 RQ+Redis 로 바꿀 때 이 모듈만 교체한다(API 는 JobStore 인터페이스만 쓴다).
"""
from __future__ import annotations

import json
import os
import logging
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "results" / "jobs"
CONFIG_DIR = ROOT / "results" / "jobs" / "configs"


jlog = logging.getLogger("tsp.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._current: str | None = None
        for p in JOBS_DIR.glob("*.json"):  # 재시작 복구
            j = json.loads(p.read_text(encoding="utf-8"))
            if j["status"] in ("queued", "running"):
                j["status"] = "interrupted"
                j["finished_at"] = _now()
                p.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # -- 공개 인터페이스 ---------------------------------------------------
    def submit(self, config: dict, *, sealed: bool = False, kind: str = "run", sealed_note: str = "", sealed_source: str | None = None) -> dict:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        cfg_path = CONFIG_DIR / f"{job_id}.json"
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
        job = {"id": job_id, "kind": kind, "name": config.get("name") or config.get("model"), "config_path": str(cfg_path), "sealed": sealed,
               "mode": config.get("mode", "trading"), "status": "queued", "submitted_at": _now(), "started_at": None, "finished_at": None, "returncode": None, "result_path": None, "model": config.get("model"),
               **({"sealed_note": sealed_note, "sealed_source": sealed_source} if sealed else {})}
        self._write(job)
        with self._lock:
            self._queue.append(job_id)
            qlen = len(self._queue)
        jlog.info("작업 접수 %s: 모델 %s · 모드 %s · 봉인 %s · 대기 %d번째 · 설정 %s", job_id, job["model"], job["mode"], sealed, qlen, cfg_path)
        return job

    def get(self, job_id: str) -> dict | None:
        p = JOBS_DIR / f"{job_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def list(self, limit: int = 100) -> list[dict]:
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in JOBS_DIR.glob("*.json")]
        jobs.sort(key=lambda j: j["submitted_at"], reverse=True)
        return jobs[:limit]

    def log(self, job_id: str, tail: int = 200) -> str:
        p = JOBS_DIR / f"{job_id}.log"
        if not p.exists():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
                j = self.get(job_id); j["status"] = "cancelled"; j["finished_at"] = _now(); self._write(j)
                return True
            if self._current == job_id and self._proc is not None:
                self._proc.terminate()
                return True
        return False

    # -- 내부 -------------------------------------------------------------
    def _write(self, job: dict) -> None:
        (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")

    def _loop(self) -> None:
        while True:
            job_id = None
            with self._lock:
                if self._queue:
                    job_id = self._queue.pop(0)
                    self._current = job_id
            if job_id is None:
                time.sleep(0.5)
                continue
            job = self.get(job_id)
            job["status"] = "running"; job["started_at"] = _now(); self._write(job)
            cmd = [sys.executable, str(ROOT / "cli.py"), "run", job["config_path"]] + (["--sealed"] if job.get("sealed") else [])
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TSP_JOB_ID=job_id)  # 하위 프로세스 로그의 run= 에 같은 ID
            if job.get("sealed"):  # 봉인 평가: 화면에서 적은 사전 기록·후보 결과를 cli 가 봉인 기록(sealed_log.json)과 결과에 남긴다
                env["TSP_SEALED_NOTE"] = job.get("sealed_note") or ""
                env["TSP_SEALED_SOURCE"] = job.get("sealed_source") or ""
            jlog.info("작업 시작 %s: %s", job_id, " ".join(cmd[1:]))
            # 하위 프로세스 출력을 파일(results/jobs/<id>.log)과 서버 콘솔(cmd 창) 양쪽에 그대로 흘린다 — 디버깅용
            with open(JOBS_DIR / f"{job_id}.log", "w", encoding="utf-8") as logf:
                try:
                    print(f"\n[작업 시작] {job_id} :: {' '.join(cmd[1:])}", flush=True)
                    self._proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
                    for line in self._proc.stdout:
                        logf.write(line); logf.flush()
                        print(f"[{job_id[:8]}] {line}", end="", flush=True)
                    rc = self._proc.wait()
                    print(f"[작업 종료] {job_id} returncode={rc}", flush=True)
                    jlog.info("작업 종료 %s: returncode=%s", job_id, rc)
                except Exception as e:  # noqa: BLE001
                    logf.write(f"\n작업 실행 실패: {e}\n"); rc = -1
            job = self.get(job_id)
            job["returncode"] = rc; job["finished_at"] = _now()
            job["status"] = "done" if rc == 0 else ("cancelled" if job["status"] == "cancelled" or rc in (-15, 15) else "failed")
            # 결과 파일 경로: cli 가 마지막에 출력한 saved 경로
            try:
                tail = self.log(job_id, 60)
                i = tail.rfind('"saved": "')
                if i >= 0:
                    job["result_path"] = tail[i + 10: tail.index('"', i + 10)]
            except Exception:  # noqa: BLE001
                pass
            self._write(job)
            if job["status"] == "failed":
                print(f"[작업 실패] {job_id} — 전체 로그: {JOBS_DIR / (job_id + '.log')}", flush=True)
            with self._lock:
                self._current = None; self._proc = None
