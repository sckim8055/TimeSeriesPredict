"""실행 기록 — 실험을 다른 사람이 나중에 그대로 다시 돌릴 수 있도록 실행 전·중·후를 남긴다(2026-11-09).

사전: 설정 전체(결과 json 의 config), 변수 사전 파일의 sha256 과 그 사본, 데이터 버전과 그 수집 기록, 변수 치환·제외 내역,
      코드·라이브러리·파이썬·장치, 실행 명령.
사중: results/<mode>/run_log.jsonl 에 시작 줄 → 끝 줄(완료/실패와 오류 메시지)을 덧붙인다. 지우거나 고치지 않는다.
사후: 같은 이름의 결과를 다시 저장하면 옛 결과를 results/<mode>/history/ 로 옮긴다(덮어써서 잃지 않음).

변수 사전 사본: data/master_history/<sha256 앞 12자리>.xlsx. 같은 내용이면 한 번만 복사한다.
결과의 run_record.master_sha256 으로 그때의 사전을 찾아 다시 쓸 수 있다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from engine.store import sha256_file

ROOT = Path(__file__).resolve().parent.parent
LIBS = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lightgbm", "torch", "pydantic", "pyarrow", "openpyxl")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def library_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in LIBS:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 — 설치 안 된 라이브러리는 기록만 비움
            out[name] = None
    return out


def snapshot_master(path: str | Path, data_root: str | Path | None = None, *, raw: bytes | None = None, sha: str | None = None) -> dict:
    """변수 사전의 sha256 과 사본(data/master_history/<sha 12자리>.<확장자>). 같은 내용이면 한 번만 쓴다.
    raw·sha 를 주면 디스크를 다시 읽지 않고 그 바이트로 사본을 만든다 — 사전을 읽은 뒤 파일이 바뀌어도 '실제로 쓴 사전'이 남는다(2026-11-10)."""
    p = Path(path)
    if raw is None:
        if not p.exists():
            return {"path": str(p), "sha256": None, "snapshot": None}
        raw = p.read_bytes()
    h = sha or hashlib.sha256(raw).hexdigest()
    hist = Path(data_root or ROOT / "data") / "master_history"
    hist.mkdir(parents=True, exist_ok=True)
    snap = hist / f"{h[:12]}{p.suffix}"
    if not snap.exists():
        tmp = snap.with_name(snap.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, snap)
    return {"path": str(p), "sha256": h, "snapshot": f"master_history/{snap.name}"}


def data_version_info(dv) -> dict:
    """데이터 버전 id 와 manifest 의 sha256, 마지막 수집 기록(언제·어느 사전으로 모았나)."""
    mf = Path(dv.dir) / "manifest.json"
    info = {"id": dv.id, "manifest_sha256": sha256_file(mf) if mf.exists() else None}
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        runs = m.get("collect_runs") or []
        info["created_utc"] = m.get("created_utc")
        info["n_collect_runs"] = len(runs)
        if runs:
            last = runs[-1]
            info["last_collect"] = {k: last.get(k) for k in ("started_utc", "finished_utc", "sources", "start", "end", "master_sha256")}
        info["collect_master_sha256"] = m.get("master_sha256")
    return info


def build_run_record(*, cfg, dv, reg, variables: dict | None, override_notes: list[str], started_utc: str, device: str | None) -> dict:
    path = getattr(reg, "source_path", "")
    master = {"path": path or None, "sha256": getattr(reg, "source_sha256", "") or None, "snapshot": None}
    if path:
        try:  # 사본을 못 써도(디스크·권한) 끝난 실험 결과를 버리지 않는다 — 오류를 기록에 남긴다
            master = snapshot_master(path, raw=getattr(reg, "source_bytes", None), sha=getattr(reg, "source_sha256", "") or None)
        except OSError as e:
            master["error"] = f"사본을 만들지 못함: {type(e).__name__}: {e}"
    dvi = data_version_info(dv)
    notes = []
    if str(path).lower().endswith((".xlsx", ".xlsm")) and master["sha256"] and dvi.get("collect_master_sha256") and master["sha256"] != dvi["collect_master_sha256"]:
        notes.append("변수 사전이 데이터 수집 뒤 바뀌었습니다(값은 데이터 버전, 정의(lag·변환·active)는 지금 사전 기준). "
                     "두 사전 모두 master_history 에 사본이 있으면 비교할 수 있습니다.")
    return {
        "run_id": os.environ.get("TSP_RUN_ID") or new_run_id(),
        "job_id": os.environ.get("TSP_JOB_ID") or None,
        "started_utc": started_utc,
        "finished_utc": None,
        "command": sys.argv,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine(), "device": device},
        "libraries": library_versions(),
        "master": master,
        "data_version": dvi,
        "variables": variables or {"requested": [], "used": [], "substituted": [], "merged": [], "excluded": []},
        "override_notes": override_notes,
        "notes": notes,
    }


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def append_log(results_mode_dir: str | Path, entry: dict) -> None:
    """results/<mode>/run_log.jsonl 에 한 줄 덧붙인다(지우거나 고치지 않음)."""
    d = Path(results_mode_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": now_utc(), **entry}, ensure_ascii=False, default=str) + "\n")


def archive_existing(out_dir: Path, stem: str, tag: str = "") -> list[str]:
    """같은 이름의 결과(json·일별 parquet)를 history/<stem>__<옛 저장 시각>[_tag].* 로 옮기고 옮긴 파일 목록을 돌려준다.
    둘 중 하나만 남아 있어도(짝 잃은 parquet) 옮긴다. 도중에 실패하면 이미 옮긴 것을 되돌리고 오류를 올린다 — 반쯤 옮겨진 상태를 남기지 않는다."""
    j, q = out_dir / f"{stem}.json", out_dir / f"{stem}_daily.parquet"
    present = [p for p in (q, j) if p.exists()]  # parquet 먼저: json 이 남아 있는 한 '결과가 있다'로 보인다
    if not present:
        return []
    ts = datetime.fromtimestamp((j if j.exists() else q).stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hist = out_dir / "history"
    hist.mkdir(exist_ok=True)
    label, n = f"{stem}__{ts}" + (f"_{tag}" if tag else ""), 1
    while any((hist / p.name.replace(stem, label if n == 1 else f"{label}_{n}", 1)).exists() for p in (q, j)):
        n += 1
    label = label if n == 1 else f"{label}_{n}"
    moved: list[tuple[Path, Path]] = []
    try:
        for src in present:
            dst = hist / src.name.replace(stem, label, 1)
            src.rename(dst)
            moved.append((src, dst))
    except OSError:
        for src, dst in reversed(moved):
            try:
                dst.rename(src)
            except OSError:
                pass
        raise
    return [str(dst.relative_to(out_dir)).replace("\\", "/") for _, dst in moved]
