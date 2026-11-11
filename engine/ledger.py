"""시도 원장 — 개발 시도 횟수 N(Deflated Sharpe 의 N)의 근거를 지울 수 없게 남긴다(2026-11-10).

전에는 results/<mode>/experiment_counter.json 의 숫자 하나였다: 무엇을 셌는지 알 수 없고, 평문이라 고치거나 지울 수 있었고(화면 서버의 삭제 API 로도),
같은 설정을 다시 돌려도 +1 이었다. 이제는 results/<mode>/trial_ledger.jsonl 에 시도마다 한 줄을 덧붙인다.

  · 줄마다 seq(차례), utc, kind, config_hash, name, model, run_id, result, prev(바로 앞 줄의 sha256) — 앞 줄을 고치거나 빼면 뒤 줄의 prev 가 맞지 않게 된다(verify).
  · kind: genesis(원장 시작 — 그 전의 횟수는 옛 카운터 값으로만 알 수 있음) · trial(실험) · ablation(변수 제거 실험) · ensemble · sealed(봉인 평가 — 개발 시도로 세지 않음) · deleted(결과 삭제 기록)
  · 개발 시도 수 N = genesis 의 count + (trial·ablation·ensemble 줄 수). 같은 설정의 재실행도 한 줄이다(다시 본 것도 본 것).
    서로 다른 설정의 수는 distinct_configs() 로 따로 센다(원장 시작 뒤의 것만 알 수 있음).
  · experiment_counter.json 은 화면 표시용 사본으로 계속 쓰지만 근거는 원장이다. 사본이 지워져도 N 은 변하지 않는다.

한계(정직하게): 로컬 파일이라 원장 전체를 지우고 새로 만들면 막을 수 없다. 그 경우 genesis 줄의 시각이 새로 찍히고, git 에 올린 원장과 달라져 드러난다.
즉 '고칠 수 없다'가 아니라 '고치면 드러난다'이다. 원장은 git 으로 관리할 것.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER = "trial_ledger.jsonl"
COUNTER = "experiment_counter.json"
DEV_KINDS = ("trial", "ablation", "ensemble")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _read(res_dir: str | Path) -> list[str]:
    p = Path(res_dir) / LEDGER
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def entries(res_dir: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in _read(res_dir)]


def _append(res_dir: Path, entry: dict) -> dict:
    lines = _read(res_dir)
    e = {"seq": len(lines), "utc": _now(), **entry, "prev": _sha(lines[-1]) if lines else None}
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / LEDGER, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    return e


def ensure(res_dir: str | Path) -> None:
    """원장이 없으면 genesis 줄로 시작한다. 그 전의 횟수는 옛 카운터 값(없으면 0)이고, 그때 폴더에 있던 결과 이름을 함께 적는다."""
    d = Path(res_dir)
    if (d / LEDGER).exists():
        return
    c = d / COUNTER
    try:
        n0 = int(json.loads(c.read_text(encoding="utf-8")).get("count", 0)) if c.exists() else 0
    except Exception:  # noqa: BLE001 — 깨진 카운터는 0 으로 두고 그 사실을 적는다
        n0 = 0
    stems = sorted(p.stem for p in d.glob("*.json") if p.name not in (COUNTER, "sealed_log.json")) if d.exists() else []
    _append(d, {"kind": "genesis", "count": n0,
                "note": "원장 시작. 이전 시도 수는 experiment_counter.json 의 값이며 개별 내역은 없다(같은 설정의 재실행·지운 결과 포함).",
                "results_on_disk": stems})


def count(res_dir: str | Path) -> int:
    """개발 시도 수 N. 원장이 없으면 옛 카운터 값(읽기만 — 원장을 만들지 않는다)."""
    d = Path(res_dir)
    es = entries(d)
    if not es:
        c = d / COUNTER
        try:
            return int(json.loads(c.read_text(encoding="utf-8")).get("count", 0)) if c.exists() else 0
        except Exception:  # noqa: BLE001
            return 0
    base = next((int(e.get("count", 0)) for e in es if e.get("kind") == "genesis"), 0)
    return base + sum(1 for e in es if e.get("kind") in DEV_KINDS)


def record(res_dir: str | Path, kind: str, **fields) -> int:
    """원장에 한 줄을 덧붙이고 개발 시도 수 N 을 돌려준다. 화면용 사본(experiment_counter.json)도 맞춘다."""
    d = Path(res_dir)
    ensure(d)
    _append(d, {"kind": kind, **fields})
    n = count(d)
    try:
        (d / COUNTER).write_text(json.dumps({"count": n}), encoding="utf-8")
    except OSError:
        pass  # 사본은 표시용 — 못 써도 원장이 근거
    return n


def distinct_configs(res_dir: str | Path) -> int:
    """원장 시작 뒤 돌린 서로 다른 설정(config_hash)의 수 — 같은 설정의 재실행은 한 번으로."""
    return len({e.get("config_hash") for e in entries(res_dir) if e.get("kind") in DEV_KINDS and e.get("config_hash")})


def verify(res_dir: str | Path) -> dict:
    """사슬 검사: 각 줄의 prev 가 바로 앞 줄의 sha256 과 같은지, seq 가 0부터 이어지는지."""
    lines = _read(res_dir)
    for i, ln in enumerate(lines):
        try:
            e = json.loads(ln)
        except Exception:  # noqa: BLE001
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": "줄을 읽을 수 없음"}
        if e.get("seq") != i:
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": f"차례 번호가 {e.get('seq')} (기대 {i}) — 줄이 빠졌거나 끼어듦"}
        want = _sha(lines[i - 1]) if i else None
        if e.get("prev") != want:
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": "앞 줄이 바뀌었음(prev 불일치)"}
    return {"ok": True, "lines": len(lines), "n_dev_trials": count(res_dir)}
